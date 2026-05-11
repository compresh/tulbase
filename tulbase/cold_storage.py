"""Cold Storage — content-addressed file write/read.

Layer 3 of the 3-layer memory hierarchy. The original content of every
compressed entry is stored on disk, addressed by sha256 hash. Same content
that recurs in many turns is stored once.

Layout (from phase22-spec.md):

    {root}/
      {session_id}/
        {hash[:2]}/
          {hash}.bin          -- raw original bytes
          {hash}.meta.json    -- minimal metadata

Default root: ~/.compresh/cold/  (override via env `COMPRESH_COLD_PATH`).

Encryption is **out of scope for this MVP** — content is written in plain.
A future tier will add AES-256-GCM with per-session keys (see spec §2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_COLD_ROOT_ENV = "COMPRESH_COLD_PATH"
DEFAULT_COLD_ROOT = "~/.compresh/cold"

# Allowed characters for session_id components: alphanumeric, dash, underscore,
# colon, dot. Anything else is rejected to prevent path-traversal attempts
# (e.g. "../../etc/passwd").
_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9_:.-]+$")
# Allowed characters for hash: hex only (sha256 output).
_SAFE_HASH_RE = re.compile(r"^[0-9a-f]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_root(root: Optional[str | os.PathLike[str]]) -> Path:
    """Resolve the cold-storage root, expanding `~` and consulting env var."""
    if root is None:
        env = os.environ.get(DEFAULT_COLD_ROOT_ENV)
        root = env if env else DEFAULT_COLD_ROOT
    return Path(root).expanduser().resolve()


def sha256_hex(data: bytes) -> str:
    """Return the hex-encoded sha256 of bytes. Stable across runs."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ColdObject:
    """A reference to a stored object on cold disk."""

    hash: str
    relative_path: str  # e.g. "v1-abdullah/ab/abc123...bin"
    absolute_path: Path
    size: int


class ColdStorage:
    """Content-addressed blob store for compressed content originals."""

    def __init__(self, root: Optional[str | os.PathLike[str]] = None):
        self.root = _resolve_root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("cold_storage root=%s", self.root)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _shard(self, session_id: str, hash_: str) -> Path:
        if not hash_ or len(hash_) < 2:
            raise ValueError(f"invalid hash: {hash_!r}")
        if not _SAFE_HASH_RE.match(hash_):
            raise ValueError(f"hash must be hex only: {hash_!r}")
        if not _SAFE_SESSION_RE.match(session_id):
            raise ValueError(
                f"invalid session_id (path-traversal protection): {session_id!r}"
            )
        candidate = (self.root / session_id / hash_[:2]).resolve()
        # Defense-in-depth: even with the regex check, ensure resolved path
        # cannot escape the cold root.
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise ValueError(
                f"resolved path escapes cold root: {candidate}"
            ) from e
        return candidate

    def _bin_path(self, session_id: str, hash_: str) -> Path:
        return self._shard(session_id, hash_) / f"{hash_}.bin"

    def _meta_path(self, session_id: str, hash_: str) -> Path:
        return self._shard(session_id, hash_) / f"{hash_}.meta.json"

    def _relative(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def save(
        self,
        session_id: str,
        content: str | bytes,
        *,
        mime: str = "text/plain",
        extra_meta: Optional[dict] = None,
    ) -> ColdObject:
        """Persist `content` and return a ColdObject reference.

        Idempotent: if the same hash already exists, the existing file is
        kept and metadata is left untouched. Useful for dedup across turns.
        """
        if not session_id:
            raise ValueError("session_id is required")
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        hash_ = sha256_hex(data)
        bin_p = self._bin_path(session_id, hash_)
        meta_p = self._meta_path(session_id, hash_)

        if bin_p.exists():
            logger.info(
                "cold_storage hit (dedup) session=%s hash=%s bytes=%d",
                session_id,
                hash_,
                len(data),
            )
            return ColdObject(
                hash=hash_,
                relative_path=self._relative(bin_p),
                absolute_path=bin_p,
                size=len(data),
            )

        bin_p.parent.mkdir(parents=True, exist_ok=True)
        # Write bytes + meta atomically-ish (write tmp, rename).
        tmp = bin_p.with_suffix(".bin.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, bin_p)

        meta = {
            "hash": hash_,
            "mime": mime,
            "size": len(data),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        logger.info(
            "cold_storage saved session=%s hash=%s bytes=%d path=%s",
            session_id,
            hash_,
            len(data),
            bin_p,
        )
        return ColdObject(
            hash=hash_,
            relative_path=self._relative(bin_p),
            absolute_path=bin_p,
            size=len(data),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read_bytes(self, session_id: str, hash_: str) -> bytes:
        """Return the raw bytes for a stored object. Raises if missing."""
        p = self._bin_path(session_id, hash_)
        if not p.exists():
            raise FileNotFoundError(
                f"cold object not found: session={session_id} hash={hash_}"
            )
        return p.read_bytes()

    def read_text(
        self,
        session_id: str,
        hash_: str,
        *,
        encoding: str = "utf-8",
    ) -> str:
        """Return content decoded as text. Raises on missing or decode error."""
        return self.read_bytes(session_id, hash_).decode(encoding)

    def read_meta(self, session_id: str, hash_: str) -> dict:
        """Return the .meta.json payload."""
        p = self._meta_path(session_id, hash_)
        if not p.exists():
            raise FileNotFoundError(
                f"cold meta not found: session={session_id} hash={hash_}"
            )
        return json.loads(p.read_text())

    def exists(self, session_id: str, hash_: str) -> bool:
        return self._bin_path(session_id, hash_).exists()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def delete(self, session_id: str, hash_: str) -> bool:
        """Remove both .bin and .meta.json. Returns True if anything was removed."""
        bin_p = self._bin_path(session_id, hash_)
        meta_p = self._meta_path(session_id, hash_)
        removed = False
        if bin_p.exists():
            bin_p.unlink()
            removed = True
        if meta_p.exists():
            meta_p.unlink()
            removed = True
        if removed:
            logger.info(
                "cold_storage deleted session=%s hash=%s",
                session_id,
                hash_,
            )
        return removed

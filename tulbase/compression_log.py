"""Compression Log — DuckDB schema + CRUD.

Layer 2 of the 3-layer memory hierarchy. Indexed metadata about each
compressed entry. Lives in the same DuckDB file as `usage_logs` (the
existing production proxy DB), but in its own table `compression_log`.

Schema (from phase22-spec.md):

    id              TEXT PRIMARY KEY     -- compr-{session_id}-{turn}-{modality}-{counter}
    session_id      TEXT NOT NULL
    turn_idx        INTEGER NOT NULL
    modality        TEXT NOT NULL        -- code | terminal_output | json_dump
    -- | image | doc | quote_detail
    summary         TEXT NOT NULL
    reason          TEXT NOT NULL
    hash            TEXT NOT NULL        -- sha256 of original content
    size_orig       INTEGER NOT NULL
    size_compressed INTEGER NOT NULL
    retrievable     BOOLEAN NOT NULL DEFAULT TRUE
    pii_filtered    BOOLEAN NOT NULL DEFAULT FALSE
    created_at      TIMESTAMP NOT NULL
    cold_path       TEXT
    metadata        JSON

All queries use parameter binding (DuckDB `?` placeholders) — never string
formatting — to avoid SQL injection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

from .provenance import Provenance

try:  # DuckDB is part of the existing proxy stack.
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover — duckdb is a runtime dependency.
    duckdb = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompressionEntry:
    """A single row of the compression_log table.

    `id` follows the pattern `compr-{session_id}-T{turn_idx}-{modality}-{counter}`.
    `hash` is the sha256 of the *original* content (cold storage key).
    `cold_path` is relative to the cold storage root.

    ``provenance`` (TUL 1.0) is persisted inside ``metadata["provenance"]``
    so the DB schema stays unchanged. The dataclass surface exposes it
    as a typed field — callers do not need to touch metadata directly.
    """

    id: str
    session_id: str
    turn_idx: int
    modality: str
    summary: str
    reason: str
    hash: str
    size_orig: int
    size_compressed: int
    retrievable: bool = True
    pii_filtered: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    cold_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (dates as ISO strings).

        Embeds ``provenance`` into ``metadata["provenance"]`` so the
        on-disk form (DuckDB metadata column) round-trips correctly.
        """
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        # Move typed provenance into metadata for persistence.
        prov = d.pop("provenance", None)
        if prov is not None:
            metadata = dict(d.get("metadata") or {})
            # asdict serializes Provenance.fetched_at as a datetime; we
            # need the JSON-friendly form.
            if self.provenance is not None:
                metadata["provenance"] = self.provenance.to_dict()
            d["metadata"] = metadata
        return d

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "CompressionEntry":
        """Build from a DuckDB row tuple ordered like `_COLUMNS`."""
        (
            id_,
            session_id,
            turn_idx,
            modality,
            summary,
            reason,
            hash_,
            size_orig,
            size_compressed,
            retrievable,
            pii_filtered,
            created_at,
            cold_path,
            metadata,
        ) = row
        metadata_dict: dict[str, Any] = (
            json.loads(metadata)
            if isinstance(metadata, str) and metadata
            else (metadata or {})
        )
        prov_data = metadata_dict.get("provenance")
        prov = Provenance.from_dict(prov_data) if prov_data else None
        return cls(
            id=id_,
            session_id=session_id,
            turn_idx=int(turn_idx),
            modality=modality,
            summary=summary,
            reason=reason,
            hash=hash_,
            size_orig=int(size_orig),
            size_compressed=int(size_compressed),
            retrievable=bool(retrievable),
            pii_filtered=bool(pii_filtered),
            created_at=(
                created_at
                if isinstance(created_at, datetime)
                else datetime.fromisoformat(str(created_at))
            ),
            cold_path=cold_path,
            metadata=metadata_dict,
            provenance=prov,
        )


_COLUMNS = (
    "id",
    "session_id",
    "turn_idx",
    "modality",
    "summary",
    "reason",
    "hash",
    "size_orig",
    "size_compressed",
    "retrievable",
    "pii_filtered",
    "created_at",
    "cold_path",
    "metadata",
)


# ---------------------------------------------------------------------------
# CRUD wrapper
# ---------------------------------------------------------------------------


class CompressionLog:
    """DuckDB-backed CRUD wrapper for the compression_log table.

    Construct with either a `duckdb.DuckDBPyConnection` (preferred — share the
    proxy's existing connection pool) or a path string to open a new file.

    Example:
        conn = duckdb.connect("tulbase.duckdb")
        log = CompressionLog(conn)
        log.ensure_schema()
        log.save(entry)
    """

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS compression_log (
      id              TEXT PRIMARY KEY,
      session_id      TEXT NOT NULL,
      turn_idx        INTEGER NOT NULL,
      modality        TEXT NOT NULL,
      summary         TEXT NOT NULL,
      reason          TEXT NOT NULL,
      hash            TEXT NOT NULL,
      size_orig       INTEGER NOT NULL,
      size_compressed INTEGER NOT NULL,
      retrievable     BOOLEAN NOT NULL DEFAULT TRUE,
      pii_filtered    BOOLEAN NOT NULL DEFAULT FALSE,
      created_at      TIMESTAMP NOT NULL,
      cold_path       TEXT,
      metadata        JSON
    );
    """

    _INDEX_SQL = (
        "CREATE INDEX IF NOT EXISTS idx_compression_session_turn "
        "ON compression_log(session_id, turn_idx);",
        "CREATE INDEX IF NOT EXISTS idx_compression_hash "
        "ON compression_log(hash);",
    )

    def __init__(self, conn_or_path: "duckdb.DuckDBPyConnection | str"):
        if duckdb is None:
            raise RuntimeError(
                "duckdb is not installed. Install it via `pip install duckdb`."
            )
        if isinstance(conn_or_path, str):
            self._conn = duckdb.connect(conn_or_path)
            self._owns_conn = True
        else:
            self._conn = conn_or_path
            self._owns_conn = False

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        """Create table + indexes if missing. Idempotent."""
        self._conn.execute(self._SCHEMA_SQL)
        for idx in self._INDEX_SQL:
            self._conn.execute(idx)
        logger.info("compression_log schema ensured")

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    def save(self, entry: CompressionEntry) -> None:
        """Insert a new compression entry. Raises on duplicate `id`.

        If ``entry.provenance`` is set, it is folded into
        ``metadata["provenance"]`` so the wider TUL 1.0 metadata
        round-trips through the existing JSON column (no schema change
        needed).
        """
        # Merge provenance into metadata for persistence.
        meta_for_db = dict(entry.metadata) if entry.metadata else {}
        if entry.provenance is not None:
            meta_for_db["provenance"] = entry.provenance.to_dict()

        self._conn.execute(
            """
            INSERT INTO compression_log
              (id, session_id, turn_idx, modality, summary, reason, hash,
               size_orig, size_compressed, retrievable, pii_filtered,
               created_at, cold_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                entry.id,
                entry.session_id,
                entry.turn_idx,
                entry.modality,
                entry.summary,
                entry.reason,
                entry.hash,
                entry.size_orig,
                entry.size_compressed,
                entry.retrievable,
                entry.pii_filtered,
                entry.created_at,
                entry.cold_path,
                json.dumps(meta_for_db) if meta_for_db else None,
            ],
        )
        logger.info(
            "compression_log saved id=%s session=%s turn=%d modality=%s",
            entry.id,
            entry.session_id,
            entry.turn_idx,
            entry.modality,
        )

    def save_many(self, entries: Iterable[CompressionEntry]) -> int:
        """Bulk insert. Returns number of inserted rows."""
        count = 0
        for e in entries:
            self.save(e)
            count += 1
        return count

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get(self, entry_id: str) -> Optional[CompressionEntry]:
        """Lookup by primary key. Returns None if missing."""
        cur = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM compression_log WHERE id = ?;",
            [entry_id],
        )
        row = cur.fetchone()
        return CompressionEntry.from_row(row) if row else None

    def list_by_session(
        self,
        session_id: str,
        *,
        turn_min: Optional[int] = None,
        turn_max: Optional[int] = None,
        modality: Optional[str] = None,
        limit: int = 1000,
    ) -> list[CompressionEntry]:
        """List entries for a session, optionally filtered by turn range / modality."""
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if turn_min is not None:
            clauses.append("turn_idx >= ?")
            params.append(turn_min)
        if turn_max is not None:
            clauses.append("turn_idx <= ?")
            params.append(turn_max)
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)
        params.append(limit)
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM compression_log "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY turn_idx ASC, id ASC LIMIT ?;"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [CompressionEntry.from_row(r) for r in rows]

    def find_by_hash(self, hash_: str) -> list[CompressionEntry]:
        """All entries that point at the same cold-storage object."""
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM compression_log WHERE hash = ?;",
            [hash_],
        ).fetchall()
        return [CompressionEntry.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Update / Delete (rare in production — log is append-only)
    # ------------------------------------------------------------------
    def mark_pii_filtered(self, entry_id: str) -> None:
        """Flip pii_filtered=true and retrievable=false."""
        self._conn.execute(
            "UPDATE compression_log SET pii_filtered = TRUE, retrievable = FALSE "
            "WHERE id = ?;",
            [entry_id],
        )
        logger.info("compression_log marked pii_filtered id=%s", entry_id)

    def delete(self, entry_id: str) -> None:
        """Hard delete (rare — used for GDPR right-to-erasure)."""
        self._conn.execute(
            "DELETE FROM compression_log WHERE id = ?;",
            [entry_id],
        )
        logger.info("compression_log deleted id=%s", entry_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    def __enter__(self) -> "CompressionLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

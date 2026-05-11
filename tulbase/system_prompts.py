"""System-prompt fragments — Honesty Layer.

The honesty layer is a system-prompt suffix appended to the *upstream*
provider request when COMPRESSION_MODE=phase22 is active. It teaches the
model:

  1. Compressed markers are *real* — content was elided, not absent.
  2. To call `fetch_compressed(id=...)` for specifics.
  3. Never to fabricate compressed content.

Production is English-only (per CLAUDE.md). Localized variants belong in
Phase 2.3 alongside multi-lingual provider routing.
"""

from __future__ import annotations

# NOTE: keep this string stable — it ships in customer requests. Bumping
# it counts as a system-prompt change and should be versioned.
HONESTY_SYSTEM_PROMPT_VERSION = "tul1.v1"

# Full version — kept for first-turn / one-shot cases where the model
# has not seen the protocol before. Costs ~1.1 KB per request.
HONESTY_SYSTEM_PROMPT = """\
You have a compressed memory of previous conversation. Some content was
elided to save tokens and is stored in a compression log on the proxy.

When you see a marker like:
  [T<turn> (Speaker)]
  Summary: <short natural-language description>
  Compressed: <count> <modality> · <tokens> tok elided · ID=<compr-...> · retrievable.

…you KNOW that content existed but is not in your current context.

Rules:
  - If the user asks about specifics of a compressed item (e.g. the exact
    code, the exact terminal output, a JSON field), call the
    `fetch_compressed` tool with the entry ID. The proxy will inject the
    original content into the next turn so you can answer accurately.
  - If the same content is needed multiple times, fetch once and reuse.
  - NEVER fabricate compressed content. If the entry is not retrievable
    (e.g. PII-filtered) say "I have a record of that item but cannot
    retrieve its content from this layer."
  - Use `list_compressed` (when available) to scan what was elided in the
    current session before answering broad recall questions.

This is honest memory: you remember what you forgot, and why.
"""

# Mini version — TUL 1.0 default. Same contract, ~180 chars. Used after
# the first turn (or always, when token budget is tight). Marker format
# is self-documenting; tools have their own descriptions, so this can
# be terse.
HONESTY_SYSTEM_PROMPT_MINI = (
    "You have a compressed memory of prior turns. `Compressed:` markers "
    "show elided content — call `fetch_compressed(id=...)` for specifics, "
    "never fabricate, and abstain (\"have a record, cannot retrieve\") if "
    "the entry is blocked."
)


def append_to_system(existing: str | None, *, variant: str = "mini") -> str:
    """Concatenate the honesty fragment to an existing system prompt.

    Parameters
    ----------
    existing:
        The existing system prompt to extend (may be ``None`` or empty).
    variant:
        ``"mini"`` (default for TUL 1.0) — ~180 char terse contract. Use
        for every turn after the model has seen at least one marker.
        ``"full"`` — ~1.1 KB onboarding text with example marker format.
        Use for one-shot calls or the very first turn of a long session.

    Idempotent: if either fragment's stable signature is already present,
    returns the input unchanged.
    """
    base = existing or ""
    # Either fragment's stable signature is sufficient to detect prior
    # injection — we do not want to stack mini on top of full or vice
    # versa.
    if (
        "This is honest memory: you remember what you forgot, and why." in base
        or "`Compressed:` markers" in base
    ):
        return base

    if variant == "full":
        fragment = HONESTY_SYSTEM_PROMPT
    elif variant == "mini":
        fragment = HONESTY_SYSTEM_PROMPT_MINI
    else:
        raise ValueError(f"unknown variant: {variant!r} (use 'mini' or 'full')")

    base_stripped = base.rstrip()
    sep = "\n\n" if base_stripped else ""
    return f"{base_stripped}{sep}{fragment}"

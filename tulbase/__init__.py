"""tulbase — depth-aware context compression for LLM proxies."""

from .cold_storage import ColdStorage
from .compression_log import CompressionEntry, CompressionLog
from .modality import Segment, classify, resolve_overlaps
from .summarizer import Tier1Summarizer, SummarizerResult
from .turn_box import TurnBox, CompressedRef, render_markdown, render_markdown_many
from .retrieval import Retriever, RetrievalResult
from .pipeline import Pipeline, PipelineResult
from .backfill import Backfiller, BackfillResult, backfill_messages
from .provenance import (
    Channel, CHANNEL_VALUES, Provenance, TrustLevel, TRUST_VALUES,
    default_trust, to_pipeline_speaker,
)
from .system_prompts import (
    HONESTY_SYSTEM_PROMPT,
    HONESTY_SYSTEM_PROMPT_MINI,
    HONESTY_SYSTEM_PROMPT_VERSION,
    append_to_system,
)
from .tools import (
    FETCH_COMPRESSED_TOOL,
    LIST_COMPRESSED_TOOL,
    FETCH_COMPRESSED_TOOL_ANTHROPIC,
    LIST_COMPRESSED_TOOL_ANTHROPIC,
    all_tools,
    all_tools_anthropic,
)

__all__ = [
    "ColdStorage", "CompressionEntry", "CompressionLog",
    "Segment", "classify", "resolve_overlaps",
    "Tier1Summarizer", "SummarizerResult",
    "TurnBox", "CompressedRef", "render_markdown", "render_markdown_many",
    "Retriever", "RetrievalResult",
    "Pipeline", "PipelineResult",
    "Backfiller", "BackfillResult", "backfill_messages",
    "Channel", "CHANNEL_VALUES", "Provenance",
    "TrustLevel", "TRUST_VALUES", "default_trust", "to_pipeline_speaker",
    "HONESTY_SYSTEM_PROMPT", "HONESTY_SYSTEM_PROMPT_MINI",
    "HONESTY_SYSTEM_PROMPT_VERSION", "append_to_system",
    "FETCH_COMPRESSED_TOOL", "LIST_COMPRESSED_TOOL",
    "FETCH_COMPRESSED_TOOL_ANTHROPIC", "LIST_COMPRESSED_TOOL_ANTHROPIC",
    "all_tools", "all_tools_anthropic",
]

__version__ = "0.2.0"

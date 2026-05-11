"""OpenAI-compatible tool definitions exposed to the upstream provider.

Two tools (spec §6):

  - fetch_compressed(id, max_tokens?)     — required, hot path
  - list_compressed(...)                  — optional, for broad recall

These follow the OpenAI / Anthropic tool-call format. The proxy is
responsible for splicing them into the request `tools` array when
COMPRESSION_MODE=phase22 is active.
"""

from __future__ import annotations

from typing import Any

FETCH_COMPRESSED_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_compressed",
        "description": (
            "Retrieve original content of a compressed entry by ID. "
            "Use when answering specifics about a compressed item "
            "(code, terminal output, JSON dump, quote details). "
            "If the entry is not retrievable, say so — do not fabricate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": (
                        "Compression entry ID (format: compr-...) shown in "
                        "the live-context Compressed marker."
                    ),
                },
                "max_tokens": {
                    "type": "integer",
                    "description": (
                        "Max tokens of original content to inject into the "
                        "next turn. Defaults to 2000."
                    ),
                    "default": 2000,
                    "minimum": 1,
                    "maximum": 32000,
                },
            },
            "required": ["id"],
        },
    },
}


LIST_COMPRESSED_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_compressed",
        "description": (
            "List compressed entries in the current session. Filter by "
            "turn range and/or modality. Use for broad recall before "
            "answering questions like \"what did we cover in turns 10-20?\"."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "turn_min": {
                    "type": "integer",
                    "description": "Lower bound of turn index (inclusive).",
                },
                "turn_max": {
                    "type": "integer",
                    "description": "Upper bound of turn index (inclusive).",
                },
                "modality": {
                    "type": "string",
                    "description": (
                        "Filter by modality "
                        "(code | terminal_output | json_dump | stack_trace)."
                    ),
                    "enum": [
                        "code",
                        "terminal_output",
                        "json_dump",
                        "stack_trace",
                    ],
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of entries to return. Default 100.",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "required": [],
        },
    },
}


def all_tools(*, include_list: bool = True) -> list[dict[str, Any]]:
    """Return the tool list (OpenAI format) to splice into a provider request."""
    tools = [FETCH_COMPRESSED_TOOL]
    if include_list:
        tools.append(LIST_COMPRESSED_TOOL)
    return tools


# ---------------------------------------------------------------------------
# Anthropic tool format (Claude messages.create)
# ---------------------------------------------------------------------------
# Anthropic uses a flat shape: {name, description, input_schema} — same JSON
# Schema for parameters but no `function` wrapper and no `type: "function"`.

FETCH_COMPRESSED_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": FETCH_COMPRESSED_TOOL["function"]["name"],
    "description": FETCH_COMPRESSED_TOOL["function"]["description"],
    "input_schema": FETCH_COMPRESSED_TOOL["function"]["parameters"],
}

LIST_COMPRESSED_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": LIST_COMPRESSED_TOOL["function"]["name"],
    "description": LIST_COMPRESSED_TOOL["function"]["description"],
    "input_schema": LIST_COMPRESSED_TOOL["function"]["parameters"],
}


def all_tools_anthropic(*, include_list: bool = True) -> list[dict[str, Any]]:
    """Return the tool list in Anthropic Messages API format."""
    tools = [FETCH_COMPRESSED_TOOL_ANTHROPIC]
    if include_list:
        tools.append(LIST_COMPRESSED_TOOL_ANTHROPIC)
    return tools

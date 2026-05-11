import tiktoken

# Cache encoders per model
_encoders: dict[str, tiktoken.Encoding] = {}


def _get_encoder(model: str) -> tiktoken.Encoding:
    """Get or create tiktoken encoder for a model."""
    if model not in _encoders:
        try:
            _encoders[model] = tiktoken.encoding_for_model(model)
        except (KeyError, Exception):
            # Fallback to cl100k_base, or rough estimate if tiktoken unavailable
            try:
                _encoders[model] = tiktoken.get_encoding("cl100k_base")
            except Exception:
                _encoders[model] = None  # Will use word-based estimation
    return _encoders[model]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate when tiktoken is unavailable (~1.3 tokens per word)."""
    return max(1, int(len(text.split()) * 1.3))


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a text string."""
    enc = _get_encoder(model)
    if enc is None:
        return _estimate_tokens(text)
    return len(enc.encode(text, disallowed_special=()))


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """
    Count tokens for a list of chat messages.
    Follows OpenAI's token counting convention.
    """
    enc = _get_encoder(model)
    tokens_per_message = 3  # <|start|>role\ncontent<|end|>
    total = 0

    def _encode(text: str) -> int:
        if enc is None:
            return _estimate_tokens(text)
        return len(enc.encode(text, disallowed_special=()))

    for msg in messages:
        total += tokens_per_message
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _encode(content)
        elif isinstance(content, list):
            # Multimodal: count text parts only
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += _encode(part.get("text", ""))
        role = msg.get("role", "")
        total += _encode(role)
        if msg.get("name"):
            total += _encode(msg["name"]) + 1
    total += 3  # reply priming
    return total

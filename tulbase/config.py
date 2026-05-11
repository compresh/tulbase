"""
Tulbase configuration — environment variables and defaults.
"""

import os


class Settings:
    """Proxy settings, loaded from environment variables."""

    # Server
    HOST: str = os.getenv("TULBASE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("TULBASE_PORT", "8000"))
    DEBUG: bool = os.getenv("TULBASE_DEBUG", "false").lower() == "true"

    # Optimizer
    OPTIMIZER_ENABLED: bool = os.getenv("TULBASE_OPTIMIZER", "true").lower() == "true"
    OPTIMIZER_MIN_TOKENS: int = int(os.getenv("TULBASE_MIN_TOKENS", "50"))

    # Injection detection
    INJECTION_ENABLED: bool = os.getenv("TULBASE_INJECTION", "true").lower() == "true"
    INJECTION_THRESHOLD: float = float(os.getenv("TULBASE_INJECTION_THRESHOLD", "0.70"))
    # sanitize | block | log
    INJECTION_ACTION: str = os.getenv("TULBASE_INJECTION_ACTION", "sanitize")

    # Compression
    # aggressive | balanced | conservative | none
    COMPRESSION_MODE: str = os.getenv("TULBASE_COMPRESSION", "balanced")

    # Logging
    LOG_LEVEL: str = os.getenv("TULBASE_LOG_LEVEL", "info")
    LOG_FILE: str = os.getenv("TULBASE_LOG_FILE", "")  # empty = stdout only

    # Provider defaults (user can override per-request via Authorization header)
    DEFAULT_BASE_URL: str = os.getenv("TULBASE_DEFAULT_BASE_URL", "https://api.openai.com/v1")
    DEFAULT_API_KEY: str = os.getenv("TULBASE_DEFAULT_API_KEY", "")

    # Optional: ML-enhanced features (install extras)
    # pip install tulbase[injection]      → DeBERTa injection + LLMLingua-2 compression
    # pip install tulbase[summarizer] → KeyBERT tag extraction
    KEYBERT_ENABLED: bool = False
    LLMLINGUA_ENABLED: bool = False
    DEBERTA_ENABLED: bool = False

    def __init__(self):
        # Auto-detect optional dependencies
        try:
            import keybert  # noqa: F401
            self.KEYBERT_ENABLED = True
        except ImportError:
            pass

        try:
            from llmlingua import PromptCompressor  # noqa: F401
            self.LLMLINGUA_ENABLED = True
        except ImportError:
            pass

        try:
            from transformers import AutoModelForSequenceClassification  # noqa: F401
            self.DEBERTA_ENABLED = True
        except ImportError:
            pass


_settings = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

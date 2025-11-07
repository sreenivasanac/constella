"""Constella package initialization with basic logging setup."""

from __future__ import annotations

import logging


logging.basicConfig(level=logging.INFO)


class _LiteLLMMessageFilter(logging.Filter):
    """Strip leading newlines from LiteLLM logs and drop empty messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "LiteLLM":
            return True
        message = record.getMessage()
        stripped = message.lstrip("\n") if isinstance(message, str) else message
        if not stripped:
            return False
        if stripped != message:
            record.msg = stripped
            record.args = ()
        return True


_litellm_logger = logging.getLogger("LiteLLM")
_litellm_logger.addFilter(_LiteLLMMessageFilter())
_litellm_logger.propagate = False

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

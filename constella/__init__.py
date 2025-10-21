"""Constella package initialization with basic logging setup."""

from __future__ import annotations

import logging


logging.basicConfig(level=logging.INFO)

_litellm_logger = logging.getLogger("LiteLLM")
_litellm_logger.propagate = False

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

"""Constella package initialization with basic logging setup."""

from __future__ import annotations

import logging


logging.basicConfig(level=logging.INFO)

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

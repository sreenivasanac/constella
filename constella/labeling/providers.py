"""LLM provider configuration helpers for auto-labeling."""

from __future__ import annotations

import os
from typing import Any, Dict


def get_provider_config(provider: str) -> Dict[str, Any]:
    """Return provider-specific configuration for LiteLLM.
    
    Configures API keys, base URLs, and custom provider settings based on the
    specified provider name. Currently supports OpenAI and Fireworks AI.
    
    Args:
        provider: Provider name (case-insensitive). Supported: "openai", "fireworks".
    
    Returns:
        Dictionary of kwargs to pass to litellm.completion() or litellm.acompletion().
        Returns empty dict for unsupported providers.
    
    Raises:
        RuntimeError: If the provider is specified but required API key is not set
            in environment variables.
    
    Examples:
        >>> config = get_provider_config("openai")
        >>> # Returns: {"api_key": "...", "custom_llm_provider": "openai"}
        
        >>> config = get_provider_config("fireworks")
        >>> # Returns: {"api_key": "...", "api_base": "...", "custom_llm_provider": "openai"}
    """
    provider_normalized = (provider or "").lower()
    
    if provider_normalized == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI provider selected but OPENAI_API_KEY is not set."
            )
        return {
            "api_key": api_key,
            "custom_llm_provider": "openai",
        }
    
    if provider_normalized == "fireworks":
        api_key = os.environ.get("FIREWORKS_AI_API_KEY") or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Fireworks provider selected but FIREWORKS_AI_API_KEY or FIREWORKS_API_KEY is not set."
            )
        api_base = os.environ.get("FIREWORKS_AI_API_BASE", "https://api.fireworks.ai/inference/v1")
        return {
            "api_key": api_key,
            "api_base": api_base,
            "custom_llm_provider": "openai",
        }
    
    return {}


__all__ = [
    "get_provider_config",
]

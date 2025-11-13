"""Response parsing and validation for LLM labeling outputs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from constella.data.results import LabelResult


LOGGER = logging.getLogger(__name__)


class LabelPayload(BaseModel):
    """Structured output schema for LLM-generated cluster labels.
    
    This Pydantic model defines the expected format for LLM responses when
    generating cluster labels. LiteLLM uses this schema to enforce structured
    outputs via the response_format parameter.
    
    Attributes:
        label: Concise human-readable label for the cluster (ideally <= 5 words).
        explanation: 1-2 sentence description of the cluster's common theme.
        confidence: Model's confidence score between 0.0 and 1.0.
        keywords: List of up to 5 keywords or phrases representing the theme.
    """
    label: str
    explanation: str
    confidence: float
    keywords: List[str] = Field(default_factory=list)

    def to_label_result(
        self,
        *,
        cluster_id: int,
        raw_response: Optional[str],
        usage: Optional[Dict[str, Any]],
    ) -> LabelResult:
        """Convert validated payload to LabelResult with bounds checking.
        
        Args:
            cluster_id: Cluster identifier to attach to the result.
            raw_response: Raw JSON string from the LLM response.
            usage: Token usage metadata from the LLM API.
        
        Returns:
            LabelResult with sanitized fields and fallback values where needed.
        """
        bounded_confidence = float(min(1.0, max(0.0, self.confidence)))
        keywords = [item.strip() for item in self.keywords if item and item.strip()]
        return LabelResult(
            cluster_id=cluster_id,
            label=self.label.strip() or f"Cluster {cluster_id}",
            explanation=self.explanation.strip() or "Automatic labeling failed",
            confidence=bounded_confidence,
            keywords=keywords,
            raw_response=raw_response,
            usage_metadata=usage,
        )


def parse_label_response(cluster_id: int, response: Dict[str, Any]) -> LabelResult:
    """Extract and validate label information from LLM response.
    
    Attempts multiple parsing strategies to extract structured label data:
    1. Native parsed field (LiteLLM structured output)
    2. Pydantic validation of parsed dict
    3. JSON parsing of raw content string
    
    Falls back to a default label if all parsing attempts fail.
    
    Args:
        cluster_id: Identifier for the cluster being labeled.
        response: Raw response dictionary from litellm.completion() or litellm.acompletion().
    
    Returns:
        LabelResult containing extracted label information or fallback values.
    
    Examples:
        >>> response = {"choices": [{"message": {"parsed": payload_obj, "content": "..."}}]}
        >>> result = parse_label_response(cluster_id=5, response=response)
        >>> print(result.label)
    """
    choice = _first_choice(response)
    if not choice:
        LOGGER.error("No completion choices returned for cluster %s", cluster_id)
        return create_fallback_label(cluster_id)

    message = choice.get("message", {})
    parsed_payload = message.get("parsed")
    raw_content = message.get("content")

    payload: Optional[LabelPayload]
    if isinstance(parsed_payload, LabelPayload):
        payload = parsed_payload
    elif parsed_payload is not None:
        try:
            payload = LabelPayload.model_validate(parsed_payload)
        except ValidationError as exc:
            LOGGER.warning(
                "Structured output validation failed for cluster %s: %s",
                cluster_id,
                exc,
            )
            return create_fallback_label(cluster_id, raw_response=raw_content)
    else:
        try:
            payload = LabelPayload.model_validate_json(raw_content or "{}")
        except (ValidationError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "Structured response missing for cluster %s; using fallback (%s)",
                cluster_id,
                exc,
            )
            return create_fallback_label(cluster_id, raw_response=raw_content)

    serialized = raw_content
    if serialized is None:
        try:
            serialized = payload.model_dump_json()
        except Exception:  # pragma: no cover - best effort serialization
            serialized = None

    return payload.to_label_result(
        cluster_id=cluster_id,
        raw_response=serialized,
        usage=response.get("usage"),
    )


def create_fallback_label(cluster_id: int, raw_response: Optional[str] = None) -> LabelResult:
    """Generate a default label when LLM labeling fails.
    
    Used as a safe fallback when API calls fail, responses are malformed,
    or validation errors occur during parsing.
    
    Args:
        cluster_id: Cluster identifier for the fallback label.
        raw_response: Optional raw response string for debugging.
    
    Returns:
        LabelResult with generic label and zero confidence.
    """
    return LabelResult(
        cluster_id=cluster_id,
        label=f"Cluster {cluster_id}",
        explanation="Automatic labeling failed",
        confidence=0.0,
        keywords=[],
        raw_response=raw_response,
        usage_metadata=None,
    )


def _first_choice(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the first choice from an LLM completion response.
    
    Args:
        response: Raw response dictionary from LiteLLM.
    
    Returns:
        First choice dictionary or empty dict if unavailable.
    """
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return choice
    return {}


def _coerce_response_to_dict(response: Any) -> Dict[str, Any]:
    """Convert various response types to dictionary format.
    
    Handles multiple response formats from LiteLLM including dicts, objects with
    model_dump() or dict() methods, and objects with __dict__ attributes.
    
    Args:
        response: Response object from LiteLLM API call.
    
    Returns:
        Dictionary representation of the response.
    """
    if isinstance(response, dict):
        return response
    for extractor in (getattr(response, "model_dump", None), getattr(response, "dict", None)):
        if callable(extractor):
            try:
                data = extractor()
                if isinstance(data, dict):
                    return data
            except Exception:  # pragma: no cover - defensive
                continue
    raw_dict = getattr(response, "__dict__", None)
    if isinstance(raw_dict, dict):
        return raw_dict
    return {"raw_response": repr(response)}


__all__ = [
    "LabelPayload",
    "parse_label_response",
    "create_fallback_label",
]

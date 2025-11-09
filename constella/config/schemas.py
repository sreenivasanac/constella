"""Configuration schemas for Constella workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ClusteringConfig:
    """Settings controlling the deterministic K-Means workflow."""

    seed: int = 42
    fallback_n_clusters: int = 8
    min_cluster_count: int = 2
    max_candidate_clusters: int = 15
    enable_silhouette_selection: bool = True
    silhouette_sample_size: Optional[int] = None
    enable_elbow_selection: bool = True
    enable_davies_bouldin_selection: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for optional UMAP visualization generation."""

    output_path: Path = Path("/tmp/constella_artifacts")
    show_plot: bool = False
    n_neighbors: int = 30
    min_dist: float = 0.3
    random_state: Optional[int] = 40

    def __post_init__(self) -> None:
        resolved = Path(self.output_path)
        if resolved.suffix:
            raise ValueError("VisualizationConfig.output_path must be a directory path")
        self.output_path = resolved


@dataclass
class RepresentativeSelectionConfig:
    """Configuration controlling representative sampling per cluster."""

    n_representatives: int = 20
    core_ratio: float = 0.6
    random_seed: int = 42
    diversity_sampling: bool = True


@dataclass
class LabelingConfig:
    """Configuration for automatic cluster labeling via LiteLLM providers."""

    llm_provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_output_tokens: int = 2000  # 16,384 is the max output tokens of gpt-4o-mini
    # max_chars_per_rep: maximum characters from ContentUnit text attribute, after which text attribute is truncated.
    # Used to save on tokens
    max_chars_per_rep: int = 700
    max_retries: int = 3
    retry_backoff_seconds: Tuple[float, float] = (1.0, 4.0)
    async_mode: bool = False
    max_concurrency: int = 6
    system_prompt_override: Optional[str] = None

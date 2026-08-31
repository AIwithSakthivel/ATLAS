"""Data models for the ACL Atlas paper-embedding clustering workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for one reproducible clustering run."""

    chroma_path: Path
    collection: str = "paper_embeddings"
    out_dir: Path = Path("clustering/runs")
    n_neighbors: int = 15
    n_components: int = 20
    min_cluster_size: int = 10
    min_samples: int = 5
    cluster_selection_method: str = "eom"
    seed: int = 42
    representatives_k: int = 5
    boundary_k: int = 5
    nearest_clusters_k: int = 5
    chroma_batch_size: int = 1000

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["chroma_path"] = str(self.chroma_path)
        data["out_dir"] = str(self.out_dir)
        return data


@dataclass
class PaperEmbeddingBatch:
    """Aligned paper records loaded from Chroma."""

    paper_ids: list[str]
    chroma_ids: list[str]
    documents: list[str | None]
    metadata: list[dict[str, Any]]
    X: np.ndarray


@dataclass(frozen=True)
class ValidationSummary:
    """Validated properties of the source embedding batch."""

    record_count: int
    embedding_dim: int
    embedding_model: str
    schema_version: str
    missing_document_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HDBSCANResult:
    """HDBSCAN outputs aligned to the source paper rows."""

    labels: np.ndarray
    probabilities: np.ndarray

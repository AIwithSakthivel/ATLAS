"""
embeddings/models.py

Description
-----------
Defines the shared data models and semantic schema constants used by the
embedding creation pipeline.

The pipeline creates two types of embeddings:

    Paper-level embedding:
        Title
        Problem Gap
        Claimed Contribution
        Technical Approach

    Facet-level embeddings:
        Problem Gap
        Claimed Contribution
        Technical Approach
        Datasets / Benchmarks
        Contribution Type
        Evaluation Summary
        Reproducibility
        Limitations / Failures

This module intentionally does not contain runtime configuration such as
OCI credentials, embedding model IDs, filesystem paths, or batch settings.

The embedding model is configured through common/config.py and exposed by
the existing OCI client.

ChromaDB paths are supplied by embeddings/run_embeddings.py.
"""

from dataclasses import asdict, dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Chroma collections
# ---------------------------------------------------------------------------

PAPER_COLLECTION = "paper_embeddings"
FACET_COLLECTION = "facet_embeddings"


# ---------------------------------------------------------------------------
# Embedding schema versions
# ---------------------------------------------------------------------------

# v1 paper embedding:
#   Title
#   Problem Gap
#   Claimed Contribution
#   Technical Approach
PAPER_SCHEMA_VERSION = "v1"

# v1 facet embedding:
#   One taxonomy field embedded independently.
FACET_SCHEMA_VERSION = "v1"


# ---------------------------------------------------------------------------
# Embedding fields
# ---------------------------------------------------------------------------

MANDATORY_TAXONOMY_FIELDS = (
    "problem_gap",
    "claimed_contribution",
    "technical_approach",
)

PAPER_EMBEDDING_FIELDS = (
    "title",
    "problem_gap",
    "claimed_contribution",
    "technical_approach",
)

FACET_EMBEDDING_FIELDS = (
    "problem_gap",
    "claimed_contribution",
    "technical_approach",
    "datasets_benchmarks",
    "contribution_type",
    "evaluation_summary",
    "reproducibility",
    "limitations_failures",
)


FIELD_LABELS = {
    "title": "Title",
    "problem_gap": "Problem Gap",
    "claimed_contribution": "Claimed Contribution",
    "technical_approach": "Technical Approach",
    "datasets_benchmarks": "Datasets / Benchmarks",
    "contribution_type": "Contribution Type",
    "evaluation_summary": "Evaluation Summary",
    "reproducibility": "Reproducibility",
    "limitations_failures": "Limitations / Failures",
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EmbeddingType = Literal["paper", "facet"]

EmbeddingStatus = Literal[
    "success",
    "skipped_existing",
    "skipped_empty",
    "failed",
]


# ---------------------------------------------------------------------------
# Pipeline models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EmbeddingUnit:
    """One semantic text unit that needs an embedding."""

    embedding_id: str
    paper_id: str
    embedding_type: EmbeddingType
    text: str
    text_hash: str
    schema_version: str
    facet: str | None = None

    @property
    def collection_name(self) -> str:
        """Return the Chroma collection for this embedding."""
        if self.embedding_type == "paper":
            return PAPER_COLLECTION

        return FACET_COLLECTION


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Processing result for one embedding unit.

    Each result becomes one line in manifest.jsonl.
    """

    run_id: str
    embedding_id: str
    paper_id: str
    embedding_type: EmbeddingType
    schema_version: str
    model: str
    text_hash: str
    status: EmbeddingStatus
    chroma_collection: str
    facet: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(slots=True)
class RunSummary:
    """Aggregate statistics for one embedding pipeline run."""

    run_id: str
    input_file: str
    embedding_model: str

    paper_schema_version: str = PAPER_SCHEMA_VERSION
    facet_schema_version: str = FACET_SCHEMA_VERSION

    papers_read: int = 0
    successful_papers_read: int = 0
    embedding_units_created: int = 0
    pending_units: int = 0

    paper_embeddings_success: int = 0
    paper_embeddings_skipped: int = 0
    paper_embeddings_failed: int = 0

    facet_embeddings_success: int = 0
    facet_embeddings_skipped: int = 0
    facet_embeddings_empty: int = 0
    facet_embeddings_failed: int = 0

    api_batches: int = 0
    verification_errors: int = 0
    embedding_dimension: int | None = None

    def record_result(self, result: EmbeddingResult) -> None:
        """Update counters from one embedding result."""
        if result.embedding_type == "paper":
            if result.status == "success":
                self.paper_embeddings_success += 1
            elif result.status == "skipped_existing":
                self.paper_embeddings_skipped += 1
            elif result.status == "failed":
                self.paper_embeddings_failed += 1

        elif result.embedding_type == "facet":
            if result.status == "success":
                self.facet_embeddings_success += 1
            elif result.status == "skipped_existing":
                self.facet_embeddings_skipped += 1
            elif result.status == "skipped_empty":
                self.facet_embeddings_empty += 1
            elif result.status == "failed":
                self.facet_embeddings_failed += 1

    @property
    def total_failures(self) -> int:
        return (
            self.paper_embeddings_failed
            + self.facet_embeddings_failed
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "input_file": self.input_file,
            "embedding_model": self.embedding_model,
            "paper_schema_version": self.paper_schema_version,
            "facet_schema_version": self.facet_schema_version,
            "papers_read": self.papers_read,
            "successful_papers_read": self.successful_papers_read,
            "embedding_units_created": self.embedding_units_created,
            "pending_units": self.pending_units,
            "paper_embeddings": {
                "success": self.paper_embeddings_success,
                "skipped_existing": self.paper_embeddings_skipped,
                "failed": self.paper_embeddings_failed,
            },
            "facet_embeddings": {
                "success": self.facet_embeddings_success,
                "skipped_existing": self.facet_embeddings_skipped,
                "skipped_empty": self.facet_embeddings_empty,
                "failed": self.facet_embeddings_failed,
            },
            "api_batches": self.api_batches,
            "embedding_dimension": self.embedding_dimension,
            "verification_errors": self.verification_errors,
        }
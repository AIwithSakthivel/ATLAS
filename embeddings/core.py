"""
embeddings/core.py

Description
-----------
Contains the core embedding pipeline operations.

Responsibilities:
    - Read extraction results from results.jsonl
    - Resolve paper titles
    - Build deterministic paper-level semantic embedding text
    - Build individual facet-level semantic embedding text
    - Compute SHA-256 hashes of exact embedding inputs
    - Create deterministic embedding IDs
    - Connect to local persistent ChromaDB
    - Detect unchanged embeddings and skip unnecessary API calls
    - Send new/changed texts to the existing OCI embedding client in batches
    - Store embeddings, semantic text, and metadata in ChromaDB
    - Produce manifest records for success, skip, and failure cases
    - Verify stored embeddings after creation

This module does not create the OCI client and does not parse CLI arguments.
Those responsibilities belong to embeddings/run_embeddings.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator, TextIO, TypeVar

import chromadb

from common.oci_client import OCIGenerativeAIClient
from embeddings.models import (
    FACET_COLLECTION,
    FACET_EMBEDDING_FIELDS,
    FACET_SCHEMA_VERSION,
    FIELD_LABELS,
    MANDATORY_TAXONOMY_FIELDS,
    PAPER_COLLECTION,
    PAPER_EMBEDDING_FIELDS,
    PAPER_SCHEMA_VERSION,
    EmbeddingResult,
    EmbeddingType,
    EmbeddingUnit,
)


logger = logging.getLogger(__name__)

DEFAULT_BATCH_DELAY_SECONDS = 0.2
DEFAULT_VERIFY_CHUNK_SIZE = 100

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _chunked(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Yield consecutive chunks from a list."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def _require_paper_id(record: dict[str, Any]) -> str:
    """Extract and validate paper_id from an extraction record."""
    paper_id = record.get("paper_id")

    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("Extraction record is missing a valid paper_id.")

    return paper_id.strip()


def _clean_text(value: Any) -> str | None:
    """Return stripped non-empty text, otherwise None."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"Expected text value, received {type(value).__name__}."
        )

    value = value.strip()
    return value or None


def compute_text_hash(text: str) -> str:
    """Compute a SHA-256 hash of the exact embedding input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_semantic_text(values: dict[str, str], fields: tuple[str, ...]) -> str:
    """Build deterministic labeled semantic text."""
    sections = []

    for field in fields:
        label = FIELD_LABELS[field]
        value = values[field]
        sections.append(f"{label}:\n{value}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------

def iter_results_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield extraction result records from results.jsonl.

    Empty lines are ignored. Invalid JSON or non-object records fail with
    the source line number so corrupted input is not silently skipped.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}."
                )

            _require_paper_id(record)
            yield record


def load_title_map(path: Path) -> dict[str, str]:
    """Load paper_id -> title mappings from JSON or JSONL metadata.

    Supported JSON:
        [
            {
                "paper_id": "2026.acl-long.1",
                "title": "..."
            }
        ]

    Supported JSONL:
        {"paper_id": "2026.acl-long.1", "title": "..."}
        {"paper_id": "2026.acl-long.2", "title": "..."}
    """
    if not path.exists():
        raise FileNotFoundError(f"Paper metadata file does not exist: {path}")

    records: list[dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected JSON object at {path}:{line_number}."
                    )

                records.append(record)

    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, list):
            raise ValueError(
                "Paper metadata JSON must contain a list of paper objects."
            )

        records = payload

    title_map: dict[str, str] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        paper_id = record.get("paper_id")
        title = record.get("title")

        if (
            isinstance(paper_id, str)
            and paper_id.strip()
            and isinstance(title, str)
            and title.strip()
        ):
            title_map[paper_id.strip()] = title.strip()

    return title_map


def resolve_title(
    record: dict[str, Any],
    title_map: dict[str, str] | None = None,
) -> str | None:
    """Resolve a paper title from available sources.

    Resolution order:
        1. record["title"]
        2. record["metadata"]["title"]
        3. external title_map keyed by paper_id
    """
    title = record.get("title")

    if isinstance(title, str) and title.strip():
        return title.strip()

    metadata = record.get("metadata")

    if isinstance(metadata, dict):
        title = metadata.get("title")

        if isinstance(title, str) and title.strip():
            return title.strip()

    if title_map:
        paper_id = _require_paper_id(record)
        title = title_map.get(paper_id)

        if title:
            return title

    return None


# ---------------------------------------------------------------------------
# Taxonomy extraction
# ---------------------------------------------------------------------------

def get_taxonomy_value(
    response: dict[str, Any],
    field: str,
) -> str | None:
    """Extract the semantic value for one taxonomy field.

    Supports the current extraction structure:

        "problem_gap": {
            "value": "...",
            "source_span": [...],
            "confidence": 1.0
        }

    Direct string values are also accepted.
    """
    field_data = response.get(field)

    if field_data is None:
        return None

    if isinstance(field_data, dict):
        value = field_data.get("value")
    else:
        value = field_data

    return _clean_text(value)


# ---------------------------------------------------------------------------
# Embedding unit construction
# ---------------------------------------------------------------------------

def build_paper_embedding_unit(
    record: dict[str, Any],
    title: str | None,
) -> EmbeddingUnit:
    """Build the single paper-level embedding unit for a paper."""
    paper_id = _require_paper_id(record)

    response = record.get("response")

    if not isinstance(response, dict):
        raise ValueError(
            f"{paper_id}: missing valid extraction response."
        )

    values: dict[str, str | None] = {
        "title": _clean_text(title),
        "problem_gap": get_taxonomy_value(response, "problem_gap"),
        "claimed_contribution": get_taxonomy_value(
            response,
            "claimed_contribution",
        ),
        "technical_approach": get_taxonomy_value(
            response,
            "technical_approach",
        ),
    }

    missing_fields = [
        field
        for field in PAPER_EMBEDDING_FIELDS
        if not values.get(field)
    ]

    if missing_fields:
        raise ValueError(
            f"{paper_id}: cannot build paper embedding; missing fields: "
            f"{', '.join(missing_fields)}"
        )

    semantic_values = {
        field: values[field]
        for field in PAPER_EMBEDDING_FIELDS
    }

    # Values have already been checked above.
    semantic_text = _build_semantic_text(
        values=semantic_values,  # type: ignore[arg-type]
        fields=PAPER_EMBEDDING_FIELDS,
    )

    return EmbeddingUnit(
        embedding_id=(
            f"{paper_id}::paper::{PAPER_SCHEMA_VERSION}"
        ),
        paper_id=paper_id,
        embedding_type="paper",
        text=semantic_text,
        text_hash=compute_text_hash(semantic_text),
        schema_version=PAPER_SCHEMA_VERSION,
    )


def build_facet_embedding_unit(
    record: dict[str, Any],
    facet: str,
) -> EmbeddingUnit:
    """Build one independent facet-level embedding unit."""
    if facet not in FACET_EMBEDDING_FIELDS:
        raise ValueError(f"Unsupported embedding facet: {facet!r}")

    paper_id = _require_paper_id(record)

    response = record.get("response")

    if not isinstance(response, dict):
        raise ValueError(
            f"{paper_id}: missing valid extraction response."
        )

    value = get_taxonomy_value(response, facet)

    if not value:
        raise ValueError(
            f"{paper_id}: taxonomy facet {facet!r} is empty."
        )

    semantic_text = _build_semantic_text(
        values={facet: value},
        fields=(facet,),
    )

    return EmbeddingUnit(
        embedding_id=(
            f"{paper_id}::{facet}::{FACET_SCHEMA_VERSION}"
        ),
        paper_id=paper_id,
        embedding_type="facet",
        facet=facet,
        text=semantic_text,
        text_hash=compute_text_hash(semantic_text),
        schema_version=FACET_SCHEMA_VERSION,
    )

def make_empty_facet_result(
    *,
    run_id: str,
    paper_id: str,
    facet: str,
    model_id: str,
) -> EmbeddingResult:
    """Create a manifest record for an optional empty taxonomy facet."""
    return EmbeddingResult(
        run_id=run_id,
        embedding_id=f"{paper_id}::{facet}::{FACET_SCHEMA_VERSION}",
        paper_id=paper_id,
        embedding_type="facet",
        facet=facet,
        schema_version=FACET_SCHEMA_VERSION,
        model=model_id,
        text_hash="",
        status="skipped_empty",
        chroma_collection=FACET_COLLECTION,
    )

def make_input_failure_result(
    *,
    run_id: str,
    paper_id: str,
    embedding_type: EmbeddingType,
    model_id: str,
    error: str,
    facet: str | None = None,
) -> EmbeddingResult:
    """Create a manifest result for an embedding input that cannot be built."""
    if embedding_type == "paper":
        embedding_id = (
            f"{paper_id}::paper::{PAPER_SCHEMA_VERSION}"
        )
        collection_name = PAPER_COLLECTION
        schema_version = PAPER_SCHEMA_VERSION

    else:
        if not facet:
            raise ValueError(
                "facet is required for a facet embedding failure."
            )

        embedding_id = (
            f"{paper_id}::{facet}::{FACET_SCHEMA_VERSION}"
        )
        collection_name = FACET_COLLECTION
        schema_version = FACET_SCHEMA_VERSION

    return EmbeddingResult(
        run_id=run_id,
        embedding_id=embedding_id,
        paper_id=paper_id,
        embedding_type=embedding_type,
        facet=facet,
        schema_version=schema_version,
        model=model_id,
        text_hash="",
        status="failed",
        chroma_collection=collection_name,
        error=error,
    )


def ensure_unique_embedding_ids(units: list[EmbeddingUnit]) -> None:
    """Fail if the current input generates duplicate embedding IDs."""
    seen: set[str] = set()
    duplicates: set[str] = set()

    for unit in units:
        if unit.embedding_id in seen:
            duplicates.add(unit.embedding_id)

        seen.add(unit.embedding_id)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates)[:10])

        raise ValueError(
            "Duplicate embedding IDs detected in input: "
            f"{duplicate_list}"
        )


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def create_chroma_collections(
    chroma_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Create/reopen the persistent ChromaDB and required collections."""
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_path),
    )

    paper_collection = client.get_or_create_collection(
        name=PAPER_COLLECTION,
        embedding_function=None,
    )

    facet_collection = client.get_or_create_collection(
        name=FACET_COLLECTION,
        embedding_function=None,
    )

    collections = {
        PAPER_COLLECTION: paper_collection,
        FACET_COLLECTION: facet_collection,
    }

    return client, collections


def _load_existing_metadata(
    units: list[EmbeddingUnit],
    collections: dict[str, Any],
    chunk_size: int = 500,
) -> dict[str, dict[str, Any]]:
    """Read existing metadata for embedding IDs from ChromaDB."""
    existing: dict[str, dict[str, Any]] = {}

    for collection_name in (PAPER_COLLECTION, FACET_COLLECTION):
        collection_units = [
            unit
            for unit in units
            if unit.collection_name == collection_name
        ]

        collection = collections[collection_name]

        for chunk in _chunked(collection_units, chunk_size):
            result = collection.get(
                ids=[unit.embedding_id for unit in chunk],
                include=["metadatas"],
            )

            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []

            for embedding_id, metadata in zip(ids, metadatas):
                existing[embedding_id] = metadata or {}

    return existing


def _result_for_unit(
    unit: EmbeddingUnit,
    *,
    run_id: str,
    model_id: str,
    status: str,
    error: str | None = None,
) -> EmbeddingResult:
    """Create an EmbeddingResult from an EmbeddingUnit."""
    return EmbeddingResult(
        run_id=run_id,
        embedding_id=unit.embedding_id,
        paper_id=unit.paper_id,
        embedding_type=unit.embedding_type,
        facet=unit.facet,
        schema_version=unit.schema_version,
        model=model_id,
        text_hash=unit.text_hash,
        status=status,  # type: ignore[arg-type]
        chroma_collection=unit.collection_name,
        error=error,
    )


def partition_existing_units(
    units: list[EmbeddingUnit],
    collections: dict[str, Any],
    *,
    run_id: str,
    model_id: str,
) -> tuple[list[EmbeddingUnit], list[EmbeddingResult]]:
    """Split units into pending work and unchanged existing embeddings.

    A record is skipped only when both:
        - text_hash matches
        - embedding_model matches

    Otherwise it is re-embedded and upserted.
    """
    existing_metadata = _load_existing_metadata(
        units,
        collections,
    )

    pending: list[EmbeddingUnit] = []
    skipped: list[EmbeddingResult] = []

    for unit in units:
        metadata = existing_metadata.get(unit.embedding_id)

        if (
            metadata
            and metadata.get("text_hash") == unit.text_hash
            and metadata.get("embedding_model") == model_id
        ):
            skipped.append(
                _result_for_unit(
                    unit,
                    run_id=run_id,
                    model_id=model_id,
                    status="skipped_existing",
                )
            )
        else:
            pending.append(unit)

    return pending, skipped


def _build_chroma_metadata(
    unit: EmbeddingUnit,
    *,
    model_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Build metadata stored with one Chroma embedding."""
    metadata: dict[str, Any] = {
        "paper_id": unit.paper_id,
        "embedding_type": unit.embedding_type,
        "schema_version": unit.schema_version,
        "embedding_model": model_id,
        "text_hash": unit.text_hash,
        "run_id": run_id,
    }

    if unit.facet:
        metadata["facet"] = unit.facet

    return metadata


# ---------------------------------------------------------------------------
# Embedding API + persistence
# ---------------------------------------------------------------------------

def embed_and_store_units(
    units: list[EmbeddingUnit],
    *,
    client: OCIGenerativeAIClient,
    collections: dict[str, Any],
    run_id: str,
    model_id: str,
    batch_size: int,
    batch_delay_seconds: float = DEFAULT_BATCH_DELAY_SECONDS,
) -> Iterator[EmbeddingResult]:
    """Embed pending units in batches and store them in ChromaDB.

    Retry behavior for each OCI request remains inside the existing
    OCIGenerativeAIClient.embed_text() implementation.

    A permanently failed API batch is recorded as failed and processing
    continues with the next batch.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    total_batches = (
        len(units) + batch_size - 1
    ) // batch_size

    for batch_number, batch in enumerate(
        _chunked(units, batch_size),
        start=1,
    ):
        logger.info(
            "Embedding batch %d/%d: %d units",
            batch_number,
            total_batches,
            len(batch),
        )

        texts = [unit.text for unit in batch]

        try:
            vectors = client.embed_text(
                inputs=texts,
                model_id=model_id,
            )
        except Exception as exc:
            logger.exception(
                "Embedding API batch %d/%d failed.",
                batch_number,
                total_batches,
            )

            error = f"{type(exc).__name__}: {exc}"

            for unit in batch:
                yield _result_for_unit(
                    unit,
                    run_id=run_id,
                    model_id=model_id,
                    status="failed",
                    error=error,
                )

            continue

        if len(vectors) != len(batch):
            error = (
                f"Embedding count mismatch: sent {len(batch)} texts, "
                f"received {len(vectors)} vectors."
            )

            logger.error(error)

            for unit in batch:
                yield _result_for_unit(
                    unit,
                    run_id=run_id,
                    model_id=model_id,
                    status="failed",
                    error=error,
                )

            continue

        dimensions = {
            len(vector)
            for vector in vectors
        }

        if len(dimensions) != 1 or 0 in dimensions:
            error = (
                "Embedding API returned empty or inconsistent "
                f"vector dimensions: {sorted(dimensions)}"
            )

            logger.error(error)

            for unit in batch:
                yield _result_for_unit(
                    unit,
                    run_id=run_id,
                    model_id=model_id,
                    status="failed",
                    error=error,
                )

            continue

        # A batch can contain both paper and facet records.
        # Store each collection as one Chroma upsert.
        indices_by_collection: dict[str, list[int]] = {}

        for index, unit in enumerate(batch):
            indices_by_collection.setdefault(
                unit.collection_name,
                [],
            ).append(index)

        for collection_name, indices in indices_by_collection.items():
            collection = collections[collection_name]

            collection_units = [
                batch[index]
                for index in indices
            ]

            collection_vectors = [
                vectors[index]
                for index in indices
            ]

            try:
                collection.upsert(
                    ids=[
                        unit.embedding_id
                        for unit in collection_units
                    ],
                    embeddings=collection_vectors,
                    documents=[
                        unit.text
                        for unit in collection_units
                    ],
                    metadatas=[
                        _build_chroma_metadata(
                            unit,
                            model_id=model_id,
                            run_id=run_id,
                        )
                        for unit in collection_units
                    ],
                )

            except Exception as exc:
                logger.exception(
                    "Chroma upsert failed for collection %s.",
                    collection_name,
                )

                error = f"{type(exc).__name__}: {exc}"

                for unit in collection_units:
                    yield _result_for_unit(
                        unit,
                        run_id=run_id,
                        model_id=model_id,
                        status="failed",
                        error=error,
                    )

            else:
                for unit in collection_units:
                    yield _result_for_unit(
                        unit,
                        run_id=run_id,
                        model_id=model_id,
                        status="success",
                    )

        if (
            batch_number < total_batches
            and batch_delay_seconds > 0
        ):
            time.sleep(batch_delay_seconds)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest_record(
    handle: TextIO,
    result: EmbeddingResult,
) -> None:
    """Append one embedding result to manifest.jsonl."""
    handle.write(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_units(
    units: list[EmbeddingUnit],
    *,
    collections: dict[str, Any],
    model_id: str,
    chunk_size: int = DEFAULT_VERIFY_CHUNK_SIZE,
) -> tuple[list[str], int | None]:
    """Read embeddings back from ChromaDB and verify stored records.

    Verifies:
        - record exists
        - exact semantic text matches
        - text hash matches
        - model matches
        - schema version matches
        - paper ID matches
        - embedding type matches
        - facet matches when applicable
        - embedding vector is non-empty
        - all vectors use one consistent dimension

    Returns:
        Tuple of:
            - verification error messages
            - detected embedding dimension
    """
    errors: list[str] = []
    detected_dimension: int | None = None

    for collection_name in (PAPER_COLLECTION, FACET_COLLECTION):
        collection_units = [
            unit
            for unit in units
            if unit.collection_name == collection_name
        ]

        collection = collections[collection_name]

        for chunk in _chunked(collection_units, chunk_size):
            result = collection.get(
                ids=[unit.embedding_id for unit in chunk],
                include=[
                    "documents",
                    "metadatas",
                    "embeddings",
                ],
            )

            ids = result.get("ids") or []
            documents = result.get("documents")
            metadatas = result.get("metadatas")
            embeddings = result.get("embeddings")

            stored: dict[str, tuple[Any, Any, Any]] = {}

            for index, embedding_id in enumerate(ids):
                document = (
                    documents[index]
                    if documents is not None
                    else None
                )

                metadata = (
                    metadatas[index]
                    if metadatas is not None
                    else None
                )

                vector = (
                    embeddings[index]
                    if embeddings is not None
                    else None
                )

                stored[embedding_id] = (
                    document,
                    metadata,
                    vector,
                )

            for unit in chunk:
                stored_record = stored.get(unit.embedding_id)

                if stored_record is None:
                    errors.append(
                        f"{unit.embedding_id}: missing from ChromaDB."
                    )
                    continue

                document, metadata, vector = stored_record
                metadata = metadata or {}

                if document != unit.text:
                    errors.append(
                        f"{unit.embedding_id}: stored document does not "
                        "match embedding input text."
                    )

                if metadata.get("text_hash") != unit.text_hash:
                    errors.append(
                        f"{unit.embedding_id}: text_hash mismatch."
                    )

                if metadata.get("embedding_model") != model_id:
                    errors.append(
                        f"{unit.embedding_id}: embedding model mismatch."
                    )

                if metadata.get("schema_version") != unit.schema_version:
                    errors.append(
                        f"{unit.embedding_id}: schema version mismatch."
                    )

                if metadata.get("paper_id") != unit.paper_id:
                    errors.append(
                        f"{unit.embedding_id}: paper_id mismatch."
                    )

                if metadata.get("embedding_type") != unit.embedding_type:
                    errors.append(
                        f"{unit.embedding_id}: embedding type mismatch."
                    )

                if unit.facet and metadata.get("facet") != unit.facet:
                    errors.append(
                        f"{unit.embedding_id}: facet mismatch."
                    )

                if vector is None or len(vector) == 0:
                    errors.append(
                        f"{unit.embedding_id}: embedding vector is empty."
                    )
                    continue

                dimension = len(vector)

                if detected_dimension is None:
                    detected_dimension = dimension

                elif dimension != detected_dimension:
                    errors.append(
                        f"{unit.embedding_id}: embedding dimension "
                        f"{dimension} != expected {detected_dimension}."
                    )

    return errors, detected_dimension
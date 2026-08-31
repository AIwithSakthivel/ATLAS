"""Core clustering operations for ACL Atlas paper embeddings.

The workflow intentionally keeps the source Chroma collection read-only:
load -> validate -> L2 normalize -> UMAP -> HDBSCAN -> diagnostics/artifacts.
Cluster centroids and paper centrality are computed in the original normalized
embedding space, not in UMAP space.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import hdbscan
import numpy as np
import umap

from .models import (
    ClusteringConfig,
    HDBSCANResult,
    PaperEmbeddingBatch,
    ValidationSummary,
)

LOGGER = logging.getLogger(__name__)

UMAP_CLUSTER_MIN_DIST = 0.0
UMAP_VISUALIZATION_MIN_DIST = 0.1
UMAP_INPUT_METRIC = "cosine"
HDBSCAN_METRIC = "euclidean"


def load_paper_embeddings(
    chroma_path: Path,
    collection_name: str,
    batch_size: int = 1000,
) -> PaperEmbeddingBatch:
    """Load all paper-level embeddings from a persisted Chroma collection.

    The returned lists and matrix preserve one shared row order.
    Only records with metadata.embedding_type == "paper" are loaded.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    LOGGER.info("Loading Chroma collection: %s", collection_name)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)

    chroma_ids: list[str] = []
    documents: list[str | None] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    offset = 0
    while True:
        result = collection.get(
            where={"embedding_type": "paper"},
            include=["embeddings", "documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )

        batch_ids = list(result.get("ids") or [])
        if not batch_ids:
            break

        batch_documents = result.get("documents")
        batch_metadatas = result.get("metadatas")
        batch_embeddings = result.get("embeddings")

        if batch_documents is None or batch_metadatas is None or batch_embeddings is None:
            raise ValueError(
                "Chroma get() did not return documents, metadatas, and embeddings."
            )

        if not (
            len(batch_ids)
            == len(batch_documents)
            == len(batch_metadatas)
            == len(batch_embeddings)
        ):
            raise ValueError("Chroma returned misaligned record columns.")

        chroma_ids.extend(batch_ids)
        documents.extend(batch_documents)

        for metadata in batch_metadatas:
            if metadata is None:
                raise ValueError("Encountered a Chroma record with missing metadata.")
            metadatas.append(dict(metadata))

        embeddings.extend(np.asarray(row) for row in batch_embeddings)

        offset += len(batch_ids)
        if len(batch_ids) < batch_size:
            break

    if not embeddings:
        raise ValueError(
            f"No paper embeddings found in collection {collection_name!r}."
        )

    paper_ids: list[str] = []
    for index, metadata in enumerate(metadatas):
        paper_id = metadata.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError(
                f"Record {chroma_ids[index]!r} has missing/invalid metadata.paper_id."
            )
        paper_ids.append(paper_id)

    try:
        X = np.asarray(embeddings, dtype=np.float32)
    except ValueError as exc:
        raise ValueError("Embeddings do not have one consistent dimensionality.") from exc

    LOGGER.info("Records found: %d", len(paper_ids))
    return PaperEmbeddingBatch(
        paper_ids=paper_ids,
        chroma_ids=chroma_ids,
        documents=documents,
        metadata=metadatas,
        X=X,
    )


def validate_embeddings(batch: PaperEmbeddingBatch) -> ValidationSummary:
    """Validate the clustering input contract and fail fast on invalid vectors."""
    LOGGER.info("Validating embeddings...")

    n_records = len(batch.paper_ids)
    if n_records == 0:
        raise ValueError("No embeddings were loaded.")

    aligned_lengths = {
        n_records,
        len(batch.chroma_ids),
        len(batch.documents),
        len(batch.metadata),
        batch.X.shape[0] if batch.X.ndim >= 1 else -1,
    }
    if len(aligned_lengths) != 1:
        raise ValueError("Loaded paper fields are not row-aligned.")

    if batch.X.ndim != 2 or batch.X.shape[1] == 0:
        raise ValueError(f"Expected a non-empty 2D embedding matrix, got {batch.X.shape}.")

    if len(set(batch.chroma_ids)) != n_records:
        raise ValueError("Duplicate Chroma record IDs found.")

    if len(set(batch.paper_ids)) != n_records:
        duplicates = _duplicate_values(batch.paper_ids)
        raise ValueError(f"Duplicate paper IDs found: {duplicates[:10]}")

    if not np.isfinite(batch.X).all():
        bad_rows = np.flatnonzero(~np.isfinite(batch.X).all(axis=1))
        raise ValueError(
            f"NaN or infinite embedding values found in rows: {bad_rows[:10].tolist()}"
        )

    norms = np.linalg.norm(batch.X, axis=1)
    zero_rows = np.flatnonzero(norms == 0.0)
    if zero_rows.size:
        raise ValueError(
            f"Zero embeddings found in rows: {zero_rows[:10].tolist()}"
        )

    duplicate_vector_pairs = _find_duplicate_vector_pairs(batch.X)
    if duplicate_vector_pairs:
        examples = [
            {
                "first_paper_id": batch.paper_ids[first],
                "duplicate_paper_id": batch.paper_ids[second],
            }
            for first, second in duplicate_vector_pairs[:10]
        ]
        raise ValueError(f"Duplicate embedding vectors found: {examples}")

    embedding_models = _single_metadata_value(batch.metadata, "embedding_model")
    schema_versions = _single_metadata_value(batch.metadata, "schema_version")

    missing_document_count = sum(
        document is None or not str(document).strip() for document in batch.documents
    )
    if missing_document_count:
        LOGGER.warning(
            "Documents missing/empty for %d records; clustering can proceed, but later "
            "cluster inspection will be incomplete.",
            missing_document_count,
        )

    LOGGER.info("Embedding dimension: %d", batch.X.shape[1])
    LOGGER.info("Invalid embeddings: 0")
    LOGGER.info("Duplicate paper IDs: 0")
    LOGGER.info("Duplicate embedding vectors: 0")
    LOGGER.info("Embedding model: %s", embedding_models)
    LOGGER.info("Schema version: %s", schema_versions)

    return ValidationSummary(
        record_count=n_records,
        embedding_dim=int(batch.X.shape[1]),
        embedding_model=embedding_models,
        schema_version=schema_versions,
        missing_document_count=missing_document_count,
    )


def normalize_embeddings(X: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Cannot normalize zero embedding vectors.")
    return (X / norms).astype(np.float32, copy=False)


def reduce_with_umap(
    X_norm: np.ndarray,
    n_neighbors: int,
    n_components: int,
    seed: int,
) -> np.ndarray:
    """Create the UMAP representation used for clustering."""
    if n_components < 2:
        raise ValueError("n_components must be >= 2")
    if n_components >= X_norm.shape[0]:
        raise ValueError("n_components must be smaller than the number of papers.")

    LOGGER.info(
        "Running UMAP: input=%s, output=(%d, %d)",
        X_norm.shape,
        X_norm.shape[0],
        n_components,
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=UMAP_CLUSTER_MIN_DIST,
        metric=UMAP_INPUT_METRIC,
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(X_norm), dtype=np.float32)


def reduce_with_umap_2d(
    X_norm: np.ndarray,
    n_neighbors: int,
    seed: int,
) -> np.ndarray:
    """Create a separate 2-D UMAP embedding for visualization only."""
    LOGGER.info("Running 2-D UMAP for visualization only")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=UMAP_VISUALIZATION_MIN_DIST,
        metric=UMAP_INPUT_METRIC,
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(X_norm), dtype=np.float32)


def cluster_with_hdbscan(
    X_umap: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str = "eom",
) -> HDBSCANResult:
    """Cluster the higher-dimensional UMAP representation with HDBSCAN."""
    if cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError("cluster_selection_method must be 'eom' or 'leaf'.")

    LOGGER.info(
        "Running HDBSCAN: min_cluster_size=%d, min_samples=%d, selection=%s",
        min_cluster_size,
        min_samples,
        cluster_selection_method,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(X_umap).astype(np.int32, copy=False)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float32)

    return HDBSCANResult(labels=labels, probabilities=probabilities)


def compute_cluster_centroids(
    X_norm: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute unit-normalized cluster centroids in original embedding space."""
    centroids: dict[int, np.ndarray] = {}
    for cluster_id in _cluster_ids(labels):
        member_vectors = X_norm[labels == cluster_id]
        centroid = member_vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0.0:
            raise ValueError(f"Cluster {cluster_id} has a zero centroid.")
        centroids[cluster_id] = (centroid / norm).astype(np.float32, copy=False)
    return centroids


def find_representative_papers(
    paper_ids: list[str],
    X_norm: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    top_k: int,
) -> dict[int, list[dict[str, Any]]]:
    """Rank cluster members by cosine similarity to their original-space centroid."""
    output: dict[int, list[dict[str, Any]]] = {}

    for cluster_id, centroid in centroids.items():
        member_indices = np.flatnonzero(labels == cluster_id)
        similarities = X_norm[member_indices] @ centroid
        order = np.argsort(-similarities)[:top_k]
        output[cluster_id] = [
            {
                "paper_id": paper_ids[int(member_indices[position])],
                "centrality_score": float(similarities[position]),
            }
            for position in order
        ]

    return output


def find_boundary_papers(
    paper_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    top_k: int,
) -> dict[int, list[dict[str, Any]]]:
    """Return the lowest-membership papers in each non-noise cluster."""
    output: dict[int, list[dict[str, Any]]] = {}

    for cluster_id in _cluster_ids(labels):
        member_indices = np.flatnonzero(labels == cluster_id)
        member_probs = probabilities[member_indices]
        order = np.argsort(member_probs)[:top_k]
        output[cluster_id] = [
            {
                "paper_id": paper_ids[int(member_indices[position])],
                "membership_probability": float(member_probs[position]),
            }
            for position in order
        ]

    return output


def find_nearest_clusters(
    centroids: dict[int, np.ndarray],
    top_k: int,
) -> dict[int, list[dict[str, Any]]]:
    """Rank neighboring clusters by original-space centroid cosine similarity."""
    cluster_ids = sorted(centroids)
    if len(cluster_ids) <= 1:
        return {cluster_id: [] for cluster_id in cluster_ids}

    centroid_matrix = np.vstack([centroids[cluster_id] for cluster_id in cluster_ids])
    similarities = centroid_matrix @ centroid_matrix.T

    output: dict[int, list[dict[str, Any]]] = {}
    for row_index, cluster_id in enumerate(cluster_ids):
        order = np.argsort(-similarities[row_index])
        neighbors: list[dict[str, Any]] = []
        for column_index in order:
            other_cluster_id = cluster_ids[int(column_index)]
            if other_cluster_id == cluster_id:
                continue
            neighbors.append(
                {
                    "cluster_id": other_cluster_id,
                    "similarity": float(similarities[row_index, column_index]),
                }
            )
            if len(neighbors) >= top_k:
                break
        output[cluster_id] = neighbors

    return output


def build_cluster_details(
    paper_ids: list[str],
    X_norm: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    representatives_k: int,
    boundary_k: int,
    nearest_clusters_k: int,
) -> list[dict[str, Any]]:
    """Build one serializable record per non-noise cluster."""
    centroids = compute_cluster_centroids(X_norm, labels)
    representatives = find_representative_papers(
        paper_ids=paper_ids,
        X_norm=X_norm,
        labels=labels,
        centroids=centroids,
        top_k=representatives_k,
    )
    boundaries = find_boundary_papers(
        paper_ids=paper_ids,
        labels=labels,
        probabilities=probabilities,
        top_k=boundary_k,
    )
    nearest = find_nearest_clusters(centroids, top_k=nearest_clusters_k)

    details: list[dict[str, Any]] = []
    for cluster_id in _cluster_ids(labels):
        member_probs = probabilities[labels == cluster_id]
        details.append(
            {
                "cluster_id": cluster_id,
                "paper_count": int(member_probs.size),
                "mean_membership_probability": float(member_probs.mean()),
                "min_membership_probability": float(member_probs.min()),
                "max_membership_probability": float(member_probs.max()),
                "representative_papers": representatives[cluster_id],
                "boundary_papers": boundaries[cluster_id],
                "nearest_clusters": nearest[cluster_id],
            }
        )

    return details


def build_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute corpus-level cluster and membership diagnostics."""
    non_noise_mask = labels != -1
    cluster_sizes = Counter(int(label) for label in labels[non_noise_mask])
    sizes = np.asarray(list(cluster_sizes.values()), dtype=np.int32)
    member_probs = probabilities[non_noise_mask]

    noise_count = int(np.sum(~non_noise_mask))
    clustered_count = int(np.sum(non_noise_mask))
    total_count = int(labels.size)

    size_histogram = Counter(int(size) for size in sizes)

    if member_probs.size:
        probability_distribution = {
            "min": float(np.min(member_probs)),
            "p25": float(np.quantile(member_probs, 0.25)),
            "median": float(np.median(member_probs)),
            "mean": float(np.mean(member_probs)),
            "p75": float(np.quantile(member_probs, 0.75)),
            "max": float(np.max(member_probs)),
        }
    else:
        probability_distribution = {}

    return {
        "paper_count": total_count,
        "cluster_count": len(cluster_sizes),
        "clustered_paper_count": clustered_count,
        "noise_paper_count": noise_count,
        "noise_percentage": (100.0 * noise_count / total_count) if total_count else 0.0,
        "cluster_size": {
            "min": int(np.min(sizes)) if sizes.size else 0,
            "median": float(np.median(sizes)) if sizes.size else 0.0,
            "max": int(np.max(sizes)) if sizes.size else 0,
        },
        "membership_probability": probability_distribution,
        "cluster_size_histogram": [
            {"cluster_size": size, "cluster_count": count}
            for size, count in sorted(size_histogram.items())
        ],
    }


def save_artifacts(
    run_dir: Path,
    config: ClusteringConfig,
    validation: ValidationSummary,
    batch: PaperEmbeddingBatch,
    X_umap: np.ndarray,
    X_umap_2d: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    cluster_details: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    """Persist immutable clustering artifacts into an existing run directory."""
    assignments = [
        {
            "paper_id": paper_id,
            "chroma_id": chroma_id,
            "cluster_id": int(cluster_id),
            "membership_probability": float(probability),
        }
        for paper_id, chroma_id, cluster_id, probability in zip(
            batch.paper_ids,
            batch.chroma_ids,
            labels,
            probabilities,
            strict=True,
        )
    ]

    noise_records = []
    for index in np.flatnonzero(labels == -1):
        i = int(index)
        noise_records.append(
            {
                "paper_id": batch.paper_ids[i],
                "chroma_id": batch.chroma_ids[i],
                "cluster_id": -1,
                "membership_probability": float(probabilities[i]),
                "document": batch.documents[i],
                "metadata": batch.metadata[i],
            }
        )

    config_payload = {
        **config.to_dict(),
        "umap_metric": UMAP_INPUT_METRIC,
        "umap_cluster_min_dist": UMAP_CLUSTER_MIN_DIST,
        "umap_visualization_min_dist": UMAP_VISUALIZATION_MIN_DIST,
        "hdbscan_metric": HDBSCAN_METRIC,
        "input": validation.to_dict(),
        "run": run_metadata,
    }

    summary_payload = {
        "run": run_metadata,
        "parameters": {
            "n_neighbors": config.n_neighbors,
            "n_components": config.n_components,
            "min_cluster_size": config.min_cluster_size,
            "min_samples": config.min_samples,
            "cluster_selection_method": config.cluster_selection_method,
            "seed": config.seed,
        },
        "input": validation.to_dict(),
        "diagnostics": diagnostics,
    }

    _write_json(run_dir / "config.json", config_payload)
    _write_jsonl(run_dir / "cluster_assignments.jsonl", assignments)
    _write_json(run_dir / "cluster_summary.json", summary_payload)
    _write_jsonl(run_dir / "cluster_details.jsonl", cluster_details)
    _write_jsonl(run_dir / "noise_papers.jsonl", noise_records)
    np.save(run_dir / "umap_embeddings.npy", X_umap)
    np.save(run_dir / "umap_2d.npy", X_umap_2d)

    LOGGER.info("Artifacts saved to: %s", run_dir)


def _cluster_ids(labels: np.ndarray) -> list[int]:
    return sorted(int(label) for label in np.unique(labels) if int(label) != -1)


def _duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _find_duplicate_vector_pairs(X: np.ndarray) -> list[tuple[int, int]]:
    """Find exact duplicate vectors without materializing an NxN comparison matrix."""
    seen: dict[bytes, int] = {}
    duplicates: list[tuple[int, int]] = []

    for index, row in enumerate(X):
        contiguous = np.ascontiguousarray(row)
        digest = hashlib.sha256(contiguous.view(np.uint8)).digest()
        first_index = seen.get(digest)
        if first_index is None:
            seen[digest] = index
        elif np.array_equal(X[first_index], row):
            duplicates.append((first_index, index))

    return duplicates


def _single_metadata_value(metadata: list[dict[str, Any]], key: str) -> str:
    values: set[str] = set()
    missing_indices: list[int] = []

    for index, item in enumerate(metadata):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            missing_indices.append(index)
        else:
            values.add(value)

    if missing_indices:
        raise ValueError(
            f"Missing/invalid metadata.{key} in rows: {missing_indices[:10]}"
        )
    if len(values) != 1:
        raise ValueError(f"Mixed metadata.{key} values found: {sorted(values)}")

    return next(iter(values))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

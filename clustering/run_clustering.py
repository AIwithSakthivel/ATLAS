"""CLI entry point for ACL Atlas paper-embedding clustering."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import (
    build_cluster_details,
    build_diagnostics,
    cluster_with_hdbscan,
    load_paper_embeddings,
    normalize_embeddings,
    reduce_with_umap,
    reduce_with_umap_2d,
    save_artifacts,
    validate_embeddings,
)
from .models import ClusteringConfig

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster ACL Atlas paper embeddings from a persisted Chroma collection."
    )
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--collection", default="paper_embeddings")
    parser.add_argument("--out-dir", type=Path, default=Path("clustering/runs"))
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument(
        "--cluster-selection-method",
        choices=("eom", "leaf"),
        default="eom",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representatives-k", type=int, default=5)
    parser.add_argument("--boundary-k", type=int, default=5)
    parser.add_argument("--nearest-clusters-k", type=int, default=5)
    parser.add_argument("--chroma-batch-size", type=int, default=1000)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ClusteringConfig:
    return ClusteringConfig(
        chroma_path=args.chroma_path,
        collection=args.collection,
        out_dir=args.out_dir,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        seed=args.seed,
        representatives_k=args.representatives_k,
        boundary_k=args.boundary_k,
        nearest_clusters_k=args.nearest_clusters_k,
        chroma_batch_size=args.chroma_batch_size,
    )


def create_run_dir(out_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_logging(run_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def validate_config(config: ClusteringConfig) -> None:
    if not config.chroma_path.exists():
        raise FileNotFoundError(f"Chroma path does not exist: {config.chroma_path}")
    if config.n_neighbors < 2:
        raise ValueError("n_neighbors must be >= 2")
    if config.n_components < 2:
        raise ValueError("n_components must be >= 2")
    if config.min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2")
    if config.min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    if min(
        config.representatives_k,
        config.boundary_k,
        config.nearest_clusters_k,
        config.chroma_batch_size,
    ) < 1:
        raise ValueError("K values and chroma_batch_size must be >= 1")


def run(config: ClusteringConfig) -> Path:
    validate_config(config)
    run_dir = create_run_dir(config.out_dir)
    configure_logging(run_dir)

    start_wall = datetime.now().astimezone()
    start_monotonic = time.perf_counter()
    LOGGER.info("Process started at: %s", start_wall.strftime("%d-%b-%y %H:%M:%S %Z"))

    try:
        batch = load_paper_embeddings(
            chroma_path=config.chroma_path,
            collection_name=config.collection,
            batch_size=config.chroma_batch_size,
        )
        validation = validate_embeddings(batch)

        X_norm = normalize_embeddings(batch.X)

        X_umap = reduce_with_umap(
            X_norm=X_norm,
            n_neighbors=config.n_neighbors,
            n_components=config.n_components,
            seed=config.seed,
        )

        clustering = cluster_with_hdbscan(
            X_umap=X_umap,
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_method=config.cluster_selection_method,
        )

        diagnostics = build_diagnostics(
            labels=clustering.labels,
            probabilities=clustering.probabilities,
        )
        LOGGER.info("Clusters discovered: %d", diagnostics["cluster_count"])
        LOGGER.info(
            "Clustered papers: %d", diagnostics["clustered_paper_count"]
        )
        LOGGER.info("Noise papers: %d", diagnostics["noise_paper_count"])

        cluster_details = build_cluster_details(
            paper_ids=batch.paper_ids,
            X_norm=X_norm,
            labels=clustering.labels,
            probabilities=clustering.probabilities,
            representatives_k=config.representatives_k,
            boundary_k=config.boundary_k,
            nearest_clusters_k=config.nearest_clusters_k,
        )

        X_umap_2d = reduce_with_umap_2d(
            X_norm=X_norm,
            n_neighbors=config.n_neighbors,
            seed=config.seed,
        )

        end_wall = datetime.now().astimezone()
        elapsed_seconds = time.perf_counter() - start_monotonic
        run_metadata = {
            "run_id": run_dir.name,
            "started_at": start_wall.isoformat(),
            "ended_at": end_wall.isoformat(),
            "duration_seconds": round(elapsed_seconds, 3),
        }

        save_artifacts(
            run_dir=run_dir,
            config=config,
            validation=validation,
            batch=batch,
            X_umap=X_umap,
            X_umap_2d=X_umap_2d,
            labels=clustering.labels,
            probabilities=clustering.probabilities,
            cluster_details=cluster_details,
            diagnostics=diagnostics,
            run_metadata=run_metadata,
        )

        LOGGER.info("Process ended at: %s", end_wall.strftime("%d-%b-%y %H:%M:%S %Z"))
        LOGGER.info("Total Time: %.2f secs", elapsed_seconds)
        return run_dir

    except Exception:
        LOGGER.exception("Clustering run failed")
        raise


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run(config)


if __name__ == "__main__":
    main()

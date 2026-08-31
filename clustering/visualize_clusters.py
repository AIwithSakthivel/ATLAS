"""Visualize ACL Atlas clustering results from cluster_details.jsonl.

This script is intentionally independent of the clustering package.

Input
-----
cluster_details.jsonl produced by the clustering pipeline.

Generated plots
---------------
1. cluster_sizes.png
2. membership_probabilities.png
3. nearest_cluster_similarity.png
4. cluster_similarity_matrix.png

Example
-------
python visualize_clusters.py \
    --cluster-details clustering/runs/20260809T182728Z/cluster_details.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_cluster_details(path: Path) -> list[dict]:
    """Load cluster details JSONL records."""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    if not records:
        raise ValueError(f"No cluster records found in {path}")

    return records


def build_cluster_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert cluster-level JSON records into a dataframe."""
    rows = []

    for record in records:
        nearest_clusters = record.get("nearest_clusters", [])

        nearest_similarity = (
            max(
                neighbor["similarity"]
                for neighbor in nearest_clusters
            )
            if nearest_clusters
            else np.nan
        )

        rows.append(
            {
                "cluster_id": int(record["cluster_id"]),
                "paper_count": int(record["paper_count"]),
                "mean_membership_probability": float(
                    record["mean_membership_probability"]
                ),
                "min_membership_probability": float(
                    record["min_membership_probability"]
                ),
                "max_membership_probability": float(
                    record["max_membership_probability"]
                ),
                "nearest_cluster_similarity": nearest_similarity,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )


def plot_cluster_sizes(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot number of papers in every cluster."""
    ordered = df.sort_values("paper_count", ascending=False)

    plt.figure(figsize=(14, 6))

    plt.bar(
        ordered["cluster_id"].astype(str),
        ordered["paper_count"],
    )

    plt.xlabel("Cluster ID")
    plt.ylabel("Number of papers")
    plt.title("Cluster Size Distribution")

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_membership_probabilities(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mean/min/max HDBSCAN membership probability by cluster."""
    ordered = df.sort_values(
        "mean_membership_probability",
        ascending=False,
    )

    x = np.arange(len(ordered))

    mean_probability = ordered[
        "mean_membership_probability"
    ].to_numpy()

    min_probability = ordered[
        "min_membership_probability"
    ].to_numpy()

    max_probability = ordered[
        "max_membership_probability"
    ].to_numpy()

    lower_error = mean_probability - min_probability
    upper_error = max_probability - mean_probability

    plt.figure(figsize=(14, 6))

    plt.errorbar(
        x,
        mean_probability,
        yerr=[lower_error, upper_error],
        fmt="o",
        capsize=2,
    )

    plt.xticks(
        x,
        ordered["cluster_id"].astype(str),
        rotation=90,
    )

    plt.ylim(0, 1.05)

    plt.xlabel("Cluster ID")
    plt.ylabel("Membership probability")
    plt.title(
        "HDBSCAN Membership Confidence "
        "(mean with min/max range)"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_nearest_cluster_similarity(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot similarity between each cluster and its closest cluster."""
    data = (
        df.dropna(subset=["nearest_cluster_similarity"])
        .sort_values(
            "nearest_cluster_similarity",
            ascending=False,
        )
    )

    plt.figure(figsize=(14, 6))

    plt.bar(
        data["cluster_id"].astype(str),
        data["nearest_cluster_similarity"],
    )

    plt.xlabel("Cluster ID")
    plt.ylabel("Cosine similarity")
    plt.title("Similarity to Nearest Cluster")

    plt.xticks(rotation=90)
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_similarity_matrix(
    records: list[dict],
) -> tuple[np.ndarray, list[int]]:
    """Build a sparse cluster-to-cluster similarity matrix."""
    cluster_ids = sorted(
        int(record["cluster_id"])
        for record in records
    )

    index_by_cluster = {
        cluster_id: index
        for index, cluster_id in enumerate(cluster_ids)
    }

    matrix = np.full(
        (len(cluster_ids), len(cluster_ids)),
        np.nan,
        dtype=float,
    )

    np.fill_diagonal(matrix, 1.0)

    for record in records:
        source_cluster = int(record["cluster_id"])
        source_index = index_by_cluster[source_cluster]

        for neighbor in record.get("nearest_clusters", []):
            target_cluster = int(neighbor["cluster_id"])

            if target_cluster not in index_by_cluster:
                continue

            target_index = index_by_cluster[target_cluster]
            similarity = float(neighbor["similarity"])

            existing = matrix[source_index, target_index]

            if np.isnan(existing):
                matrix[source_index, target_index] = similarity
            else:
                matrix[source_index, target_index] = max(
                    existing,
                    similarity,
                )

            reverse_existing = matrix[
                target_index,
                source_index,
            ]

            if np.isnan(reverse_existing):
                matrix[target_index, source_index] = similarity
            else:
                matrix[target_index, source_index] = max(
                    reverse_existing,
                    similarity,
                )

    return matrix, cluster_ids


def plot_cluster_similarity_matrix(
    records: list[dict],
    output_path: Path,
) -> None:
    """Visualize nearest-cluster cosine similarities as a matrix."""
    matrix, cluster_ids = build_similarity_matrix(records)

    masked_matrix = np.ma.masked_invalid(matrix)

    figure_size = max(8, len(cluster_ids) * 0.18)

    plt.figure(
        figsize=(figure_size, figure_size),
    )

    image = plt.imshow(
        masked_matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Cosine similarity",
    )

    positions = np.arange(len(cluster_ids))

    plt.xticks(
        positions,
        cluster_ids,
        rotation=90,
        fontsize=7,
    )

    plt.yticks(
        positions,
        cluster_ids,
        fontsize=7,
    )

    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster ID")
    plt.title("Nearest-Cluster Similarity Matrix")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact clustering summary."""
    print()
    print("Cluster summary")
    print("=" * 60)

    print(f"Clusters:              {len(df)}")
    print(f"Clustered papers:      {df['paper_count'].sum()}")
    print(f"Min cluster size:      {df['paper_count'].min()}")
    print(
        f"Median cluster size:   "
        f"{df['paper_count'].median():.1f}"
    )
    print(f"Max cluster size:      {df['paper_count'].max()}")

    print(
        f"Mean membership prob:  "
        f"{df['mean_membership_probability'].mean():.3f}"
    )

    print()

    print("Largest clusters")
    print("-" * 60)

    largest = (
        df.nlargest(10, "paper_count")[
            [
                "cluster_id",
                "paper_count",
                "mean_membership_probability",
            ]
        ]
    )

    print(
        largest.to_string(
            index=False,
        )
    )

    print()

    print("Lowest-confidence clusters")
    print("-" * 60)

    lowest_confidence = (
        df.nsmallest(
            10,
            "mean_membership_probability",
        )[
            [
                "cluster_id",
                "paper_count",
                "mean_membership_probability",
                "min_membership_probability",
            ]
        ]
    )

    print(
        lowest_confidence.to_string(
            index=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ACL Atlas clustering results "
            "from cluster_details.jsonl."
        )
    )

    parser.add_argument(
        "--cluster-details",
        type=Path,
        required=True,
        help="Path to cluster_details.jsonl",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for plots. "
            "Defaults to <cluster-details-parent>/visualizations."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cluster_details.exists():
        raise FileNotFoundError(
            f"File not found: {args.cluster_details}"
        )

    output_dir = (
        args.out_dir
        if args.out_dir is not None
        else args.cluster_details.parent / "visualizations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_cluster_details(
        args.cluster_details
    )

    df = build_cluster_dataframe(records)

    print_summary(df)

    plot_cluster_sizes(
        df,
        output_dir / "cluster_sizes.png",
    )

    plot_membership_probabilities(
        df,
        output_dir / "membership_probabilities.png",
    )

    plot_nearest_cluster_similarity(
        df,
        output_dir / "nearest_cluster_similarity.png",
    )

    plot_cluster_similarity_matrix(
        records,
        output_dir / "cluster_similarity_matrix.png",
    )

    print()
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
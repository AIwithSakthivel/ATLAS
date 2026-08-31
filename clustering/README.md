# Atlas semantic clustering

This is stage five of [Atlas](../): it turns the paper-level vectors produced
by [`embeddings`](../embeddings) into candidate research groups. The stage
discovers structure only — it does not name or interpret what a group is
about. That's deliberately left to the next stage, cluster interpretation, so
structure discovery stays separate from semantic naming.

## Pipeline

The source Chroma collection is treated as read-only throughout:

```
load paper embeddings
    -> validate the embedding space
    -> L2-normalize
    -> UMAP (clustering representation)
    -> HDBSCAN
    -> cluster centroids + representative/boundary papers + nearest clusters
    -> corpus/cluster diagnostics
    -> persist run artifacts
```

Only records with `metadata.embedding_type == "paper"` are loaded from the
`paper_embeddings` collection — one 3,072-dimensional vector per paper — kept
aligned with paper ID, source text, and metadata throughout.

## Validate before clustering

Clustering is fundamentally about relationships between vectors, so before
running anything the module checks that every vector actually belongs to the
same representation space: consistent dimensionality, no NaN/infinite or
zero-norm rows, no duplicate paper IDs or duplicate vectors, and a single
consistent `embedding_model` / `schema_version` across the batch. If vectors
came from different models or dimensions, their distances wouldn't be
meaningfully comparable, so this runs before any dimensionality reduction.

## Normalize, then reduce

Every embedding is L2-normalized so it has unit length. What matters
semantically is the *direction* of a vector, not its magnitude, and once
vectors are unit-length, a dot product is equivalent to cosine similarity —
used throughout the rest of the pipeline for centroids, representative
papers, and cluster relationships. This normalized space is retained
separately from anything UMAP produces, specifically for those later
calculations.

The normalized vectors are still high-dimensional, so the module runs UMAP
(cosine distance on the normalized space) to produce a smaller, configurable
representation — 20 dimensions by default. This isn't compression for
storage; UMAP is used as a representation-learning step that preserves local
neighborhood structure, making it easier for a density-based clustering
algorithm to model.

## Two UMAP representations, not one

There are two separate UMAP outputs, and they are never used interchangeably:

- A higher-dimensional representation (20-D by default) that HDBSCAN actually
  clusters on.
- An independent 2-D representation that exists only so a human can inspect
  the result visually.

A 2-D projection necessarily discards most of the structure in the space, so
the module never clusters what gets plotted — it clusters the higher-
dimensional manifold and creates the 2-D view as a separate, purely visual
artifact.

## HDBSCAN over k-means, and noise is kept

The reduced vectors are clustered with HDBSCAN rather than a method like
k-means, specifically to avoid having to decide the number of clusters in
advance. HDBSCAN instead looks for dense regions in the representation space;
dense regions become clusters, and points that don't belong strongly enough
to any of them are left unassigned (`cluster_id == -1`) rather than forced
into the nearest group. Noise papers are reported alongside clustered ones
rather than discarded, since an unassigned paper may be genuinely novel,
cross-cutting, or simply under-represented in the corpus.

HDBSCAN also returns a **membership probability** per paper, not just a
label — a low-confidence assignment and a high-confidence one are very
different situations, and that probability is stored for every assignment
and summarized at both the corpus and cluster level.

## Cluster centroids, representative papers, boundary papers

For each discovered cluster, the module computes a unit-normalized centroid
— but in the original normalized embedding space, not UMAP space, so semantic
interpretation doesn't depend on coordinates created by a dimensionality-
reduction step.

Every member paper's cosine similarity to its cluster's centroid gives two
complementary views:

- **Representative papers** — members ranked highest by centroid similarity.
  These are the papers to read first to understand what a cluster is about.
- **Boundary papers** — members ranked lowest by HDBSCAN membership
  probability. These are useful for checking whether a cluster should
  actually be split, or whether it's blending two related themes.

## Relationships between clusters

Cluster centroids are also compared against each other with cosine
similarity in the original normalized space, giving each cluster a ranked
list of its nearest semantic neighbors — useful for later spotting closely
related or possibly redundant clusters.

## Diagnostics and resolution selection

Every run computes corpus- and cluster-level diagnostics: cluster and noise
counts, cluster-size distribution, and the membership-probability
distribution. There's no single universally correct clustering for
unsupervised semantic discovery, so the main HDBSCAN parameters
(`min_cluster_size`, `min_samples`, `n_neighbors`, `n_components`,
`cluster_selection_method`) are treated as controlling resolution — the CLI
exposes all of them so different runs can be swept, compared, and
reproduced.

## Visualization is independent

[`visualize_clusters.py`](visualize_clusters.py) is intentionally decoupled
from the clustering package itself — it only reads the `cluster_details.jsonl`
a run produces, not any clustering internals. It renders cluster-size
distribution, membership-probability confidence per cluster, nearest-cluster
similarity, and a full cluster-to-cluster similarity matrix (useful for
spotting dense blocks of closely related clusters).

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. Reads from a Chroma collection already populated by
[`embeddings`](../embeddings).

## Usage

```bash
python -m clustering.run_clustering \
  --chroma-path embeddings/chroma_db \
  --collection paper_embeddings
```

Run as a module (`-m clustering.run_clustering`) from the directory
containing `clustering/`, so its relative imports resolve.

| Flag | Default | Purpose |
|---|---|---|
| `--chroma-path` | *(required)* | Path to the persisted Chroma store |
| `--collection` | `paper_embeddings` | Chroma collection to read from |
| `--out-dir` | `clustering/runs` | Where run artifacts are written |
| `--n-neighbors` | `15` | UMAP neighborhood size |
| `--n-components` | `20` | Dimensionality of the clustering UMAP representation |
| `--min-cluster-size` | `10` | Minimum HDBSCAN cluster size |
| `--min-samples` | `5` | HDBSCAN density/noise sensitivity |
| `--cluster-selection-method` | `eom` | `eom` or `leaf` |
| `--seed` | `42` | Random seed for UMAP |
| `--representatives-k` | `5` | Representative papers stored per cluster |
| `--boundary-k` | `5` | Boundary papers stored per cluster |
| `--nearest-clusters-k` | `5` | Nearest-neighbor clusters stored per cluster |
| `--chroma-batch-size` | `1000` | Page size for reading Chroma |

To visualize a completed run:

```bash
python visualize_clusters.py \
  --cluster-details clustering/runs/<run_id>/cluster_details.jsonl
```

## Output

```
clustering/runs/<run_id>/
├── config.json                # full resolved config + input validation summary + run timing
├── cluster_assignments.jsonl  # one row per paper: cluster_id, membership_probability
├── cluster_summary.json       # parameters + corpus-level diagnostics
├── cluster_details.jsonl      # one row per cluster: representatives, boundary papers, nearest clusters
├── noise_papers.jsonl         # unassigned papers (cluster_id == -1), with document + metadata
├── umap_embeddings.npy        # the higher-dimensional UMAP representation used for clustering
├── umap_2d.npy                # the 2-D UMAP representation used for visualization only
├── run.log
└── visualizations/            # created by visualize_clusters.py
    ├── cluster_sizes.png
    ├── membership_probabilities.png
    ├── nearest_cluster_similarity.png
    └── cluster_similarity_matrix.png
```

Each run gets its own timestamped directory, so a clustering run is treated
as a reproducible experiment: parameters, timing, input validation, and every
artifact needed to inspect or compare it later are all persisted together.

## Known limitations

- Assumes a `paper_embeddings` Chroma collection already populated in the
  shape [`embeddings`](../embeddings) produces (embedding + `paper_id`,
  `embedding_model`, `schema_version` metadata); it does not build that
  collection itself.
- The clustering stage stops at structure discovery — cluster IDs carry no
  semantic label. Turning a cluster into a named topic is the responsibility
  of the next stage, cluster interpretation.

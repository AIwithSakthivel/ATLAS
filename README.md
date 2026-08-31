# ATLAS

ATLAS is a research-intelligence system for understanding how a research field
is evolving through its published papers. Starting with ACL 2026, it turns a
conference corpus into structured, traceable evidence that helps answer
questions such as:

- Where are researchers concentrating their effort?
- Which research topics, methods, and application areas are growing or emerging?
- Which areas appear underexplored or differentiated?
- Where should further research, product investment, or strategic attention go?
- Which papers and topics are most similar to a question or area of interest?

The goal is to give research and business stakeholders a grounded way to
navigate a large body of work, identify meaningful patterns, and make more
informed decisions about where to explore, collaborate, or invest.

ATLAS is organized as a staged, file-backed pipeline. Each stage has a focused
responsibility, produces durable and reviewable artifacts, and can be rerun or
resumed independently — building on the evidence produced by the previous
stage while preserving a link back to the underlying papers.

## Pipeline stages

1. **Document ingestion** — available now
2. **Research extraction** — available now
3. **Embedding creation** — available now
4. **Semantic clustering** — available now
5. **Cluster interpretation**
6. **Canonical taxonomy and assignment**
7. **Research-landscape statistics**
8. **Atlas web delivery**

## End-to-end data flow

```
Document ingestion
    -> Research extraction
    -> Embedding creation
    -> Semantic clustering
    -> Cluster interpretation
    -> Canonical taxonomy and assignment
    -> Research-landscape statistics
    -> Atlas web delivery
```

Concretely, each stage hands the next a specific artifact:

```
Ingestion artifacts (llm_context.txt + selected page images)
    -> extraction/results.jsonl
    -> ChromaDB: paper_embeddings and facet_embeddings
    -> clustering run: assignments, cluster details, diagnostics
    -> taxonomy run: cluster interpretations
    -> paper-to-taxonomy run: taxonomy_v1.json and atlas_papers_v1.jsonl
    -> statistics run: themes, patterns, opportunities, global landscape
    -> Atlas UI: generated/atlas.json and compact retrieval index
```

## Core framework principles

- PDFs are the authoritative source for research content.
- Every extracted field is evidence-grounded with source spans and confidence.
- Intermediate files make each stage inspectable, resumable, and auditable.
- Embeddings preserve semantic meaning while excluding operational metadata.
- Clustering discovers candidate themes; taxonomy stages turn them into stable
  categories.
- Statistics operates on a frozen taxonomy output, preserving reproducibility.
- The Atlas UI validates completeness before presenting a research run.

## Available: document ingestion

[`document_ingestion/`](https://github.com/AIwithSakthivel/ATLAS/blob/main/document_ingestion) is the first implemented stage.
Given a paper landing-page URL, it creates a normalized, page-traceable package
containing extracted text and selected page images where visual evidence is
needed.

The ingestion process is deterministic: it does not use generative-model calls.
It validates source PDFs, reconstructs academic-paper reading order, handles
tables and OCR when necessary, and preserves links back to the original paper
pages.

See the [document ingestion README](https://github.com/AIwithSakthivel/ATLAS/blob/main/document_ingestion/README.md) for
installation, usage, outputs, and implementation details.

## Available: research extraction

[`extraction/`](https://github.com/AIwithSakthivel/ATLAS/blob/main/extraction) converts each paper's ingested
representation into structured, evidence-grounded research information: eight
standard research dimensions (problem gap, contribution, technical approach,
datasets/benchmarks, contribution type, evaluation summary, reproducibility,
and limitations), each backed by exact source-text evidence and a confidence
score.

Extraction is powered by a pluggable LLM client that you supply — the module
itself has no third-party dependencies and defines only the extraction
contract, concurrency, validation, and checkpointing around it. Every
extracted value is validated against a strict schema before being accepted,
and runs are resumable: papers that already succeeded are skipped
automatically.

See the [extraction README](https://github.com/AIwithSakthivel/ATLAS/blob/main/extraction/README.md) for
installation, usage, the field contract, and output format.

## Available: embedding creation

[`embeddings/`](https://github.com/AIwithSakthivel/ATLAS/blob/main/embeddings) converts each successfully
extracted paper into vector representations: one paper-level embedding
capturing overall research identity (title, problem gap, contribution,
technical approach), and up to eight independent facet-level embeddings
capturing individual research dimensions.

Embeddings are stored in a local, persistent ChromaDB alongside the exact
semantic text and metadata that produced them, so every vector remains
traceable back to its source paper and field. Unchanged embeddings are
detected and skipped automatically, so reruns only touch new or changed
records.

See the [embedding creation README](https://github.com/AIwithSakthivel/ATLAS/blob/main/embeddings/README.md) for
the design rationale, installation, usage, and the required embedding-client
interface.

## Available: semantic clustering

[`clustering/`](https://github.com/AIwithSakthivel/ATLAS/blob/main/clustering) turns the paper-level
vectors produced by embedding creation into candidate research groups. It
validates and L2-normalizes the embedding space, reduces it with UMAP, and
discovers dense regions with HDBSCAN — without having to specify the number
of clusters in advance and without forcing every paper into a group.

Each discovered cluster gets a centroid computed in the original normalized
embedding space, representative papers ranked by centroid similarity,
boundary papers ranked by HDBSCAN membership probability, and a ranked list
of its nearest neighboring clusters. Corpus- and cluster-level diagnostics,
and a separate visualization/inspection layer, are generated from the same
run artifacts. Clustering discovers structure only; naming what a cluster
represents is left to the next stage, cluster interpretation.

See the [semantic clustering README](https://github.com/AIwithSakthivel/ATLAS/blob/main/clustering/README.md) for
the design rationale, installation, usage, and output format.

## Project status

**Pushed so far:** document ingestion, research extraction, embedding
creation, and semantic clustering — stages 1–4 above.

**Not yet in this repository:** cluster interpretation, canonical taxonomy
and assignment, research-landscape statistics, and Atlas web delivery. These
are listed to show the intended flow; their implementations are still under
review and will be added separately.

Conference acquisition (fetching accepted-paper metadata and PDF URLs from
ACL/ICLR sources) exists as working code ahead of document ingestion in the
full framework, but isn't part of this roadmap yet — it will be added once
it's packaged and reviewed.
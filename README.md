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

ATLAS is organized as a staged pipeline. Each stage builds on the evidence
produced by the previous one, while preserving a link back to the underlying
papers.

## Pipeline stages

1. **Document ingestion** — available now
2. **Research extraction** — available now
3. **Taxonomy extraction**
4. **Statistical insights**
5. **Reasoning**

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

## Project status

Document ingestion and research extraction are currently included in this
repository. The remaining pipeline stages (taxonomy extraction, statistical
insights, reasoning) are listed to show the intended flow; their
implementations are still under review and will be added separately.
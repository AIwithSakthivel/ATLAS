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
2. **Taxonomy extraction**
3. **Statistical insights**
4. **Reasoning**

## Available: document ingestion

[`document_ingestion/`](document_ingestion/) is the first implemented stage.
Given a paper landing-page URL, it creates a normalized, page-traceable package
containing extracted text and selected page images where visual evidence is
needed.

The ingestion process is deterministic: it does not use generative-model calls.
It validates source PDFs, reconstructs academic-paper reading order, handles
tables and OCR when necessary, and preserves links back to the original paper
pages.

See the [document ingestion README](document_ingestion/README.md) for
installation, usage, outputs, and implementation details.

## Project status

Only document ingestion is currently included in this repository. The remaining
pipeline stages are listed to show the intended flow; their implementations are
still under review and will be added separately.

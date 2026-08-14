# Atlas

Atlas gives insights into the recently concluded ACL 2026 conference — scraped
papers, statistical analysis, and the pipeline stages that produce them.

## Stages

- [`document_ingestion/`](document_ingestion/) — Stage one. Deterministic
  PDF-to-LLM-context ingestion for each paper: text extraction, table
  detection, selective page rendering, and OCR fallback.

More stages (taxonomy extraction, statistical insights, reasoning) will be
added here as they're built.

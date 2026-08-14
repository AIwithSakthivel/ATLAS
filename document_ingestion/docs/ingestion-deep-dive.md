# Atlas document ingestion: from paper URL to page-traceable multimodal LLM context

*This is stage one of Atlas, a project I've been building end to end. Document ingestion is the deterministic foundation everything downstream in Atlas relies on — more on the later stages in future posts.*

Most "PDF ingestion for RAG" pipelines do the same thing: dump every page through a text extractor, chunk it, embed it, move on. That works fine for a product manual. It falls apart on academic papers, where a single page might have a two-column body, a nine-column results table, an inline equation, and a vector-drawn architecture diagram, all competing for the same physical space.

Atlas's ingestion stage — the part that runs before any LLM sees the paper — is built around a different premise: **every decision about how to represent a page should be deterministic, inspectable, and made *before* a generative model gets involved.** No summarization, no chunking heuristics tuned by vibes, no silent failures. Just a pipeline of explicit thresholds you can read, tune, and audit.

To make this concrete, I ran it against a real ACL 2026 paper — [OctoTools](https://octotools.github.io) (Lu et al., *"OctoTools: A Multi-Agent Framework with Extensible Tools for Complex Reasoning,"* ACL 2026, paper ID `2026.acl-long.1`) — an 86-page paper with dense results tables, framework comparison figures, and a project page linking out to a GitHub repo and a Hugging Face demo. It's a good stress test: two-column body text, tables with 5+ columns, vector diagrams, and reproducibility links buried in footnotes. Every example below is pulled from that actual run.

## Stage 1 — fetch and validate, fail loud and early

A paper URL and an output directory go in. Network requests get a hard 30-second timeout — an unresponsive landing page or PDF server shouldn't stall the whole pipeline. Once a file lands, it has to be at least 1,024 bytes (filters out empty responses and redirect stubs) and its first bytes have to match the PDF signature — checking magic bytes rather than trusting the HTTP `Content-Type` header, which lies more often than you'd expect.

The file then gets opened with PyMuPDF to confirm it's structurally readable. Password-protected PDFs are rejected outright, zero-page PDFs are rejected, and if any of this fails after the file's already been written to disk, the temp file gets deleted immediately. Nothing invalid sticks around to be picked up by a later stage.

## Stage 2 — metadata from the landing page, content from the PDF

The landing page (not the PDF) is the source for metadata: paper ID, title, authors, venue, year, and the PDF URL itself, pulled primarily from citation `<meta>` tags with fallbacks to visible headings or the page `<title>`. Everything else — the actual scientific content — comes exclusively from the PDF. The landing page is metadata-and-discovery only.

For the OctoTools run, this resolved cleanly: paper ID `2026.acl-long.1`, six listed authors, venue "*Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*", year 2026, and a PDF URL matching the paper identifier — which is exactly the deterministic-matching path this stage is designed for on ACL Anthology pages specifically (prefer citation metadata, fall back to filename-identifier matching on landing-page links).

## Stage 3 — the two-column problem

This is the part that actually separates "PDF parsing" from "PDF understanding." Academic papers are two-column. A naive top-to-bottom, left-to-right text dump alternates between columns mid-sentence, which silently corrupts every downstream use of the text.

Atlas asks PyMuPDF for text blocks *with bounding-box coordinates* and reconstructs reading order geometrically:

1. **Full-width detection** — any block spanning ≥70% of page width (titles, section headers, wide captions, table surroundings) is pulled out as a special block that can split the page into vertical regions.
2. **Column classification** — every remaining block's horizontal center is compared against the page midpoint to assign it left or right.
3. **Two-column confidence check** — if either side has fewer than two blocks, the system doesn't force a two-column read; it just sorts everything top-to-bottom, left-to-right. This stops single-column or irregular pages from getting mangled by an interpretation that doesn't apply.
4. **Region-aware ordering** — on confirmed two-column pages, full-width blocks divide the page into regions; within each region, column blocks above a wide block are ordered first, then the wide block itself, repeating down the page.
5. Within each region, left column top-to-bottom, then right column top-to-bottom, concatenated into the page's raw text.

On the OctoTools PDF, page 1 alone has this exact shape: title and author line as full-width blocks at the top, then the abstract and a results bar chart running as two columns beneath. Get the 70%-width threshold or the midpoint split wrong, and you'd get abstract text interleaved with figure-adjacent numbers. This is also why the raw text is never discarded — it's the closest available evidence to what's actually on the page, and it's preserved through every later cleaning step.

## Stage 4 — cleaning without touching the evidence

A second, *cleaned* representation is derived from raw text, never in place of it. NFKC Unicode normalization, ligature repair, and a conservative hyphenation-repair rule that only rejoins alphabetic fragments split by a line-wrapping hyphen — deliberately narrow, because a rule that strips every line-ending hyphen will just as happily mangle genuine hyphenated terms. Whitespace gets normalized last: collapsed repeated spaces, trimmed line edges, blank-line runs reduced to one separator.

## Stage 5 — tables get two scores, not one

Tables are extracted as a separate pass, since PyMuPDF's table-region detection recovers row/column structure that plain text blocks lose. But the interesting design choice is that **complexity and confidence are tracked as two independent scores**, because they answer two different questions.

**Complexity (0–1)** asks: *how risky is a text-only reconstruction of this table?* It increments for:
- ≥5 columns: **+0.20**
- ≥10 rows: **+0.15**
- ≥15% empty cells: **+0.20**
- ≥10% multiline cells: **+0.15**
- inconsistent row widths: **+0.30**

capped at 1.0. A score ≥0.60 marks the table's page for visual (image) representation instead of trusting the text reconstruction.

**Confidence (starts at 1.0)** asks a different question: *does the extracted structure look right at all?* It's reduced by empty cells (up to −0.40), irregular row lengths (up to −0.30), and a degenerate single-row-or-column shape (−0.15 flat). A score below 0.70 also routes the page to visual selection — for a different reason: not because the table is inherently hard to represent as text, but because the extraction itself looks untrustworthy.

OctoTools' Table 1 (main results across 16 benchmarks, ~10 method rows × 8+ metric columns) is a textbook complexity trigger. Table 7 (optimized tool sets per benchmark, dense checkmark grid across many columns) is exactly the shape that tends to produce irregular extraction and low confidence. Both patterns showed up as visual-selection triggers in the actual run — pages tagged `complex_table` and `poor_table_extraction` respectively.

Table captions are located by searching a narrow window — up to 70pt above and 50pt below the detected table region — for lines matching a table-number pattern, so extracted tables keep their real published numbers (`Table 1`, `Table 6`, `Table 7`) instead of being anonymized into `table_p7_1`-style fallback IDs.

## Stage 6 — visual selection: six independent signals, one flag

After text and tables are extracted, Atlas decides which pages also need to be rendered as images for the multimodal model. Rendering every page is wasteful — more storage, more processing time, more multimodal tokens for no benefit — so a page only gets rasterized when a deterministic signal says text alone isn't trustworthy:

| Signal | Trigger |
|---|---|
| Poor native text | fails the 80-char / 85%-printable / <2%-replacement-char thresholds |
| Scanned-page pattern | single embedded image covers ≥50% of page area |
| Complex table | table complexity ≥ 0.60 |
| Poor table extraction | table confidence < 0.70 |
| Figure | a numbered/abbreviated figure caption **and** an embedded image ≥5% of page area |
| Vector figure | a figure caption **and** ≥20 vector drawing elements on the page (catches architecture diagrams stored as vector art, not raster images) |
| Equation content | equation-like symbols present, only checked when native text already failed |

These reasons all stack on the same page rather than overwriting each other — a page can be flagged for a complex table *and* a figure simultaneously. In the OctoTools run, the visual-pages summary reads (abbreviated):

```
Page 1: poor_table_extraction, figure
Page 9: complex_table, poor_table_extraction, figure
Page 20: poor_table_extraction, figure
Page 21–24: figure
```

Note the asymmetric thresholds: 50% page-area for a *scanned-page* signal versus 5% for a *figure* signal. That 10x gap is intentional — a full-page scan legitimately covers most of the page, while a typical scientific figure (a bar chart, a small architecture box) might occupy a fraction of the layout and still be the single most important thing on that page.

Only flagged pages get rasterized, at 175 DPI, no alpha channel, into a per-paper visual directory. Every image filename encodes the one-based physical page number — traceability back to the source PDF page is non-negotiable throughout this pipeline.

## Stage 7 — OCR as a last resort, not a default

OCR only runs on pages that already failed native-text-quality checks. Pages with usable born-digital text skip it entirely — OCR introduces error modes that clean PDF text extraction doesn't have, so it's only invoked when there's genuinely nothing better. For flagged pages, PyMuPDF hands off to Tesseract at 200 DPI (English), and the resulting text goes through the *same* cleaning function as native text — no separate, less-tested code path.

Successful OCR replaces the page's *cleaned* text only; the original raw native text is never overwritten, and a `text_source` field on the page records that this particular page's cleaned text came from OCR rather than native extraction. OCR failures are page-level warnings, not pipeline-level failures — an exception or an empty OCR result gets logged against that page, and ingestion continues for the rest of the document.

## Stage 8 — killing running headers without touching real content

Repeated-noise removal runs after OCR, so both native and OCR-derived text get identical treatment. It only looks at the first and last three non-empty lines of each page — deliberately narrow, since that's where running headers and footers live.

Candidate lines get normalized before counting: whitespace collapsed, lowercased, and any standalone number replaced with a generic token — so `"page 14"` and `"page 87"` are recognized as the same repeated pattern despite different digits. Blank lines, lines over 120 characters, and any line containing an `http`/`https` URL are excluded from consideration entirely. That last exclusion matters more than it sounds — OctoTools' own header block on page 1 carries five separate links (project site, GitHub, a Hugging Face demo space, tool-card and visualization anchors), and footnote-style reproducibility links like that are exactly what this rule is protecting from accidental deletion.

A normalized line only gets removed if it recurs on at least `max(3, 30% of total pages)` — an adaptive floor that gets stricter as papers get longer, so a genuinely repeated running header on an 86-page paper needs to show up on ~26 pages before it's touched, while a 5-page paper only needs 3.

## Stage 9 — sections, walked forward across page boundaries

Section detection recognizes a fixed list of common academic headings (abstract, introduction, related work, methods, results, discussion, limitations, references, appendix, etc.) plus regex rules for numbered and lettered sub/appendix headings. Headings over 120 characters or 14+ words are rejected outright — mostly to stop table and figure captions from being misread as section headings.

Section state carries forward: a page with no new heading inherits whichever section was most recently detected. Worth being honest about the limitation here — each page stores exactly one section label, so a page where two sections both begin isn't represented with full fidelity. That's a known, accepted tradeoff, not an oversight.

## Stage 10 — URLs and artifact links, extracted twice and merged

URLs are pulled from the raw (pre-cleaning) text via pattern match, and separately from the PDF's actual hyperlink annotations — because scientific PDFs often have clickable links whose full URL isn't visible in the rendered text at all. Both sources merge, trailing punctuation gets stripped, and duplicates are removed while preserving first-discovery order.

A second pass highlights a subset of those URLs as **artifacts**: links to GitHub, GitLab, Hugging Face, Zenodo, Figshare, OSF, Kaggle, or GitHub Pages domains. DOI links and ACL Anthology links are explicitly excluded from this category — they describe publication identity, not reproducibility. For OctoTools, this pass correctly isolated the actual reproducibility surface from the paper's identity metadata: the GitHub repo, the Hugging Face demo space, and the project-page anchors, each tagged with the physical page number they were found on.

## Stage 11 — tying tables back to pages, and one more sanity check

Once visual pages are finalized, any table sitting on a page that got rendered visually (for *any* reason — even a figure elsewhere on that page) is marked to use the rendered image instead of its extracted Markdown. This is deliberately conservative: it's safer to hand the model an authoritative page image than to send a possibly-imperfect table reconstruction alongside it and risk contradictory representations.

Before the ingestion package is returned, Atlas checks its own work: the PDF still has to exist, page count has to be ≥1, extracted page objects must exactly match the stored page count, physical page numbers must form a sequential range starting at 1, every table's page reference has to point to a real page, and the PDF gets reopened one final time to cross-check its real page count against the package's. A fully deterministic pipeline still doesn't trust itself blindly.

## Stage 12 — assembling context, still with no LLM in the loop

The final text context starts with metadata (explicitly marking unknown fields as unknown rather than guessing), then the complete cleaned paper page by page under physical-page markers — main text, references, appendices, supplementary material, all of it. **No chunking, no retrieval step, no summarization.** Tables not already covered by a rendered page get their own section, each with page number, identifier, caption, and Markdown. Artifact links get surfaced with their originating page. The visual-selection reasons are summarized so the reason a page was rendered is never opaque.

Rendered images aren't embedded in the text — their file paths are returned as a separate visual-pages component, so text evidence and image evidence stay distinct while remaining aligned by physical page number.

## Why this is worth doing the "boring" way

The architectural bet here is that **ingestion should be entirely deterministic** — metadata resolution, layout reconstruction, cleaning, table scoring, visual selection, OCR, section detection, URL extraction, and context assembly all happen with zero generative calls. The first place a generative model enters the picture is the *next* stage, once this normalized, invariant-checked, page-traceable package already exists.

That ordering is the whole point. When something looks wrong three stages downstream, you can point at a specific threshold, a specific score, a specific page — not a prompt that might have hallucinated its way past the problem. For a system meant to be trusted with the actual scientific content of a paper, that auditability is worth more than whatever a smarter, fuzzier ingestion step could save you in code.

---

*Ingestion run on: [OctoTools: A Multi-Agent Framework with Extensible Tools for Complex Reasoning](https://aclanthology.org/2026.acl-long.1/), Lu, Chen, Liu, Thapa, Boen & Zou, ACL 2026.*

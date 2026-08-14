"""Atlas document ingestion.

The module keeps the full scientific-document workflow visible in one place:
landing page -> PDF -> page text -> tables/visuals/OCR -> normalized package ->
LLM context.

The ingestion layer intentionally contains no Atlas taxonomy extraction logic.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pymupdf
import requests
from bs4 import BeautifulSoup

from ingestion.models import (
    DocumentPackage,
    LLMContext,
    PageContent,
    PaperMetadata,
    TableContent,
)


# -----------------------------
# Tunable ingestion heuristics
# -----------------------------

# Network / Download
REQUEST_TIMEOUT_SECONDS = 30  # Maximum time allowed for HTTP requests when fetching the paper page or PDF before treating the request as failed.
# PDF Validation
MIN_PDF_SIZE_BYTES = 1024  # Rejects suspiciously small downloads that are unlikely to be valid research PDFs.
# Text Quality
MIN_NATIVE_TEXT_CHARS = 80  # Requires at least 80 non-whitespace characters on a page before native PDF text is considered potentially usable.
MIN_PRINTABLE_RATIO = 0.85  # Requires at least 85% of extracted characters to be printable; otherwise the page may trigger visual/OCR fallback.
# Noise Removal
REPEATED_NOISE_MIN_PAGES = 3  # A header/footer line must appear on at least 3 pages before it can be classified as repeated document noise.
REPEATED_NOISE_PAGE_RATIO = 0.30  # Repeated noise must occur on at least 30% of document pages, preventing accidental removal of legitimate content.
REPEATED_NOISE_EDGE_LINES = 3  # Only the first and last 3 non-empty lines of each page are considered when detecting repeated headers/footers.
# Table Handling
TABLE_COMPLEXITY_THRESHOLD = 0.60  # Tables scoring 0.60 or higher are considered complex enough that the rendered page image is preferred over extracted table text.
TABLE_CONFIDENCE_THRESHOLD = 0.70  # Table extractions below 0.70 confidence are treated as unreliable and trigger visual-page representation.
# Visual Rendering
RENDER_DPI = 175  # Resolution used to render selected visual pages for the multimodal LLM.
# OCR
OCR_DPI = 200  # Resolution used by OCR for pages where native PDF text extraction is unusable.
# Layout Analysis
FULL_WIDTH_BLOCK_RATIO = 0.70  # Text blocks spanning at least 70% of page width are treated as full-width blocks instead of left/right-column content during reading-order reconstruction.
# Figure Detection
FIGURE_IMAGE_AREA_RATIO = 0.05  # A page with a detected figure caption and an image occupying at least 5% of page area is considered a likely visual figure page.
FIGURE_DRAWING_COUNT = 20  # A page with a figure caption and at least 20 vector drawing objects is treated as containing a diagram/chart worth rendering.

USER_AGENT = "AtlasDocumentIngestion/0.1"
# Finds HTTP/HTTPS URLs in extracted page text so code/project/dataset/artifact links can be preserved in page.urls
URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+")
# Detects lines that look like figure captions such as Figure 2 or Fig. 3a; used as a signal that the page may contain a meaningful visual and should be considered for rendering.
FIGURE_CAPTION_PATTERN = re.compile(
    r"^\s*(?:Figure|Fig\.)\s*\d+[A-Za-z]?\b",
    re.IGNORECASE | re.MULTILINE,
)
# Detects table references/captions such as Table 1 or Table 2A; used to associate detected table regions with human-readable table IDs/captions.
TABLE_CAPTION_PATTERN = re.compile(r"\bTable\s+\d+[A-Za-z]?\b", re.IGNORECASE)

# Recognizes common academic section headings even when they are not numbered
KNOWN_SECTION_TITLES = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "analysis",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "references",
    "appendix",
    "acknowledgments",
    "acknowledgements",
}
# Detects numbered section/subsection headings such as 1 Introduction, 3.2 Training Setup, or A.1 Prompt Details.
NUMBERED_SECTION_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)[.)]?\s+[A-Z].{1,100}$"
)
# Detects appendix-style headings such as Appendix A Additional Results or A.2 Hyperparameters.
APPENDIX_SECTION_PATTERN = re.compile(
    r"^(?:Appendix(?:\s+[A-Z])?|[A-Z](?:\.\d+)*)\s+.{1,100}$",
    re.IGNORECASE,
)
# -----------------------------
# Metadata and PDF
# -----------------------------


def resolve_metadata(paper_url: str) -> PaperMetadata:
    """Resolve stable paper metadata and the official PDF URL."""

    response = requests.get(
        paper_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    paper_id = _paper_id_from_url(response.url)
    title = _first_meta(soup, "citation_title") or _page_title(soup)
    authors = _all_meta(soup, "citation_author")

    pdf_url = _first_meta(soup, "citation_pdf_url") or _find_pdf_link(
        soup,
        base_url=response.url,
        paper_id=paper_id,
    )

    venue = (
        _first_meta(soup, "citation_conference_title")
        or _first_meta(soup, "citation_journal_title")
    )
    year = _extract_year(
        _first_meta(soup, "citation_publication_date")
        or _first_meta(soup, "citation_date")
        or paper_id
    )

    if not paper_id:
        raise ValueError(f"Could not resolve paper ID from URL: {paper_url}")
    if not title:
        raise ValueError(f"Could not resolve paper title from: {paper_url}")
    if not pdf_url:
        raise ValueError(f"Could not resolve PDF URL from: {paper_url}")

    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        paper_url=response.url,
        pdf_url=pdf_url,
    )


def download_pdf(metadata: PaperMetadata, output_dir: Path) -> Path:
    """Download the authoritative PDF and validate that it is readable."""

    paper_dir = output_dir / metadata.paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "source.pdf"

    response = requests.get(
        metadata.pdf_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    pdf_bytes = response.content
    if len(pdf_bytes) < MIN_PDF_SIZE_BYTES:
        raise ValueError(
            f"Downloaded PDF is unexpectedly small: {len(pdf_bytes)} bytes"
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        content_type = response.headers.get("Content-Type", "unknown")
        raise ValueError(
            "Downloaded content is not a PDF "
            f"(Content-Type={content_type!r}, URL={metadata.pdf_url})"
        )

    pdf_path.write_bytes(pdf_bytes)

    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                raise ValueError("PDF is encrypted and requires a password")
            if document.page_count <= 0:
                raise ValueError("PDF contains zero pages")
    except Exception:
        pdf_path.unlink(missing_ok=True)
        raise

    return pdf_path


# -----------------------------
# Page extraction and cleaning
# -----------------------------


def extract_pages(pdf_path: Path) -> list[PageContent]:
    """Extract one layout-aware native-text representation per PDF page."""

    pages: list[PageContent] = []

    with pymupdf.open(pdf_path) as document:
        for page_index, pdf_page in enumerate(document):
            raw_text = _extract_layout_text(pdf_page)
            pages.append(
                PageContent(
                    page_number=page_index + 1,
                    raw_text=raw_text,
                    clean_text=clean_text(raw_text),
                )
            )

    return pages


def clean_text(raw_text: str) -> str:
    """Create a lightly normalized text representation for LLM/search use by fixing Unicode artifacts, repairing likely line-break hyphenation, normalizing whitespace, and preserving only meaningful paragraph breaks without modifying the original raw text."""

    text = unicodedata.normalize("NFKC", raw_text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = _repair_line_break_hyphenation(text)

    clean_lines: list[str] = []
    previous_blank = False

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            if clean_lines and not previous_blank:
                clean_lines.append("")
            previous_blank = True
            continue

        clean_lines.append(line)
        previous_blank = False

    return "\n".join(clean_lines).strip()


def evaluate_page_quality(page: PageContent) -> bool:
    """Check whether a page's native PDF text is usable by requiring enough visible characters, a high printable-character ratio, and very few replacement-character artifacts."""

    visible_chars = [char for char in page.raw_text if not char.isspace()]
    if len(visible_chars) < MIN_NATIVE_TEXT_CHARS:
        return False

    printable_count = sum(char.isprintable() for char in visible_chars)
    printable_ratio = printable_count / len(visible_chars)

    replacement_count = page.raw_text.count("�")
    replacement_ratio = replacement_count / max(len(visible_chars), 1)

    return printable_ratio >= MIN_PRINTABLE_RATIO and replacement_ratio < 0.02


def remove_repeated_noise(pages: list[PageContent]) -> None:
    """Remove repeated header and footer noise from each page's clean text by identifying normalized edge lines that recur across enough pages, while preserving the original raw text."""

    if len(pages) < REPEATED_NOISE_MIN_PAGES:
        return

    pages_by_key: dict[str, set[int]] = defaultdict(set)

    for page in pages:
        lines = page.clean_text.splitlines()
        for index in _edge_line_indexes(lines):
            key = _noise_key(lines[index])
            if key:
                pages_by_key[key].add(page.page_number)

    min_occurrences = max(
        REPEATED_NOISE_MIN_PAGES,
        math.ceil(len(pages) * REPEATED_NOISE_PAGE_RATIO),
    )

    repeated_keys = {
        key
        for key, page_numbers in pages_by_key.items()
        if len(page_numbers) >= min_occurrences
    }

    if not repeated_keys:
        return

    for page in pages:
        lines = page.clean_text.splitlines()
        edge_indexes = set(_edge_line_indexes(lines))
        kept_lines: list[str] = []

        for index, line in enumerate(lines):
            key = _noise_key(line)
            if index in edge_indexes and key in repeated_keys:
                continue
            kept_lines.append(line)

        page.clean_text = "\n".join(kept_lines).strip()


def detect_sections(pages: list[PageContent]) -> None:
    """Attach a lightweight current-section label to each page."""

    current_section: str | None = None

    for page in pages:
        first_heading_on_page: str | None = None

        for line in page.clean_text.splitlines():
            if _looks_like_section_heading(line):
                if first_heading_on_page is None:
                    first_heading_on_page = line.strip()
                current_section = line.strip()

        page.section = first_heading_on_page or current_section


# -----------------------------
# Tables
# -----------------------------


def extract_tables(pdf_path: Path) -> list[TableContent]:
    """Detect and extract tables from each PDF page, convert them to Markdown, associate nearby captions, and compute simple complexity and extraction-confidence scores for downstream visual/text representation decisions."""

    tables: list[TableContent] = []

    with pymupdf.open(pdf_path) as document:
        for page_index, pdf_page in enumerate(document):
            page_number = page_index + 1

            try:
                finder = pdf_page.find_tables()
            except Exception:
                continue

            for table_index, table in enumerate(finder.tables, start=1):
                cells = table.extract()
                markdown = table.to_markdown(clean=True).strip()
                caption = _find_table_caption(pdf_page, table.bbox)

                table_id = _table_id_from_caption(caption)
                if not table_id:
                    table_id = f"page_{page_number}_table_{table_index}"

                complexity = calculate_table_complexity(
                    cells=cells,
                    row_count=table.row_count,
                    column_count=table.col_count,
                )
                confidence = estimate_table_confidence(
                    cells=cells,
                    row_count=table.row_count,
                    column_count=table.col_count,
                )

                tables.append(
                    TableContent(
                        table_id=table_id,
                        page_number=page_number,
                        caption=caption,
                        text=markdown or None,
                        complexity_score=complexity,
                        extraction_confidence=confidence,
                    )
                )

    return tables


def calculate_table_complexity(
    cells: list[list[str | None]],
    row_count: int,
    column_count: int,
) -> float:
    """Estimate how complex or risky a text-only table representation is using table size, empty cells, multiline cells, and irregular row structure."""

    flat_cells = [cell for row in cells for cell in row]
    total_cells = max(len(flat_cells), 1)

    empty_ratio = sum(not (cell or "").strip() for cell in flat_cells) / total_cells
    multiline_ratio = sum("\n" in (cell or "") for cell in flat_cells) / total_cells
    irregular_rows = any(len(row) != column_count for row in cells)

    score = 0.0
    if column_count >= 5:
        score += 0.20
    if row_count >= 10:
        score += 0.15
    if empty_ratio >= 0.15:
        score += 0.20
    if multiline_ratio >= 0.10:
        score += 0.15
    if irregular_rows:
        score += 0.30

    return min(score, 1.0)


def estimate_table_confidence(
    cells: list[list[str | None]],
    row_count: int,
    column_count: int,
) -> float:
    """Estimate how reliable the extracted table structure is based on valid dimensions, empty-cell ratio, irregular rows, and degenerate one-row or one-column tables."""

    if row_count <= 0 or column_count <= 0 or not cells:
        return 0.0

    flat_cells = [cell for row in cells for cell in row]
    total_cells = max(len(flat_cells), 1)

    empty_ratio = sum(not (cell or "").strip() for cell in flat_cells) / total_cells
    irregular_rows = sum(len(row) != column_count for row in cells)

    confidence = 1.0
    confidence -= min(empty_ratio * 0.50, 0.40)
    confidence -= min(irregular_rows * 0.15, 0.30)

    if row_count == 1 or column_count == 1:
        confidence -= 0.15

    return max(0.0, min(confidence, 1.0))


# -----------------------------
# Visual selection, rendering, OCR
# -----------------------------


def select_visual_pages(
    pages: list[PageContent],
    tables: list[TableContent],
    pdf_path: Path,
) -> None:
    """Mark pages that need visual evidence by checking native-text quality, table complexity or extraction confidence, figure candidates, scanned-page signals, and equation-like content that may not be reliably represented by text alone."""
    tables_by_page: dict[int, list[TableContent]] = defaultdict(list)
    for table in tables:
        tables_by_page[table.page_number].append(table)

    with pymupdf.open(pdf_path) as document:
        for page, pdf_page in zip(pages, document, strict=True):
            native_text_usable = evaluate_page_quality(page)

            if not native_text_usable:
                _add_visual_reason(page, "corrupted_text")

                if _large_image_ratio(pdf_page) >= 0.50:
                    _add_visual_reason(page, "scanned_page")

            for table in tables_by_page.get(page.page_number, []):
                if table.complexity_score >= TABLE_COMPLEXITY_THRESHOLD:
                    _add_visual_reason(page, "complex_table")
                elif table.extraction_confidence < TABLE_CONFIDENCE_THRESHOLD:
                    _add_visual_reason(page, "poor_table_extraction")

            if _page_contains_figure_candidate(pdf_page, page.raw_text):
                _add_visual_reason(page, "figure")

            if not native_text_usable and _contains_equation_like_text(page.raw_text):
                _add_visual_reason(page, "equation")


def render_visual_pages(
    pdf_path: Path,
    pages: list[PageContent],
) -> None:
    """Render only pages already selected as visual evidence."""

    visual_dir = pdf_path.parent / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as document:
        for page in pages:
            if not page.needs_visual:
                continue

            pdf_page = document[page.page_number - 1]
            pixmap = pdf_page.get_pixmap(dpi=RENDER_DPI, alpha=False)
            image_path = visual_dir / f"page_{page.page_number:03d}.png"
            pixmap.save(str(image_path))
            page.image_path = image_path


def apply_ocr_fallback(
    pdf_path: Path,
    pages: list[PageContent],
) -> None:
    """Use OCR only for pages whose native text is unusable.

    PyMuPDF delegates OCR to Tesseract. If Tesseract is unavailable, the page is
    retained with its native extraction and a warning.
    """

    with pymupdf.open(pdf_path) as document:
        for page in pages:
            if evaluate_page_quality(page):
                continue

            pdf_page = document[page.page_number - 1]

            try:
                text_page = pdf_page.get_textpage_ocr(
                    language="eng",
                    dpi=OCR_DPI,
                    full=True,
                )
                ocr_text = pdf_page.get_text(
                    "text",
                    textpage=text_page,
                    sort=True,
                ).strip()
            except Exception as exc:
                page.warnings.append(f"OCR failed: {exc}")
                continue

            if not ocr_text:
                page.warnings.append("OCR returned no text")
                continue

            page.clean_text = clean_text(ocr_text)
            page.text_source = "ocr"


def resolve_table_representations(
    pages: list[PageContent],
    tables: list[TableContent],
) -> None:
    """Mark tables to use the rendered page image instead of separate extracted table text whenever their page has been selected and rendered as visual evidence, preventing duplicate or conflicting table representations in the LLM context."""
    visual_page_numbers = {
        page.page_number
        for page in pages
        if page.needs_visual and page.image_path is not None
    }

    for table in tables:
        table.use_visual = table.page_number in visual_page_numbers


# -----------------------------
# URLs, validation, context
# -----------------------------


def extract_urls(pages: list[PageContent], pdf_path: Path) -> None:
    """Preserve visible URLs and PDF hyperlink annotations per page."""

    with pymupdf.open(pdf_path) as document:
        for page, pdf_page in zip(pages, document, strict=True):
            urls = [_strip_url_punctuation(url) for url in URL_PATTERN.findall(page.raw_text)]

            for link in pdf_page.get_links():
                uri = link.get("uri")
                if isinstance(uri, str) and uri.startswith(("http://", "https://")):
                    urls.append(uri)

            page.urls = _stable_unique(urls)


def validate_document(document: DocumentPackage) -> None:
    """Validate important ingestion invariants before downstream use."""

    if not document.pdf_path.exists():
        raise ValueError(f"PDF does not exist: {document.pdf_path}")

    if document.page_count <= 0:
        raise ValueError("Document contains zero pages")

    if len(document.pages) != document.page_count:
        raise ValueError("Page count does not match extracted pages")

    expected_pages = list(range(1, document.page_count + 1))
    actual_pages = [page.page_number for page in document.pages]

    if actual_pages != expected_pages:
        raise ValueError("PDF pages are missing or out of order")

    for table in document.tables:
        if not 1 <= table.page_number <= document.page_count:
            raise ValueError(
                f"{table.table_id} references invalid page {table.page_number}"
            )

    with pymupdf.open(document.pdf_path) as pdf:
        if pdf.page_count != document.page_count:
            raise ValueError("Stored PDF page count changed during ingestion")


def build_llm_context(document: DocumentPackage) -> LLMContext:
    """Assemble page-traceable text plus selected visual-page attachments."""

    parts: list[str] = []

    metadata = document.metadata
    parts.append(
        "PAPER METADATA\n\n"
        f"Paper ID: {metadata.paper_id}\n"
        f"Title: {metadata.title}\n"
        f"Authors: {', '.join(metadata.authors) if metadata.authors else 'Unknown'}\n"
        f"Venue: {metadata.venue or 'Unknown'}\n"
        f"Year: {metadata.year or 'Unknown'}\n"
        f"PDF URL: {metadata.pdf_url}"
    )

    page_parts = ["FULL PAPER TEXT"]
    for page in document.pages:
        page_parts.append(
            f"=== PAGE {page.page_number} ===\n"
            f"{page.clean_text}"
        )
    parts.append("\n\n".join(page_parts))

    text_tables = [
        table
        for table in document.tables
        if not table.use_visual and table.text
    ]
    if text_tables:
        table_parts = ["TABLES NOT REPRESENTED BY IMAGES"]
        for table in text_tables:
            content = [
                f"=== PAGE {table.page_number} / {table.table_id} ==="
            ]
            if table.caption:
                content.append(table.caption)
            content.append(table.text or "")
            table_parts.append("\n".join(content))
        parts.append("\n\n".join(table_parts))

    artifact_lines: list[str] = []
    for page in document.pages:
        artifact_urls = [url for url in page.urls if _looks_like_artifact_url(url)]
        if artifact_urls:
            artifact_lines.append(
                f"=== PAGE {page.page_number} ===\n" + "\n".join(artifact_urls)
            )

    if artifact_lines:
        parts.append("ARTIFACT LINKS\n\n" + "\n\n".join(artifact_lines))

    visual_pages = [
        page.image_path
        for page in document.pages
        if page.needs_visual and page.image_path is not None
    ]

    if visual_pages:
        visual_lines = ["VISUAL PAGES"]
        visual_lines.extend(
            f"Page {page.page_number}: {', '.join(page.visual_reason)}"
            for page in document.pages
            if page.needs_visual and page.image_path is not None
        )
        parts.append("\n".join(visual_lines))

    return LLMContext(
        text="\n\n".join(parts).strip(),
        visual_pages=visual_pages,
    )


# -----------------------------
# Main orchestration
# -----------------------------


def ingest_document(
    paper_url: str,
    output_dir: Path,
) -> DocumentPackage:
    """Run the complete Atlas document-ingestion workflow."""

    # Resolve stable metadata and official PDF URL.
    metadata = resolve_metadata(paper_url)

    # Download and validate the authoritative PDF.
    pdf_path = download_pdf(metadata, output_dir)

    # Preserve pages and build native raw + clean text.
    pages = extract_pages(pdf_path)

    # Tables are extracted separately from normal page prose.
    tables = extract_tables(pdf_path)

    # Decide which pages genuinely need visual evidence.
    select_visual_pages(pages, tables, pdf_path)

    # Render selected pages once; these images are also useful for OCR/debugging.
    render_visual_pages(pdf_path, pages)

    # OCR only when native PDF text is unusable.
    apply_ocr_fallback(pdf_path, pages)

    # Suppress repeated header/footer noise from final clean text only.
    remove_repeated_noise(pages)

    # Detect structure while keeping all pages, including appendices.
    detect_sections(pages)

    # Preserve reproducibility/artifact URLs and footnote links.
    extract_urls(pages, pdf_path)

    # Avoid sending separate table text when the page image is authoritative.
    resolve_table_representations(pages, tables)

    # Build and validate the normalized document package.
    document = DocumentPackage(
        metadata=metadata,
        pdf_path=pdf_path,
        page_count=len(pages),
        pages=pages,
        tables=tables,
    )
    validate_document(document)

    return document


# -----------------------------
# Internal helpers
# -----------------------------


def _first_meta(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.find("meta", attrs={"name": name})
    if tag and tag.get("content"):
        return str(tag["content"]).strip()
    return None


def _all_meta(soup: BeautifulSoup, name: str) -> list[str]:
    values: list[str] = []
    for tag in soup.find_all("meta", attrs={"name": name}):
        content = tag.get("content")
        if content:
            values.append(str(content).strip())
    return values


def _paper_id_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.rsplit("/", maxsplit=1)[-1].removesuffix(".pdf")


def _page_title(soup: BeautifulSoup) -> str | None:
    heading = soup.find("h2") or soup.find("h1")
    if heading:
        return heading.get_text(" ", strip=True)
    if soup.title:
        return soup.title.get_text(" ", strip=True)
    return None


def _find_pdf_link(
    soup: BeautifulSoup,
    base_url: str,
    paper_id: str,
) -> str | None:
    preferred_suffix = f"/{paper_id}.pdf"

    candidates: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        absolute_url = urljoin(base_url, href)

        if urlparse(absolute_url).path.lower().endswith(".pdf"):
            candidates.append(absolute_url)

    for candidate in candidates:
        if urlparse(candidate).path.endswith(preferred_suffix):
            return candidate

    for candidate in candidates:
        if "checklist" not in candidate.lower():
            return candidate

    return None


def _extract_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group()) if match else None


def _extract_layout_text(page: pymupdf.Page) -> str:
    """Extract text blocks with their coordinates, keep only non-empty text blocks, reorder them using page layout, and return readable page text for downstream cleaning and LLM context."""
    raw_blocks = page.get_text("blocks")

    blocks: list[tuple[float, float, float, float, str]] = []
    for block in raw_blocks:
        if len(block) < 7:
            continue

        x0, y0, x1, y1, text, _block_no, block_type = block[:7]
        if block_type != 0:
            continue

        text = str(text).strip()
        if not text:
            continue

        blocks.append((float(x0), float(y0), float(x1), float(y1), text))

    if not blocks:
        return ""

    ordered = _order_blocks(blocks, page.rect.width)
    return "\n\n".join(block[4] for block in ordered).strip()


def _order_blocks(
    blocks: list[tuple[float, float, float, float, str]],
    page_width: float,
) -> list[tuple[float, float, float, float, str]]:
    """Reconstruct page reading order by separating full-width blocks from column blocks, detecting likely two-column layouts, and ordering content from top to bottom in the correct column sequence."""

    mid_x = page_width / 2

    full_width = [
        block
        for block in blocks
        if (block[2] - block[0]) >= page_width * FULL_WIDTH_BLOCK_RATIO
    ]
    column_blocks = [block for block in blocks if block not in full_width]

    left_count = sum(_block_center_x(block) < mid_x for block in column_blocks)
    right_count = len(column_blocks) - left_count

    if left_count < 2 or right_count < 2:
        return sorted(blocks, key=lambda block: (block[1], block[0]))

    ordered: list[tuple[float, float, float, float, str]] = []
    remaining = list(column_blocks)

    for wide_block in sorted(full_width, key=lambda block: (block[1], block[0])):
        above = [block for block in remaining if block[1] < wide_block[1]]
        ordered.extend(_order_column_region(above, mid_x))

        above_ids = {id(block) for block in above}
        remaining = [block for block in remaining if id(block) not in above_ids]

        ordered.append(wide_block)

    ordered.extend(_order_column_region(remaining, mid_x))
    return ordered


def _order_column_region(
    blocks: list[tuple[float, float, float, float, str]],
    mid_x: float,
) -> list[tuple[float, float, float, float, str]]:
    """Order blocks within a two-column region by placing left-column content first, then right-column content, with each column sorted from top to bottom."""
    left = [block for block in blocks if _block_center_x(block) < mid_x]
    right = [block for block in blocks if _block_center_x(block) >= mid_x]

    left.sort(key=lambda block: (block[1], block[0]))
    right.sort(key=lambda block: (block[1], block[0]))
    return left + right


def _block_center_x(block: tuple[float, float, float, float, str]) -> float:
    """Return the horizontal center position of a text block so it can be classified as belonging to the left or right side of the page."""
    return (block[0] + block[2]) / 2


def _repair_line_break_hyphenation(text: str) -> str:
    """Repair likely words split across PDF line boundaries by removing a hyphen-newline sequence when it appears between alphabetic word fragments."""
    pattern = re.compile(r"\b([A-Za-z]{3,})-\n([a-z]{2,5})\b")
    return pattern.sub(lambda match: match.group(1) + match.group(2), text)


def _edge_line_indexes(lines: list[str]) -> list[int]:
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    if not non_empty:
        return []

    indexes = (
        non_empty[:REPEATED_NOISE_EDGE_LINES]
        + non_empty[-REPEATED_NOISE_EDGE_LINES:]
    )
    return sorted(set(indexes))


def _noise_key(line: str) -> str | None:
    normalized = re.sub(r"\s+", " ", line).strip().lower()

    if not normalized or len(normalized) > 120:
        return None
    if "http://" in normalized or "https://" in normalized:
        return None

    # Treat changing physical page numbers as the same repeated footer/header.
    normalized = re.sub(r"\b\d+\b", "<number>", normalized)
    return normalized


def _looks_like_section_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 120:
        return False

    lowered = text.lower().rstrip(":")
    if lowered in KNOWN_SECTION_TITLES:
        return True

    if text.lower().startswith(("table ", "figure ", "fig. ")):
        return False

    if len(text.split()) > 14:
        return False

    return bool(
        NUMBERED_SECTION_PATTERN.match(text)
        or APPENDIX_SECTION_PATTERN.match(text)
    )


def _find_table_caption(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float],
) -> str | None:
    """Find the most likely table caption by searching a small text region immediately above and below the detected table bounding box."""
    x0, y0, x1, y1 = bbox

    above = pymupdf.Rect(
        0,
        max(0, y0 - 70),
        page.rect.width,
        y0,
    )
    below = pymupdf.Rect(
        0,
        y1,
        page.rect.width,
        min(page.rect.height, y1 + 50),
    )

    for region in (above, below):
        text = page.get_textbox(region)
        for line in text.splitlines():
            if TABLE_CAPTION_PATTERN.search(line):
                return re.sub(r"\s+", " ", line).strip()

    return None


def _table_id_from_caption(caption: str | None) -> str | None:
    """Extract a human-readable table identifier such as 'Table 1' from a detected table caption."""
    if not caption:
        return None
    match = TABLE_CAPTION_PATTERN.search(caption)
    return match.group(0).title() if match else None


def _add_visual_reason(page: PageContent, reason: str) -> None:
    """Mark a page as requiring visual evidence and add the given reason once without creating duplicate reason entries."""
    if reason not in page.visual_reason:
        page.visual_reason.append(reason)
    page.needs_visual = True


def _large_image_ratio(page: pymupdf.Page) -> float:
    """Return the largest embedded-image area as a fraction of the total page area, used to identify image-heavy or potentially scanned pages."""
    page_area = max(page.rect.width * page.rect.height, 1.0)
    max_ratio = 0.0

    for info in page.get_image_info():
        bbox = info.get("bbox")
        if not bbox:
            continue

        rect = pymupdf.Rect(bbox)
        ratio = max(rect.width * rect.height, 0.0) / page_area
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def _page_contains_figure_candidate(page: pymupdf.Page, raw_text: str) -> bool:
    """Detect whether a page likely contains a meaningful figure by requiring a figure caption plus either a sufficiently large embedded image or many vector drawing elements."""
    if not FIGURE_CAPTION_PATTERN.search(raw_text):
        return False

    if _large_image_ratio(page) >= FIGURE_IMAGE_AREA_RATIO:
        return True

    try:
        return len(page.get_drawings()) >= FIGURE_DRAWING_COUNT
    except Exception:
        return False


def _contains_equation_like_text(text: str) -> bool:
    math_symbols = ("=", "∑", "∏", "∫", "≤", "≥", "λ", "α", "β")
    return any(symbol in text for symbol in math_symbols)


def _strip_url_punctuation(url: str) -> str:
    return url.rstrip(".,;:!?)]}")


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _looks_like_artifact_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")

    if host in {"doi.org", "dx.doi.org", "aclanthology.org"}:
        return False

    artifact_hosts = {
        "github.com",
        "gitlab.com",
        "huggingface.co",
        "zenodo.org",
        "figshare.com",
        "osf.io",
        "kaggle.com",
    }

    return host in artifact_hosts or host.endswith(".github.io")

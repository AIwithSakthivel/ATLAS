from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PaperMetadata:
    """Stable metadata resolved from the paper landing page."""

    paper_id: str
    title: str
    authors: list[str]
    venue: str | None
    year: int | None
    paper_url: str
    pdf_url: str


@dataclass
class PageContent:
    """Content and evidence retained for one physical PDF page."""

    page_number: int
    raw_text: str
    clean_text: str

    section: str | None = None
    text_source: str = "native"

    needs_visual: bool = False
    visual_reason: list[str] = field(default_factory=list)
    image_path: Path | None = None

    urls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TableContent:
    """Separate representation of one detected table."""

    table_id: str
    page_number: int
    caption: str | None
    text: str | None
    complexity_score: float
    extraction_confidence: float
    use_visual: bool = False


@dataclass
class DocumentPackage:
    """Normalized output of the document-ingestion layer."""

    metadata: PaperMetadata
    pdf_path: Path
    page_count: int
    pages: list[PageContent]
    tables: list[TableContent]
    warnings: list[str] = field(default_factory=list)


@dataclass
class LLMContext:
    """Text plus selected page images sent to the downstream model."""

    text: str
    visual_pages: list[Path]

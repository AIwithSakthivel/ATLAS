"""
Run Atlas document ingestion for a list of papers.

Usage
-----
    python run_ingestion.py --input results/test.json --output data/ingestion

--input and --output are required on the command line rather than
hardcoded, so the same script runs against any venue's paper list
(acl2026/results/*.json, iclr2026/results/*.json, a one-off test file,
etc.) without editing this file.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from ingestion.core import build_llm_context, ingest_document

# Log folder lives under this script's own directory (the project root),
# independent of --output -- so it's the same place regardless of which
# papers/venue you're ingesting into.
LOG_DIR = Path(__file__).resolve().parent / "logs"

DATETIME_FORMAT = "%d-%m-%Y %H:%M"

logger = logging.getLogger("atlas.ingestion")


def setup_logging() -> Path:
    """Configures the module logger to write to both the console and a
    timestamped file under LOG_DIR. Returns the log file path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if setup runs twice

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_path


def format_duration(seconds: float) -> str:
    """Minutes by default; switches to hours once the total exceeds 60
    minutes, per the requested reporting rule."""
    minutes = seconds / 60
    if minutes > 60:
        return f"{minutes / 60:.2f} hours"
    return f"{minutes:.2f} minutes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Atlas document ingestion for a list of papers.")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the JSON file listing papers to ingest (e.g. results/test.json)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Directory to write ingestion output into (e.g. data/ingestion)",
    )
    return parser.parse_args()


def load_papers(input_path: Path) -> list[dict]:
    """Load paper records from the JSON input file."""
    papers = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(papers, list):
        raise ValueError(f"Expected a list of papers in {input_path}")

    return papers


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output

    log_path = setup_logging()
    start_time = datetime.now()
    logger.info(f"Ingestion started at {start_time.strftime(DATETIME_FORMAT)}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")

    papers = load_papers(input_path)
    logger.info(f"Found {len(papers)} papers in {input_path}")

    successful = 0
    failed = 0

    for index, paper in enumerate(papers, start=1):
        anthology_id = paper.get("anthology_id", f"paper_{index}")
        paper_url = paper.get("page_url")
        title = paper.get("title", "")

        # Expected source.pdf location created by ingest_document().
        # Keeping this outside the try block lets us clean it up even
        # if ingestion fails after downloading the PDF.
        source_pdf_path = output_dir / anthology_id / "source.pdf"

        logger.info(f"[{index}/{len(papers)}] {anthology_id} | Title: {title}")

        if not paper_url:
            failed += 1
            logger.warning(f"[{index}/{len(papers)}] {anthology_id}: skipped, missing page_url")
            continue

        try:
            document = ingest_document(
                paper_url=paper_url,
                output_dir=output_dir,
            )

            context = build_llm_context(document)

            paper_output_dir = document.pdf_path.parent
            context_path = paper_output_dir / "llm_context.txt"

            context_path.write_text(
                context.text,
                encoding="utf-8",
            )

            visual_pages = [
                page.page_number
                for page in document.pages
                if page.needs_visual
            ]

            successful += 1

            logger.info(
                f"[{index}/{len(papers)}] {anthology_id}: OK | "
                f"pages={document.page_count} tables={len(document.tables)} "
                f"visual_pages={visual_pages} output={paper_output_dir} "
                f"context={context_path}"
            )

        except Exception as exc:
            failed += 1
            logger.error(f"[{index}/{len(papers)}] {anthology_id}: FAILED, {exc}")

        finally:
            # source.pdf is temporary and is no longer needed after
            # ingestion/context generation.
            if source_pdf_path.exists():
                source_pdf_path.unlink()
                logger.info(f"[{index}/{len(papers)}] {anthology_id}: deleted temporary PDF {source_pdf_path}")

    end_time = datetime.now()
    elapsed_seconds = (end_time - start_time).total_seconds()

    logger.info(f"Ingestion finished at {end_time.strftime(DATETIME_FORMAT)}")
    logger.info(f"Total time taken: {format_duration(elapsed_seconds)}")
    logger.info("Ingestion complete")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")


if __name__ == "__main__":
    main()
"""
embeddings/run_embeddings.py

Description
-----------
Command-line entry point for the embedding creation pipeline.

The script:
    1. Reads extraction/results.jsonl
    2. Processes only records with status="success"
    3. Resolves the paper title
    4. Builds one paper-level embedding unit
    5. Builds eight independent facet-level embedding units
    6. Checks ChromaDB for unchanged existing embeddings
    7. Sends only new/changed texts to the existing OCI embedding client
    8. Stores successful embeddings in local persistent ChromaDB
    9. Writes a per-unit manifest and human-readable run log
    10. Verifies embeddings by reading them back from ChromaDB
    11. Writes a final summary.json

Default storage:
    embeddings/chroma_db/
        Local persistent ChromaDB

    embeddings/runs/<run_id>/
        manifest.jsonl
        run.log
        summary.json

The embedding model is not configured here. It comes from:

    .env
        -> common/config.py
        -> common/oci_client.py
        -> client.embedding_model_id

Typical usage:

    python -m embeddings.run_embeddings

If results.jsonl does not contain paper titles:

    python -m embeddings.run_embeddings \
        --paper-metadata test.json

Custom paths:

    python -m embeddings.run_embeddings \
        --input extraction/results.jsonl \
        --paper-metadata test.json \
        --chroma-path embeddings/chroma_db
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from common.oci_client import (
    DEFAULT_EMBED_BATCH_SIZE,
    build_client,
)
from embeddings.core import (
    build_facet_embedding_unit,
    build_paper_embedding_unit,
    create_chroma_collections,
    embed_and_store_units,
    ensure_unique_embedding_ids,
    get_taxonomy_value,
    iter_results_jsonl,
    load_title_map,
    make_empty_facet_result,
    make_input_failure_result,
    partition_existing_units,
    resolve_title,
    verify_units,
    write_manifest_record,
)
from embeddings.models import (
    FACET_EMBEDDING_FIELDS,
    MANDATORY_TAXONOMY_FIELDS,
    EmbeddingResult,
    EmbeddingUnit,
    RunSummary,
)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

EMBEDDINGS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EMBEDDINGS_DIR.parent

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "extraction"
    / "results"
    / "results.jsonl"
)

DEFAULT_CHROMA_PATH = (
    EMBEDDINGS_DIR
    / "chroma_db"
)

DEFAULT_RUNS_PATH = (
    EMBEDDINGS_DIR
    / "runs"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    """argparse validator for positive integer arguments."""
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-level and facet-level embeddings "
            "from taxonomy extraction results."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Extraction results.jsonl path. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--paper-metadata",
        type=Path,
        default=None,
        help=(
            "Optional JSON/JSONL file containing paper_id and title. "
            "Required only when titles are not already available in "
            "results.jsonl."
        ),
    )

    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=DEFAULT_CHROMA_PATH,
        help=(
            "Local ChromaDB persistence directory. "
            f"Default: {DEFAULT_CHROMA_PATH}"
        ),
    )

    parser.add_argument(
        "--runs-path",
        type=Path,
        default=DEFAULT_RUNS_PATH,
        help=(
            "Directory containing embedding run logs/manifests. "
            f"Default: {DEFAULT_RUNS_PATH}"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help=(
            "Number of embedding texts sent per OCI request. "
            f"Default: {DEFAULT_EMBED_BATCH_SIZE}"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def create_run_id() -> str:
    """Create a unique human-readable run ID."""
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    suffix = uuid.uuid4().hex[:8]

    return f"{timestamp}_{suffix}"


def configure_logging(log_path: Path) -> None:
    """Configure console + file logging for this run."""
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s - "
            "%(message)s"
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicated handlers when executed repeatedly in an
    # interactive Python process.
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------

def record_result(
    *,
    manifest_handle,
    summary: RunSummary,
    result: EmbeddingResult,
) -> None:
    """Write one manifest record and update run counters."""
    write_manifest_record(
        manifest_handle,
        result,
    )

    summary.record_result(result)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
) -> int:
    """Execute one embedding creation run."""
    logger = logging.getLogger(__name__)

    logger.info(
        "Embedding run started: run_id=%s",
        run_id,
    )
    logger.info(
        "Input file: %s",
        args.input,
    )
    logger.info(
        "Chroma path: %s",
        args.chroma_path,
    )
    logger.info(
        "Batch size: %d",
        args.batch_size,
    )

    # Existing common/oci_client.py handles OCI configuration,
    # authentication, API calls, and retry behavior.
    client = build_client()
    model_id = client.embedding_model_id

    logger.info(
        "Embedding model: %s",
        model_id,
    )

    title_map: dict[str, str] = {}

    if args.paper_metadata is not None:
        title_map = load_title_map(
            args.paper_metadata
        )

        logger.info(
            "Loaded %d paper titles from %s",
            len(title_map),
            args.paper_metadata,
        )

    # Keep the Chroma client alive for the full pipeline run.
    _chroma_client, collections = create_chroma_collections(
        args.chroma_path
    )

    summary = RunSummary(
        run_id=run_id,
        input_file=str(args.input.resolve()),
        embedding_model=model_id,
    )

    manifest_path = run_dir / "manifest.jsonl"
    summary_path = run_dir / "summary.json"

    units: list[EmbeddingUnit] = []

    with manifest_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as manifest_handle:

        # ---------------------------------------------------------------
        # Build embedding units
        # ---------------------------------------------------------------

        for record in iter_results_jsonl(args.input):
            summary.papers_read += 1

            if record.get("status") != "success":
                logger.info(
                    "Skipping paper_id=%s status=%s",
                    record["paper_id"],
                    record.get("status"),
                )
                continue

            summary.successful_papers_read += 1

            paper_id = record["paper_id"]

            # -----------------------------------------------------------
            # Paper-level embedding
            # -----------------------------------------------------------

            title = resolve_title(
                record,
                title_map,
            )

            try:
                paper_unit = build_paper_embedding_unit(
                    record,
                    title,
                )

            except ValueError as exc:
                logger.error(
                    "Paper embedding input failed: paper_id=%s error=%s",
                    paper_id,
                    exc,
                )

                result = make_input_failure_result(
                    run_id=run_id,
                    paper_id=paper_id,
                    embedding_type="paper",
                    model_id=model_id,
                    error=str(exc),
                )

                record_result(
                    manifest_handle=manifest_handle,
                    summary=summary,
                    result=result,
                )

            else:
                units.append(paper_unit)

            # -----------------------------------------------------------
            # Facet-level embeddings
            # -----------------------------------------------------------

            response = record.get("response")

            if not isinstance(response, dict):
                raise ValueError(
                    f"{paper_id}: missing valid extraction response."
                )

            for facet in FACET_EMBEDDING_FIELDS:
                value = get_taxonomy_value(
                    response,
                    facet,
                )

                # Optional taxonomy fields are allowed to be empty.
                if not value and facet not in MANDATORY_TAXONOMY_FIELDS:
                    logger.info(
                        "Skipping empty optional facet: "
                        "paper_id=%s facet=%s",
                        paper_id,
                        facet,
                    )

                    result = make_empty_facet_result(
                        run_id=run_id,
                        paper_id=paper_id,
                        facet=facet,
                        model_id=model_id,
                    )

                    record_result(
                        manifest_handle=manifest_handle,
                        summary=summary,
                        result=result,
                    )

                    continue

                # Missing mandatory taxonomy fields indicate an extraction/data
                # contract problem and should remain failures.
                if not value:
                    error = (
                        f"{paper_id}: mandatory taxonomy facet "
                        f"{facet!r} is empty."
                    )

                    logger.error(error)

                    result = make_input_failure_result(
                        run_id=run_id,
                        paper_id=paper_id,
                        embedding_type="facet",
                        facet=facet,
                        model_id=model_id,
                        error=error,
                    )

                    record_result(
                        manifest_handle=manifest_handle,
                        summary=summary,
                        result=result,
                    )

                    continue

                facet_unit = build_facet_embedding_unit(
                    record,
                    facet,
                )

                units.append(facet_unit)

        ensure_unique_embedding_ids(units)

        summary.embedding_units_created = len(units)

        logger.info(
            "Papers read=%d successful_papers=%d embedding_units=%d",
            summary.papers_read,
            summary.successful_papers_read,
            summary.embedding_units_created,
        )

        # Fast lookup used during verification selection.
        unit_by_id = {
            unit.embedding_id: unit
            for unit in units
        }

        # ---------------------------------------------------------------
        # Skip unchanged existing embeddings
        # ---------------------------------------------------------------

        pending_units, skipped_results = partition_existing_units(
            units,
            collections,
            run_id=run_id,
            model_id=model_id,
        )

        summary.pending_units = len(pending_units)

        logger.info(
            "Existing unchanged=%d pending=%d",
            len(skipped_results),
            len(pending_units),
        )

        verification_units: list[EmbeddingUnit] = []

        for result in skipped_results:
            record_result(
                manifest_handle=manifest_handle,
                summary=summary,
                result=result,
            )

            verification_units.append(
                unit_by_id[result.embedding_id]
            )

        # ---------------------------------------------------------------
        # API embedding + Chroma persistence
        # ---------------------------------------------------------------

        if pending_units:
            summary.api_batches = math.ceil(
                len(pending_units)
                / args.batch_size
            )

            for result in embed_and_store_units(
                pending_units,
                client=client,
                collections=collections,
                run_id=run_id,
                model_id=model_id,
                batch_size=args.batch_size,
            ):
                record_result(
                    manifest_handle=manifest_handle,
                    summary=summary,
                    result=result,
                )

                if result.status == "success":
                    verification_units.append(
                        unit_by_id[result.embedding_id]
                    )

        # ---------------------------------------------------------------
        # Verification
        # ---------------------------------------------------------------

        logger.info(
            "Verifying %d stored embeddings.",
            len(verification_units),
        )

        verification_errors, embedding_dimension = verify_units(
            verification_units,
            collections=collections,
            model_id=model_id,
        )

        summary.embedding_dimension = embedding_dimension
        summary.verification_errors = len(
            verification_errors
        )

        for error in verification_errors:
            logger.error(
                "Verification error: %s",
                error,
            )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary.to_dict(),
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    logger.info(
        "Embedding run completed: "
        "papers=%d "
        "paper_success=%d "
        "paper_skipped=%d "
        "paper_failed=%d "
        "facet_success=%d "
        "facet_skipped=%d "
        "facet_failed=%d "
        "verification_errors=%d",
        summary.successful_papers_read,
        summary.paper_embeddings_success,
        summary.paper_embeddings_skipped,
        summary.paper_embeddings_failed,
        summary.facet_embeddings_success,
        summary.facet_embeddings_skipped,
        summary.facet_embeddings_failed,
        summary.verification_errors,
    )

    logger.info(
        "Manifest: %s",
        manifest_path,
    )
    logger.info(
        "Summary: %s",
        summary_path,
    )

    if (
        summary.total_failures > 0
        or summary.verification_errors > 0
    ):
        return 1

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""
    args = parse_args()

    run_id = create_run_id()
    run_dir = args.runs_path / run_id
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    configure_logging(
        run_dir / "run.log"
    )

    try:
        return run_pipeline(
            args=args,
            run_id=run_id,
            run_dir=run_dir,
        )

    except Exception:
        logging.getLogger(__name__).exception(
            "Embedding run aborted due to an unexpected error."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
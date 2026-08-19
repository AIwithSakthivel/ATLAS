#!/usr/bin/env python3
"""Concurrent, resumable LLM-based multimodal extraction runner.

This script processes one artifact directory per model request. Each artifact is
expected to contain ``llm_context.txt`` or ``llm_context.py`` and may contain a
``Visuals`` directory with page images.

The runner is intentionally a single-file science solution:
- deterministic artifact discovery and 1-based inclusive ``--limit`` slicing;
- bounded concurrent LLM requests with adaptive AIMD concurrency control;
- thread-local LLM clients so workers do not share an SDK client instance;
- provider-specific authentication, model settings, and transport/service retries stay inside the client adapter;
- strict validation of the extraction schema;
- append-only ``results.jsonl`` / ``failures.jsonl`` for checkpointing;
- automatic resume by skipping paper IDs already present in ``results.jsonl``;
- run metadata with start/end/elapsed time and prompt/model provenance.

The supplied LLM client is dynamically imported. An optional client config
module can be loaded into the same synthetic package so relative ``.config``
imports keep working.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import mimetypes
import re
import sys
import threading
import time
import types
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


LOGGER = logging.getLogger("extraction")

REQUIRED_FIELDS = (
    "problem_gap",
    "claimed_contribution",
    "technical_approach",
    "datasets_benchmarks",
    "contribution_type",
    "evaluation_summary",
    "reproducibility",
    "limitations_failures",
)
REQUIRED_FIELD_KEYS = {"value", "source_span", "confidence"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CONTEXT_FILENAMES = ("llm_context.txt", "llm_context.py")

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Job:
    """One paper/artifact extraction request."""

    position: int
    paper_id: str
    artifact_dir: Path
    context_path: Path
    image_paths: tuple[Path, ...]


@dataclass
class JobOutcome:
    """Result returned from a worker to the main writer thread."""

    paper_id: str
    success: bool
    record: dict[str, Any]
    latency_seconds: float
    attempts: int
    throttles: int


@dataclass
class AdaptiveConcurrency:
    """Small AIMD controller for request concurrency.

    Additive increase is intentionally conservative: after a window of clean
    successes, concurrency rises by one. Any observed throttling triggers a
    multiplicative decrease. This adapts to changing model-service capacity
    without flooding the service with all jobs at once.
    """

    current: int
    minimum: int
    maximum: int
    success_window: int
    decrease_factor: float = 0.7
    clean_successes: int = 0

    def observe(self, outcome: JobOutcome) -> int:
        old = self.current

        if outcome.throttles > 0:
            self.current = max(
                self.minimum,
                int(max(self.minimum, self.current * self.decrease_factor)),
            )
            self.clean_successes = 0
        elif outcome.success:
            self.clean_successes += 1
            if self.clean_successes >= self.success_window:
                self.current = min(self.maximum, self.current + 1)
                self.clean_successes = 0
        else:
            self.clean_successes = 0

        if self.current != old:
            LOGGER.info(
                "Adaptive concurrency changed %d -> %d",
                old,
                self.current,
            )
        return self.current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent, resumable multimodal extraction through a configurable LLM client."
    )
    parser.add_argument("--prompt-template", required=True, type=Path)
    parser.add_argument("--artifacts-path", required=True, type=Path)
    parser.add_argument(
        "--llm-client",
        required=True,
        type=Path,
        help=(
            "Python module that exposes build_client(). The returned client must expose "
            "complete(prompt=..., system=..., images=...) -> str."
        ),
    )
    parser.add_argument(
        "--client-config",
        type=Path,
        default=None,
        help=(
            "Optional Python config module loaded as '.config' beside the runtime client. "
            "Use this only when the client implementation depends on relative config imports."
        ),
    )
    parser.add_argument(
        "--limit",
        required=True,
        help="1-based inclusive selection: '5' => 1..5, '10' => 1..10, '5-20' => 5..20.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <artifacts parent>/extraction.",
    )
    parser.add_argument(
        "--initial-concurrency",
        type=int,
        default=8,
        help="Initial number of in-flight paper requests (default: 8).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=24,
        help="Maximum adaptive concurrency / thread-pool size (default: 24).",
    )
    parser.add_argument(
        "--min-concurrency",
        type=int,
        default=2,
        help="Minimum adaptive concurrency after throttling (default: 2).",
    )
    parser.add_argument(
        "--success-window",
        type=int,
        default=8,
        help="Clean completions required before increasing concurrency by one (default: 8).",
    )
    parser.add_argument(
        "--contract-retries",
        type=int,
        default=1,
        help="Additional full model calls after JSON/schema validation failure (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover/validate the selected artifacts without calling the LLM service.",
    )
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    for label in ("prompt_template", "artifacts_path", "llm_client"):
        path = getattr(args, label)
        if not path.exists():
            raise FileNotFoundError(f"--{label.replace('_', '-')} does not exist: {path}")

    if args.client_config is not None and not args.client_config.exists():
        raise FileNotFoundError(f"--client-config does not exist: {args.client_config}")

    if not args.artifacts_path.is_dir():
        raise NotADirectoryError(f"--artifacts-path is not a directory: {args.artifacts_path}")

    if args.min_concurrency < 1:
        raise ValueError("--min-concurrency must be >= 1")
    if args.initial_concurrency < args.min_concurrency:
        raise ValueError("--initial-concurrency must be >= --min-concurrency")
    if args.max_concurrency < args.initial_concurrency:
        raise ValueError("--max-concurrency must be >= --initial-concurrency")
    if args.success_window < 1:
        raise ValueError("--success-window must be >= 1")
    if args.contract_retries < 0:
        raise ValueError("--contract-retries must be >= 0")


def parse_limit(spec: str, total: int) -> tuple[int, int]:
    """Parse a 1-based inclusive CLI range and return Python slice bounds.

    ``5`` means positions 1..5; ``5-20`` means positions 5..20.
    The returned tuple is ``(start_index, stop_index)`` using Python's
    zero-based, stop-exclusive slicing convention.
    """
    text = spec.strip()
    if re.fullmatch(r"\d+", text):
        start, end = 1, int(text)
    else:
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
        if not match:
            raise ValueError("--limit must look like '5', '10', or '5-20'")
        start, end = map(int, match.groups())

    if start < 1 or end < 1:
        raise ValueError("--limit positions are 1-based and must be >= 1")
    if end < start:
        raise ValueError("--limit end must be >= start")
    if start > total:
        raise ValueError(f"--limit starts at {start}, but only {total} artifacts were discovered")

    end = min(end, total)
    return start - 1, end


def natural_key(value: str) -> list[Any]:
    """Natural sort key so suffixes .2, .10, .100 sort numerically."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def find_context_file(artifact_dir: Path) -> Optional[Path]:
    for filename in CONTEXT_FILENAMES:
        path = artifact_dir / filename
        if path.is_file():
            return path
    return None


def find_images(artifact_dir: Path) -> tuple[Path, ...]:
    visuals_dir = artifact_dir / "Visuals"
    if not visuals_dir.is_dir():
        visuals_dir = artifact_dir / "visuals"
    if not visuals_dir.is_dir():
        return ()

    images = [
        path
        for path in visuals_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda path: natural_key(path.name))
    return tuple(images)


def discover_jobs(artifacts_path: Path) -> list[Job]:
    """Discover immediate artifact subdirectories containing an LLM context."""
    artifact_dirs = [path for path in artifacts_path.iterdir() if path.is_dir()]
    artifact_dirs.sort(key=lambda path: natural_key(path.name))

    jobs: list[Job] = []
    for artifact_dir in artifact_dirs:
        context_path = find_context_file(artifact_dir)
        if context_path is None:
            continue
        jobs.append(
            Job(
                position=len(jobs) + 1,
                paper_id=artifact_dir.name,
                artifact_dir=artifact_dir,
                context_path=context_path,
                image_paths=find_images(artifact_dir),
            )
        )
    return jobs


def preflight_job(job: Job) -> list[str]:
    errors: list[str] = []
    if not job.context_path.is_file():
        errors.append(f"missing context: {job.context_path}")
    elif job.context_path.stat().st_size == 0:
        errors.append(f"empty context: {job.context_path}")

    for image_path in job.image_paths:
        if not image_path.is_file():
            errors.append(f"missing image: {image_path}")
        elif image_path.stat().st_size == 0:
            errors.append(f"empty image: {image_path}")
    return errors


def load_prompt(prompt_template_path: Path) -> str:
    spec = importlib.util.spec_from_file_location("_runtime_prompt_template", prompt_template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import prompt template: {prompt_template_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prompt = getattr(module, "EXTRACTION_SYSTEM_PROMPT", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"{prompt_template_path} must define a non-empty EXTRACTION_SYSTEM_PROMPT string"
        )
    return prompt.strip()


def load_llm_client_module(
    llm_client_path: Path,
    client_config_path: Optional[Path],
) -> types.ModuleType:
    """Dynamically load a provider-neutral LLM client module.

    The client module must expose ``build_client()``. The returned client is
    expected to expose ``complete(prompt=..., system=..., images=...) -> str``.

    If ``client_config_path`` is supplied, it is loaded as ``.config`` inside
    the same synthetic package. This supports client implementations that use
    a relative ``from .config import ...`` without coupling this runner to a
    specific SDK or model provider.
    """
    package_name = f"_runtime_llm_{uuid.uuid4().hex}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(llm_client_path.parent)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package

    import_parents = {llm_client_path.parent}
    if client_config_path is not None:
        import_parents.add(client_config_path.parent)

    for parent in import_parents:
        parent_str = str(parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)

    if client_config_path is not None:
        config_module_name = f"{package_name}.config"
        config_spec = importlib.util.spec_from_file_location(
            config_module_name, client_config_path
        )
        if config_spec is None or config_spec.loader is None:
            raise ImportError(f"Cannot import client config module: {client_config_path}")
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_module_name] = config_module
        config_spec.loader.exec_module(config_module)

    client_module_name = f"{package_name}.llm_client"
    client_spec = importlib.util.spec_from_file_location(client_module_name, llm_client_path)
    if client_spec is None or client_spec.loader is None:
        raise ImportError(f"Cannot import LLM client module: {llm_client_path}")
    client_module = importlib.util.module_from_spec(client_spec)
    sys.modules[client_module_name] = client_module
    client_spec.loader.exec_module(client_module)

    if not callable(getattr(client_module, "build_client", None)):
        raise AttributeError(f"{llm_client_path} must expose build_client()")
    return client_module


def get_thread_client(client_module: types.ModuleType) -> Any:
    client = getattr(_THREAD_LOCAL, "llm_client", None)
    if client is None:
        client = client_module.build_client()
        _THREAD_LOCAL.llm_client = client
    return client


def encode_images(image_paths: Iterable[Path]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for image_path in image_paths:
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        images.append({"base64": encoded, "media_type": media_type})
    return images


def is_retryable_exception(exc: Exception) -> tuple[bool, bool]:
    """Return (retryable, throttled) for an LLM-service/network exception."""
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None

    if status_int == 429:
        return True, True
    if status_int in {408, 409, 425}:
        return True, False
    if status_int is not None and 500 <= status_int <= 599:
        return True, False

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    transient_markers = (
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection error",
        "temporarily unavailable",
        "service unavailable",
        "too many requests",
        "throttl",
    )
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        marker in name or marker in message for marker in transient_markers
    ):
        throttled = "429" in message or "too many requests" in message or "throttl" in message
        return True, throttled

    return False, False


def parse_and_validate_response(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc

    validate_extraction_schema(payload)
    return payload


def validate_extraction_schema(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("top-level response must be a JSON object")

    actual_fields = set(payload)
    required_fields = set(REQUIRED_FIELDS)
    missing = required_fields - actual_fields
    extra = actual_fields - required_fields
    if missing or extra:
        raise ValueError(
            f"top-level fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    for field_name in REQUIRED_FIELDS:
        item = payload[field_name]
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} must be an object")

        keys = set(item)
        if keys != REQUIRED_FIELD_KEYS:
            raise ValueError(
                f"{field_name} keys must be exactly {sorted(REQUIRED_FIELD_KEYS)}; got {sorted(keys)}"
            )

        value = item["value"]
        source_span = item["source_span"]
        confidence = item["confidence"]

        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field_name}.value must be null or a non-empty string")
        if not isinstance(source_span, list):
            raise ValueError(f"{field_name}.source_span must be a list")
        if len(source_span) > 3:
            raise ValueError(f"{field_name}.source_span may contain at most 3 spans")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"{field_name}.confidence must be numeric")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"{field_name}.confidence must be between 0.0 and 1.0")

        if value is None:
            if source_span:
                raise ValueError(f"{field_name}: null value requires empty source_span")
            if float(confidence) != 0.0:
                raise ValueError(f"{field_name}: null value requires confidence 0.0")

        for span_index, span in enumerate(source_span):
            if not isinstance(span, dict) or set(span) != {"page", "text"}:
                raise ValueError(
                    f"{field_name}.source_span[{span_index}] must contain exactly page and text"
                )
            page = span["page"]
            if page is not None and (isinstance(page, bool) or not isinstance(page, int)):
                raise ValueError(
                    f"{field_name}.source_span[{span_index}].page must be an integer or null"
                )
            text = span["text"]
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"{field_name}.source_span[{span_index}].text must be a non-empty string"
                )


def run_one_job(
    job: Job,
    *,
    client_module: types.ModuleType,
    system_prompt: str,
    contract_retries: int,
) -> JobOutcome:
    """Run one paper extraction using the configured LLM client.

    Transport/service retries and model request construction are owned by the
    supplied client implementation. This runner handles paper-level orchestration,
    response validation, checkpointing, and output-contract retries.
    """
    started = time.perf_counter()
    started_at = utc_now_iso()
    model_call_rounds = 0
    throttles = 0
    raw_response: Optional[str] = None

    try:
        context = job.context_path.read_text(encoding="utf-8")
        images = encode_images(job.image_paths)
        client = get_thread_client(client_module)
        correction_suffix = ""

        for contract_attempt in range(contract_retries + 1):
            model_call_rounds += 1

            try:
                raw_response = client.complete(
                    prompt=context,
                    system=system_prompt + correction_suffix,
                    images=images,
                )
            except Exception as exc:
                _, throttled = is_retryable_exception(exc)
                throttles += int(throttled)
                raise

            try:
                parsed = parse_and_validate_response(raw_response)
            except ValueError as exc:
                if contract_attempt >= contract_retries:
                    raise
                LOGGER.warning(
                    "%s response contract failed (%s); retrying full model call (%d/%d)",
                    job.paper_id,
                    exc,
                    contract_attempt + 1,
                    contract_retries,
                )
                correction_suffix = (
                    "\n\nIMPORTANT RETRY CORRECTION: The previous response failed output "
                    f"validation because: {exc}. Return only the exact JSON object required by "
                    "the extraction schema; do not add fields or markdown."
                )
                continue

            elapsed = time.perf_counter() - started
            return JobOutcome(
                paper_id=job.paper_id,
                success=True,
                latency_seconds=elapsed,
                attempts=model_call_rounds,
                throttles=throttles,
                record={
                    "paper_id": job.paper_id,
                    "position": job.position,
                    "status": "success",
                    "response": parsed,
                    "metadata": {
                        "artifact_dir": str(job.artifact_dir),
                        "context_file": str(job.context_path),
                        "image_files": [str(path) for path in job.image_paths],
                        "image_count": len(job.image_paths),
                        "client_calls": model_call_rounds,
                        "contract_retries_used": model_call_rounds - 1,
                        "throttles_surfaced": throttles,
                        "started_at": started_at,
                        "completed_at": utc_now_iso(),
                        "latency_seconds": round(elapsed, 3),
                    },
                },
            )

        raise RuntimeError("unreachable: contract retry loop exited without a result")

    except Exception as exc:
        elapsed = time.perf_counter() - started
        retryable, throttled = is_retryable_exception(exc)
        throttles = max(throttles, int(throttled))
        error_record: dict[str, Any] = {
            "paper_id": job.paper_id,
            "position": job.position,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retryable_at_failure": retryable,
            "metadata": {
                "artifact_dir": str(job.artifact_dir),
                "context_file": str(job.context_path),
                "image_files": [str(path) for path in job.image_paths],
                "image_count": len(job.image_paths),
                "client_calls": model_call_rounds,
                "throttles_surfaced": throttles,
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "latency_seconds": round(elapsed, 3),
            },
        }
        if raw_response is not None:
            error_record["raw_response"] = raw_response

        return JobOutcome(
            paper_id=job.paper_id,
            success=False,
            record=error_record,
            latency_seconds=elapsed,
            attempts=model_call_rounds,
            throttles=throttles,
        )

def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and flush one checkpoint record; called only by the main thread."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def load_completed_ids(results_path: Path) -> set[str]:
    completed: set[str] = set()
    if not results_path.exists():
        return completed

    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed results.jsonl line %d", line_number)
                continue
            if record.get("status") == "success" and isinstance(record.get("paper_id"), str):
                completed.add(record["paper_id"])
    return completed


def count_success_records(results_path: Path) -> int:
    return len(load_completed_ids(results_path))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_elapsed(seconds: float) -> tuple[str, float | None]:
    if seconds >= 3600:
        return f"{seconds / 3600:.3f} hours", round(seconds / 3600, 6)
    if seconds >= 60:
        return f"{seconds / 60:.2f} minutes", None
    return f"{seconds:.1f} seconds", None


def update_run_metadata(metadata_path: Path, run_record: dict[str, Any]) -> None:
    existing: dict[str, Any]
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        existing = {}

    runs = existing.get("runs")
    if not isinstance(runs, list):
        runs = []
    runs.append(run_record)

    payload = {
        "last_run": run_record,
        "runs": runs,
    }
    temp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(metadata_path)


def run_concurrent(
    jobs: list[Job],
    *,
    client_module: types.ModuleType,
    system_prompt: str,
    args: argparse.Namespace,
    results_path: Path,
    failures_path: Path,
) -> tuple[int, int, int]:
    """Run pending jobs with bounded adaptive concurrency.

    Returns ``(succeeded, failed, total_throttles)`` for this invocation.
    """
    controller = AdaptiveConcurrency(
        current=args.initial_concurrency,
        minimum=args.min_concurrency,
        maximum=args.max_concurrency,
        success_window=args.success_window,
    )

    succeeded = 0
    failed = 0
    total_throttles = 0
    next_job_index = 0
    in_flight: dict[concurrent.futures.Future[JobOutcome], Job] = {}

    worker_kwargs = {
        "client_module": client_module,
        "system_prompt": system_prompt,
        "contract_retries": args.contract_retries,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        while next_job_index < len(jobs) or in_flight:
            while next_job_index < len(jobs) and len(in_flight) < controller.current:
                job = jobs[next_job_index]
                future = executor.submit(run_one_job, job, **worker_kwargs)
                in_flight[future] = job
                next_job_index += 1

            if not in_flight:
                break

            done, _ = concurrent.futures.wait(
                in_flight,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            for future in done:
                job = in_flight.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:  # Defensive: run_one_job should already contain failures.
                    outcome = JobOutcome(
                        paper_id=job.paper_id,
                        success=False,
                        latency_seconds=0.0,
                        attempts=0,
                        throttles=0,
                        record={
                            "paper_id": job.paper_id,
                            "position": job.position,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": f"unexpected worker exception: {exc}",
                            "metadata": {"completed_at": utc_now_iso()},
                        },
                    )

                total_throttles += outcome.throttles
                controller.observe(outcome)

                if outcome.success:
                    append_jsonl(results_path, outcome.record)
                    succeeded += 1
                    LOGGER.info(
                        "[%d/%d] SUCCESS %s | %.1fs | attempts=%d | throttles=%d",
                        succeeded + failed,
                        len(jobs),
                        outcome.paper_id,
                        outcome.latency_seconds,
                        outcome.attempts,
                        outcome.throttles,
                    )
                else:
                    append_jsonl(failures_path, outcome.record)
                    failed += 1
                    LOGGER.error(
                        "[%d/%d] FAILED  %s | %.1fs | %s",
                        succeeded + failed,
                        len(jobs),
                        outcome.paper_id,
                        outcome.latency_seconds,
                        outcome.record.get("error"),
                    )

    return succeeded, failed, total_throttles


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    validate_cli_args(args)
    artifacts_path = args.artifacts_path.resolve()
    prompt_template_path = args.prompt_template.resolve()
    llm_client_path = args.llm_client.resolve()
    client_config_path = args.client_config.resolve() if args.client_config else None
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.prompt_template.resolve().parent / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    failures_path = output_dir / "failures.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    results_path.touch(exist_ok=True)
    failures_path.touch(exist_ok=True)

    run_id = uuid.uuid4().hex
    run_start_perf = time.perf_counter()
    run_start = utc_now_iso()
    run_status = "running"

    all_jobs = discover_jobs(artifacts_path)
    if not all_jobs:
        raise RuntimeError(
            f"No artifact directories containing {CONTEXT_FILENAMES} were found under {artifacts_path}"
        )

    start_index, stop_index = parse_limit(args.limit, len(all_jobs))
    selected_jobs = all_jobs[start_index:stop_index]

    LOGGER.info("Discovered %d artifacts with LLM context", len(all_jobs))
    LOGGER.info(
        "Selected positions %d..%d (%d artifacts) via --limit %s",
        start_index + 1,
        stop_index,
        len(selected_jobs),
        args.limit,
    )
    LOGGER.info("Output directory: %s", output_dir)

    preflight_errors: list[tuple[Job, list[str]]] = []
    for job in selected_jobs:
        errors = preflight_job(job)
        if errors:
            preflight_errors.append((job, errors))
    if preflight_errors:
        for job, errors in preflight_errors[:20]:
            LOGGER.error("Preflight failed for %s: %s", job.paper_id, "; ".join(errors))
        raise RuntimeError(f"Preflight failed for {len(preflight_errors)} selected artifacts")

    completed_before = load_completed_ids(results_path)
    pending_jobs = [job for job in selected_jobs if job.paper_id not in completed_before]
    skipped_completed = len(selected_jobs) - len(pending_jobs)

    LOGGER.info(
        "Resume check: %d already successful, %d pending in selected range",
        skipped_completed,
        len(pending_jobs),
    )

    system_prompt = load_prompt(prompt_template_path)
    prompt_sha256 = sha256_text(system_prompt)

    if args.dry_run:
        for job in selected_jobs:
            LOGGER.info(
                "DRY RUN | #%d %s | context=%s | images=%d%s",
                job.position,
                job.paper_id,
                job.context_path.name,
                len(job.image_paths),
                " | already completed" if job.paper_id in completed_before else "",
            )
        LOGGER.info("Dry run complete. No LLM requests were made.")
        return 0

    client_module = load_llm_client_module(llm_client_path, client_config_path)

    # Build one client on the main thread for interface validation and provenance.
    probe_client = client_module.build_client()
    if not callable(getattr(probe_client, "complete", None)):
        raise AttributeError(
            "build_client() must return an object exposing "
            "complete(prompt=..., system=..., images=...) -> str"
        )

    model_id = (
        getattr(probe_client, "model_id", None)
        or getattr(probe_client, "chat_model_id", None)
        or getattr(probe_client, "model", None)
    )

    LOGGER.info("Model configured by LLM client: %s", model_id or "<not exposed>")
    LOGGER.info(
        "Concurrency settings: initial=%d, max=%d",
        args.initial_concurrency,
        args.max_concurrency,
    )
    del probe_client

    succeeded = 0
    failed = 0
    throttles = 0
    try:
        if pending_jobs:
            succeeded, failed, throttles = run_concurrent(
                pending_jobs,
                client_module=client_module,
                system_prompt=system_prompt,
                args=args,
                results_path=results_path,
                failures_path=failures_path,
            )
        run_status = "completed"
    except KeyboardInterrupt:
        run_status = "interrupted"
        LOGGER.warning("Interrupted by user. Completed JSONL checkpoints remain resumable.")
    except Exception:
        run_status = "failed"
        LOGGER.exception("Run failed")
        raise
    finally:
        run_end = utc_now_iso()
        elapsed_seconds = time.perf_counter() - run_start_perf
        elapsed_display, elapsed_hours = format_elapsed(elapsed_seconds)
        final_success_count = count_success_records(results_path)

        run_record = {
            "run_id": run_id,
            "status": run_status,
            "start_time_utc": run_start,
            "end_time_utc": run_end,
            "total_time_seconds": round(elapsed_seconds, 3),
            "total_time_minutes": round(elapsed_seconds / 60, 3),
            "total_time_hours": elapsed_hours,
            "total_time_display": elapsed_display,
            "paths": {
                "prompt_template": str(prompt_template_path),
                "artifacts": str(artifacts_path),
                "llm_client": str(llm_client_path),
                "client_config": str(client_config_path) if client_config_path else None,
                "output_dir": str(output_dir),
                "results": str(results_path),
                "failures": str(failures_path),
            },
            "provenance": {
                "prompt_sha256": prompt_sha256,
                "llm_client_sha256": sha256_file(llm_client_path),
                "client_config_sha256": sha256_file(client_config_path) if client_config_path else None,
                "model_id": model_id,
            },
            "selection": {
                "limit": args.limit,
                "discovered_artifacts": len(all_jobs),
                "selected_start_position": start_index + 1,
                "selected_end_position": stop_index,
                "selected_count": len(selected_jobs),
                "skipped_already_completed": skipped_completed,
                "pending_at_start": len(pending_jobs),
            },
            "concurrency": {
                "min": args.min_concurrency,
                "initial": args.initial_concurrency,
                "max": args.max_concurrency,
                "success_window": args.success_window,
                "strategy": "AIMD",
            },
            "retry_policy": {
                "transport_service_retries": "owned_by_llm_client",
                "contract_retries": args.contract_retries,
            },
            "this_run": {
                "succeeded": succeeded,
                "failed": failed,
                "throttles_observed": throttles,
            },
            "checkpoint_state": {
                "successful_papers_in_results_jsonl": final_success_count,
            },
        }
        update_run_metadata(metadata_path, run_record)

        LOGGER.info("=" * 72)
        LOGGER.info("FINAL STATUS: %s", run_status.upper())
        LOGGER.info("Start time (UTC): %s", run_start)
        LOGGER.info("End time   (UTC): %s", run_end)
        LOGGER.info("Total time: %s", elapsed_display)
        LOGGER.info(
            "This run: success=%d failed=%d skipped=%d throttles=%d",
            succeeded,
            failed,
            skipped_completed,
            throttles,
        )
        LOGGER.info("Successful checkpoints now in results.jsonl: %d", final_success_count)
        LOGGER.info("Metadata: %s", metadata_path)
        LOGGER.info("=" * 72)

    return 0 if run_status == "completed" else 130 if run_status == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())

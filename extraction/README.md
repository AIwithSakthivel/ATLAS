# Atlas extraction

![Research extraction pipeline](https://raw.githubusercontent.com/AIwithSakthivel/atlas-assets/main/images/atlas_extraction.png)

Converts each paper's ingested representation into structured, evidence-grounded research information. Given an artifact directory produced by [`document_ingestion`](../document_ingestion), it extracts eight standard research dimensions per paper, each backed by exact source-text evidence and a confidence score — through a pluggable LLM client you supply.

This is stage two of [Atlas](../), inserted ahead of taxonomy extraction: taxonomy construction, statistical insights, and reasoning all consume the structured corpus this stage produces.

## What it does

- Discovers artifact directories under `--artifacts-path` that contain a valid `llm_context.txt` (or `llm_context.py`) and optional page images, and turns each into an independent extraction job.
- Runs a preflight check per job — context file exists and is non-empty, referenced images exist and are non-empty — before any model call is made.
- Sends one multimodal request per paper: the ingested text plus any page images, governed by the fixed system prompt in `prompt_template.py`.
- Validates every response against a strict schema: exactly eight required fields, each with `value` / `source_span` / `confidence`, confidence in `[0.0, 1.0]`, at most 3 source spans, and null values requiring empty evidence and zero confidence.
- Retries schema/JSON failures ("contract" failures) with an explicit correction message, separately from transport/service retries — which stay inside your LLM client implementation.
- Runs papers concurrently with adaptive concurrency (AIMD: increases after a window of clean successes, backs off on throttling).
- Checkpoints every successful paper immediately to `results.jsonl`; failures go to `failures.jsonl` with diagnostics. Reruns automatically skip papers already present in `results.jsonl`.
- Records full run provenance to `run_metadata.json`: timing, concurrency settings, and SHA-256 hashes of the prompt template, LLM client, and client config used.

## The eight extracted dimensions

1. **problem_gap** — the limitation or unresolved problem in prior work that motivates the paper
2. **claimed_contribution** — the paper's main new contribution relative to prior work
3. **technical_approach** — how the work is technically carried out
4. **datasets_benchmarks** — datasets/benchmarks introduced or materially used
5. **contribution_type** — the shape of the contribution (method, dataset, benchmark, system, analysis, etc.), derived from how the authors frame it
6. **evaluation_summary** — how the work is evaluated and the headline results
7. **reproducibility** — explicitly disclosed code/data/model releases, artifact URLs, compute, hyperparameters
8. **limitations_failures** — concrete, explicitly reported limitations or failure modes

Full field definitions and extraction rules live in [`prompt_template.py`](prompt_template.py) — it's the actual contract sent to the model, not just documentation of one.

## Install

No third-party dependencies — `run_extraction.py` and `prompt_template.py` use only the Python standard library. Python 3.10+.

You do need an LLM client module of your own. It must expose:

```python
def build_client():
    """Return an object with a .complete(prompt, system, images) -> str method."""
```

`images` is a list of `{"base64": ..., "media_type": ...}` dicts. Transport retries, authentication, and provider-specific request construction all live inside your client — the runner only owns contract retries (schema/JSON failures) and orchestration.

## Usage

```bash
python run_extraction.py \
  --prompt-template prompt_template.py \
  --artifacts-path data/ingestion \
  --llm-client my_llm_client.py \
  --limit 50 \
  --dry-run
```

Drop `--dry-run` to actually call the model. `--limit` is 1-based and inclusive: `50` means positions 1–50, `5-20` means positions 5–20 — useful for sampling, batching a large corpus, or rerunning a specific range.

Each artifact directory under `--artifacts-path` is expected to contain:

```
data/ingestion/2026.acl-long.1/
├── llm_context.txt
└── visuals/            # or Visuals/ — both are checked
    └── page_009.png
```

— exactly what `document_ingestion` produces.

Useful flags:

| Flag | Purpose |
|---|---|
| `--client-config` | Optional config module loaded as `.config` beside your client, for clients using relative config imports |
| `--output-dir` | Where results are written (default: `results/` next to the prompt template) |
| `--initial-concurrency` / `--max-concurrency` / `--min-concurrency` | Bound the adaptive concurrency controller |
| `--success-window` | Clean completions required before concurrency increases by one (default 8) |
| `--contract-retries` | Extra full model calls after a schema/JSON validation failure (default 1) |

## Output

```
results/
├── results.jsonl        # one line per successful paper, checkpointed as it completes
├── failures.jsonl       # one line per failed paper, with error diagnostics
└── run_metadata.json    # timing, concurrency, prompt/client hashes, per-run history
```

Rerunning the same command resumes automatically — paper IDs already in `results.jsonl` are skipped.

## Known limitations

- Requires a user-supplied LLM client module; there's no reference/example client included in this folder.
- Logs to console only (`logging.basicConfig`, no file handler) — unlike `document_ingestion`'s `run_ingestion.py`, nothing is written to a `logs/` directory. Redirect stdout if you want a persistent log file.
- `--contract-retries` re-sends the full prompt and paper content on every retry; there's no cheaper partial-repair path for a response that's almost valid.

"""
Entry point. Configure your models in model_config.py, then:

    python main.py                      # run all configured models
    python main.py --model gemma-4-31b-it
    python main.py --single             # run just the default model (gemma-4-31b-it)

To use a real benchmark, swap SAMPLE_TASKS for load_bfcl(...) or load_gaia(...).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_config import MODELS
from runner import run_task, summarize, save_results
from tasks import SAMPLE_TASKS  # , load_bfcl, load_gaia


DEFAULT_SINGLE_MODEL = "gemma-4-31b-it"


def _select_models(name: str | None, single: bool):
    if name is None and not single:
        return MODELS
    target = name or DEFAULT_SINGLE_MODEL
    selected = [m for m in MODELS if m.name == target]
    if not selected:
        available = ", ".join(m.name for m in MODELS)
        raise SystemExit(f"Model {target!r} not found. Available: {available}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent eval against configured models.")
    parser.add_argument("--model", help="Run a single model by its config name.")
    parser.add_argument(
        "--single",
        action="store_true",
        help=f"Run only the default model ({DEFAULT_SINGLE_MODEL}).",
    )
    args = parser.parse_args()

    models = _select_models(args.model, args.single)
    tasks = SAMPLE_TASKS
    # tasks = load_bfcl("data/bfcl_simple.json", limit=50)
    # tasks = load_gaia("data/gaia_validation.jsonl", limit=20)

    jobs = [(m, t) for m in models for t in tasks]
    results = []

    # Parallelize across (model, task) pairs. Drop max_workers if rate-limited.
    # run_task prints its own per-step trace; we just collect results here.
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_task, m, t): (m.name, t.task_id) for m, t in jobs}
        for fut in as_completed(futures):
            name, tid = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[{name}/{tid}] ✗ run_task raised: {e}", flush=True)

    save_results(results, "results.json")
    summary = summarize(results)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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
from datetime import datetime
from pathlib import Path

from model_config import MODELS
from runner import run_task, save_results, summarize
from tasks import SAMPLE_TASKS, load_bfcl  # , load_gaia

DEFAULT_SINGLE_MODEL = "gemma-4-31b-it"
RESULTS_ROOT = Path("results")


def _results_paths(dataset: str, now: datetime | None = None) -> tuple[Path, Path]:
    """Return (per-task results, summary) paths under ./results/YYYY-MM-DD/."""
    now = now or datetime.now()
    day_dir = RESULTS_ROOT / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{now.strftime('%H-%M-%S')}_{dataset}"
    return day_dir / f"{stem}.json", day_dir / f"{stem}_summary.json"


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
    parser = argparse.ArgumentParser(
        description="Run agent eval against configured models."
    )
    parser.add_argument("--model", help="Run a single model by its config name.")
    parser.add_argument(
        "--single",
        action="store_true",
        help=f"Run only the default model ({DEFAULT_SINGLE_MODEL}).",
    )
    parser.add_argument(
        "--dataset",
        choices=["sample", "bfcl", "bfcl-multiple"],
        default="sample",
        help="Which task set to run.",
    )
    parser.add_argument(
        "--bfcl-path",
        default="data/bfcl/BFCL_v4_simple.json",
        help="BFCL questions JSONL path (used when --dataset=bfcl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on tasks loaded (BFCL only).",
    )
    args = parser.parse_args()

    models = _select_models(args.model, args.single)
    if args.dataset == "bfcl":
        dataset = "bfcl"
        tasks = load_bfcl(args.bfcl_path, limit=args.limit)
    elif args.dataset == "bfcl-multiple":
        dataset = "bfcl-multiple"
        bfcl_path = "data/bfcl/BFCL_v4_multiple.json"
        tasks = load_bfcl(bfcl_path, limit=args.limit)
    else:
        dataset, tasks = "sample", SAMPLE_TASKS

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

    results_path, summary_path = _results_paths(dataset)
    save_results(results, str(results_path))
    summary = summarize(results)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults  → {results_path}")
    print(f"Summary  → {summary_path}")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

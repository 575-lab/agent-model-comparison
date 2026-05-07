"""
Agent evaluation runner.

Measures three metrics across open-source models:
  1. Tool-use accuracy (correct tool + correct args)
  2. Multi-step task completion (end-to-end success)
  3. Cost & latency per successful task

Models are accessed via any OpenAI-compatible endpoint (vLLM, together.ai, etc.)
so swapping models is just a config change.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


# ---------- Retry & timeout config ----------

RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
MAX_RETRIES = 5
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0
# Per-request timeout (seconds). Hosted open-source models can cold-start; tune via env.
REQUEST_TIMEOUT_S = float(os.environ.get("LLM_REQUEST_TIMEOUT_S", "20"))
# How often to print a "still waiting" heartbeat during a long LLM call.
HEARTBEAT_EVERY_S = 5.0


def _completion_with_retry(client: OpenAI, label: str = "", **kwargs: Any) -> Any:
    """Call chat.completions.create with exponential backoff and a heartbeat."""
    for attempt in range(MAX_RETRIES + 1):
        done = threading.Event()
        call_start = time.time()

        def heartbeat() -> None:
            while not done.wait(HEARTBEAT_EVERY_S):
                elapsed = time.time() - call_start
                print(
                    f"{label} … still waiting ({elapsed:.0f}s / {REQUEST_TIMEOUT_S:.0f}s timeout)",
                    flush=True,
                )

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        try:
            return client.chat.completions.create(timeout=REQUEST_TIMEOUT_S, **kwargs)
        except RETRYABLE_ERRORS as e:
            if attempt == MAX_RETRIES:
                raise
            delay = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
            delay += random.uniform(0, delay * 0.25)  # jitter
            print(
                f"{label} ! {type(e).__name__} — retry {attempt + 1}/{MAX_RETRIES} "
                f"in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
        finally:
            done.set()


def _truncate(s: str | None, n: int = 120) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------- Config ----------

@dataclass
class ModelConfig:
    name: str                      # display name, e.g. "llama-3.3-70b"
    base_url: str                  # OpenAI-compatible endpoint
    api_key: str                   # may be a dummy value for local vLLM
    model_id: str                  # the id the server expects
    input_price_per_1m: float      # USD per 1M input tokens
    output_price_per_1m: float     # USD per 1M output tokens


# ---------- Per-task result ----------

@dataclass
class TaskResult:
    task_id: str
    model: str
    success: bool
    steps: int
    tool_calls_correct: int
    tool_calls_total: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


# ---------- Task definition ----------

@dataclass
class Task:
    """
    A single eval task. `tools` are passed to the model; `verifier` decides
    success; `expected_calls` enables tool-use accuracy scoring (optional).
    """
    task_id: str
    prompt: str
    tools: list[dict[str, Any]]
    tool_impls: dict[str, Callable[..., Any]]      # name -> python fn
    verifier: Callable[[list[dict[str, Any]]], bool]  # trace -> success
    expected_calls: list[dict[str, Any]] | None = None  # for BFCL-style scoring
    max_steps: int = 10


# ---------- The agent loop ----------

def run_task(model: ModelConfig, task: Task, verbose: bool = True) -> TaskResult:
    """Run one task with one model. Tracks tokens, latency, tool correctness."""
    client = OpenAI(base_url=model.base_url, api_key=model.api_key)
    label = f"[{model.name}/{task.task_id}]"

    def log(msg: str) -> None:
        if verbose:
            print(f"{label} {msg}", flush=True)

    log(f"START prompt={_truncate(task.prompt, 100)!r}")

    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    trace: list[dict[str, Any]] = []
    in_tokens = out_tokens = 0
    tc_correct = tc_total = 0
    start = time.time()
    error: str | None = None
    success = False
    steps = 0

    try:
        for step in range(task.max_steps):
            steps = step + 1
            log(f"step {steps} → calling LLM")
            resp = _completion_with_retry(
                client,
                label=label,
                model=model.model_id,
                messages=messages,
                tools=task.tools,
                tool_choice="auto",
                temperature=0.0,
            )
            usage = resp.usage
            in_tokens += usage.prompt_tokens
            out_tokens += usage.completion_tokens
            log(
                f"step {steps} ← LLM responded "
                f"(in={usage.prompt_tokens} out={usage.completion_tokens} "
                f"cum_in={in_tokens} cum_out={out_tokens})"
            )

            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            # No tool calls → model thinks it's done
            if not msg.tool_calls:
                trace.append({"role": "assistant", "content": msg.content})
                log(f"step {steps} ✓ final answer: {_truncate(msg.content)!r}")
                break

            # Execute each tool call, log to trace, score correctness
            for call in msg.tool_calls:
                tc_total += 1
                fn_name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Score against expected calls if provided (BFCL-style)
                expected_match = ""
                if task.expected_calls and tc_total <= len(task.expected_calls):
                    expected = task.expected_calls[tc_total - 1]
                    if (
                        expected["name"] == fn_name
                        and _args_match(expected.get("arguments", {}), args)
                    ):
                        tc_correct += 1
                        expected_match = " ✓expected"
                    else:
                        expected_match = " ✗expected"

                log(f"step {steps} → tool {fn_name}({_truncate(json.dumps(args), 80)}){expected_match}")

                # Execute the tool
                impl = task.tool_impls.get(fn_name)
                if impl is None:
                    result = f"ERROR: unknown tool '{fn_name}'"
                else:
                    try:
                        result = impl(**args)
                    except Exception as e:  # tool runtime errors are part of the eval
                        result = f"ERROR: {e}"

                log(f"step {steps} ← tool {fn_name} → {_truncate(str(result))!r}")

                trace.append({"tool": fn_name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })
        else:
            log(f"⚠ reached max_steps={task.max_steps} without finishing")

        success = task.verifier(trace)

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log(f"✗ ERROR {error}")

    latency = time.time() - start
    cost = (
        in_tokens * model.input_price_per_1m / 1_000_000
        + out_tokens * model.output_price_per_1m / 1_000_000
    )

    status = "PASS" if success else ("ERROR" if error else "FAIL")
    log(
        f"DONE {status} steps={steps} tools={tc_correct}/{tc_total} "
        f"cost=${cost:.4f} t={latency:.1f}s"
    )

    return TaskResult(
        task_id=task.task_id,
        model=model.name,
        success=success,
        steps=steps,
        tool_calls_correct=tc_correct,
        tool_calls_total=tc_total,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        latency_s=latency,
        cost_usd=cost,
        error=error,
        trace=trace,
    )


def _args_match(expected: dict, actual: dict) -> bool:
    """Loose match: every expected key/value is present in actual."""
    return all(actual.get(k) == v for k, v in expected.items())


# ---------- Aggregation ----------

def summarize(results: list[TaskResult]) -> dict[str, dict[str, float]]:
    """Compute the three headline metrics per model."""
    by_model: dict[str, list[TaskResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    summary = {}
    for name, rs in by_model.items():
        n = len(rs)
        successes = [r for r in rs if r.success]
        n_succ = len(successes)
        total_calls = sum(r.tool_calls_total for r in rs)
        correct_calls = sum(r.tool_calls_correct for r in rs)

        summary[name] = {
            # Metric 1: tool-use accuracy
            "tool_accuracy": correct_calls / total_calls if total_calls else 0.0,
            # Metric 2: multi-step completion rate
            "completion_rate": n_succ / n if n else 0.0,
            "avg_steps_on_success": (
                sum(r.steps for r in successes) / n_succ if n_succ else 0.0
            ),
            # Metric 3: cost & latency per successful task
            "cost_per_success_usd": (
                sum(r.cost_usd for r in rs) / n_succ if n_succ else float("inf")
            ),
            "latency_per_success_s": (
                sum(r.latency_s for r in rs) / n_succ if n_succ else float("inf")
            ),
            "n_tasks": n,
        }
    return summary


def save_results(results: list[TaskResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

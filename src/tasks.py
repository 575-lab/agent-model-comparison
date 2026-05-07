"""
Task definitions and dataset loaders.

Includes:
  - 2 hand-written sample tasks (smoke test, no internet needed)
  - load_bfcl():  Berkeley Function Calling Leaderboard  -> tool-use accuracy
  - load_gaia():  GAIA benchmark                          -> multi-step completion
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runner import Task


# ---------- Tool implementations used by sample tasks ----------

def _search_flights(origin: str, destination: str, date: str) -> str:
    return json.dumps([
        {"flight": "AA100", "price": 320, "depart": "08:00"},
        {"flight": "UA250", "price": 410, "depart": "11:30"},
    ])


def _book_flight(flight: str) -> str:
    return json.dumps({"status": "confirmed", "flight": flight, "ref": "ABC123"})


def _calculator(expression: str) -> str:
    try:
        # eval is fine here — it's a sandboxed eval task
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERROR: {e}"


# ---------- Tool schemas ----------

FLIGHT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights between two cities on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a specific flight by its identifier.",
            "parameters": {
                "type": "object",
                "properties": {"flight": {"type": "string"}},
                "required": ["flight"],
            },
        },
    },
]

CALC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a Python arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


# ---------- Verifiers ----------

def _verify_flight_booking(trace: list[dict[str, Any]]) -> bool:
    """Success = booked the cheaper flight (AA100)."""
    for step in trace:
        if step.get("tool") == "book_flight" and step.get("args", {}).get("flight") == "AA100":
            return True
    return False


def _verify_calc(expected: float):
    def verify(trace: list[dict[str, Any]]) -> bool:
        # final assistant message should contain the answer
        for step in reversed(trace):
            content = step.get("content") or ""
            if str(int(expected)) in content or f"{expected:g}" in content:
                return True
        return False
    return verify


# ---------- Sample tasks (run these first to smoke-test) ----------

SAMPLE_TASKS: list[Task] = [
    Task(
        task_id="flight-cheapest",
        prompt=(
            "Find flights from SFO to JFK on 2026-06-01 and book the cheapest one. "
            "Tell me the booking reference when done."
        ),
        tools=FLIGHT_TOOLS,
        tool_impls={"search_flights": _search_flights, "book_flight": _book_flight},
        verifier=_verify_flight_booking,
        expected_calls=[
            {"name": "search_flights",
             "arguments": {"origin": "SFO", "destination": "JFK", "date": "2026-06-01"}},
            {"name": "book_flight", "arguments": {"flight": "AA100"}},
        ],
        max_steps=6,
    ),
    Task(
        task_id="calc-compound",
        prompt=(
            "What is (1500 * 1.07 ** 5) rounded to the nearest dollar? "
            "Use the calculator and tell me the number."
        ),
        tools=CALC_TOOLS,
        tool_impls={"calculator": _calculator},
        verifier=_verify_calc(2103),  # 1500 * 1.07^5 ≈ 2103.83
        max_steps=4,
    ),
]


# ---------- BFCL loader (tool-use accuracy) ----------

def load_bfcl(path: str | Path, limit: int | None = None) -> list[Task]:
    """
    Load Berkeley Function Calling Leaderboard tasks.

    Get the data:
        git clone https://github.com/ShishirPatil/gorilla
        # use files in: gorilla/berkeley-function-call-leaderboard/data/

    Each line is JSON with: id, question (list of messages), function (schemas).
    Pair with the `_possible_answer` file for `expected_calls`.
    Verifier is just "did the expected calls happen" — pure tool-use scoring.
    """
    tasks: list[Task] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            row = json.loads(line)
            prompt = row["question"][0][0]["content"]  # BFCL nests messages
            schemas = [{"type": "function", "function": fn} for fn in row["function"]]
            expected = row.get("expected_calls", [])

            def make_verifier(exp):
                def verify(trace):
                    calls = [s for s in trace if "tool" in s]
                    return len(calls) >= len(exp) and all(
                        calls[j]["tool"] == exp[j]["name"] for j in range(len(exp))
                    )
                return verify

            tasks.append(Task(
                task_id=f"bfcl-{row['id']}",
                prompt=prompt,
                tools=schemas,
                tool_impls={fn["name"]: lambda **kw: "ok" for fn in row["function"]},
                verifier=make_verifier(expected),
                expected_calls=expected,
                max_steps=8,
            ))
    return tasks


# ---------- GAIA loader (multi-step completion) ----------

def load_gaia(path: str | Path, limit: int | None = None) -> list[Task]:
    """
    Load GAIA tasks. Get from HuggingFace:
        huggingface-cli download gaia-benchmark/GAIA --repo-type dataset

    GAIA tasks need real tools (web browser, file reader, python).
    This stub shows the structure; you'd plug in your actual tool implementations.
    """
    tasks: list[Task] = []
    with open(path) as f:
        data = json.load(f) if str(path).endswith(".json") else [json.loads(l) for l in f]

    for i, row in enumerate(data):
        if limit and i >= limit:
            break
        expected_answer = row["Final answer"].strip().lower()

        def make_verifier(ans):
            def verify(trace):
                for step in reversed(trace):
                    content = (step.get("content") or "").lower()
                    if ans in content:
                        return True
                return False
            return verify

        tasks.append(Task(
            task_id=f"gaia-{row['task_id']}",
            prompt=row["Question"],
            tools=[],          # plug in browser/python/file tools here
            tool_impls={},     # ditto
            verifier=make_verifier(expected_answer),
            max_steps=20,
        ))
    return tasks

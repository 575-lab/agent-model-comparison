"""
Task definitions and dataset loaders.

Includes:
  - 2 hand-written sample tasks (smoke test, no internet needed)
  - load_bfcl():  Berkeley Function Calling Leaderboard  -> tool-use accuracy
  - load_gaia():  GAIA benchmark                          -> multi-step completion
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from runner import Task

# ---------- Tool implementations used by sample tasks ----------


def _search_flights(origin: str, destination: str, date: str) -> str:
    return json.dumps(
        [
            {"flight": "AA100", "price": 320, "depart": "08:00"},
            {"flight": "UA250", "price": 410, "depart": "11:30"},
        ]
    )


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
        if (
            step.get("tool") == "book_flight"
            and step.get("args", {}).get("flight") == "AA100"
        ):
            return True
    return False


def _verify_calc(expected: float):
    def verify(trace: list[dict[str, Any]]) -> bool:
        # final assistant message should contain the answer.
        # Strip thousands separators so "2,104" matches "2104".
        for step in reversed(trace):
            content = (step.get("content") or "").replace(",", "")
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
            {
                "name": "search_flights",
                "arguments": {
                    "origin": "SFO",
                    "destination": "JFK",
                    "date": "2026-06-01",
                },
            },
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
        verifier=_verify_calc(2104),  # 1500 * 1.07^5 ≈ 2103.83 → rounds to 2104
        max_steps=4,
    ),
]


# ---------- BFCL loader (tool-use accuracy) ----------

# BFCL ground truth has shape:
#   [{"fn_name": {"arg1": [allowed1, allowed2, ...], "arg2": [...]}}, ...]
# Each call is a dict with one key (the function name) mapping to a dict of
# argument -> list of accepted values. The list captures BFCL's allowance for
# semantically equivalent inputs (e.g. ["units", "unit"] for a unit string).
BFCLCall = dict[str, dict[str, Any]]


def load_bfcl(
    questions_path: str | Path,
    answers_path: str | Path | None = None,
    limit: int | None = None,
) -> list[Task]:
    """
    Load Berkeley Function Calling Leaderboard tasks.

    Get the data:
        git clone https://github.com/ShishirPatil/gorilla
        # questions:  gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json
        # answers:    gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json

    Args:
        questions_path: JSONL file with id / question / function fields.
        answers_path:   matching possible_answer file. Defaults to
                        `<questions_dir>/possible_answer/<filename>`.
        limit:          cap on tasks loaded.
    """
    questions_path = Path(questions_path)
    if answers_path is None:
        answers_path = questions_path.parent / "possible_answer" / questions_path.name
    answers_path = Path(answers_path)

    answers_by_id: dict[str, list[BFCLCall]] = {}
    if answers_path.exists():
        with open(answers_path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                answers_by_id[row["id"]] = row.get("ground_truth", [])

    tasks: list[Task] = []
    with open(questions_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row["id"]
            ground_truth = answers_by_id.get(tid, [])

            sanitized_fns = [_sanitize_bfcl_function(fn) for fn in row["function"]]
            sanitized_gt = _sanitize_bfcl_ground_truth(ground_truth)

            tasks.append(
                Task(
                    task_id=f"bfcl-{tid}",
                    prompt=_bfcl_prompt(row["question"]),
                    tools=[
                        {"type": "function", "function": fn} for fn in sanitized_fns
                    ],
                    # BFCL is a tool-use eval: implementations are stubs — the agent
                    # never gets to chain on real outputs. Returning "ok" keeps the
                    # loop moving for any post-tool reasoning step.
                    tool_impls={
                        fn["name"]: (lambda **kw: "ok") for fn in sanitized_fns
                    },
                    verifier=_make_bfcl_verifier(sanitized_gt),
                    expected_calls=_bfcl_to_expected_calls(sanitized_gt),
                    max_steps=max(4, 2 * len(ground_truth) or 4),
                )
            )
    return tasks


# ---------- BFCL → JSON-Schema sanitization ----------
#
# BFCL schemas use Python type names (`dict`, `float`, `int`, `tuple`, `any`)
# and dotted function names (`number_analysis.prime_factors`). OpenAI's strict
# tools validator rejects both. We rewrite to JSON Schema types and a safe name
# pattern, applying the same rewrite to the ground truth so verification still
# matches what the model actually called.

_PY_TO_JSONSCHEMA_TYPE = {
    "int": "integer",
    "integer": "integer",
    "long": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "tuple": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}

_INVALID_FN_NAME_CHAR = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_fn_name(name: str) -> str:
    return _INVALID_FN_NAME_CHAR.sub("_", name)


def _normalize_schema(node: Any) -> Any:
    """Recursively coerce Python type names to JSON Schema types."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                t = v.lower()
                if t == "any":
                    # OpenAI rejects unknown types; drop the constraint.
                    continue
                out[k] = _PY_TO_JSONSCHEMA_TYPE.get(t, v)
            else:
                out[k] = _normalize_schema(v)
        return out
    if isinstance(node, list):
        return [_normalize_schema(x) for x in node]
    return node


def _sanitize_bfcl_function(fn: dict[str, Any]) -> dict[str, Any]:
    out = dict(fn)
    out["name"] = _sanitize_fn_name(fn["name"])
    params = _normalize_schema(fn.get("parameters", {}) or {})
    # OpenAI requires the top-level parameters to declare type=object.
    if "properties" in params and params.get("type") != "object":
        params["type"] = "object"
    out["parameters"] = params
    return out


def _sanitize_bfcl_ground_truth(gt: list[BFCLCall]) -> list[BFCLCall]:
    """Apply the same name sanitization to ground truth so the verifier matches the wire name."""
    out: list[BFCLCall] = []
    for entry in gt:
        if not entry:
            out.append(entry)
            continue
        fn_name, args = next(iter(entry.items()))
        out.append({_sanitize_fn_name(fn_name): args})
    return out


def _bfcl_prompt(question: Any) -> str:
    """BFCL nests messages as question[turn][message]. Concatenate user turns."""
    if not question:
        return ""
    # Single-turn (most categories): question = [[{role, content}, ...]]
    turn = question[0] if isinstance(question[0], list) else question
    return "\n".join(
        m["content"]
        for m in turn
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
    )


def _bfcl_to_expected_calls(ground_truth: list[BFCLCall]) -> list[dict[str, Any]]:
    """Pick the first allowed value per arg so the runner's per-call scorer has something concrete."""
    out: list[dict[str, Any]] = []
    for entry in ground_truth:
        if not entry:
            continue
        fn_name, arg_options = next(iter(entry.items()))
        args = {
            k: (v[0] if isinstance(v, list) and v else v)
            for k, v in arg_options.items()
        }
        out.append({"name": fn_name, "arguments": args})
    return out


def _make_bfcl_verifier(ground_truth: list[BFCLCall]):
    """
    AST-style match: for each ground-truth position, the model's tool call must
    use the right function and every required argument must be one of BFCL's
    allowed values. Extra calls past len(ground_truth) are ignored.
    """

    def verify(trace: list[dict[str, Any]]) -> bool:
        if not ground_truth:
            return False
        calls = [s for s in trace if "tool" in s]
        if len(calls) < len(ground_truth):
            return False
        for i, expected in enumerate(ground_truth):
            if not expected:
                continue
            fn_name, arg_options = next(iter(expected.items()))
            if calls[i]["tool"] != fn_name:
                return False
            actual = calls[i].get("args") or {}
            for arg, allowed in arg_options.items():
                allowed_list = allowed if isinstance(allowed, list) else [allowed]
                # BFCL marks optional args by including "" in the alternatives;
                # if the model omitted such an arg, accept it.
                if arg not in actual:
                    if "" in allowed_list:
                        continue
                    return False
                if actual[arg] not in allowed_list:
                    return False
        return True

    return verify


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
        data = (
            json.load(f) if str(path).endswith(".json") else [json.loads(l) for l in f]
        )

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

        tasks.append(
            Task(
                task_id=f"gaia-{row['task_id']}",
                prompt=row["Question"],
                tools=[],  # plug in browser/python/file tools here
                tool_impls={},  # ditto
                verifier=make_verifier(expected_answer),
                max_steps=20,
            )
        )
    return tasks

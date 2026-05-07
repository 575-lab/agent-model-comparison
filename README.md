# Agent Evaluation Runner

Compares open-source models on three agent metrics:

1. **Tool-use accuracy** — % of tool calls with correct name + arguments
2. **Multi-step completion rate** — % of tasks solved end-to-end
3. **Cost & latency per successful task** — true efficiency, not per-call cost

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
cp .env.example .env
# Edit .env and set DOUBLEWORD_API_KEY=...
```

For local serving with vLLM:

```bash
uv sync --extra vllm
```

## Running

The runner reads model configs from `src/model_config.py` (Doubleword-hosted by default) and the API key from `.env`.

```bash
uv run python src/main.py                          # all configured models
uv run python src/main.py --single                 # just the default model (gemma-4-31b-it)
uv run python src/main.py --model qwen3.6-35b-a3b  # one arbitrary model by config name
```

- **No flag** — fans out every configured model × every task in parallel (`ThreadPoolExecutor(max_workers=4)`).
- **`--single`** — quickest smoke test. Runs only `gemma-4-31b-it` (cheap + fast on Doubleword realtime pricing) so you can verify the loop works without spending much.
- **`--model NAME`** — pass any `name` from `MODELS` in `src/model_config.py` (e.g. `deepseek-v4-pro`, `kimi-k2.6`, `gpt-oss-20b`). Unknown names print the available list and exit.

Each run prints per-task PASS/FAIL with steps/cost/latency, writes the full traces to `results.json`, and prints an aggregated summary.

## Real benchmarks

**BFCL** (tool-use accuracy):
```bash
git clone https://github.com/ShishirPatil/gorilla
# In src/main.py, swap SAMPLE_TASKS for:
#   tasks = load_bfcl("gorilla/berkeley-function-call-leaderboard/data/BFCL_v4_simple_python.json", limit=100)
```

**GAIA** (multi-step completion):
```bash
huggingface-cli download gaia-benchmark/GAIA --repo-type dataset --local-dir data/gaia
# Plug your browser/python tool implementations into load_gaia()
```

**SWE-bench Verified** (coding agents): use the official harness ([swebench.com](https://www.swebench.com)) — it has its own Docker-based evaluation that's hard to replicate. Run it separately and import the results.

## Local model serving

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --enable-auto-tool-choice --tool-call-parser llama3_json
```

Then add a `ModelConfig` in `src/model_config.py` pointing at `http://localhost:8000/v1`.

## Files

- `src/runner.py` — agent loop, retry-with-backoff, metrics, aggregation
- `src/tasks.py` — sample tasks + dataset loaders
- `src/model_config.py` — model definitions (Doubleword endpoints + pricing)
- `src/main.py` — CLI + entry point

## Extending

- **More tools**: add to `tool_impls` dict per task
- **Different scoring**: replace the `verifier` callable
- **New benchmark**: write a loader that returns `list[Task]`
- **More models**: append a `ModelConfig` to `MODELS` in `src/model_config.py`

"""
Model configurations for evaluation.

Models served via Doubleword's OpenAI-compatible inference API.
Pricing is realtime tier (USD per 1M tokens) — see
https://docs.doubleword.ai/inference-api/models for current rates.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from runner import ModelConfig


load_dotenv()

DOUBLEWORD_BASE_URL = os.environ.get("DOUBLEWORD_BASE_URL", "https://api.doubleword.ai/v1")
DOUBLEWORD_API_KEY = os.environ.get("DOUBLEWORD_API_KEY", "YOUR_DOUBLEWORD_KEY")


MODELS = [
    ModelConfig(
        name="deepseek-v4-pro",
        base_url=DOUBLEWORD_BASE_URL,
        api_key=DOUBLEWORD_API_KEY,
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        input_price_per_1m=1.74,
        output_price_per_1m=3.48,
    ),
    ModelConfig(
        name="kimi-k2.6",
        base_url=DOUBLEWORD_BASE_URL,
        api_key=DOUBLEWORD_API_KEY,
        model_id="moonshotai/Kimi-K2.6",
        input_price_per_1m=0.95,
        output_price_per_1m=4.00,
    ),
    ModelConfig(
        name="gemma-4-31b-it",
        base_url=DOUBLEWORD_BASE_URL,
        api_key=DOUBLEWORD_API_KEY,
        model_id="google/gemma-4-31B-it",
        input_price_per_1m=0.14,
        output_price_per_1m=0.40,
    ),
    ModelConfig(
        name="nemotron-3-super-120b-a12b",
        base_url=DOUBLEWORD_BASE_URL,
        api_key=DOUBLEWORD_API_KEY,
        model_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        input_price_per_1m=0.30,
        output_price_per_1m=0.75,
    ),
    ModelConfig(
        name="gpt-oss-20b",
        base_url=DOUBLEWORD_BASE_URL,
        api_key=DOUBLEWORD_API_KEY,
        model_id="openai/gpt-oss-20b",
        input_price_per_1m=0.04,
        output_price_per_1m=0.30,
    ),
    ModelConfig(
        name="qwen3.6-35b-a3b",
        base_url=DOUBLEWORD_BASE_URL,
        api_key=DOUBLEWORD_API_KEY,
        model_id="Qwen/Qwen3.6-35B-A3B-FP8",
        input_price_per_1m=0.25,
        output_price_per_1m=2.00,
    ),
    ModelConfig(
        name="glm-5.1",
        base_url=DOUBLEWORD_BASE_URL,
        api_key=DOUBLEWORD_API_KEY,
        model_id="zai-org/GLM-5.1-FP8",
        input_price_per_1m=1.40,
        output_price_per_1m=4.40,
    ),
]

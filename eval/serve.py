"""vLLM / Ollama serving integration for fine-tuned LoRA models."""
from __future__ import annotations
import os
from typing import List, Optional

try:
    import httpx
    _OK = True
except ImportError:
    _OK = False

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")


def generate_ollama(model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Generate text via Ollama API."""
    if not _OK:
        return f"[mock] Response to: {prompt[:50]}..."
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "options": {
                "num_predict": max_tokens, "temperature": temperature
            }, "stream": False},
            timeout=60,
        )
        return resp.json().get("response", "")
    except Exception as e:
        return f"[error] {e}"


def generate_vllm(model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Generate text via vLLM OpenAI-compatible API."""
    if not _OK:
        return f"[mock] Response to: {prompt[:50]}..."
    try:
        resp = httpx.post(
            f"{VLLM_URL}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature},
            timeout=60,
        )
        data = resp.json()
        return data["choices"][0]["text"] if data.get("choices") else ""
    except Exception as e:
        return f"[error] {e}"


def batch_evaluate_serving(
    model: str,
    examples: List[dict],
    backend: str = "ollama",
    max_tokens: int = 200,
) -> List[str]:
    """Run batch inference on eval examples using the specified backend."""
    generate_fn = generate_ollama if backend == "ollama" else generate_vllm
    predictions = []
    for ex in examples:
        from preprocessing.tokenizer import format_example
        prompt = format_example({**ex, "output": ""}, format="alpaca").rstrip()
        pred = generate_fn(model, prompt, max_tokens=max_tokens)
        predictions.append(pred)
    return predictions

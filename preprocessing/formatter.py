"""Dataset format converter (raw datasets → Alpaca/ShareGPT format)."""
from __future__ import annotations
import json, os
from typing import List, Dict, Iterator


def convert_to_alpaca(raw_examples: List[Dict]) -> List[Dict]:
    """
    Convert raw Q&A or instruction examples to Alpaca format.
    Input: [{"question": "...", "answer": "..."}, ...]
    Output: [{"instruction": "...", "input": "", "output": "..."}, ...]
    """
    result = []
    for ex in raw_examples:
        if "instruction" in ex and "output" in ex:
            result.append(ex)  # already Alpaca
        elif "question" in ex and "answer" in ex:
            result.append({"instruction": ex["question"], "input": "", "output": ex["answer"]})
        elif "prompt" in ex and "completion" in ex:
            result.append({"instruction": ex["prompt"], "input": "", "output": ex["completion"]})
        else:
            continue
    return result


def load_jsonl(path: str) -> Iterator[Dict]:
    """Load a JSONL file line by line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(examples: List[Dict], path: str):
    """Save examples as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def split_dataset(examples: List[Dict], train_ratio: float = 0.9) -> tuple:
    """Split into train/eval sets."""
    n_train = int(len(examples) * train_ratio)
    return examples[:n_train], examples[n_train:]

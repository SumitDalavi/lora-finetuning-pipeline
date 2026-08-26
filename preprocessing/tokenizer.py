"""Dataset tokenizer for instruction fine-tuning with LoRA."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from transformers import AutoTokenizer
    _HF = True
except ImportError:
    _HF = False


@dataclass
class TokenizedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    original_text: str


def tokenize_instruction_dataset(
    examples: List[Dict[str, str]],
    model_name: str = "gpt2",
    max_length: int = 512,
    format: str = "alpaca",
) -> List[TokenizedExample]:
    """
    Tokenize a list of instruction examples for supervised fine-tuning.

    Args:
        examples: List of dicts with 'instruction', 'input' (optional), 'output'
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length (pad/truncate to this)
        format: Input format ('alpaca' or 'sharegpt')
    Returns:
        List of TokenizedExample with input_ids, attention_mask, labels
    """
    if not _HF:
        # Return mock tokenization for testing without HF installed
        results = []
        for ex in examples:
            text = format_example(ex, format=format)
            ids = [ord(c) % 1000 for c in text[:max_length]]
            results.append(TokenizedExample(
                input_ids=ids, attention_mask=[1]*len(ids),
                labels=ids, original_text=text,
            ))
        return results

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for ex in examples:
        text = format_example(ex, format=format)
        encoded = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors=None,
        )
        # For causal LM: labels = input_ids (shifted inside model)
        labels = encoded["input_ids"].copy()
        # Mask instruction part so loss is only on output
        if "output" in ex:
            output_start = text.find(ex["output"])
            if output_start > 0:
                output_ids = tokenizer(text[:output_start])["input_ids"]
                labels[:len(output_ids)] = [-100] * len(output_ids)

        results.append(TokenizedExample(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=labels,
            original_text=text,
        ))
    return results


def format_example(ex: Dict[str, str], format: str = "alpaca") -> str:
    """Format a dict example into a prompt string."""
    if format == "alpaca":
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        output = ex.get("output", "")
        prompt = f"### Instruction:\n{instruction}"
        if inp:
            prompt += f"\n\n### Input:\n{inp}"
        prompt += f"\n\n### Response:\n{output}"
        return prompt
    elif format == "sharegpt":
        msgs = ex.get("conversations", [])
        return " ".join(f"{m['from'].upper()}: {m['value']}" for m in msgs)
    else:
        return str(ex)

"""Tests for preprocessing pipeline."""
import pytest
from preprocessing.formatter import convert_to_alpaca, split_dataset
from preprocessing.tokenizer import format_example, tokenize_instruction_dataset
from eval.benchmarks import compute_bleu, compute_rouge_l, evaluate_batch, compute_perplexity


def test_convert_qa_to_alpaca():
    raw = [{"question": "What is LoRA?", "answer": "Low-Rank Adaptation."}]
    result = convert_to_alpaca(raw)
    assert len(result) == 1
    assert result[0]["instruction"] == "What is LoRA?"
    assert result[0]["output"] == "Low-Rank Adaptation."


def test_convert_passthrough_alpaca():
    ex = {"instruction": "Summarize", "input": "", "output": "Summary."}
    assert convert_to_alpaca([ex]) == [ex]


def test_format_example_alpaca():
    ex = {"instruction": "Say hello", "input": "", "output": "Hello!"}
    formatted = format_example(ex, format="alpaca")
    assert "### Instruction" in formatted
    assert "### Response" in formatted
    assert "Hello!" in formatted


def test_split_dataset():
    examples = [{"instruction": str(i)} for i in range(100)]
    train, eval_ = split_dataset(examples, train_ratio=0.8)
    assert len(train) == 80
    assert len(eval_) == 20


def test_compute_bleu_identical():
    score = compute_bleu("hello world", "hello world")
    assert score == pytest.approx(1.0, abs=0.05)


def test_compute_bleu_no_overlap():
    score = compute_bleu("apple orange", "banana grape")
    assert score == pytest.approx(0.0, abs=0.1)


def test_compute_rouge_l():
    score = compute_rouge_l("the cat sat on the mat", "the cat sat on the mat")
    assert score > 0.8


def test_evaluate_batch():
    examples = [{"output": "hello world"}, {"output": "goodbye world"}]
    preds = ["hello world", "goodbye world"]
    report = evaluate_batch(examples, preds)
    assert report["sample_count"] == 2
    assert report["avg_bleu"] > 0.5
    assert "per_example" in report


def test_compute_perplexity():
    log_probs = [-1.0, -1.0, -1.0]  # avg nll=1 → ppl=e≈2.718
    ppl = compute_perplexity(log_probs)
    assert abs(ppl - 2.718) < 0.1


def test_tokenize_without_hf(monkeypatch):
    monkeypatch.setattr("preprocessing.tokenizer._HF", False)
    examples = [{"instruction": "Test", "input": "", "output": "Result"}]
    tokenized = tokenize_instruction_dataset(examples, max_length=50)
    assert len(tokenized) == 1
    assert len(tokenized[0].input_ids) <= 50

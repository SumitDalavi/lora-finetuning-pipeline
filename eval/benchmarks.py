"""Evaluation metrics for fine-tuned model outputs."""
from __future__ import annotations
import math
from typing import List, Dict

try:
    from rouge_score import rouge_scorer
    _ROUGE = True
except ImportError:
    _ROUGE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _BLEU = True
except ImportError:
    _BLEU = False


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score (0.0–1.0)."""
    if not _BLEU:
        # Simple overlap approximation
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if not hyp_tokens:
            return 0.0
        return len(ref_tokens & hyp_tokens) / len(hyp_tokens)

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    sf = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=sf)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score."""
    if not _ROUGE:
        # LCS-based approximation
        ref, hyp = reference.lower().split(), hypothesis.lower().split()
        common = len(set(ref) & set(hyp))
        if not ref or not hyp:
            return 0.0
        precision = common / len(hyp)
        recall = common / len(ref)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_perplexity(log_probs: List[float]) -> float:
    """Compute perplexity from a list of log probabilities."""
    if not log_probs:
        return float("inf")
    avg_nll = -sum(log_probs) / len(log_probs)
    return math.exp(avg_nll)


def evaluate_batch(examples: List[Dict], predictions: List[str]) -> Dict:
    """
    Evaluate a batch of predictions against ground truth.

    Args:
        examples: List of Alpaca examples with 'output' field
        predictions: List of generated strings (same order as examples)
    Returns:
        Dict with avg_bleu, avg_rouge_l, sample_count, per_example scores
    """
    assert len(examples) == len(predictions), "examples and predictions must be same length"
    results = []
    for ex, pred in zip(examples, predictions):
        ref = ex.get("output", "")
        bleu = compute_bleu(ref, pred)
        rouge = compute_rouge_l(ref, pred)
        results.append({"bleu": bleu, "rouge_l": rouge, "reference": ref, "prediction": pred})

    avg_bleu = sum(r["bleu"] for r in results) / len(results) if results else 0.0
    avg_rouge = sum(r["rouge_l"] for r in results) / len(results) if results else 0.0

    return {
        "sample_count": len(results),
        "avg_bleu": round(avg_bleu, 4),
        "avg_rouge_l": round(avg_rouge, 4),
        "per_example": results,
    }

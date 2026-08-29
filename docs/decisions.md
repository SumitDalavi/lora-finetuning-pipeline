# Decisions

## ADR-001: LoRA Rank (r) Selection
**Date:** 2026-08-29  
**Status:** Accepted

**Context:**  
Fine-tuning full weights (e.g. 7B parameters) is too slow and expensive. We use PEFT/LoRA. We need to choose a rank `r`.

**Decision:**  
We default to `r=8` for the LoRA adapter.

**Consequences:**  
- ✅ Dramatically reduces trainable parameters (<1% of base model).
- ✅ Prevents catastrophic forgetting.
- ⚠️ May underfit highly complex novel reasoning tasks compared to `r=32` or `r=64`.

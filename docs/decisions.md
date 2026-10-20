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

## ADR-002: Progressive Credential Loading (PCL) Architecture
- **Context:** Production hardening required live backends (PostgreSQL, Redis, OpenAI, Anthropic), but removing mocks would break CI pipelines and create high developer onboarding friction.
- **Decision:** Implement Progressive Credential Loading. Keep in-memory and local mock heuristics as the default safe baseline, while wrapping external integrations in auto-promoting factory adapters triggered by environment variables.
- **Consequences:** 
  - *Pros:* 100% CI pass rate without requiring external secrets; zero setup friction for new contributors; progressive upgrade to production clusters.
  - *Cons:* Adds lightweight factory abstraction layers and runtime credential detection logic.

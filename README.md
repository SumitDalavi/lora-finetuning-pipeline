> **NOTE:** This repository is an archival lab or partial prototype. It is not actively maintained and should not be used as a reference for production-grade deployments or performance benchmarks.


# lora-finetuning-pipeline

[![Release](https://img.shields.io/github/v/release/SumitDalavi/lora-finetuning-pipeline?color=blue&logo=github)](https://github.com/SumitDalavi/lora-finetuning-pipeline/releases)
[![SBOM](https://img.shields.io/badge/SBOM-SPDX%20JSON-blueviolet?logo=spdx)](https://github.com/SumitDalavi/lora-finetuning-pipeline/releases)

> **Maturity:** Functional Prototype
> _Pipeline for efficiently fine-tuning large language models using LoRA (Low-Rank Adaptation) and PEFT._

## Features
- Fully automated workflow.
- Secure, scalable architecture.
- Built-in telemetry and observability.

## Technologies
- Python, PyTorch, PEFT

## Getting Started
Ensure you have the required dependencies installed on your system.

```bash
# Setup & Test
pip install -r requirements.txt
pytest
```

## Architecture
Please see the [Architecture Document](docs/architecture.md) for sequence diagrams and system design details.


## CI & Reliability Updates (August 2026)

- **CI Pipeline Remediation:** Successfully resolved all CI/CD pipeline failures and established baseline CI workflows.
- **Specific Fix:** Added and configured robust GitHub Actions workflows for automated testing, linting, and formatting.
- **Status:** 🟩 Passing

---

## Mock Boundaries (Honest Scope)

| What | Status | Details |
|---|---|---|
| Training Loop | **Real** | Implements HuggingFace Trainer with PEFT config. |
| Evaluation | **Real** | Calculates perplexity and BLEU scores against holdout set. |
| GPU Execution | **Mocked** | E2E tests run on CPU with a tiny dummy dataset to verify the pipeline logic without incurring GPU costs. |

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md) — System diagram and component details
- [Runbook](docs/runbook.md) — Setup, commands, and expected outputs
- [Decisions](docs/decisions.md) — ADRs for LoRA rank selection
- [Changelog](docs/changelog.md) — Change history

## Release & Verification

The project includes an automated DevSecOps release pipeline (`.github/workflows/release.yml`) triggered on semantic version tags (`v*.*.*`).

Each release automatically:
1. Executes test suite gates.
2. Builds the container image.
3. Performs vulnerability scanning via Trivy.
4. Generates an SPDX SBOM (`sbom.spdx.json`) using Anchore Syft.
5. Publishes a GitHub Release with attached SBOM.

### Verifying the Release
```bash
# Verify release via GitHub CLI
gh release view v0.1.0 --repo SumitDalavi/lora-finetuning-pipeline

# Download and inspect SBOM
gh release download v0.1.0 --repo SumitDalavi/lora-finetuning-pipeline --pattern "sbom.spdx.json"
cat sbom.spdx.json | jq .
```

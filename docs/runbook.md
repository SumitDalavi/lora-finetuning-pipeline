# Runbook — lora-finetuning-pipeline
> Last updated: 2026-08-29

## Quick Start
```bash
pip install -r requirements.txt
python src/train.py --config config.yaml
```

## Run Tests
```bash
pytest
bash tests/e2e/test_lora_training.sh
```

## Environment Variables
| Variable | Default | Purpose |
|---|---|---|
| MODEL_NAME | `gpt2` | Model used for testing locally |
| LORA_RANK | `8` | Rank of the update matrices |

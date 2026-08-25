import pytest
import os
from src.pipeline import LoRAFinetuner

def test_setup_model():
    tuner = LoRAFinetuner(model_name="test-model")
    assert tuner.setup_model() == True

def test_train_success(tmp_path):
    tuner = LoRAFinetuner()
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"text": "test"}')
    
    result = tuner.train(str(data_file), epochs=1)
    assert result["status"] == "success"
    assert result["epochs_completed"] == 1
    assert tuner.is_trained == True

def test_train_file_not_found():
    tuner = LoRAFinetuner()
    with pytest.raises(FileNotFoundError):
        tuner.train("nonexistent.jsonl")

def test_save_adapters(tmp_path):
    tuner = LoRAFinetuner()
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"text": "test"}')
    
    tuner.train(str(data_file))
    
    out_dir = tmp_path / "weights"
    adapter_path = tuner.save_adapters(str(out_dir))
    
    assert os.path.exists(adapter_path)
    with open(adapter_path) as f:
        assert "mock_lora_weights" in f.read()

def test_save_adapters_untrained(tmp_path):
    tuner = LoRAFinetuner()
    out_dir = tmp_path / "weights"
    with pytest.raises(RuntimeError):
        tuner.save_adapters(str(out_dir))

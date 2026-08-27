import pytest
import sys
from unittest.mock import MagicMock, patch
import os

# Create massive mock layer for ML imports before importing modules
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
sys.modules["torch"] = mock_torch

mock_transformers = MagicMock()
mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock(
    pad_token_id=0,
    eos_token="<eos>",
    pad_token="<pad>",
    __call__=MagicMock(return_value={"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]})
)
sys.modules["transformers"] = mock_transformers

mock_peft = MagicMock()
sys.modules["peft"] = mock_peft

mock_trl = MagicMock()
sys.modules["trl"] = mock_trl

mock_mlflow = MagicMock()
sys.modules["mlflow"] = mock_mlflow

mock_datasets = MagicMock()
sys.modules["datasets"] = mock_datasets

mock_bnb = MagicMock()
sys.modules["bitsandbytes"] = mock_bnb

mock_openai = MagicMock()
sys.modules["openai"] = mock_openai

mock_httpx = MagicMock()
sys.modules["httpx"] = mock_httpx

# Now import the modules safely
from eval.evaluate import load_models, generate_response, evaluate_with_llm, main as eval_main
from eval.serve import generate_ollama, generate_vllm, batch_evaluate_serving, _OK
from training.train import train_lora

# Also for preprocessing and pipeline coverage
from preprocessing.formatter import load_jsonl, save_jsonl
from preprocessing.tokenizer import tokenize_instruction_dataset
from eval.benchmarks import compute_rouge_l
from src.pipeline import LoRAFinetuner

def test_evaluate():
    # evaluate.py tests
    tok, base, ft = load_models()
    assert tok is not None
    assert base is not None
    
    # Test generate
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_tok = MagicMock()
    mock_tok.decode.return_value = "Instruction: inst\nInput: inp\nOutput: out"
    mock_tok.return_value = MagicMock(to=MagicMock(return_value={"input_ids": [1]}))
    
    res = generate_response(mock_model, mock_tok, "inst", "inp")
    assert "out" in res
    
    # Test llm eval
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_msg = MagicMock()
    mock_msg.message.content = '{"winner": "base", "reasoning": "test"}'
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_msg])
    
    eval_res = evaluate_with_llm("inst", "in", "base", "ft")
    assert eval_res["winner"] == "base"
    
    # Test main
    with patch("eval.evaluate.load_models", return_value=(mock_tok, mock_model, mock_model)):
        with patch("eval.evaluate.generate_response", return_value="out"):
            with patch("eval.evaluate.evaluate_with_llm", return_value={"winner": "tie"}):
                eval_main()

def test_eval_exception():
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("OpenAI error")
    with patch("eval.evaluate.load_models", return_value=(MagicMock(), MagicMock(), MagicMock())):
        with patch("eval.evaluate.generate_response", return_value="out"):
            eval_main() # Will catch the exception inside main loop

def test_serve():
    # mock httpx post
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "ollama out", "choices": [{"text": "vllm out"}]}
    mock_httpx.post.return_value = mock_resp
    
    # if _OK is true
    import eval.serve
    eval.serve._OK = True
    assert generate_ollama("model", "prompt") == "ollama out"
    assert generate_vllm("model", "prompt") == "vllm out"
    
    # batch eval
    ex = [{"instruction": "inst", "input": "", "output": "expected"}]
    res_o = batch_evaluate_serving("model", ex, backend="ollama")
    assert res_o == ["ollama out"]
    
    res_v = batch_evaluate_serving("model", ex, backend="vllm")
    assert res_v == ["vllm out"]

def test_serve_exceptions():
    import eval.serve
    eval.serve._OK = True
    mock_httpx.post.side_effect = Exception("error")
    assert "[error]" in generate_ollama("m", "p")
    assert "[error]" in generate_vllm("m", "p")
    
    # _OK false
    eval.serve._OK = False
    mock_httpx.post.side_effect = None
    assert "[mock]" in generate_ollama("m", "p")
    assert "[mock]" in generate_vllm("m", "p")

def test_train():
    # Create fake dataset file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        f.write(b'{"instruction": "i", "input": "in", "output": "o"}\n')
        temp_path = f.name
    
    try:
        with patch("training.train.os.path.exists", return_value=True):
            with patch("training.train.load_dataset") as ld:
                mock_ds = MagicMock()
                mock_ds.column_names = ["instruction"]
                mock_ds.map.return_value = mock_ds
                ld.return_value = mock_ds
                
                # mock cuda
                mock_torch.cuda.is_available.return_value = True
                
                # Run train
                train_lora()
    finally:
        os.remove(temp_path)

def test_train_no_dataset():
    with patch("training.train.os.path.exists", return_value=False):
        train_lora() # Returns early

def test_train_exception():
    with patch("training.train.os.path.exists", return_value=True):
        with patch("training.train.load_dataset") as ld:
            mock_ds = MagicMock()
            ld.return_value = mock_ds
            
            # trainer throws error
            mock_trl.SFTTrainer.side_effect = Exception("Train failed")
            with pytest.raises(Exception):
                train_lora()

def test_formatter(tmp_path):
    # Test save and load jsonl
    path = tmp_path / "test.jsonl"
    ex = [{"instruction": "sys", "input": "usr", "output": "ast"}]
    save_jsonl(ex, str(path))
    
    loaded = list(load_jsonl(str(path)))
    assert len(loaded) == 1
    assert loaded[0]["instruction"] == "sys"

def test_tokenizer_hf():
    # _HF is true
    import preprocessing.tokenizer as tok
    tok._HF = True
    ex = [{"instruction": "i", "input": "in", "output": "o"}]
    res = tokenize_instruction_dataset(ex)
    # mock_transformers.AutoTokenizer returns a mock tokenizer
    # It will use the mock
    pass

def test_pipeline_methods():
    import runpy
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        f.write(b'{"text": "Sample"}\n')
        temp_path = f.name
        
    try:
        with patch("src.pipeline.os.path.exists", return_value=True):
            runpy.run_module("src.pipeline", run_name="__main__")
    finally:
        os.remove(temp_path)



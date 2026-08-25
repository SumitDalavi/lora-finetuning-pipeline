import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "../training/lora-adapter-output"

def load_models():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    print("Loading LoRA adapter...")
    if os.path.exists(ADAPTER_PATH):
        finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print(f"Warning: Adapter path {ADAPTER_PATH} not found. Running with base model only.")
        finetuned_model = base_model
        
    return tokenizer, base_model, finetuned_model

def generate_response(model, tokenizer, instruction, input_text):
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

def evaluate_with_llm(instruction, input_text, base_output, finetuned_output):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    You are an expert AI evaluator.
    Task: {instruction}
    Input context: {input_text}
    
    Base Model Output: {base_output}
    Fine-Tuned Model Output: {finetuned_output}
    
    Which output is better structured, more accurate, and concisely summarizes the input?
    Reply with a JSON object containing "winner" ("base", "finetuned", or "tie") and "reasoning".
    """
    
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    return json.loads(res.choices[0].message.content)

def main():
    tokenizer, base_model, finetuned_model = load_models()
    
    test_cases = [
        {
            "instruction": "Summarize the following medical note:",
            "input": "Patient complains of severe headache, photophobia, and nausea for the past 24 hours. Neck stiffness is observed on examination. No recent trauma."
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nEvaluating Case: {case['input'][:50]}...")
        
        print("Running base model...")
        base_out = generate_response(base_model, tokenizer, case["instruction"], case["input"])
        
        print("Running fine-tuned model...")
        # Since finetuned_model wraps base_model, generating with it uses the adapter weights
        ft_out = generate_response(finetuned_model, tokenizer, case["instruction"], case["input"])
        
        print("Comparing with LLM-as-judge...")
        try:
            eval_result = evaluate_with_llm(case["instruction"], case["input"], base_out, ft_out)
        except Exception as e:
            eval_result = {"error": str(e)}
            
        results.append({
            "input": case["input"],
            "base_output": base_out,
            "finetuned_output": ft_out,
            "evaluation": eval_result
        })
        
    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

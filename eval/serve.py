from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="Medical LoRA Inference Server")

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "../training/lora-adapter-output"

tokenizer = None
model = None

class InferenceRequest(BaseModel):
    instruction: str
    input_text: str

@app.on_event("startup")
def load_models():
    global tokenizer, model
    print("Initializing inference server...")
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    if os.path.exists(ADAPTER_PATH):
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print("Warning: LoRA adapter not found. Serving base model only.")
        model = base_model

@app.post("/v1/completions")
async def generate(req: InferenceRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    prompt = f"Instruction: {req.instruction}\nInput: {req.input_text}\nOutput: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1)
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_response = response.replace(prompt, "").strip()
    
    return {
        "model": "TinyLlama-LoRA",
        "output": clean_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

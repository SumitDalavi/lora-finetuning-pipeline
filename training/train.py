import os
import torch
import mlflow
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import bitsandbytes as bnb

def train_lora():
    # 1. Configuration & MLflow Setup
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "lora_medical_summarization"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Tiny model for demo purposes
    output_dir = "./lora-adapter-output"
    
    # LoRA Hyperparameters
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    
    # Training Hyperparameters
    learning_rate = 2e-4
    batch_size = 4
    num_epochs = 3

    mlflow.start_run(run_name="tinyllama-lora-run-1")
    mlflow.log_params({
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs
    })

    print("Loading Dataset...")
    dataset_path = "../data/train.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run generate_dataset.py first.")
        mlflow.end_run()
        return
        
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_prompt(example):
        return f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"

    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization Config (simulate 4-bit loading, requires CUDA usually, 
    # but we will just load normally if CUDA is unavailable for demonstration)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] # Target attention layers
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=2,
        num_train_epochs=num_epochs,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        save_strategy="epoch",
        report_to="mlflow"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=lambda x: [format_prompt(item) for item in x] if isinstance(x, list) else [format_prompt(x)]
    )
    
    # Note: To avoid crashes on formatting_func, trl uses standard dataset mapping.
    # A safer way in TRL is mapping the dataset beforehand:
    def map_format(examples):
        return {"text": [format_prompt({"instruction": i, "input": inp, "output": o}) 
                         for i, inp, o in zip(examples["instruction"], examples["input"], examples["output"])]}
    
    formatted_dataset = dataset.map(map_format, batched=True, remove_columns=dataset.column_names)
    
    # Re-init trainer with pre-formatted dataset
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting Training Loop...")
    # This will likely crash or take forever on CPU, but the pipeline is complete.
    try:
        trainer.train()
        print(f"Training Complete. Saving LoRA adapter to {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_artifact(output_dir)
    except Exception as e:
        print("Training execution failed (likely due to missing GPU/CUDA):", e)
    
    mlflow.end_run()

if __name__ == "__main__":
    train_lora()

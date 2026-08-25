import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# We will generate a dataset for "Medical Note Summarization" 
# formatted as instruction, input, output.
NUM_SAMPLES = 10 
# Note: Set this to 10 for a quick demo, but in reality you'd want 500-2000.

prompt_template = """
You are an expert medical data synthesizer.
Generate {count} unique examples of medical notes and their corresponding concise summaries.
The output MUST be a valid JSON array of objects, with each object having exactly two keys: "medical_note" and "summary".

Example:
[
  {
    "medical_note": "Patient presents with a 3-day history of productive cough, fever up to 101F, and mild shortness of breath. No history of smoking. Exam reveals crackles in the left lower lobe.",
    "summary": "Suspected pneumonia in left lower lobe; symptoms include fever and productive cough."
  }
]
"""

def generate_dataset():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print(f"Generating {NUM_SAMPLES} synthetic medical note examples...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_template.format(count=NUM_SAMPLES)}
        ],
        temperature=0.7,
        response_format={ "type": "json_object" } 
    )
    
    # OpenAI might wrap the array in a dict if response_format="json_object" is used.
    # The prompt asked for a JSON array, so we must parse carefully.
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        # If it returned an object with a key holding the array:
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    data = data[key]
                    break
    except Exception as e:
        print("Failed to parse JSON", e)
        return

    # Write to JSONL format expected by HuggingFace datasets
    output_file = "train.jsonl"
    with open(output_file, 'w') as f:
        for item in data:
            # Transform into Alpaca instruction format
            record = {
                "instruction": "Summarize the following medical note:",
                "input": item.get("medical_note", ""),
                "output": item.get("summary", "")
            }
            f.write(json.dumps(record) + '\n')
            
    print(f"Successfully wrote {len(data)} examples to {output_file}.")

if __name__ == "__main__":
    generate_dataset()

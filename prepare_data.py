import json
import pandas as pd
from datasets import Dataset

def prepare_elden_ring_dataset(input_file, output_file):
    # Load your raw data (assuming a JSON list of objects with 'name' and 'description')
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    formatted_data = []

    for item in raw_data:
        name = item.get('name', 'Unknown Item')
        desc = item.get('description', 'No description available.')
        
        # We create multiple instruction variations for better model flexibility
        templates = [
            {"instruction": f"Tell me about the {name} in Elden Ring.", "output": desc},
            {"instruction": f"What is the lore behind the {name}?", "output": desc},
            {"instruction": f"Describe the {name}.", "output": desc}
        ]
        
        for template in templates:
            formatted_data.append({
                "instruction": template["instruction"],
                "input": "", # Leave empty for general QA
                "output": template["output"]
            })

    # Convert to Hugging Face Dataset format
    hf_dataset = Dataset.from_list(formatted_data)
    
    # Split into Train and Test (Validation)
    split_dataset = hf_dataset.train_test_split(test_size=0.1)
    
    # Save to disk
    split_dataset.save_to_disk(output_file)
    print(f"Dataset saved! Total samples: {len(formatted_data)}")
    
    return split_dataset

# Usage
# prepare_elden_ring_dataset('elden_ring_raw.json', 'elden_ring_finetune_ds')
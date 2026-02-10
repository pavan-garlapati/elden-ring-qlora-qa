import os
import json
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Strip HTML tags and Elden Ring specific markers like [DLC]
    text = re.sub(r'<[^>]+>|\[.*?\]', '', text)
    return text.strip()

def build_lore_library(folder_name, filenames):
    lore_library = {}
    
    for fname in filenames:
        # Construct the full path: data/Carian_Master.html
        file_path = os.path.join(folder_name, fname)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping...")
            continue
            
        print(f"Parsing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                # Finding item names in <h3> and descriptions in the following <p>
                for entry in soup.find_all('h3'):
                    name = entry.get_text(strip=True).lower()
                    desc_tag = entry.find_next('p')
                    if desc_tag:
                        lore_library[name] = clean_text(desc_tag.get_text())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return lore_library

# Execution
data_dir = "data"
html_files = ["Carian_Master.html", "Impaler_Master.html"]

master_lore = build_lore_library(data_dir, html_files)

# Save the results to your local directory for the next step
with open('master_lore.json', 'w') as f:
    json.dump(master_lore, f, indent=4)

print(f"Done! Extracted {len(master_lore)} items to master_lore.json")
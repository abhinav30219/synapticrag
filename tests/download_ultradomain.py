"""Script to download UltraDomain dataset"""
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_ultradomain(output_dir: str = "data/ultradomain"):
    """Download UltraDomain dataset and extract unique contexts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of domains
    domains = ["agriculture", "cs", "legal", "mix"]
    
    print("Downloading UltraDomain dataset...")
    
    for domain in tqdm(domains):
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("TommyChien/UltraDomain", domain, split="train")
            
            # Extract unique contexts
            contexts = set()
            for item in dataset:
                if "context" in item:
                    contexts.add(item["context"])
            
            # Convert to list and save
            contexts = list(contexts)
            output_path = output_dir / f"{domain}_unique_contexts.json"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(contexts, f, indent=2, ensure_ascii=False)
                
            print(f"Downloaded {domain}: {len(contexts)} unique contexts")
            
        except Exception as e:
            print(f"Error downloading {domain}: {str(e)}")
            continue

if __name__ == "__main__":
    download_ultradomain()

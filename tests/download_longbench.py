"""Script to download LongBench datasets"""
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_longbench(output_dir: str = "data/longbench"):
    """Download LongBench datasets"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of datasets to download
    datasets = [
        "narrativeqa",
        "qasper", 
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news"
    ]
    
    print("Downloading LongBench datasets...")
    
    for dataset_name in tqdm(datasets):
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("THUDM/longbench", dataset_name, split="test")
            
            # Convert to list of dicts
            data = []
            for item in dataset:
                example = {
                    "context": item["context"],
                    "question": item.get("input", ""),  # Some datasets don't have questions
                }
                
                # Handle different answer formats
                if "answers" in item:
                    example["answers"] = item["answers"]
                elif "answer" in item:
                    example["answer"] = item["answer"]
                    
                data.append(example)
            
            # Save to JSON
            output_path = output_dir / f"{dataset_name}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"Downloaded {dataset_name}: {len(data)} examples")
            
        except Exception as e:
            print(f"Error downloading {dataset_name}: {str(e)}")
            continue

if __name__ == "__main__":
    download_longbench()

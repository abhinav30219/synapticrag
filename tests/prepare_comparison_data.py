"""Script to prepare data for RAG system comparison"""
import os
import json
import shutil
import requests
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

def download_ultradomain(output_dir: str = "data/comparison"):
    """Download UltraDomain dataset files directly from GitHub"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for raw files
    base_url = "https://huggingface.co/datasets/TommyChien/UltraDomain/raw/main"
    
    # List of domain files to download
    domains = ["agriculture", "cs", "legal", "mix"]
    
    print("Downloading UltraDomain dataset...")
    
    for domain in tqdm(domains):
        try:
            # Download JSONL file
            url = f"{base_url}/{domain}.jsonl"
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse JSONL and extract contexts
            contexts = []
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "context" in data:
                            contexts.append(data["context"])
                    except json.JSONDecodeError:
                        continue
            
            # Save contexts
            with open(output_dir / f"{domain}_context.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(contexts))
                
            print(f"Saved {len(contexts)} contexts for {domain}")
            
        except Exception as e:
            print(f"Error downloading {domain}: {str(e)}")

def generate_queries(
    openai_api_key: str,
    output_dir: str = "data/comparison"
):
    """Generate queries for each domain using GPT-4"""
    output_dir = Path(output_dir)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Process each domain
    domains = ["agriculture", "cs", "legal", "mix"]
    
    print("\nGenerating queries...")
    
    for domain in tqdm(domains):
        try:
            # Read context
            context_file = output_dir / f"{domain}_context.txt"
            if not context_file.exists():
                print(f"Context file not found for {domain}, skipping...")
                continue
                
            with open(context_file) as f:
                context = f.read()
            
            # Create prompt
            prompt = f"""Given the following text from the {domain} domain:

{context[:4000]}  # Use first 4000 chars to stay within context limit

Generate 10 high-level questions that:
1. Require understanding the entire text
2. Cannot be answered by simple keyword matching
3. Need complex reasoning and synthesis
4. Cover different aspects and difficulty levels

Format each question on a new line starting with "- Question N: "
"""
            
            # Generate questions
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at creating high-quality evaluation questions."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Save questions
            with open(output_dir / f"{domain}_questions.txt", "w") as f:
                f.write(completion.choices[0].message.content)
                
            print(f"Generated questions for {domain}")
            
        except Exception as e:
            print(f"Error generating questions for {domain}: {str(e)}")

if __name__ == "__main__":
    # Download dataset
    download_ultradomain()
    
    # Generate queries if OpenAI key is available
    if "OPENAI_API_KEY" in os.environ:
        generate_queries(openai_api_key=os.environ["OPENAI_API_KEY"])
    else:
        print("\nSkipping query generation - OPENAI_API_KEY not found")

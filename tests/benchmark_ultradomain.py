"""Benchmark script for evaluating SynapticRAG on UltraDomain dataset"""
import os
import json
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import datasets
from transformers import GPT2Tokenizer
from openai import OpenAI

from synaptic.pipeline import Pipeline
from synaptic.config import Config

@dataclass
class UltraDomainResult:
    """Container for UltraDomain benchmark results"""
    domain: str
    query: str
    response: str
    metrics: Dict[str, Dict[str, str]]  # e.g. {"Comprehensiveness": {"Winner": "Answer 1", "Explanation": "..."}}

class UltraDomainEvaluator:
    """Evaluator for UltraDomain dataset"""
    
    def __init__(
        self,
        config: Config,
        openai_api_key: str,
        data_dir: str = "data/ultradomain",
        output_dir: str = "results/ultradomain"
    ):
        self.pipeline = Pipeline(config=config)
        self.openai = OpenAI(api_key=openai_api_key)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer for summary generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Domains to evaluate
        self.domains = ["agriculture", "cs", "legal", "mix"]
        
    def get_context_summary(self, context: str, total_tokens: int = 2000) -> str:
        """Get summary of context by combining start and end tokens"""
        tokens = self.tokenizer.tokenize(context)
        half_tokens = total_tokens // 2
        
        # Get tokens from start and end
        start_tokens = tokens[1000:1000 + half_tokens]
        end_tokens = tokens[-(1000 + half_tokens):-1000]
        
        # Combine tokens
        summary_tokens = start_tokens + end_tokens
        summary = self.tokenizer.convert_tokens_to_string(summary_tokens)
        
        return summary
    
    def generate_queries(self, context_summary: str) -> List[str]:
        """Generate high-level queries using GPT-4"""
        prompt = f"""Given the following description of a dataset:

{context_summary}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output the results in the following structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1:
        - Question 2:
        - Question 3:
        - Question 4:
        - Question 5:
    - Task 2: [task description]
        ...
    - Task 5: [task description]
- User 2: [user description]
    ...
- User 5: [user description]
    ..."""
        
        completion = self.openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at generating insightful questions about datasets."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract questions from response
        response = completion.choices[0].message.content
        queries = []
        for line in response.split("\n"):
            if line.strip().startswith("- Question"):
                query = line.split(":", 1)[1].strip()
                queries.append(query)
        
        return queries
    
    def evaluate_response(
        self,
        query: str,
        response1: str,
        response2: str,
        system1: str = "NaiveRAG",
        system2: str = "SynapticRAG"
    ) -> Dict:
        """Evaluate responses using GPT-4"""
        prompt = f"""You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Question: {query}

Answer 1 ({system1}):
{response1}

Answer 2 ({system2}):
{response2}

Evaluate both answers using these criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects?
- Diversity: How varied and rich is the answer in providing different perspectives?
- Empowerment: How well does the answer help understand and make judgments?

Output your evaluation in JSON format:
{{
    "Comprehensiveness": {{
        "Winner": "[{system1} or {system2}]",
        "Explanation": "[explanation]"
    }},
    "Diversity": {{
        "Winner": "[{system1} or {system2}]",
        "Explanation": "[explanation]"
    }},
    "Empowerment": {{
        "Winner": "[{system1} or {system2}]",
        "Explanation": "[explanation]"
    }},
    "Overall": {{
        "Winner": "[{system1} or {system2}]",
        "Explanation": "[summary explanation]"
    }}
}}"""
        
        completion = self.openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of question-answering systems."},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            return json.loads(completion.choices[0].message.content)
        except:
            print("Error parsing evaluation response")
            return {}
    
    def evaluate_domain(self, domain: str) -> List[UltraDomainResult]:
        """Evaluate on a specific domain"""
        # Load domain data
        data_path = self.data_dir / f"{domain}_unique_contexts.json"
        if not data_path.exists():
            print(f"Domain data not found: {data_path}")
            return []
            
        with open(data_path) as f:
            contexts = json.load(f)
            
        results = []
        
        # Process each context
        for context in tqdm(contexts, desc=f"Evaluating {domain}"):
            # Get context summary
            summary = self.get_context_summary(context)
            
            # Generate queries
            queries = self.generate_queries(summary)
            
            # Get responses for each query
            for query in queries:
                # Get SynapticRAG response
                synaptic_response = self.pipeline(
                    query=query,
                    context=context,
                    retrieval_mode="hybrid"
                )
                
                # Get NaiveRAG response (using only vector retrieval)
                naive_response = self.pipeline(
                    query=query,
                    context=context,
                    retrieval_mode="naive"
                )
                
                # Evaluate responses
                evaluation = self.evaluate_response(
                    query=query,
                    response1=naive_response,
                    response2=synaptic_response,
                    system1="NaiveRAG",
                    system2="SynapticRAG"
                )
                
                results.append(UltraDomainResult(
                    domain=domain,
                    query=query,
                    response=synaptic_response,
                    metrics=evaluation
                ))
                
        # Save domain results
        output_path = self.output_dir / f"{domain}_results.json"
        with open(output_path, "w") as f:
            json.dump([{
                "domain": r.domain,
                "query": r.query,
                "response": r.response,
                "metrics": r.metrics
            } for r in results], f, indent=2)
            
        return results
    
    def evaluate_all(self) -> Dict:
        """Evaluate on all domains"""
        all_results = {}
        metrics_by_domain = {}
        
        for domain in self.domains:
            results = self.evaluate_domain(domain)
            all_results[domain] = results
            
            # Calculate domain metrics
            domain_metrics = {
                "Comprehensiveness": 0,
                "Diversity": 0,
                "Empowerment": 0,
                "Overall": 0
            }
            
            for result in results:
                for metric in domain_metrics:
                    if result.metrics.get(metric, {}).get("Winner") == "SynapticRAG":
                        domain_metrics[metric] += 1
                        
            # Convert to percentages
            total = len(results)
            domain_metrics = {
                k: round(100 * v / total, 2) if total > 0 else 0
                for k, v in domain_metrics.items()
            }
            
            metrics_by_domain[domain] = domain_metrics
            
        # Calculate overall metrics
        overall_metrics = {}
        for metric in ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]:
            scores = [m[metric] for m in metrics_by_domain.values()]
            overall_metrics[metric] = round(sum(scores) / len(scores), 2)
            
        summary = {
            "domain_metrics": metrics_by_domain,
            "overall_metrics": overall_metrics
        }
        
        # Save summary
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary

if __name__ == "__main__":
    # Initialize evaluator
    config = Config()
    evaluator = UltraDomainEvaluator(
        config=config,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Print summary
    print("\nEvaluation Results:")
    print("\nOverall Metrics:")
    for metric, score in results["overall_metrics"].items():
        print(f"{metric}: {score:.2f}%")
    print("\nDomain Metrics:")
    for domain, metrics in results["domain_metrics"].items():
        print(f"\n{domain}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.2f}%")

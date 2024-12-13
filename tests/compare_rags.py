"""Script to compare SynapticRAG against naive vector-based RAG using GPT-4 as judge"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI

from synaptic.pipeline import SynapticPipeline
from synaptic.config import SynapticConfig, StorageConfig
from synaptic.storage.kv_storage import KVStorage
from synaptic.storage.vector_storage import VectorStorage
from synaptic.storage.graph_storage import GraphStorage
from tests.naive_rag import NaiveRAG

@dataclass
class RAGSystem:
    """Container for RAG system"""
    name: str
    system: object
    query_kwargs: Dict

class RAGComparator:
    """Compare RAG systems using GPT-4 as judge"""
    
    def __init__(
        self,
        openai_api_key: str,
        data_dir: str = "data/comparison",
        output_dir: str = "results/comparison",
        cache_dir: str = "cache/comparison"
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RAG systems
        self.systems = self._init_rag_systems()
    
    def _init_rag_systems(self) -> List[RAGSystem]:
        """Initialize RAG systems"""
        systems = []
        
        # SynapticRAG
        storage_config = StorageConfig(
            kv_storage_cls=KVStorage,
            vector_storage_cls=VectorStorage,
            graph_storage_cls=GraphStorage,
            working_dir=str(self.cache_dir / "synaptic"),
            cache_dir=str(self.cache_dir / "synaptic" / "cache")
        )
        config = SynapticConfig(storage=storage_config)
        synaptic = SynapticPipeline(config=config)
        systems.append(RAGSystem(
            name="SynapticRAG",
            system=synaptic,
            query_kwargs={"retrieval_mode": "hybrid"}
        ))
        
        # Naive RAG
        naive = NaiveRAG(
            llm_model_func=synaptic.rag.llm.generate  # Use same LLM
        )
        systems.append(RAGSystem(
            name="NaiveRAG",
            system=naive,
            query_kwargs={}
        ))
        
        return systems
    
    def evaluate_responses(
        self,
        query: str,
        responses: Dict[str, str]
    ) -> Dict:
        """Evaluate responses using GPT-4"""
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """
        
        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the answers:

        **SynapticRAG:**
        {responses["SynapticRAG"]}

        **NaiveRAG:**
        {responses["NaiveRAG"]}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[SynapticRAG or NaiveRAG]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[SynapticRAG or NaiveRAG]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[SynapticRAG or NaiveRAG]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall": {{
                "Winner": "[SynapticRAG or NaiveRAG]",
                "Explanation": "[Summarize why this system is the overall winner]"
            }}
        }}"""
        
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            return json.loads(completion.choices[0].message.content)
        except:
            print("Error parsing evaluation response")
            return {}
    
    def compare_systems(self, domain: str) -> Dict:
        """Compare RAG systems on given domain"""
        # Load context
        with open(self.data_dir / f"{domain}_context.txt") as f:
            context = f.read()
            
        # Load queries
        with open(self.data_dir / f"{domain}_questions.txt") as f:
            queries = [q.strip() for q in f.readlines() if q.strip()]
            
        # Initialize systems with context
        for system in self.systems:
            if isinstance(system.system, SynapticPipeline):
                system.system.process_documents(context)
            elif isinstance(system.system, NaiveRAG):
                system.system.add_documents([context])
            
        results = []
        
        # Compare systems
        print(f"\nComparing systems on {len(queries)} queries for {domain} domain")
        
        for query in tqdm(queries):
            # Get responses from all systems
            responses = {}
            for system in self.systems:
                if isinstance(system.system, SynapticPipeline):
                    result = system.system.query(
                        query=query,
                        **system.query_kwargs
                    )
                    responses[system.name] = result.response
                else:
                    response = system.system.query(
                        query=query,
                        **system.query_kwargs
                    )
                    responses[system.name] = response
            
            # Evaluate responses
            evaluation = self.evaluate_responses(
                query=query,
                responses=responses
            )
            
            results.append({
                "query": query,
                "responses": responses,
                "evaluation": evaluation
            })
        
        # Save detailed results
        output_path = self.output_dir / f"{domain}_detailed_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        # Calculate statistics
        stats = self._calculate_stats(results)
        
        # Save overall results
        with open(self.output_dir / f"{domain}_results.json", "w") as f:
            json.dump({
                "detailed_results": results,
                "statistics": stats
            }, f, indent=2)
            
        return stats
    
    def _calculate_stats(self, results: List[Dict]) -> Dict:
        """Calculate statistics from results"""
        stats = {
            "Comprehensiveness": {name: 0 for name in ["SynapticRAG", "NaiveRAG"]},
            "Diversity": {name: 0 for name in ["SynapticRAG", "NaiveRAG"]},
            "Empowerment": {name: 0 for name in ["SynapticRAG", "NaiveRAG"]},
            "Overall": {name: 0 for name in ["SynapticRAG", "NaiveRAG"]}
        }
        
        # Count wins
        for result in results:
            evaluation = result["evaluation"]
            
            # Count wins for each metric
            for metric in ["Comprehensiveness", "Diversity", "Empowerment", "Overall"]:
                if metric in evaluation and "Winner" in evaluation[metric]:
                    winner = evaluation[metric]["Winner"]
                    stats[metric][winner] += 1
        
        # Convert to percentages
        total = len(results)
        for metric in stats:
            for system in stats[metric]:
                stats[metric][system] = round(100 * stats[metric][system] / total, 2)
                
        return stats

if __name__ == "__main__":
    # Initialize comparator
    comparator = RAGComparator(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run comparison on each domain
    domains = ["agriculture", "cs", "legal", "mix"]
    for domain in domains:
        print(f"\nEvaluating {domain} domain:")
        stats = comparator.compare_systems(domain)
        
        # Print results
        print("\nResults:")
        for metric, results in stats.items():
            print(f"\n{metric}:")
            for system, score in results.items():
                print(f"  {system}: {score:.2f}%")
                
        # Clean up cache
        shutil.rmtree(comparator.cache_dir)
        comparator.cache_dir.mkdir()

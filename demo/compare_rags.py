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
from synaptic.config import SynapticConfig, StorageConfig, MemoryConfig, RetrieverConfig, GraphConfig
from synaptic.storage.kv_storage import JsonKVStorage
from synaptic.storage.vector_storage import FaissVectorStorage
from synaptic.storage.graph_storage import NetworkXStorage
from synaptic.llm.claude_model import ClaudeModel, ClaudeConfig
from naive_rag import NaiveRAG  # Import from local file

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
        data_dir: str = "../data/comparison",  # Updated path
        output_dir: str = "../results/comparison",  # Updated path
        cache_dir: str = "../cache/comparison"  # Updated path
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
            working_dir="./workspace",
            cache_dir="./cache",
            kv_storage_cls=JsonKVStorage,
            vector_storage_cls=FaissVectorStorage,
            graph_storage_cls=NetworkXStorage,
            kv_namespace="kv_storage",
            vector_namespace="vector_storage",
            graph_namespace="graph_storage",
            enable_compression=True,
            compression_level=3,
            max_cache_size=1000,
            cache_ttl=3600
        )
        
        # Create memory config with higher weight for memory
        memory_config = MemoryConfig(
            model_name_or_path="claude-3-5-sonnet-20241022",
            memory_weight=0.4,  # Higher weight for memory
            clue_boost=0.2,
            min_memory_relevance=0.3,
            enable_cache=True
        )
        
        # Create retriever config with balanced weights
        retriever_config = RetrieverConfig(
            model_name_or_path="text-embedding-3-large",
            vector_weight=0.3,  # Balanced weight for vector
            graph_weight=0.3,   # Balanced weight for graph
            enable_reranking=True,
            similarity_threshold=0.3,
            rerank_top_k=5,
            max_context_length=2000
        )
        
        # Create graph config with lower thresholds for better coverage
        graph_config = GraphConfig(
            entity_confidence_threshold=0.3,
            relationship_confidence_threshold=0.2,
            enable_node2vec=True,
            node2vec_dimensions=1536  # Match OpenAI embedding dimension
        )
        
        # Create main config
        config = SynapticConfig(
            memory=memory_config,
            retriever=retriever_config,
            graph=graph_config,
            storage=storage_config,
            enable_llm_cache=True
        )
        
        synaptic = SynapticPipeline(config=config)
        systems.append(RAGSystem(
            name="SynapticRAG",
            system=synaptic,
            query_kwargs={"retrieval_mode": "hybrid"}  # Use hybrid mode to combine memory, graph, and vector
        ))
        
        # Create Claude model for naive RAG
        claude_config = ClaudeConfig(model_name="claude-3-5-sonnet-20241022")
        claude_model = ClaudeModel(config=claude_config)
        
        # Naive RAG
        naive = NaiveRAG(
            llm_model_func=claude_model.generate,  # Use Claude model directly
            similarity_threshold=0.3  # Use same threshold as SynapticRAG
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
            model="gpt-4",  # Fixed model name
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
        context_file = self.data_dir / f"{domain}_context.txt"
        if not context_file.exists():
            print(f"No context file found for {domain} domain")
            return {}
            
        with open(context_file) as f:
            context = f.read().strip()
            
        if not context:
            print(f"Empty context for {domain} domain")
            return {}
            
        # Load queries
        query_file = self.data_dir / f"{domain}_questions.txt"
        if not query_file.exists():
            print(f"No questions file found for {domain} domain")
            return {}
            
        with open(query_file) as f:
            queries = [q.strip() for q in f.readlines() if q.strip()]
            
        if not queries:
            print(f"No questions found for {domain} domain")
            return {}
            
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
                    # Handle empty response
                    if not result.response:
                        responses[system.name] = "I could not find any relevant information to answer your question."
                    else:
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
    
    # Run comparison only on mix domain
    domain = "mix"
    print(f"\nEvaluating {domain} domain:")
    stats = comparator.compare_systems(domain)
    
    if stats:
        # Print results
        print("\nResults:")
        for metric, results in stats.items():
            print(f"\n{metric}:")
            for system, score in results.items():
                print(f"  {system}: {score:.2f}%")
                
        # Clean up cache
        shutil.rmtree(comparator.cache_dir)
        comparator.cache_dir.mkdir()

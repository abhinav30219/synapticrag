"""Hybrid retrieval combining memory, graph and vector search"""
import torch
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from torch_geometric.data import HeteroData
from langchain_openai import OpenAIEmbeddings
from ..utils import cos_sim, normalize_embeddings

@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    text: str
    score: float
    source_id: str
    metadata: Optional[Dict] = None

class HybridRetriever:
    """Hybrid retrieval combining memory, graph and vector search"""
    
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-large",
        top_k: int = 3,
        device: Optional[str] = None,
        memory_weight: float = 0.4,
        vector_weight: float = 0.3,
        graph_weight: float = 0.3
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.memory_weight = memory_weight
        self.vector_weight = vector_weight 
        self.graph_weight = graph_weight
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
        
        # Storage for chunks and embeddings
        self.chunks: Dict[str, str] = {}
        self.chunk_embeddings: Dict[str, torch.Tensor] = {}
    
    def add_chunks(self, chunks: Dict[str, str]) -> None:
        """Add text chunks to retriever"""
        if not chunks:
            return
            
        # Store chunks
        self.chunks.update(chunks)
        
        # Generate embeddings for new chunks
        new_embeddings = self.embeddings.embed_documents(list(chunks.values()))
        
        # Store embeddings
        for chunk_id, embedding in zip(chunks.keys(), new_embeddings):
            self.chunk_embeddings[chunk_id] = torch.tensor(embedding, device=self.device)
    
    def retrieve_by_query(
        self,
        query: str,
        clues: Optional[List[str]] = None,
        memory_response: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Retrieve chunks by query similarity using low-level clues"""
        if not self.chunks:
            return []
            
        # Get query embedding
        query_embedding = torch.tensor(
            self.embeddings.embed_query(query),
            device=self.device
        )
        
        # Get clue embeddings if provided
        clue_embeddings = None
        if clues:
            clue_embeddings = torch.tensor(
                self.embeddings.embed_documents(clues),
                device=self.device
            )
        
        # Get memory embedding if provided
        memory_embedding = None
        if memory_response:
            memory_embedding = torch.tensor(
                self.embeddings.embed_query(memory_response),
                device=self.device
            )
        
        # Stack chunk embeddings
        chunk_ids = list(self.chunk_embeddings.keys())
        chunk_embeddings = torch.stack([
            self.chunk_embeddings[chunk_id] for chunk_id in chunk_ids
        ])
        
        # Compute similarities
        similarities = cos_sim(query_embedding, chunk_embeddings)[0]
        
        # Add clue similarities if available
        if clue_embeddings is not None:
            clue_similarities = cos_sim(clue_embeddings, chunk_embeddings)
            # Take max similarity across clues for each chunk
            max_clue_similarities, _ = clue_similarities.max(dim=0)
            # Combine with query similarities
            similarities = (similarities + max_clue_similarities) / 2
        
        # Add memory similarities if available
        if memory_embedding is not None:
            memory_similarities = cos_sim(memory_embedding, chunk_embeddings)[0]
            # Weight memory similarities
            similarities = (
                self.vector_weight * similarities + 
                self.memory_weight * memory_similarities
            ) / (self.vector_weight + self.memory_weight)
        
        # Get top results
        k = top_k or self.top_k
        scores, indices = similarities.topk(min(k, len(chunk_ids)))
        
        # Format results
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            chunk_id = chunk_ids[idx]
            results.append(RetrievalResult(
                text=self.chunks[chunk_id],
                score=score,
                source_id=chunk_id.split("_chunk_")[0],
                metadata={"type": "local"}
            ))
        
        return results
    
    def retrieve_by_graph(
        self,
        query: str,
        graph: HeteroData,
        entity_embeddings: Dict[str, torch.Tensor],
        clues: Optional[List[str]] = None,
        memory_response: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Retrieve chunks using graph structure and high-level clues"""
        if not self.chunks or not graph.node_types:
            return []
            
        # Get query embedding
        query_embedding = torch.tensor(
            self.embeddings.embed_query(query),
            device=self.device
        )
        
        # Get clue embeddings if provided
        clue_embeddings = None
        if clues:
            clue_embeddings = torch.tensor(
                self.embeddings.embed_documents(clues),
                device=self.device
            )
        
        # Get memory embedding if provided
        memory_embedding = None
        if memory_response:
            memory_embedding = torch.tensor(
                self.embeddings.embed_query(memory_response),
                device=self.device
            )
        
        # Compute similarities with entity embeddings and get node degrees
        top_entities = []
        for node_type, embeddings in entity_embeddings.items():
            node_similarities = cos_sim(query_embedding, embeddings)[0]
            
            # Add clue influence if available
            if clue_embeddings is not None:
                clue_similarities = cos_sim(clue_embeddings, embeddings)
                max_clue_similarities, _ = clue_similarities.max(dim=0)
                node_similarities = (node_similarities + max_clue_similarities) / 2
            
            # Add memory influence if available
            if memory_embedding is not None:
                memory_similarities = cos_sim(memory_embedding, embeddings)[0]
                node_similarities = (
                    self.graph_weight * node_similarities + 
                    self.memory_weight * memory_similarities
                ) / (self.graph_weight + self.memory_weight)
            
            # Get node metadata
            metadata = graph[node_type].metadata
            
            # Combine scores with node degrees
            for idx, (score, meta) in enumerate(zip(node_similarities, metadata)):
                source_id = meta.get("source_id", "")
                if source_id:
                    # Find chunks for this source
                    source_chunks = {
                        chunk_id: chunk
                        for chunk_id, chunk in self.chunks.items()
                        if chunk_id.startswith(source_id)
                    }
                    if source_chunks:
                        # Use highest chunk ID as representative text
                        chunk_id = max(source_chunks.keys())
                        top_entities.append((score.item(), source_id, source_chunks[chunk_id]))
        
        # Sort by score and limit to top_k
        top_entities.sort(reverse=True, key=lambda x: x[0])
        k = top_k or self.top_k
        top_entities = top_entities[:k]
        
        # Format results
        results = []
        for score, source_id, text in top_entities:
            results.append(RetrievalResult(
                text=text,
                score=score,
                source_id=source_id,
                metadata={"type": "global"}
            ))
        
        return results
    
    def hybrid_retrieve(
        self,
        query: str,
        clues: Optional[List[str]] = None,
        graph: Optional[HeteroData] = None,
        entity_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        memory_response: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Combine memory, query and graph-based retrieval"""
        # Get results from both methods
        query_results = self.retrieve_by_query(
            query, 
            clues=clues,
            memory_response=memory_response,
            top_k=top_k
        )
        
        graph_results = []
        if graph is not None and entity_embeddings is not None:
            graph_results = self.retrieve_by_graph(
                query, 
                graph, 
                entity_embeddings,
                clues=clues,
                memory_response=memory_response,
                top_k=top_k
            )
        
        # Combine and deduplicate results
        combined_results = {}
        for result in query_results + graph_results:
            if result.source_id not in combined_results:
                combined_results[result.source_id] = result
            else:
                # Keep result with higher score
                if result.score > combined_results[result.source_id].score:
                    combined_results[result.source_id] = result
        
        # Sort by score and limit to top_k
        k = top_k or self.top_k
        results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )[:k]
        
        return results

import torch
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from .memory_model import MemoryModel, MemoryState

@dataclass
class MemoryBlock:
    """Container for memory blocks"""
    text: str
    state: MemoryState
    embedding: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None

@dataclass
class RecallResult:
    """Container for recall results"""
    text: str
    score: float
    metadata: Optional[Dict] = None

@dataclass
class ConsolidatedMemory:
    """Container for consolidated memories"""
    texts: List[str]
    metadata: Optional[Dict] = None
    score: Optional[float] = None

class Memorizer:
    """Manages memory consolidation and retrieval"""
    
    def __init__(
        self,
        llm: Optional[MemoryModel] = None,
        max_memory_tokens: int = 1024,
        consolidation_threshold: float = 0.9,
        device: Optional[str] = None
    ):
        self.llm = llm or MemoryModel()
        self.max_memory_tokens = max_memory_tokens
        self.consolidation_threshold = consolidation_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize storage
        self.memory_blocks: List[MemoryBlock] = []
        self.consolidated_memories: List[ConsolidatedMemory] = []
        self._current_timestamp = 0
    
    def memorize(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> MemoryBlock:
        """Add text to memory"""
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
            
        # Generate memory state
        state = self.llm.memorize(text, metadata)
        
        # Generate embedding
        embedding = self.llm.get_memory_embeddings(text)
        
        # Create metadata with timestamp
        block_metadata = metadata.copy() if metadata else {}
        block_metadata.update({
            "timestamp": self._current_timestamp,
            "token_count": state.memory_tokens.size(1),
            "original_text": text  # Store original text for keyword matching
        })
        self._current_timestamp += 1
        
        # Create memory block
        block = MemoryBlock(
            text=text,
            state=state,
            embedding=embedding,
            metadata=block_metadata
        )
        
        # Add to storage
        self.memory_blocks.append(block)
        
        # Consolidate if needed
        total_tokens = sum(
            block.state.memory_tokens.size(1)
            for block in self.memory_blocks
        )
        if total_tokens >= self.max_memory_tokens:
            self.consolidate_memories()
            
        return block
    
    def consolidate_memories(self) -> None:
        """Consolidate similar memories"""
        if len(self.memory_blocks) < 2:
            return
            
        # Calculate similarity matrix
        similarity_matrix = torch.zeros(
            (len(self.memory_blocks), len(self.memory_blocks)),
            device=self.device
        )
        
        for i, block_i in enumerate(self.memory_blocks):
            for j, block_j in enumerate(self.memory_blocks):
                if i != j:
                    # Calculate semantic similarity
                    # Ensure embeddings are 2D
                    emb_i = block_i.embedding.unsqueeze(0) if block_i.embedding.dim() == 1 else block_i.embedding
                    emb_j = block_j.embedding.unsqueeze(0) if block_j.embedding.dim() == 1 else block_j.embedding
                    
                    # Calculate cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=1).mean()
                    similarity_matrix[i, j] = similarity.item()
        
        # Find groups of similar memories
        consolidated = set()
        for i in range(len(self.memory_blocks)):
            if i in consolidated:
                continue
                
            # Find similar memories
            similar = []
            for j in range(len(self.memory_blocks)):
                if i != j and j not in consolidated:
                    # Use >= for threshold comparison
                    if similarity_matrix[i, j] >= self.consolidation_threshold:
                        similar.append(j)
            
            # Only consolidate if we found similar memories
            if similar:
                # Create consolidated memory
                texts = [self.memory_blocks[idx].text for idx in [i] + similar]
                metadata = {
                    "source_blocks": [i] + similar,
                    "similarity_scores": [1.0] + [
                        similarity_matrix[i, j].item() for j in similar
                    ],
                    "original_metadata": [
                        self.memory_blocks[idx].metadata for idx in [i] + similar
                    ],
                    "timestamp": min(
                        self.memory_blocks[idx].metadata["timestamp"]
                        for idx in [i] + similar
                    ),
                    "original_texts": [
                        self.memory_blocks[idx].metadata["original_text"]
                        for idx in [i] + similar
                    ]
                }
                self.consolidated_memories.append(
                    ConsolidatedMemory(texts=texts, metadata=metadata)
                )
                consolidated.update([i] + similar)
        
        # Keep unconsolidated memories
        new_blocks = []
        for i in range(len(self.memory_blocks)):
            if i not in consolidated:
                new_blocks.append(self.memory_blocks[i])
        
        self.memory_blocks = new_blocks
    
    def recall(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RecallResult]:
        """Retrieve most relevant memories"""
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
            
        # Get query embedding
        query_embedding = self.llm.get_memory_embeddings(query)
        
        # Score all memories
        results = []
        
        # Score unconsolidated memories
        for block in self.memory_blocks:
            # Ensure embeddings are 2D
            q_emb = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding
            b_emb = block.embedding.unsqueeze(0) if block.embedding.dim() == 1 else block.embedding
            
            # Calculate semantic similarity
            score = torch.nn.functional.cosine_similarity(q_emb, b_emb, dim=1).mean().item()
            
            # Boost score for relevant keywords
            query_terms = set(query.lower().split())
            text_terms = set(block.metadata["original_text"].lower().split())
            overlap = len(query_terms & text_terms)
            if overlap > 0:
                score *= (1 + 0.2 * overlap)  # Boost by 20% per matching term
            
            results.append(RecallResult(
                text=block.text,
                score=score,
                metadata=block.metadata
            ))
        
        # Score consolidated memories
        for consolidated in self.consolidated_memories:
            # Calculate embedding for all texts in consolidated memory
            embeddings = []
            for text in consolidated.metadata["original_texts"]:
                memory_embedding = self.llm.get_memory_embeddings(text)
                embeddings.append(memory_embedding)
            
            # Calculate max similarity across all texts
            max_score = float('-inf')
            for memory_embedding in embeddings:
                # Ensure embeddings are 2D
                q_emb = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding
                m_emb = memory_embedding.unsqueeze(0) if memory_embedding.dim() == 1 else memory_embedding
                
                score = torch.nn.functional.cosine_similarity(q_emb, m_emb, dim=1).mean().item()
                
                # Boost score for relevant keywords
                query_terms = set(query.lower().split())
                text_terms = set(" ".join(consolidated.metadata["original_texts"]).lower().split())
                overlap = len(query_terms & text_terms)
                if overlap > 0:
                    score *= (1 + 0.2 * overlap)  # Boost by 20% per matching term
                
                max_score = max(max_score, score)
            
            results.append(RecallResult(
                text=" | ".join(consolidated.texts),
                score=max_score,
                metadata=consolidated.metadata
            ))
        
        # Sort by score and timestamp (newer first for equal scores)
        results.sort(key=lambda x: (-x.score, -x.metadata["timestamp"]))
        return results[:top_k]
    
    def score_memories(
        self,
        query: str,
        top_k: int = 5
    ) -> List[float]:
        """Score memories based on relevance to query"""
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
            
        # Get all scores without limiting to top_k
        results = self.recall(query, top_k=len(self.memory_blocks) + len(self.consolidated_memories))
        
        # Return all scores
        return [result.score for result in results]
    
    def save(self, path: str) -> None:
        """Save memorizer state"""
        # Convert tensors to CPU before saving
        memory_blocks = []
        for block in self.memory_blocks:
            memory_blocks.append(MemoryBlock(
                text=block.text,
                state=MemoryState(
                    memory_tokens=block.state.memory_tokens.cpu(),
                    memory_mask=block.state.memory_mask.cpu(),
                    metadata=block.state.metadata
                ),
                embedding=block.embedding.cpu() if block.embedding is not None else None,
                metadata=block.metadata
            ))
        
        save_dict = {
            'memory_blocks': memory_blocks,
            'consolidated_memories': self.consolidated_memories,
            'current_timestamp': self._current_timestamp
        }
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load memorizer state"""
        load_dict = torch.load(path)
        
        # Move tensors to device
        self.memory_blocks = []
        for block in load_dict['memory_blocks']:
            self.memory_blocks.append(MemoryBlock(
                text=block.text,
                state=MemoryState(
                    memory_tokens=block.state.memory_tokens.to(self.device),
                    memory_mask=block.state.memory_mask.to(self.device),
                    metadata=block.state.metadata
                ),
                embedding=block.embedding.to(self.device) if block.embedding is not None else None,
                metadata=block.metadata
            ))
        self.consolidated_memories = load_dict['consolidated_memories']
        self._current_timestamp = load_dict['current_timestamp']
    
    def clear(self) -> None:
        """Clear all memories"""
        self.memory_blocks.clear()
        self.consolidated_memories.clear()
        self._current_timestamp = 0

"""Memory model using Claude API for storing and retrieving information"""
import json
import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from ..llm.claude_model import ClaudeModel, ClaudeConfig

@dataclass
class MemoryState:
    """Container for memory state"""
    memory_tokens: torch.Tensor
    memory_mask: torch.Tensor
    metadata: Optional[Dict] = None

    def __eq__(self, other):
        if not isinstance(other, MemoryState):
            return False
        # Compare metadata
        if self.metadata != other.metadata:
            return False
        # Compare tensors
        if self.memory_tokens.size() != other.memory_tokens.size():
            return False
        if self.memory_mask.size() != other.memory_mask.size():
            return False
        return (torch.allclose(self.memory_tokens, other.memory_tokens) and 
                torch.allclose(self.memory_mask, other.memory_mask))

    def __ne__(self, other):
        return not self.__eq__(other)

class MemoryModel:
    """Memory model using Claude API for storing and retrieving information"""
    
    def __init__(
        self,
        model_name_or_path: str = "claude-3-5-sonnet-20241022",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Claude model
        self.config = ClaudeConfig(model_name=model_name_or_path)
        self.model = ClaudeModel(config=self.config)
        
        # Initialize memory state
        self.memory_state = None
        self.memory_states: List[MemoryState] = []
    
    def generate(self, *args, **kwargs):
        """Forward to Claude model's generate method"""
        return self.model.generate(*args, **kwargs)
    
    def memorize(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> MemoryState:
        """Store text in memory"""
        # Create memory tokens using OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        memory_tokens = torch.tensor(embeddings.embed_documents([text])[0]).unsqueeze(0)
        memory_mask = torch.ones(1, memory_tokens.size(1), device=self.device)
        
        # Create memory state
        state = MemoryState(
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            metadata=metadata
        )
        
        # Add to memory states list
        self.memory_states.append(state)
        
        # Update global memory state by concatenating all states
        if len(self.memory_states) == 1:
            self.memory_state = state
        else:
            # Concatenate along sequence dimension
            all_tokens = torch.cat([s.memory_tokens for s in self.memory_states], dim=1)
            all_masks = torch.cat([s.memory_mask for s in self.memory_states], dim=1)
            
            # Create combined metadata
            combined_metadata = {}
            for i, s in enumerate(self.memory_states):
                if s.metadata:
                    combined_metadata[f"block_{i}"] = s.metadata
            
            self.memory_state = MemoryState(
                memory_tokens=all_tokens,
                memory_mask=all_masks,
                metadata=combined_metadata
            )
        
        return self.memory_state
    
    def recall(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """Retrieve relevant information from memory"""
        if not self.memory_state:
            return ""
        
        # Create prompt for memory recall using LightRAG's approach
        prompt = f"""Given the query below, please recall and synthesize relevant information from your memory. Focus on both:

1. High-level concepts and themes that relate to the query's broader implications
2. Low-level specific details, entities, and facts that directly answer the query

Query: {query}

Please provide a concise response that:
- Combines both abstract concepts and concrete details
- Maintains factual accuracy based on your memory
- Directly addresses the query's information needs
- Avoids speculation or unsupported claims

Response:"""
        
        # Generate response using Claude
        response = self.model.generate(prompt)
        return response
    
    def generate_response(
        self,
        query: str,
        context: str,
        max_length: int = 512
    ) -> str:
        """Generate response using query and context"""
        # Create prompt using LightRAG's style
        prompt = f"""Please analyze the following query and context to provide a comprehensive response.

Query: {query}

Context: {context}

Instructions:
1. Focus on directly answering the query using information from the context
2. Incorporate both high-level concepts and specific details
3. Maintain factual accuracy and avoid speculation
4. Structure the response clearly and logically
5. Use concrete examples from the context when relevant

Response:"""
        
        # Generate response using Claude
        response = self.model.generate(prompt)
        return response
    
    def generate_clues(
        self,
        prompt: str,
        memory_state: Optional[Dict] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate clues using Claude"""
        # Create prompt for keyword extraction
        clue_prompt = f"""Given the following text, extract two types of keywords:

1. High-level keywords: Abstract concepts, themes, and key topics that capture the broader meaning
2. Low-level keywords: Specific entities, terms, and details that are directly mentioned

Text: {prompt}

Please format your response as JSON with two lists:
{{
    "high_level_keywords": ["keyword1", "keyword2", ...],
    "low_level_keywords": ["keyword1", "keyword2", ...]
}}

Focus on the most relevant and important keywords. Each list should have 3-5 keywords.

Examples:

Text: "The new AI model demonstrated remarkable performance in natural language processing tasks, achieving state-of-the-art results on multiple benchmarks."
{{
    "high_level_keywords": ["artificial intelligence advancement", "technological innovation", "performance evaluation"],
    "low_level_keywords": ["AI model", "natural language processing", "benchmarks", "state-of-the-art results"]
}}

Text: "What are the environmental consequences of deforestation on biodiversity?"
{{
    "high_level_keywords": ["environmental impact", "ecosystem disruption", "conservation"],
    "low_level_keywords": ["deforestation", "biodiversity", "species loss", "habitat destruction"]
}}

Now extract keywords from the following text:
{prompt}"""

        # Add memory context if available
        if memory_state and "memory_embeddings" in memory_state:
            clue_prompt += "\n\nConsider the following memory context when extracting keywords:"
            clue_prompt += "\nMemory embeddings are available and should influence keyword extraction."
        
        # Generate clues using Claude
        response = self.model.generate(
            clue_prompt,
            temperature=temperature,
            keyword_extraction=True  # Enable special JSON handling
        )
        
        # Extract keywords from response
        try:
            # Get high and low level keywords
            high_level = response.get("high_level_keywords", [])
            low_level = response.get("low_level_keywords", [])
            
            # Combine keywords
            clues = []
            clues.extend(high_level)
            clues.extend(low_level)
            
            return clues[:num_return_sequences]
            
        except Exception as e:
            print(f"Error parsing clues: {str(e)}")
            return []
    
    def clear(self) -> None:
        """Clear memory state"""
        self.memory_state = None
        self.memory_states.clear()

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
import numpy as np
from dataclasses import dataclass

@dataclass
class EmbeddingOutput:
    """Container for embedding outputs"""
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    token_embeddings: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None

class GraphEmbedder:
    """Handles embedding generation for graph components"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        pooling_strategy: str = "mean",
        normalize: bool = True,
        cache_size: int = 10000
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Initialize cache
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_size = cache_size
        
    def embed_text(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> Union[EmbeddingOutput, List[EmbeddingOutput]]:
        """Generate embeddings for text"""
        
        # Handle single text
        if isinstance(texts, str):
            return self._embed_single(texts)
        
        # Process in batches
        outputs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_outputs = self._embed_batch(batch_texts)
            outputs.extend(batch_outputs)
        
        return outputs
    
    def embed_entity(
        self,
        name: str,
        entity_type: str,
        description: str
    ) -> EmbeddingOutput:
        """Generate embeddings for an entity"""
        
        # Create cache key
        cache_key = f"entity_{name}_{entity_type}"
        
        # Check cache
        if cache_key in self.cache:
            return EmbeddingOutput(embeddings=self.cache[cache_key])
        
        # Combine entity information
        text = f"{name} ({entity_type}): {description}"
        
        # Generate embedding
        output = self._embed_single(text)
        
        # Cache embedding
        self._update_cache(cache_key, output.embeddings)
        
        return output
    
    def embed_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        description: str,
        keywords: List[str]
    ) -> EmbeddingOutput:
        """Generate embeddings for a relationship"""
        
        # Create cache key
        cache_key = f"rel_{source}_{rel_type}_{target}"
        
        # Check cache
        if cache_key in self.cache:
            return EmbeddingOutput(embeddings=self.cache[cache_key])
        
        # Combine relationship information
        text = (
            f"{source} {rel_type} {target}\n"
            f"Description: {description}\n"
            f"Keywords: {', '.join(keywords)}"
        )
        
        # Generate embedding
        output = self._embed_single(text)
        
        # Cache embedding
        self._update_cache(cache_key, output.embeddings)
        
        return output
    
    def embed_subgraph(
        self,
        nodes: Dict[str, Dict],
        edges: List[Tuple[str, str, Dict]]
    ) -> Dict[str, torch.Tensor]:
        """Generate embeddings for a subgraph"""
        
        embeddings = {}
        
        # Embed nodes
        for node_id, node_data in nodes.items():
            output = self.embed_entity(
                name=node_data.get("name", node_id),
                entity_type=node_data.get("type", "UNKNOWN"),
                description=node_data.get("description", "")
            )
            embeddings[f"node_{node_id}"] = output.embeddings
        
        # Embed edges
        for src, dst, edge_data in edges:
            output = self.embed_relationship(
                source=nodes[src].get("name", src),
                target=nodes[dst].get("name", dst),
                rel_type=edge_data.get("type", "UNKNOWN"),
                description=edge_data.get("description", ""),
                keywords=edge_data.get("keywords", [])
            )
            embeddings[f"edge_{src}_{dst}"] = output.embeddings
        
        return embeddings
    
    def _embed_single(self, text: str) -> EmbeddingOutput:
        """Generate embeddings for a single text"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pool embeddings
        if self.pooling_strategy == "mean":
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        elif self.pooling_strategy == "cls":
            # CLS token pooling
            embeddings = outputs.last_hidden_state[:, 0]
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Normalize if requested
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return EmbeddingOutput(
            embeddings=embeddings,
            attention_mask=inputs["attention_mask"],
            token_embeddings=outputs.last_hidden_state
        )
    
    def _embed_batch(self, texts: List[str]) -> List[EmbeddingOutput]:
        """Generate embeddings for a batch of texts"""
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pool embeddings
        if self.pooling_strategy == "mean":
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        elif self.pooling_strategy == "cls":
            # CLS token pooling
            embeddings = outputs.last_hidden_state[:, 0]
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Normalize if requested
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Create output for each text
        batch_outputs = []
        for i in range(len(texts)):
            batch_outputs.append(EmbeddingOutput(
                embeddings=embeddings[i].unsqueeze(0),
                attention_mask=inputs["attention_mask"][i].unsqueeze(0),
                token_embeddings=outputs.last_hidden_state[i].unsqueeze(0)
            ))
        
        return batch_outputs
    
    def _update_cache(self, key: str, value: torch.Tensor) -> None:
        """Update embedding cache"""
        
        # Remove oldest item if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # Add new item
        self.cache[key] = value
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self.cache.clear()

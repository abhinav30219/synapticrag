"""Utility functions for SynapticRAG"""
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Initialize OpenAI embeddings
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def get_embeddings_model(model_name: str, device: Optional[str] = None) -> OpenAIEmbeddings:
    """Get embeddings model"""
    return OpenAIEmbeddings(model=model_name)

def encode_text(text: str, model: OpenAIEmbeddings) -> torch.Tensor:
    """Encode text to embeddings"""
    embedding = model.embed_query(text)
    return torch.tensor(embedding)

def get_openai_embeddings(text: str) -> torch.Tensor:
    """Get embeddings using OpenAI API via LangChain"""
    # Get embeddings
    embedding = embeddings.embed_query(text)
    return torch.tensor(embedding)

def batch_get_openai_embeddings(texts: List[str], batch_size: int = 128) -> torch.Tensor:
    """Get embeddings for multiple texts using OpenAI API via LangChain"""
    # Process in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Get embeddings
        batch_embeddings = embeddings.embed_documents(batch)
        # Convert to tensors
        batch_tensors = [torch.tensor(emb) for emb in batch_embeddings]
        all_embeddings.extend(batch_tensors)
    
    # Stack all embeddings
    return torch.stack(all_embeddings)

def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """Compute cosine similarity between two tensors"""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    
    # Ensure tensors are 2D
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    # Normalize and compute similarity
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def dot_score(a: Tensor, b: Tensor) -> Tensor:
    """Compute dot product between two tensors"""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    
    # Ensure tensors are 2D
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    return torch.mm(a, b.transpose(0, 1))

def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """Normalize embeddings tensor"""
    return F.normalize(embeddings, p=2, dim=1)

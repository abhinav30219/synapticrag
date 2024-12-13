"""Simple vector-based RAG implementation for comparison"""
import os
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from tqdm import tqdm

@dataclass
class Document:
    """Container for document and its embedding"""
    text: str
    embedding: np.ndarray

class NaiveRAG:
    """Simple vector-based RAG implementation"""
    
    def __init__(
        self,
        llm_model_func: callable,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3
    ):
        self.client = OpenAI()
        self.llm_model_func = llm_model_func
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.documents: List[Document] = []
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size
            
            # If not at end of text, try to break at sentence
            if end < len(text):
                # Look for sentence break within last 100 chars
                look_back = min(100, self.chunk_size//10)
                break_chars = ['.', '!', '?', '\n']
                
                # Search for last break char
                last_break = -1
                for i in range(end-1, end-look_back-1, -1):
                    if i < len(text) and text[i] in break_chars:
                        last_break = i + 1
                        break
                        
                if last_break != -1:
                    end = last_break
            
            # Add chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start
            start = end - self.chunk_overlap
            
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def add_documents(self, documents: List[str]) -> None:
        """Process and add documents"""
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc)
            all_chunks.extend(chunks)
            
        # Get embeddings for chunks
        print(f"Getting embeddings for {len(all_chunks)} chunks...")
        for chunk in tqdm(all_chunks):
            embedding = self._get_embedding(chunk)
            self.documents.append(Document(
                text=chunk,
                embedding=embedding
            ))
    
    def query(
        self,
        query: str,
        **kwargs
    ) -> str:
        """Query the system"""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Get similarities
        similarities = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((similarity, doc))
            
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Get top k chunks
        top_chunks = [doc.text for _, doc in similarities[:self.top_k]]
        context = "\n\n".join(top_chunks)
        
        # Create prompt
        prompt = f"""Use the following context to answer the question. Be detailed and comprehensive in your response.

Context:
{context}

Question:
{query}

Answer:"""
        
        # Generate response
        response = self.llm_model_func(prompt)
        return response

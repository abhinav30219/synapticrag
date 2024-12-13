"""Simple vector-based RAG implementation for comparison"""
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
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
        top_k: int = 3,
        similarity_threshold: float = 0.3
    ):
        self.client = OpenAI()
        self.llm_model_func = llm_model_func
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
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
            model="text-embedding-3-large",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
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
    
    def _get_top_chunks(self, query_embedding: np.ndarray) -> List[Tuple[float, str]]:
        """Get top-k most similar chunks"""
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            # Only include chunks above threshold
            if similarity >= self.similarity_threshold:
                similarities.append((similarity, doc.text))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return similarities[:self.top_k]
    
    def query(
        self,
        query: str,
        **kwargs
    ) -> str:
        """Query the system"""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Get top chunks
        top_chunks = self._get_top_chunks(query_embedding)
        
        # If no relevant chunks found, return a message
        if not top_chunks:
            return "I could not find any relevant information to answer your question."
            
        # Create context from chunks
        context = "\n\n".join(chunk for _, chunk in top_chunks)
        
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

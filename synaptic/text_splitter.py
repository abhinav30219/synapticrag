"""Text splitting utilities for SynapticRAG"""
from typing import List
import tiktoken
from semantic_text_splitter import TextSplitter

class HybridTextSplitter:
    """Hybrid text splitter combining token-based and semantic splitting"""
    
    def __init__(
        self,
        chunk_token_size: int = 1200,
        chunk_overlap_tokens: int = 100,
        semantic_chunk_size: int = 512,
        tiktoken_model: str = "cl100k_base",  # For token-based splitting
        language: str = "english"
    ):
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.semantic_chunk_size = semantic_chunk_size
        if language.lower() == "chinese":
            self.semantic_chunk_size = 2048
        
        # Initialize tokenizers
        self.tokenizer = tiktoken.get_encoding(tiktoken_model)
        # Create semantic splitter with GPT-3.5-turbo model and capacity
        self.semantic_splitter = TextSplitter.from_tiktoken_model(
            "gpt-3.5-turbo",  # Use GPT-3.5-turbo for semantic splitting
            capacity=self.semantic_chunk_size
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text using both token-based and semantic approaches"""
        # First do semantic splitting
        semantic_chunks = self.semantic_splitter.chunks(text)
        
        # Then do token-based splitting on each semantic chunk
        final_chunks = []
        for chunk in semantic_chunks:
            tokens = self.tokenizer.encode(chunk)
            
            # Create chunks with overlap
            for i in range(0, len(tokens), self.chunk_token_size - self.chunk_overlap_tokens):
                chunk_tokens = tokens[i:i + self.chunk_token_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                if chunk_text.strip():  # Only add non-empty chunks
                    final_chunks.append(chunk_text)
        
        return final_chunks
    
    def split_documents(self, documents: List[str]) -> List[str]:
        """Split multiple documents"""
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
        return all_chunks

"""FAISS-based vector storage implementation"""
import os
import json
import numpy as np
import faiss
from typing import Dict, List, Any, Optional
from tqdm.asyncio import tqdm
import asyncio
from .base import BaseVectorStorage, StorageConfig
from langchain_openai import OpenAIEmbeddings

class FaissVectorStorage(BaseVectorStorage):
    """FAISS-based vector storage"""
    
    def __init__(
        self,
        config: StorageConfig,
        embeddings: OpenAIEmbeddings,
        dimension: int = 1536,  # Default for OpenAI embeddings
        index_type: str = "Flat",
        metric: str = "cosine",
        batch_size: int = 64,
        meta_fields: Optional[set] = None
    ):
        super().__init__(config=config, embeddings=embeddings, meta_fields=meta_fields)
        self.dimension = dimension
        self.batch_size = batch_size
        
        # Initialize FAISS index
        if metric == "cosine":
            self.metric = faiss.METRIC_INNER_PRODUCT
            self.normalize = True
        elif metric == "l2":
            self.metric = faiss.METRIC_L2
            self.normalize = False
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self.index = self._create_index(index_type)
        
        # Storage for metadata
        self._metadata_file = os.path.join(
            config.working_dir,
            f"vector_store_{config.namespace}_metadata.json"
        )
        self._index_file = os.path.join(
            config.working_dir,
            f"vector_store_{config.namespace}_index.faiss"
        )
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._id_to_index: Dict[str, int] = {}
        self._next_index = 0
        
        # Load existing data if available
        self._load_data()
    
    def _create_index(self, index_type: str) -> faiss.Index:
        """Create FAISS index"""
        if index_type == "Flat":
            return faiss.IndexFlatIP(self.dimension) if self.normalize else faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = max(4096, self.dimension * 4)  # Rule of thumb
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, self.metric)
            if not index.is_trained:
                # Generate random vectors for training
                train_size = max(256 * nlist, 4096)
                train_vectors = np.random.normal(size=(train_size, self.dimension)).astype(np.float32)
                if self.normalize:
                    faiss.normalize_L2(train_vectors)
                index.train(train_vectors)
            return index
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector if using cosine similarity"""
        if self.normalize:
            faiss.normalize_L2(vector)
        return vector
    
    def _load_data(self):
        """Load index and metadata from disk"""
        if os.path.exists(self._metadata_file) and os.path.exists(self._index_file):
            # Load metadata
            with open(self._metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._metadata = {int(k): v for k, v in data['metadata'].items()}
                self._id_to_index = {k: int(v) for k, v in data['id_to_index'].items()}
                self._next_index = data['next_index']
            
            # Load index
            self.index = faiss.read_index(self._index_file)
    
    def _save_data(self):
        """Save index and metadata to disk"""
        # Create directory if needed
        os.makedirs(os.path.dirname(self._metadata_file), exist_ok=True)
        
        # Save metadata
        with open(self._metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self._metadata,
                'id_to_index': self._id_to_index,
                'next_index': self._next_index
            }, f, indent=2)
        
        # Save index
        faiss.write_index(self.index, self._index_file)
    
    async def query(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query vectors by similarity"""
        # Generate query embedding
        query_embedding = np.array(self.embeddings.embed_query(query))
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        query_embedding = self._normalize_vector(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for not enough results
                result = {
                    **self._metadata[idx],
                    'score': float(dist)
                }
                results.append(result)
        
        return results
    
    async def upsert(
        self,
        data: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Insert or update vectors"""
        if not data:
            return []
        
        # Process in batches
        all_ids = []
        batch_size = self.batch_size
        items = list(data.items())
        
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_ids = [item[0] for item in batch_items]
            batch_texts = [item[1]['content'] for item in batch_items]
            
            # Generate embeddings
            embeddings = np.array(self.embeddings.embed_documents(batch_texts))
            embeddings = embeddings.astype(np.float32)
            embeddings = self._normalize_vector(embeddings)
            
            # Add to index
            for j, (id_, embedding) in enumerate(zip(batch_ids, embeddings)):
                if id_ in self._id_to_index:
                    # Update existing vector
                    idx = self._id_to_index[id_]
                    self.index.remove_ids(np.array([idx]))
                else:
                    # Add new vector
                    idx = self._next_index
                    self._next_index += 1
                    self._id_to_index[id_] = idx
                
                self.index.add(embedding.reshape(1, -1))
                
                # Store metadata
                self._metadata[idx] = {
                    'id': id_,
                    **{k: v for k, v in data[id_].items() if k in self.meta_fields}
                }
                
                all_ids.append(id_)
        
        # Save changes
        self._save_data()
        
        return all_ids
    
    async def delete(self, ids: List[str]):
        """Delete vectors by ID"""
        indices = []
        for id_ in ids:
            if id_ in self._id_to_index:
                idx = self._id_to_index[id_]
                indices.append(idx)
                del self._id_to_index[id_]
                del self._metadata[idx]
        
        if indices:
            self.index.remove_ids(np.array(indices))
            self._save_data()
    
    async def clear(self):
        """Clear all data"""
        self.index = self._create_index("Flat")
        self._metadata.clear()
        self._id_to_index.clear()
        self._next_index = 0
        self._save_data()

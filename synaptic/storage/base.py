"""Base storage classes for SynapticRAG"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
from langchain_openai import OpenAIEmbeddings

@dataclass
class StorageConfig:
    """Base configuration for storage"""
    namespace: str
    working_dir: str
    cache_dir: Optional[str] = None

@dataclass
class BaseStorage:
    """Base class for all storage types"""
    config: StorageConfig
    
    async def index_done_callback(self):
        """Called when indexing is complete"""
        pass
    
    async def query_done_callback(self):
        """Called when querying is complete"""
        pass

@dataclass
class BaseVectorStorage(BaseStorage):
    """Base class for vector storage"""
    embeddings: OpenAIEmbeddings
    meta_fields: Set[str] = None
    
    def __post_init__(self):
        self.meta_fields = self.meta_fields or set()
    
    async def query(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query vectors by similarity"""
        raise NotImplementedError
    
    async def upsert(
        self,
        data: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Insert or update vectors"""
        raise NotImplementedError

@dataclass
class BaseKVStorage(BaseStorage):
    """Base class for key-value storage"""
    
    async def all_keys(self) -> List[str]:
        """Get all keys"""
        raise NotImplementedError
    
    async def get_by_id(
        self,
        id: str
    ) -> Optional[Dict[str, Any]]:
        """Get value by key"""
        raise NotImplementedError
    
    async def get_by_ids(
        self,
        ids: List[str],
        fields: Optional[Set[str]] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """Get multiple values by keys"""
        raise NotImplementedError
    
    async def filter_keys(
        self,
        keys: List[str]
    ) -> Set[str]:
        """Filter out existing keys"""
        raise NotImplementedError
    
    async def upsert(
        self,
        data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Insert or update key-value pairs"""
        raise NotImplementedError
    
    async def drop(self):
        """Drop all data"""
        raise NotImplementedError

@dataclass
class BaseGraphStorage(BaseStorage):
    """Base class for graph storage"""
    
    async def has_node(
        self,
        node_id: str
    ) -> bool:
        """Check if node exists"""
        raise NotImplementedError
    
    async def has_edge(
        self,
        source_node_id: str,
        target_node_id: str
    ) -> bool:
        """Check if edge exists"""
        raise NotImplementedError
    
    async def node_degree(
        self,
        node_id: str
    ) -> int:
        """Get node degree"""
        raise NotImplementedError
    
    async def edge_degree(
        self,
        src_id: str,
        tgt_id: str
    ) -> int:
        """Get edge degree"""
        raise NotImplementedError
    
    async def get_node(
        self,
        node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get node data"""
        raise NotImplementedError
    
    async def get_edge(
        self,
        source_node_id: str,
        target_node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get edge data"""
        raise NotImplementedError
    
    async def get_node_edges(
        self,
        source_node_id: str
    ) -> Optional[List[tuple[str, str]]]:
        """Get edges connected to node"""
        raise NotImplementedError
    
    async def upsert_node(
        self,
        node_id: str,
        node_data: Dict[str, Any]
    ):
        """Insert or update node"""
        raise NotImplementedError
    
    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: Dict[str, Any]
    ):
        """Insert or update edge"""
        raise NotImplementedError
    
    async def delete_node(
        self,
        node_id: str
    ):
        """Delete node"""
        raise NotImplementedError
    
    async def embed_nodes(
        self,
        algorithm: str
    ) -> tuple[np.ndarray, List[str]]:
        """Generate node embeddings"""
        raise NotImplementedError("Node embedding is not implemented.")

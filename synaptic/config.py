"""Configuration for SynapticRAG"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Type
from .storage.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage

@dataclass
class MemoryConfig:
    """Configuration for memory components"""
    model_name_or_path: str = "claude-3-5-sonnet-20241022"
    max_clues: int = 3
    min_clue_score: float = 0.5
    temperature: float = 0.7
    max_tokens: int = 100
    consolidation_threshold: float = 0.9
    max_memory_tokens: int = 1024
    memory_weight: float = 0.4  # Weight for memory-based results
    clue_boost: float = 0.2  # Boost for memory-influenced clues
    min_memory_relevance: float = 0.3  # Minimum relevance for memory recall
    enable_cache: bool = True  # Enable memory caching
    cache_similarity_threshold: float = 0.95  # Threshold for memory cache hits

@dataclass
class GraphConfig:
    """Configuration for graph components"""
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    entity_extract_max_gleaning: int = 1
    entity_summary_max_tokens: int = 500
    gnn_hidden_channels: int = 256
    gnn_num_layers: int = 3
    dropout: float = 0.1
    entity_confidence_threshold: float = 0.3  # Lowered threshold for entity extraction
    relationship_confidence_threshold: float = 0.3  # Lowered threshold for relationship extraction
    max_entities_per_chunk: int = 10  # Maximum entities to extract per chunk
    max_relationships_per_entity: int = 5  # Maximum relationships per entity
    enable_node2vec: bool = True  # Enable node2vec embeddings
    node2vec_dimensions: int = 1536  # Dimensions for node2vec embeddings
    node2vec_walk_length: int = 40  # Walk length for node2vec
    node2vec_num_walks: int = 10  # Number of walks per node
    node2vec_window_size: int = 2  # Context window size
    node2vec_iterations: int = 3  # Number of epochs

@dataclass
class RetrieverConfig:
    """Configuration for retriever components"""
    model_name_or_path: str = "text-embedding-3-large"  # OpenAI embeddings model
    hits: int = 3
    similarity_threshold: float = 0.3  # Lowered threshold for better recall
    vector_weight: float = 0.3  # Weight for vector similarity
    graph_weight: float = 0.3  # Weight for graph-based results
    hybrid_threshold: float = 0.3  # Lowered threshold for combined score
    rerank_top_k: int = 5  # Number of results to rerank
    max_context_length: int = 2000  # Maximum context length for final response
    enable_reranking: bool = True  # Enable result reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Model for reranking
    chunk_overlap: float = 0.1  # Chunk overlap ratio
    min_chunk_length: int = 100  # Minimum chunk length in tokens

@dataclass
class StorageConfig:
    """Configuration for storage components"""
    kv_storage_cls: Type[BaseKVStorage]
    vector_storage_cls: Type[BaseVectorStorage]
    graph_storage_cls: Type[BaseGraphStorage]
    working_dir: str
    cache_dir: str
    kv_namespace: str = "kv_storage"  # Namespace for KV storage
    vector_namespace: str = "vector_storage"  # Namespace for vector storage
    graph_namespace: str = "graph_storage"  # Namespace for graph storage
    enable_compression: bool = True  # Enable storage compression
    compression_level: int = 3  # Compression level (1-9)
    max_cache_size: int = 1000  # Maximum number of items in cache
    cache_ttl: int = 3600  # Cache time-to-live in seconds

@dataclass
class SynapticConfig:
    """Main configuration class"""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    batch_size: int = 2
    max_retries: int = 3  # Maximum retries for failed operations
    validation_threshold: float = 0.8  # Threshold for validation checks
    log_level: str = "INFO"
    enable_llm_cache: bool = True
    addon_params: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration"""
        # Validate working directory
        if not self.storage.working_dir:
            raise ValueError("Working directory must be specified")
            
        # Validate batch size
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
            
        # Validate cache directory
        if not self.storage.cache_dir:
            raise ValueError("Cache directory must be specified")
            
        # Validate weights sum to 1.0
        total_weight = (
            self.memory.memory_weight + 
            self.retriever.vector_weight + 
            self.retriever.graph_weight
        )
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point variance
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
        # Validate thresholds
        if not 0 <= self.memory.min_memory_relevance <= 1:
            raise ValueError("Memory relevance threshold must be between 0 and 1")
        if not 0 <= self.graph.entity_confidence_threshold <= 1:
            raise ValueError("Entity confidence threshold must be between 0 and 1")
        if not 0 <= self.graph.relationship_confidence_threshold <= 1:
            raise ValueError("Relationship confidence threshold must be between 0 and 1")
        if not 0 <= self.retriever.hybrid_threshold <= 1:
            raise ValueError("Hybrid threshold must be between 0 and 1")
        if not 0 <= self.validation_threshold <= 1:
            raise ValueError("Validation threshold must be between 0 and 1")
        
        # Validate cache settings
        if self.storage.enable_compression:
            if not 1 <= self.storage.compression_level <= 9:
                raise ValueError("Compression level must be between 1 and 9")
        if self.storage.max_cache_size < 1:
            raise ValueError("Maximum cache size must be positive")
        if self.storage.cache_ttl < 0:
            raise ValueError("Cache TTL must be non-negative")
        
        # Validate node2vec parameters
        if self.graph.enable_node2vec:
            if self.graph.node2vec_dimensions < 1:
                raise ValueError("Node2vec dimensions must be positive")
            if self.graph.node2vec_walk_length < 1:
                raise ValueError("Node2vec walk length must be positive")
            if self.graph.node2vec_num_walks < 1:
                raise ValueError("Node2vec number of walks must be positive")
            if self.graph.node2vec_window_size < 1:
                raise ValueError("Node2vec window size must be positive")
            if self.graph.node2vec_iterations < 1:
                raise ValueError("Node2vec iterations must be positive")

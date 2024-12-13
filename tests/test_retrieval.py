import pytest
import torch
from torch_geometric.data import HeteroData
from synaptic.graph.retrieval import HybridRetriever, RetrievalResult
from synaptic.config import GraphConfig

@pytest.fixture
def retriever():
    return HybridRetriever(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",  # Smaller model for testing
        top_k=3,
        similarity_threshold=0.6
    )

@pytest.fixture
def sample_chunks():
    return {
        "chunk1": "Neural networks are a type of machine learning model inspired by biological neurons.",
        "chunk2": "Deep learning is a subset of machine learning using neural networks.",
        "chunk3": "PyTorch is a popular framework for implementing deep learning models."
    }

@pytest.fixture
def sample_graph():
    data = HeteroData()
    
    # Add node features
    data['CONCEPT'].x = torch.randn(3, 64)  # 3 concept nodes
    data['TOOL'].x = torch.randn(2, 64)     # 2 tool nodes
    
    # Add edge indices
    data['CONCEPT', 'RELATES_TO', 'CONCEPT'].edge_index = torch.tensor([[0, 1], [1, 2]])
    data['TOOL', 'USED_IN', 'CONCEPT'].edge_index = torch.tensor([[0, 1], [0, 2]])
    
    # Add node metadata
    data['CONCEPT'].metadata = [
        {"name": "Neural Networks", "description": "A type of machine learning model"},
        {"name": "Deep Learning", "description": "A subset of machine learning"},
        {"name": "Machine Learning", "description": "A field of AI"}
    ]
    data['TOOL'].metadata = [
        {"name": "PyTorch", "description": "A deep learning framework"},
        {"name": "TensorFlow", "description": "Another deep learning framework"}
    ]
    
    return data

@pytest.fixture
def sample_entity_embeddings():
    return {
        "Neural Networks": torch.randn(1, 768),
        "Deep Learning": torch.randn(1, 768),
        "Machine Learning": torch.randn(1, 768),
        "PyTorch": torch.randn(1, 768),
        "TensorFlow": torch.randn(1, 768)
    }

def test_retriever_initialization(retriever):
    assert retriever is not None
    assert retriever.model is not None
    assert retriever.tokenizer is not None
    assert retriever.device in ['cuda', 'cpu']
    assert retriever.top_k == 3
    assert retriever.similarity_threshold == 0.6
    assert isinstance(retriever.chunks, dict)
    assert isinstance(retriever.chunk_embeddings, dict)

def test_add_chunks(retriever, sample_chunks):
    # Add chunks
    retriever.add_chunks(sample_chunks)
    
    # Verify chunks were added
    assert len(retriever.chunks) == len(sample_chunks)
    assert len(retriever.chunk_embeddings) == len(sample_chunks)
    assert all(chunk_id in retriever.chunks for chunk_id in sample_chunks)
    assert all(chunk_id in retriever.chunk_embeddings for chunk_id in sample_chunks)
    
    # Verify embeddings shape
    for embedding in retriever.chunk_embeddings.values():
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dim() == 2  # [batch_size=1, embedding_dim]
        assert embedding.size(0) == 1

def test_retrieve_by_query(retriever, sample_chunks):
    # Add chunks
    retriever.add_chunks(sample_chunks)
    
    # Test query retrieval
    query = "What is deep learning?"
    results = retriever.retrieve_by_query(query)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
    assert all(r.source_id in sample_chunks for r in results)

def test_retrieve_by_query_with_clues(retriever, sample_chunks):
    # Add chunks
    retriever.add_chunks(sample_chunks)
    
    # Test query retrieval with clues
    query = "What is deep learning?"
    clues = ["neural networks", "machine learning"]
    results = retriever.retrieve_by_query(query, clues=clues)
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
    assert all(r.source_id in sample_chunks for r in results)

def test_retrieve_by_graph(retriever, sample_chunks, sample_graph, sample_entity_embeddings):
    # Add chunks
    retriever.add_chunks(sample_chunks)
    
    # Test graph-based retrieval
    query = "What is PyTorch?"
    results = retriever.retrieve_by_graph(query, sample_graph, sample_entity_embeddings)
    
    assert isinstance(results, list)
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
    assert all(r.source_id in sample_chunks for r in results)
    assert all(r.metadata is not None for r in results)

def test_hybrid_retrieve(retriever, sample_chunks, sample_graph, sample_entity_embeddings):
    # Add chunks
    retriever.add_chunks(sample_chunks)
    
    # Test hybrid retrieval
    query = "What is deep learning?"
    clues = ["neural networks", "machine learning"]
    results = retriever.hybrid_retrieve(
        query,
        clues=clues,
        graph=sample_graph,
        entity_embeddings=sample_entity_embeddings
    )
    
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
    assert all(r.source_id in sample_chunks for r in results)

def test_empty_retrieval(retriever):
    # Test retrieval with no chunks
    query = "test query"
    
    assert retriever.retrieve_by_query(query) == []
    assert retriever.retrieve_by_graph(query, HeteroData(), {}) == []
    assert retriever.hybrid_retrieve(query) == []

def test_similarity_threshold(retriever, sample_chunks):
    # Add chunks
    retriever.add_chunks(sample_chunks)
    
    # Test with high threshold
    retriever.similarity_threshold = 0.99
    results_high = retriever.retrieve_by_query("test query")
    
    # Test with low threshold
    retriever.similarity_threshold = 0.1
    results_low = retriever.retrieve_by_query("test query")
    
    assert len(results_high) <= len(results_low)

def test_top_k_limit(retriever, sample_chunks):
    # Add chunks
    retriever.add_chunks(sample_chunks)
    
    # Test with different top_k values
    query = "test query"
    
    results1 = retriever.retrieve_by_query(query, top_k=1)
    results2 = retriever.retrieve_by_query(query, top_k=2)
    
    assert len(results1) <= 1
    assert len(results2) <= 2

def test_duplicate_chunks(retriever):
    # Add same chunk multiple times with different IDs
    chunks = {
        "chunk1": "Test content",
        "chunk2": "Test content",  # Same content
        "chunk3": "Different content"
    }
    retriever.add_chunks(chunks)
    
    # Verify embeddings
    emb1 = retriever.chunk_embeddings["chunk1"]
    emb2 = retriever.chunk_embeddings["chunk2"]
    emb3 = retriever.chunk_embeddings["chunk3"]
    
    assert torch.allclose(emb1, emb2)  # Same content should have same embedding
    assert not torch.allclose(emb1, emb3)  # Different content should have different embedding

def test_connected_chunks(retriever, sample_graph):
    # Test internal method for getting connected chunks
    chunks = retriever._get_connected_chunks("Neural Networks", sample_graph)
    assert isinstance(chunks, set)

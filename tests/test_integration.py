import pytest
import torch
import os
import shutil
from synaptic.config import (
    SynapticConfig,
    MemoryConfig,
    GraphConfig,
    RetrieverConfig
)
from synaptic.pipeline import SynapticPipeline, ProcessingResult
from synaptic.adapter import GenerationResult

# Use GPT2 for testing LLM functionality
TEST_LLM = "gpt2"
# Use sentence transformer for embeddings
TEST_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"

@pytest.fixture
def config():
    return SynapticConfig(
        working_dir="./test_workspace",
        memory=MemoryConfig(
            model_name_or_path=TEST_LLM,  # Use GPT2 for LLM
            load_in_4bit=False,  # Disable 4-bit for testing
            enable_flash_attn=False  # Disable flash attention for testing
        ),
        graph=GraphConfig(
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            entity_extract_max_gleaning=1,
            entity_summary_max_tokens=500
        ),
        retriever=RetrieverConfig(
            model_name_or_path=TEST_EMBEDDING,  # Use sentence transformer for embeddings
            hits=3
        ),
        llm_model_name=TEST_LLM,  # Use GPT2 for LLM
        enable_4bit=False,  # Disable 4-bit for testing
        enable_flash_attn=False,  # Disable flash attention for testing
        batch_size=2
    )

@pytest.fixture
def pipeline(config):
    pipeline = SynapticPipeline(config)
    yield pipeline
    # Cleanup
    pipeline.clear()
    if os.path.exists(config.working_dir):
        shutil.rmtree(config.working_dir)

@pytest.fixture
def sample_documents():
    return [
        "Neural networks are a type of machine learning model inspired by biological neurons. They are fundamental to deep learning.",
        "PyTorch is a popular framework for implementing deep learning models. It provides tools for building neural networks.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    ]

def test_pipeline_initialization(pipeline):
    assert pipeline is not None
    assert pipeline.config is not None
    assert pipeline.rag is not None
    assert os.path.exists(pipeline.config.working_dir)
    
    # Verify RAG components
    assert pipeline.rag.memory_model is not None
    assert pipeline.rag.clue_generator is not None
    assert pipeline.rag.graph_builder is not None
    assert pipeline.rag.retriever is not None

def test_document_processing(pipeline, sample_documents):
    # Process documents
    result = pipeline.process_documents(sample_documents)
    
    assert isinstance(result, ProcessingResult)
    assert result.num_documents == len(sample_documents)
    assert result.num_entities > 0
    assert result.num_relationships > 0
    assert result.memory_size > 0
    assert result.graph_size["nodes"] > 0
    assert result.graph_size["edges"] > 0
    
    # Verify RAG state
    assert len(pipeline.rag.documents) == len(sample_documents)
    assert pipeline.rag.memory_model.memory_state is not None
    assert len(pipeline.rag.graph_builder.entities) > 0
    assert len(pipeline.rag.retriever.chunks) > 0

def test_incremental_processing(pipeline):
    # Process first document
    doc1 = "Neural networks are fundamental to deep learning."
    result1 = pipeline.process_documents(doc1)
    initial_entities = result1.num_entities
    initial_memory = result1.memory_size
    
    # Process second document
    doc2 = "Deep learning uses multiple layers of neural networks."
    result2 = pipeline.process_documents(doc2)
    
    # Verify incremental updates
    assert result2.num_entities > initial_entities
    assert result2.memory_size > initial_memory
    assert len(pipeline.rag.documents) == 2

def test_query_processing(pipeline, sample_documents):
    # First process documents
    pipeline.process_documents(sample_documents)
    
    # Test single query
    query = "What is deep learning?"
    result = pipeline.query(query, return_sources=True)
    
    assert isinstance(result, GenerationResult)
    assert result.response is not None
    assert len(result.response) > 0
    assert result.sources is not None
    assert len(result.sources) > 0
    assert result.metadata is not None
    assert "clues" in result.metadata
    assert "retrieval_scores" in result.metadata

def test_batch_query_processing(pipeline, sample_documents):
    # Process documents
    pipeline.process_documents(sample_documents)
    
    # Test batch queries
    queries = [
        "What are neural networks?",
        "How is deep learning used?",
        "What is machine learning?"
    ]
    
    results = pipeline.batch_query(
        queries,
        batch_size=2,
        return_sources=True
    )
    
    assert len(results) == len(queries)
    assert all(isinstance(r, GenerationResult) for r in results)
    assert all(r.response is not None for r in results)
    assert all(r.sources is not None for r in results)
    assert all(r.metadata is not None for r in results)

def test_state_persistence(pipeline, sample_documents, tmp_path):
    # Process documents
    pipeline.process_documents(sample_documents)
    
    # Save state
    save_path = tmp_path / "pipeline_state.pt"
    pipeline.save(str(save_path))
    
    # Create new pipeline and load state
    new_pipeline = SynapticPipeline(pipeline.config)
    new_pipeline.load(str(save_path))
    
    # Verify state
    assert len(new_pipeline.rag.documents) == len(pipeline.rag.documents)
    assert len(new_pipeline.rag.graph_builder.entities) == len(pipeline.rag.graph_builder.entities)
    assert len(new_pipeline.rag.retriever.chunks) == len(pipeline.rag.retriever.chunks)
    
    # Verify functionality after loading
    query = "What is deep learning?"
    result = new_pipeline.query(query)
    assert isinstance(result, GenerationResult)
    assert result.response is not None

def test_error_handling(pipeline):
    # Test empty document
    with pytest.raises(ValueError):
        pipeline.process_documents("")
    
    # Test empty query
    result = pipeline.query("")
    assert len(result.response) == 0
    assert len(result.sources) == 0
    
    # Test query before processing documents
    result = pipeline.query("test query")
    assert len(result.response) == 0
    assert len(result.sources) == 0
    
    # Test invalid batch size
    with pytest.raises(ValueError):
        pipeline.batch_query(["test"], batch_size=0)

def test_memory_graph_integration(pipeline, sample_documents):
    # Process documents
    pipeline.process_documents(sample_documents)
    
    # Verify memory and graph interaction through RAG
    query = "How are neural networks used in deep learning?"
    result = pipeline.query(query)
    
    # Check that both memory and graph were used
    assert result.metadata["clues"] is not None  # Memory clues
    assert len(result.sources) > 0  # Retrieved chunks
    
    # Verify graph state
    assert pipeline.rag.graph_builder.pyg_graph is not None
    assert pipeline.rag.gnn is not None

def test_component_interaction(pipeline, sample_documents):
    """Test component interaction following LightRAG/MemoRAG approach"""
    # Process documents
    pipeline.process_documents(sample_documents)
    
    # Get initial state
    initial_memory = pipeline.rag.memory_model.memory_state
    initial_entities = len(pipeline.rag.graph_builder.entities)
    
    # Add new related document
    new_doc = "Deep learning architectures include convolutional neural networks."
    pipeline.process_documents(new_doc)
    
    # Verify cross-component updates
    assert pipeline.rag.memory_model.memory_state != initial_memory
    assert len(pipeline.rag.graph_builder.entities) > initial_entities
    
    # Test retrieval with updated state
    query = "What types of neural networks are used in deep learning?"
    result = pipeline.query(query)
    
    # Verify response contains information from both old and new documents
    assert result.response is not None
    assert len(result.response) > 0
    assert len(result.sources) > 0
    # Verify sources include both old and new documents
    assert any(source_id.startswith("doc_") for source_id in result.sources)

def test_document_deduplication(pipeline):
    """Test document deduplication following LightRAG's approach"""
    # Add same document multiple times
    doc = "Neural networks are fundamental to deep learning."
    
    # First processing
    pipeline.process_documents(doc)
    initial_doc_count = len(pipeline.rag.documents)
    
    # Process same document again
    pipeline.process_documents(doc)
    
    # Verify document count hasn't changed (deduplication worked)
    assert len(pipeline.rag.documents) == initial_doc_count

def test_clear_state(pipeline, sample_documents):
    # Process documents
    pipeline.process_documents(sample_documents)
    
    # Clear state
    pipeline.clear()
    
    # Verify cleared state
    assert pipeline.rag.memory_model.memory_state is None
    assert len(pipeline.rag.graph_builder.entities) == 0
    assert len(pipeline.rag.documents) == 0
    assert len(pipeline.rag.retriever.chunks) == 0
    assert pipeline.rag.gnn is None

def test_empty_query_handling(pipeline, sample_documents):
    # Process documents
    pipeline.process_documents(sample_documents)
    
    # Test empty query
    result = pipeline.query("")
    assert len(result.response) == 0
    assert len(result.sources) == 0
    
    # Test whitespace query
    result = pipeline.query("   ")
    assert len(result.response) == 0
    assert len(result.sources) == 0

import pytest
import torch
import os
from synaptic.memory.memorizer import Memorizer, MemoryBlock, RecallResult
from synaptic.memory.memory_model import MemoryModel, MemoryState

@pytest.fixture
def memorizer():
    return Memorizer()

def test_memory_initialization(memorizer):
    assert memorizer.llm is not None
    assert memorizer.max_memory_tokens > 0
    assert memorizer.consolidation_threshold > 0
    assert len(memorizer.memory_blocks) == 0
    assert len(memorizer.consolidated_memories) == 0

def test_memory_block_creation(memorizer):
    text = "Neural networks are fundamental to deep learning."
    block = memorizer.memorize(text)
    
    assert isinstance(block, MemoryBlock)
    assert block.text == text
    assert block.state is not None
    assert block.embedding is not None
    assert block.metadata is not None
    assert "timestamp" in block.metadata
    assert "token_count" in block.metadata

def test_memory_recall(memorizer):
    texts = [
        "Neural networks are fundamental to deep learning.",
        "Machine learning enables systems to learn from data.",
        "Deep learning uses multiple neural network layers."
    ]
    
    # Add memories
    for text in texts:
        memorizer.memorize(text)
    
    # Test recall
    query = "Tell me about neural networks"
    results = memorizer.recall(query)
    
    assert len(results) > 0
    assert all(isinstance(r, RecallResult) for r in results)
    assert all(r.score > 0 for r in results)
    assert all(r.text in texts for r in results)

def test_memory_recall_ordering(memorizer):
    texts = [
        "Neural networks are fundamental to deep learning.",
        "The weather is nice today.",
        "Deep learning uses multiple neural network layers."
    ]
    
    # Add memories
    for text in texts:
        memorizer.memorize(text)
    
    # Test recall ordering
    query = "Tell me about neural networks"
    results = memorizer.recall(query)
    
    # Verify relevant texts are ranked higher
    relevant_scores = []
    irrelevant_scores = []
    for result in results:
        if "neural" in result.text.lower():
            relevant_scores.append(result.score)
        else:
            irrelevant_scores.append(result.score)
    
    # Check that relevant texts have higher average score
    if relevant_scores and irrelevant_scores:
        avg_relevant = sum(relevant_scores) / len(relevant_scores)
        avg_irrelevant = sum(irrelevant_scores) / len(irrelevant_scores)
        assert avg_relevant > avg_irrelevant

def test_memory_consolidation(memorizer):
    # Add similar memories
    texts = [
        "Neural networks process data.",
        "Neural networks analyze information.",
        "Neural networks handle computations."
    ]
    
    for text in texts:
        memorizer.memorize(text)
    
    # Force consolidation
    memorizer.consolidate_memories()
    
    # Verify memories were consolidated
    assert len(memorizer.consolidated_memories) > 0
    
    # Verify consolidated memory contains original texts
    consolidated = memorizer.consolidated_memories[0]
    assert all(text in consolidated.texts for text in texts)

def test_memory_state_persistence(memorizer, tmp_path):
    # Add some memories
    texts = [
        "Neural networks are fundamental to deep learning.",
        "Machine learning enables systems to learn from data."
    ]
    
    for text in texts:
        memorizer.memorize(text)
    
    # Save state
    save_path = tmp_path / "memory_state.pt"
    memorizer.save(str(save_path))
    
    # Create new memorizer and load state
    new_memorizer = Memorizer()
    new_memorizer.load(str(save_path))
    
    # Verify state
    assert len(new_memorizer.memory_blocks) == len(memorizer.memory_blocks)
    assert len(new_memorizer.consolidated_memories) == len(memorizer.consolidated_memories)
    
    # Verify functionality after loading
    query = "Tell me about neural networks"
    results = new_memorizer.recall(query)
    assert len(results) > 0

def test_memory_clear(memorizer):
    # Add some memories
    texts = [
        "Neural networks are fundamental to deep learning.",
        "Machine learning enables systems to learn from data."
    ]
    
    for text in texts:
        memorizer.memorize(text)
    
    # Clear memory
    memorizer.clear()
    
    # Verify cleared state
    assert len(memorizer.memory_blocks) == 0
    assert len(memorizer.consolidated_memories) == 0

def test_empty_input_handling(memorizer):
    # Test empty text
    with pytest.raises(ValueError):
        memorizer.memorize("")
    
    # Test None text
    with pytest.raises(ValueError):
        memorizer.memorize(None)
    
    # Test whitespace text
    with pytest.raises(ValueError):
        memorizer.memorize("   ")

def test_memory_metadata(memorizer):
    text = "Neural networks are fundamental to deep learning."
    metadata = {"source": "textbook", "chapter": 1}
    
    block = memorizer.memorize(text, metadata)
    
    assert block.metadata["source"] == "textbook"
    assert block.metadata["chapter"] == 1
    assert "timestamp" in block.metadata
    assert "token_count" in block.metadata

def test_memory_token_limit(memorizer):
    # Set small token limit for testing
    memorizer.max_memory_tokens = 10
    
    # Add memories until limit is reached
    texts = [
        "Neural networks process data.",
        "Neural networks analyze information.",
        "Neural networks handle computations."
    ]
    
    for text in texts:
        memorizer.memorize(text)
    
    # Verify consolidation happened
    assert len(memorizer.consolidated_memories) > 0
    assert len(memorizer.memory_blocks) < len(texts)

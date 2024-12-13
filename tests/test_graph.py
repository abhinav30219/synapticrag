import pytest
import torch
from torch_geometric.data import HeteroData
from synaptic.config import GraphConfig
from synaptic.graph.build_graph import GraphBuilder, Entity, Relationship
from synaptic.graph.pyG_utils import HeteroGNN, create_pyg_graph, get_subgraph
from synaptic.graph.embeddings import GraphEmbedder, EmbeddingOutput

@pytest.fixture
def graph_config():
    return GraphConfig(
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=1,
        entity_summary_max_tokens=500,
        gnn_hidden_channels=256,
        gnn_num_layers=3
    )

@pytest.fixture
def embedder():
    return GraphEmbedder(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Smaller model for testing
        pooling_strategy="mean",
        normalize=True,
        cache_size=100
    )

@pytest.fixture
def graph_builder(graph_config, embedder):
    return GraphBuilder(
        config=graph_config,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"  # Use same model as embedder
    )

@pytest.fixture
def sample_entities():
    return [
        Entity(
            name="Neural Network",
            type="CONCEPT",
            description="A computational model inspired by biological neural networks",
            source_id="doc1",
            embedding=None
        ),
        Entity(
            name="Deep Learning",
            type="CONCEPT",
            description="A subset of machine learning using neural networks",
            source_id="doc1",
            embedding=None
        ),
        Entity(
            name="PyTorch",
            type="TOOL",
            description="A machine learning framework",
            source_id="doc2",
            embedding=None
        )
    ]

@pytest.fixture
def sample_relationships():
    return [
        Relationship(
            source="Neural Network",
            target="Deep Learning",
            type="IS_USED_IN",
            description="Neural networks are fundamental to deep learning",
            keywords=["architecture", "learning"],
            weight=1.0,
            source_id="doc1",
            embedding=None
        ),
        Relationship(
            source="PyTorch",
            target="Deep Learning",
            type="ENABLES",
            description="PyTorch is used to implement deep learning models",
            keywords=["framework", "implementation"],
            weight=1.0,
            source_id="doc2",
            embedding=None
        )
    ]

def test_embedder_initialization(embedder):
    assert embedder is not None
    assert embedder.model is not None
    assert embedder.tokenizer is not None
    assert embedder.device in ['cuda', 'cpu']
    assert embedder.pooling_strategy == "mean"
    assert embedder.normalize is True
    assert isinstance(embedder.cache, dict)

def test_embedder_text_embedding(embedder):
    text = "This is a test text"
    output = embedder.embed_text(text)
    
    assert isinstance(output, EmbeddingOutput)
    assert output.embeddings.shape[0] == 1  # Batch size
    assert output.attention_mask is not None
    assert output.token_embeddings is not None

def test_embedder_batch_embedding(embedder):
    texts = ["First text", "Second text", "Third text"]
    outputs = embedder.embed_text(texts, batch_size=2)
    
    assert isinstance(outputs, list)
    assert len(outputs) == 3
    assert all(isinstance(o, EmbeddingOutput) for o in outputs)
    assert all(o.embeddings.shape[0] == 1 for o in outputs)

def test_embedder_entity_embedding(embedder):
    output = embedder.embed_entity(
        name="Test Entity",
        entity_type="TEST",
        description="This is a test entity"
    )
    
    assert isinstance(output, EmbeddingOutput)
    assert output.embeddings.shape[0] == 1
    assert "entity_Test Entity_TEST" in embedder.cache

def test_embedder_relationship_embedding(embedder):
    output = embedder.embed_relationship(
        source="Entity A",
        target="Entity B",
        rel_type="TEST_REL",
        description="Test relationship",
        keywords=["test", "relationship"]
    )
    
    assert isinstance(output, EmbeddingOutput)
    assert output.embeddings.shape[0] == 1
    assert "rel_Entity A_TEST_REL_Entity B" in embedder.cache

def test_graph_builder_initialization(graph_builder):
    assert graph_builder is not None
    assert graph_builder.config is not None
    assert graph_builder.embedding_model is not None
    assert graph_builder.embedding_tokenizer is not None
    assert isinstance(graph_builder.entities, dict)
    assert isinstance(graph_builder.relationships, list)
    assert isinstance(graph_builder.node_to_idx, dict)

def test_add_entity(graph_builder):
    # Test adding valid entity
    graph_builder.add_entity(
        name="Test Entity",
        entity_type="TEST",
        description="Test description",
        source_id="test_doc"
    )
    
    assert "Test Entity" in graph_builder.entities
    assert graph_builder.entities["Test Entity"].embedding is not None
    assert "Test Entity" in graph_builder.nx_graph.nodes
    
    # Test adding empty entity
    with pytest.raises(ValueError, match="Entity name and type cannot be empty"):
        graph_builder.add_entity("", "", "desc", "test")

def test_add_relationship(graph_builder, sample_entities):
    # Add required entities first
    for entity in sample_entities[:2]:  # Add Neural Network and Deep Learning
        graph_builder.add_entity(
            name=entity.name,
            entity_type=entity.type,
            description=entity.description,
            source_id=entity.source_id
        )
    
    # Test adding valid relationship
    graph_builder.add_relationship(
        source="Neural Network",
        target="Deep Learning",
        rel_type="IS_USED_IN",
        description="Test relationship",
        keywords=["test"],
        weight=1.0,
        source_id="test_doc"
    )
    
    assert len(graph_builder.relationships) == 1
    assert graph_builder.nx_graph.has_edge("Neural Network", "Deep Learning")
    
    # Test adding relationship with missing source
    with pytest.raises(ValueError, match="Source entity .* not found"):
        graph_builder.add_relationship(
            source="Missing Entity",
            target="Deep Learning",
            rel_type="TEST",
            description="Test",
            keywords=[],
            weight=1.0,
            source_id="test"
        )

def test_build_pyg_graph(graph_builder, sample_entities, sample_relationships):
    # Add entities and relationships
    for entity in sample_entities:
        graph_builder.add_entity(
            name=entity.name,
            entity_type=entity.type,
            description=entity.description,
            source_id=entity.source_id
        )
    
    for rel in sample_relationships:
        graph_builder.add_relationship(
            source=rel.source,
            target=rel.target,
            rel_type=rel.type,
            description=rel.description,
            keywords=rel.keywords,
            weight=rel.weight,
            source_id=rel.source_id
        )
    
    # Build PyG graph
    pyg_graph = graph_builder.build_pyg_graph()
    
    # Verify graph structure
    assert isinstance(pyg_graph, HeteroData)
    assert set(pyg_graph.node_types) == {"CONCEPT", "TOOL"}
    assert ("CONCEPT", "IS_USED_IN", "CONCEPT") in pyg_graph.edge_types
    assert ("TOOL", "ENABLES", "CONCEPT") in pyg_graph.edge_types
    
    # Verify node features
    assert pyg_graph["CONCEPT"].x.size(0) == 2  # Neural Network and Deep Learning
    assert pyg_graph["TOOL"].x.size(0) == 1     # PyTorch
    
    # Verify edge indices
    edge_index = pyg_graph["CONCEPT", "IS_USED_IN", "CONCEPT"].edge_index
    assert edge_index.size(1) == 1  # One edge
    assert edge_index.size(0) == 2  # Source and target indices
    
    edge_index = pyg_graph["TOOL", "ENABLES", "CONCEPT"].edge_index
    assert edge_index.size(1) == 1  # One edge
    assert edge_index.size(0) == 2  # Source and target indices

def test_node2vec_embeddings(graph_builder, sample_entities, sample_relationships):
    # Add entities and relationships
    for entity in sample_entities:
        graph_builder.add_entity(
            name=entity.name,
            entity_type=entity.type,
            description=entity.description,
            source_id=entity.source_id
        )
    
    for rel in sample_relationships:
        graph_builder.add_relationship(
            source=rel.source,
            target=rel.target,
            rel_type=rel.type,
            description=rel.description,
            keywords=rel.keywords,
            weight=rel.weight,
            source_id=rel.source_id
        )
    
    # Build graph and compute Node2Vec embeddings
    graph_builder.build_pyg_graph()
    embeddings = graph_builder.compute_node2vec_embeddings(
        embedding_dim=64,
        walk_length=10,
        context_size=5,
        walks_per_node=5,
        epochs=5
    )
    
    # Verify embeddings
    assert isinstance(embeddings, dict)
    assert "CONCEPT" in embeddings
    assert "TOOL" in embeddings
    assert embeddings["CONCEPT"].size(0) == 2  # Two concept nodes
    assert embeddings["TOOL"].size(0) == 1     # One tool node
    assert embeddings["CONCEPT"].size(1) == 64  # Embedding dimension
    assert embeddings["TOOL"].size(1) == 64     # Embedding dimension

def test_node2vec_persistence(graph_builder, sample_entities, sample_relationships, tmp_path):
    # Add entities and relationships
    for entity in sample_entities:
        graph_builder.add_entity(
            name=entity.name,
            entity_type=entity.type,
            description=entity.description,
            source_id=entity.source_id
        )
    
    for rel in sample_relationships:
        graph_builder.add_relationship(
            source=rel.source,
            target=rel.target,
            rel_type=rel.type,
            description=rel.description,
            keywords=rel.keywords,
            weight=rel.weight,
            source_id=rel.source_id
        )
    
    # Build graph and compute Node2Vec embeddings
    graph_builder.build_pyg_graph()
    original_embeddings = graph_builder.compute_node2vec_embeddings()
    
    # Save and load graph
    save_path = tmp_path / "test_graph.pt"
    graph_builder.save(str(save_path))
    
    new_builder = GraphBuilder(
        config=graph_builder.config,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
    new_builder.load(str(save_path))
    
    # Verify Node2Vec embeddings were preserved
    assert new_builder.node2vec_embeddings is not None
    for node_type in original_embeddings:
        assert torch.equal(
            original_embeddings[node_type],
            new_builder.node2vec_embeddings[node_type]
        )

def test_empty_graph(graph_builder):
    with pytest.raises(ValueError, match="No entities in graph"):
        graph_builder.build_pyg_graph()

def test_node_mapping(graph_builder, sample_entities):
    # Add entities
    for entity in sample_entities:
        graph_builder.add_entity(
            name=entity.name,
            entity_type=entity.type,
            description=entity.description,
            source_id=entity.source_id
        )
    
    # Build graph
    graph_builder.build_pyg_graph()
    
    # Verify node mappings
    assert "CONCEPT" in graph_builder.node_to_idx
    assert "TOOL" in graph_builder.node_to_idx
    assert len(graph_builder.node_to_idx["CONCEPT"]) == 2
    assert len(graph_builder.node_to_idx["TOOL"]) == 1
    
    # Verify mapping correctness
    concept_mapping = graph_builder.node_to_idx["CONCEPT"]
    assert "Neural Network" in concept_mapping
    assert "Deep Learning" in concept_mapping
    assert concept_mapping["Neural Network"] != concept_mapping["Deep Learning"]

def test_graph_persistence(graph_builder, sample_entities, sample_relationships, tmp_path):
    # Add entities and relationships
    for entity in sample_entities:
        graph_builder.add_entity(
            name=entity.name,
            entity_type=entity.type,
            description=entity.description,
            source_id=entity.source_id
        )
    
    for rel in sample_relationships:
        graph_builder.add_relationship(
            source=rel.source,
            target=rel.target,
            rel_type=rel.type,
            description=rel.description,
            keywords=rel.keywords,
            weight=rel.weight,
            source_id=rel.source_id
        )
    
    # Build and save graph
    graph_builder.build_pyg_graph()
    save_path = tmp_path / "test_graph.pt"
    graph_builder.save(str(save_path))
    
    # Create new builder and load
    new_builder = GraphBuilder(
        config=graph_builder.config,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
    new_builder.load(str(save_path))
    
    # Verify loaded state
    assert len(new_builder.entities) == len(sample_entities)
    assert len(new_builder.relationships) == len(sample_relationships)
    assert isinstance(new_builder.pyg_graph, HeteroData)
    assert new_builder.node_to_idx == graph_builder.node_to_idx

def test_heterogeneous_graph(graph_builder):
    # Add entities of different types
    entities = [
        ("Data", "RESOURCE", "Input data"),
        ("Model", "ALGORITHM", "Processing unit"),
        ("Result", "OUTPUT", "Generated output")
    ]
    
    for name, type_, desc in entities:
        graph_builder.add_entity(name, type_, desc, "test")
    
    # Add relationships between different types
    relationships = [
        ("Data", "Model", "FEEDS_INTO"),
        ("Model", "Result", "PRODUCES")
    ]
    
    for src, dst, type_ in relationships:
        graph_builder.add_relationship(
            source=src,
            target=dst,
            rel_type=type_,
            description="Test relationship",
            keywords=[],
            weight=1.0,
            source_id="test"
        )
    
    # Build and verify graph
    pyg_graph = graph_builder.build_pyg_graph()
    
    # Verify node types
    assert set(pyg_graph.node_types) == {"RESOURCE", "ALGORITHM", "OUTPUT"}
    
    # Verify edge types
    assert ("RESOURCE", "FEEDS_INTO", "ALGORITHM") in pyg_graph.edge_types
    assert ("ALGORITHM", "PRODUCES", "OUTPUT") in pyg_graph.edge_types
    
    # Verify node features
    assert pyg_graph["RESOURCE"].x.size(0) == 1
    assert pyg_graph["ALGORITHM"].x.size(0) == 1
    assert pyg_graph["OUTPUT"].x.size(0) == 1
    
    # Verify edge indices
    for edge_type in pyg_graph.edge_types:
        edge_index = pyg_graph[edge_type].edge_index
        assert edge_index.size(0) == 2  # Source and target indices
        assert edge_index.size(1) == 1  # One edge

def test_embedder_cache(embedder):
    # Test cache functionality
    text = "Test text for caching"
    
    # First embedding should add to cache
    output1 = embedder.embed_text(text)
    cache_size1 = len(embedder.cache)
    
    # Second embedding should use cache
    output2 = embedder.embed_text(text)
    cache_size2 = len(embedder.cache)
    
    assert cache_size1 == cache_size2
    assert torch.equal(output1.embeddings, output2.embeddings)
    
    # Clear cache
    embedder.clear_cache()
    assert len(embedder.cache) == 0

def test_embedder_cache_limit(embedder):
    # Test cache size limit
    cache_size = embedder.cache_size
    
    # Add more items than cache size
    for i in range(cache_size + 10):
        embedder.embed_entity(
            name=f"Entity{i}",
            entity_type="TEST",
            description="Test entity"
        )
    
    assert len(embedder.cache) <= cache_size

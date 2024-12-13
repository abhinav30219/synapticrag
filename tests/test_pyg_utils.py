import pytest
import torch
from torch_geometric.data import HeteroData
from synaptic.graph.pyG_utils import (
    HeteroGNN,
    create_pyg_graph,
    aggregate_node_embeddings,
    get_subgraph,
    merge_graphs
)

@pytest.fixture
def sample_node_features():
    return {
        'CONCEPT': torch.randn(3, 64),  # 3 concept nodes with 64 features
        'TOOL': torch.randn(2, 64)      # 2 tool nodes with 64 features
    }

@pytest.fixture
def sample_edge_indices():
    return {
        ('CONCEPT', 'RELATES_TO', 'CONCEPT'): torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        ('TOOL', 'USED_IN', 'CONCEPT'): torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    }

@pytest.fixture
def sample_hetero_data(sample_node_features, sample_edge_indices):
    data = HeteroData()
    
    # Add node features
    for node_type, features in sample_node_features.items():
        data[node_type].x = features
    
    # Add edge indices
    for edge_type, edge_index in sample_edge_indices.items():
        src_type, edge_name, dst_type = edge_type
        data[src_type, edge_name, dst_type].edge_index = edge_index
    
    return data

@pytest.fixture
def hetero_gnn():
    node_types = ['CONCEPT', 'TOOL']
    edge_types = [
        ('CONCEPT', 'RELATES_TO', 'CONCEPT'),
        ('TOOL', 'USED_IN', 'CONCEPT')
    ]
    metadata = (node_types, edge_types)
    
    return HeteroGNN(
        hidden_channels=32,
        num_layers=2,
        metadata=metadata,
        dropout=0.1
    )

def test_hetero_gnn_initialization(hetero_gnn):
    assert hetero_gnn is not None
    assert hetero_gnn.hidden_channels == 32
    assert hetero_gnn.num_layers == 2
    assert hetero_gnn.dropout == 0.1
    assert len(hetero_gnn.convs) == 2
    assert isinstance(hetero_gnn.linear_dict, torch.nn.ModuleDict)

def test_hetero_gnn_forward(hetero_gnn, sample_hetero_data):
    # Prepare input dictionaries
    x_dict = {
        node_type: data.x
        for node_type, data in sample_hetero_data.node_items()
    }
    
    edge_index_dict = {
        edge_type: data.edge_index
        for edge_type, data in sample_hetero_data.edge_items()
    }
    
    # Forward pass
    out_dict = hetero_gnn(x_dict, edge_index_dict)
    
    # Verify output
    assert isinstance(out_dict, dict)
    assert set(out_dict.keys()) == {'CONCEPT', 'TOOL'}
    assert all(isinstance(v, torch.Tensor) for v in out_dict.values())
    assert all(v.size(-1) == hetero_gnn.hidden_channels for v in out_dict.values())

def test_create_pyg_graph(sample_node_features, sample_edge_indices):
    node_types = ['CONCEPT', 'TOOL']
    edge_types = [
        ('CONCEPT', 'RELATES_TO', 'CONCEPT'),
        ('TOOL', 'USED_IN', 'CONCEPT')
    ]
    
    graph = create_pyg_graph(
        nodes=sample_node_features,
        edges=sample_edge_indices,
        node_types=node_types,
        edge_types=edge_types
    )
    
    assert isinstance(graph, HeteroData)
    assert set(graph.node_types) == set(node_types)
    assert all(graph[node_type].x.shape == features.shape 
              for node_type, features in sample_node_features.items())
    assert all(torch.equal(graph[edge_type].edge_index, edge_index)
              for edge_type, edge_index in sample_edge_indices.items())

def test_aggregate_node_embeddings():
    node_embeddings = {
        'CONCEPT': torch.randn(3, 64),
        'TOOL': torch.randn(2, 64)
    }
    node_types = ['CONCEPT', 'TOOL']
    
    # Test mean aggregation
    mean_emb = aggregate_node_embeddings(node_embeddings, node_types, "mean")
    assert mean_emb.shape == (64,)
    
    # Test max aggregation
    max_emb = aggregate_node_embeddings(node_embeddings, node_types, "max")
    assert max_emb.shape == (64,)
    
    # Test concat aggregation
    concat_emb = aggregate_node_embeddings(node_embeddings, node_types, "concat")
    assert concat_emb.shape == (64 * len(node_types),)
    
    # Test invalid aggregation
    with pytest.raises(ValueError):
        aggregate_node_embeddings(node_embeddings, node_types, "invalid")

def test_get_subgraph(sample_hetero_data):
    # Create node masks
    node_idx = {
        'CONCEPT': torch.tensor([True, True, False]),  # Select first two concept nodes
        'TOOL': torch.tensor([True, False])            # Select first tool node
    }
    
    # Extract subgraph
    subgraph = get_subgraph(sample_hetero_data, node_idx, num_hops=1)
    
    assert isinstance(subgraph, HeteroData)
    assert set(subgraph.node_types) == {'CONCEPT', 'TOOL'}
    assert subgraph['CONCEPT'].x.size(0) == 2
    assert subgraph['TOOL'].x.size(0) == 1

def test_merge_graphs():
    # Create two sample graphs
    graphs = []
    for i in range(2):
        data = HeteroData()
        data['CONCEPT'].x = torch.randn(2, 64)
        data['TOOL'].x = torch.randn(1, 64)
        data['CONCEPT', 'RELATES_TO', 'CONCEPT'].edge_index = torch.tensor([[0], [1]])
        graphs.append(data)
    
    # Merge graphs
    merged = merge_graphs(graphs)
    
    assert isinstance(merged, HeteroData)
    assert merged['CONCEPT'].x.size(0) == 4  # 2 nodes from each graph
    assert merged['TOOL'].x.size(0) == 2     # 1 node from each graph
    assert merged['CONCEPT', 'RELATES_TO', 'CONCEPT'].edge_index.size(1) == 2

def test_merge_graphs_empty():
    assert merge_graphs([]) is None

def test_hetero_gnn_different_conv_types(sample_hetero_data):
    # Test GCN
    gnn_gcn = HeteroGNN(
        hidden_channels=32,
        num_layers=2,
        metadata=sample_hetero_data.metadata(),
        conv_type="gcn"
    )
    
    # Test GAT
    gnn_gat = HeteroGNN(
        hidden_channels=32,
        num_layers=2,
        metadata=sample_hetero_data.metadata(),
        conv_type="gat"
    )
    
    x_dict = {
        node_type: data.x
        for node_type, data in sample_hetero_data.node_items()
    }
    
    edge_index_dict = {
        edge_type: data.edge_index
        for edge_type, data in sample_hetero_data.edge_items()
    }
    
    # Verify both models can process the data
    out_gcn = gnn_gcn(x_dict, edge_index_dict)
    out_gat = gnn_gat(x_dict, edge_index_dict)
    
    assert all(v.size(-1) == 32 for v in out_gcn.values())
    assert all(v.size(-1) == 32 for v in out_gat.values())

def test_subgraph_multi_hop(sample_hetero_data):
    # Test with different numbers of hops
    node_idx = {
        'CONCEPT': torch.tensor([True, False, False]),
        'TOOL': torch.tensor([True, False])
    }
    
    # One hop
    subgraph_1hop = get_subgraph(sample_hetero_data, node_idx, num_hops=1)
    
    # Two hops
    subgraph_2hop = get_subgraph(sample_hetero_data, node_idx, num_hops=2)
    
    # Two hop subgraph should include more nodes than one hop
    assert subgraph_2hop['CONCEPT'].x.size(0) >= subgraph_1hop['CONCEPT'].x.size(0)

def test_graph_merge_with_attributes():
    # Create graphs with node and edge attributes
    graphs = []
    for i in range(2):
        data = HeteroData()
        # Node features and attributes
        data['CONCEPT'].x = torch.randn(2, 64)
        data['CONCEPT'].attr = torch.randn(2, 32)
        data['TOOL'].x = torch.randn(1, 64)
        data['TOOL'].attr = torch.randn(1, 32)
        # Edge indices and attributes
        data['CONCEPT', 'RELATES_TO', 'CONCEPT'].edge_index = torch.tensor([[0], [1]])
        data['CONCEPT', 'RELATES_TO', 'CONCEPT'].attr = torch.randn(1, 16)
        graphs.append(data)
    
    # Merge graphs
    merged = merge_graphs(graphs)
    
    # Verify node attributes
    assert merged['CONCEPT'].x.size(0) == 4
    assert merged['CONCEPT'].attr.size(0) == 4
    assert merged['TOOL'].x.size(0) == 2
    assert merged['TOOL'].attr.size(0) == 2
    
    # Verify edge attributes
    assert merged['CONCEPT', 'RELATES_TO', 'CONCEPT'].edge_index.size(1) == 2
    assert merged['CONCEPT', 'RELATES_TO', 'CONCEPT'].attr.size(0) == 2

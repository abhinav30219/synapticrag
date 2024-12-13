import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv, GATConv
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

class HeteroGNN(nn.Module):
    """Heterogeneous Graph Neural Network for processing knowledge graphs"""
    
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        dropout: float = 0.1,
        conv_type: str = "sage"
    ):
        super().__init__()
        
        # Store metadata
        self.node_types, self.edge_types = metadata[0], metadata[1]
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create initial linear transformations for each node type
        self.linear_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.linear_dict[node_type] = nn.Linear(64, hidden_channels)
        
        # Create convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Create a convolution for each edge type
            conv_dict = {}
            for edge_type in self.edge_types:
                # Always use SAGEConv as it handles both homogeneous and heterogeneous edges
                conv_dict[edge_type] = SAGEConv(hidden_channels, hidden_channels, root_weight=True)
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
    
    def _get_max_nodes(self, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, int]:
        """Get the maximum node index for each node type from edge indices"""
        max_nodes = {}
        for (src_type, _, dst_type), edge_index in edge_index_dict.items():
            if src_type not in max_nodes:
                max_nodes[src_type] = 0
            if dst_type not in max_nodes:
                max_nodes[dst_type] = 0
            
            max_nodes[src_type] = max(max_nodes[src_type], edge_index[0].max().item() + 1)
            max_nodes[dst_type] = max(max_nodes[dst_type], edge_index[1].max().item() + 1)
        return max_nodes
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the GNN"""
        
        # Get maximum number of nodes for each type
        max_nodes = self._get_max_nodes(edge_index_dict)
        
        # Initial feature transformation
        hidden_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                x = x_dict[node_type]
                hidden_dict[node_type] = F.dropout(
                    F.relu(self.linear_dict[node_type](x)),
                    p=self.dropout,
                    training=self.training
                )
            else:
                # Initialize missing features with correct number of nodes
                num_nodes = max_nodes.get(node_type, 1)
                hidden_dict[node_type] = torch.zeros(num_nodes, self.hidden_channels)
        
        # Message passing layers
        for conv in self.convs:
            # Let HeteroConv handle the message passing
            out_dict = conv(hidden_dict, edge_index_dict)
            
            # Apply non-linearity and dropout, preserving all node types
            hidden_dict = {}
            for node_type in self.node_types:
                if node_type in out_dict:
                    h = out_dict[node_type]
                    hidden_dict[node_type] = F.dropout(
                        F.relu(h),
                        p=self.dropout,
                        training=self.training
                    )
                else:
                    # Initialize missing features with correct number of nodes
                    num_nodes = max_nodes.get(node_type, 1)
                    hidden_dict[node_type] = torch.zeros(num_nodes, self.hidden_channels)
        
        return hidden_dict

def create_pyg_graph(
    nodes: Dict[str, torch.Tensor],
    edges: Dict[Tuple[str, str, str], torch.Tensor],
    node_types: List[str],
    edge_types: List[Tuple[str, str, str]]
) -> HeteroData:
    """Create a PyG heterogeneous graph from nodes and edges"""
    
    data = HeteroData()
    
    # Add nodes
    for node_type, node_features in nodes.items():
        data[node_type].x = node_features
        
    # Add edges
    for edge_type, edge_index in edges.items():
        src_type, edge_name, dst_type = edge_type
        data[edge_type].edge_index = edge_index
        
    return data

def aggregate_node_embeddings(
    node_embeddings: Dict[str, torch.Tensor],
    node_types: List[str],
    aggregation: str = "mean"
) -> torch.Tensor:
    """Aggregate node embeddings across different types"""
    
    embeddings_list = []
    for node_type in node_types:
        if node_type in node_embeddings:
            # Ensure 2D tensor
            emb = node_embeddings[node_type]
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embeddings_list.append(emb)
    
    if not embeddings_list:
        return None
        
    # Concatenate along batch dimension
    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    if aggregation == "mean":
        return torch.mean(all_embeddings, dim=0)
    elif aggregation == "max":
        return torch.max(all_embeddings, dim=0)[0]
    elif aggregation == "concat":
        # Flatten and limit to 128 dimensions
        flat = all_embeddings.view(-1)
        return flat[:128] if flat.size(0) > 128 else flat
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

def get_subgraph(
    data: HeteroData,
    node_idx: Dict[str, torch.Tensor],
    num_hops: int = 2
) -> HeteroData:
    """Extract a subgraph around specified nodes"""
    
    # Create boolean masks for nodes
    node_masks = {}
    for node_type, mask in node_idx.items():
        if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
            node_masks[node_type] = mask.clone()
        else:
            # Convert indices to boolean mask
            full_mask = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
            full_mask[mask] = True
            node_masks[node_type] = full_mask.clone()
    
    # Create subgraph
    sub_data = HeteroData()
    
    # Copy selected node features and attributes
    for node_type in data.node_types:
        mask = node_masks[node_type]
        sub_data[node_type].num_nodes = int(mask.sum())
        for key, value in data[node_type].items():
            if isinstance(value, torch.Tensor) and value.size(0) == data[node_type].num_nodes:
                sub_data[node_type][key] = value[mask]
    
    # Create node index mapping
    node_idx_map = {}
    for node_type in data.node_types:
        mask = node_masks[node_type]
        idx_map = torch.zeros(data[node_type].num_nodes, dtype=torch.long)
        idx_map[mask] = torch.arange(mask.sum())
        node_idx_map[node_type] = idx_map
    
    # Copy edge indices and attributes
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        
        # Get edges where both nodes are in the subgraph
        src_mask = node_masks[src_type][edge_index[0]]
        dst_mask = node_masks[dst_type][edge_index[1]]
        edge_mask = src_mask & dst_mask
        
        if edge_mask.any():
            # Update edge indices
            new_edge_index = edge_index[:, edge_mask]
            new_edge_index[0] = node_idx_map[src_type][new_edge_index[0]]
            new_edge_index[1] = node_idx_map[dst_type][new_edge_index[1]]
            sub_data[edge_type].edge_index = new_edge_index
            
            # Copy edge attributes
            for key, value in data[edge_type].items():
                if key != 'edge_index' and isinstance(value, torch.Tensor):
                    if value.size(0) == edge_index.size(1):
                        sub_data[edge_type][key] = value[edge_mask]
    
    return sub_data

def merge_graphs(graphs: List[HeteroData]) -> HeteroData:
    """Merge multiple heterogeneous graphs"""
    
    if not graphs:
        return None
        
    merged = HeteroData()
    
    # Track number of nodes for each type
    num_nodes = {node_type: 0 for node_type in graphs[0].node_types}
    
    # Merge nodes and their attributes
    for node_type in graphs[0].node_types:
        # Get all attributes for this node type
        attr_keys = set()
        for graph in graphs:
            attr_keys.update(graph[node_type].keys())
        
        # Merge each attribute
        for key in attr_keys:
            tensors = []
            for graph in graphs:
                if key in graph[node_type]:
                    tensors.append(graph[node_type][key])
            
            if tensors:
                merged[node_type][key] = torch.cat(tensors, dim=0)
        
        # Update node count
        merged[node_type].num_nodes = sum(
            graph[node_type].num_nodes for graph in graphs
        )
        num_nodes[node_type] = merged[node_type].num_nodes
    
    # Merge edges and their attributes
    for edge_type in graphs[0].edge_types:
        src_type, _, dst_type = edge_type
        
        # Track edge attributes
        attr_keys = set()
        for graph in graphs:
            if edge_type in graph.edge_types:
                attr_keys.update(graph[edge_type].keys())
        
        # Initialize offset for node indices
        offset = {node_type: 0 for node_type in graphs[0].node_types}
        
        # Merge edge indices and attributes
        for key in attr_keys:
            tensors = []
            for graph in graphs:
                if edge_type in graph.edge_types and key in graph[edge_type]:
                    if key == 'edge_index':
                        # Adjust edge indices
                        edge_index = graph[edge_type].edge_index.clone()
                        edge_index[0] += offset[src_type]
                        edge_index[1] += offset[dst_type]
                        tensors.append(edge_index)
                    else:
                        tensors.append(graph[edge_type][key])
                
                # Update offsets
                if edge_type in graph.edge_types:
                    offset[src_type] += graph[src_type].num_nodes
                    offset[dst_type] += graph[dst_type].num_nodes
            
            if tensors:
                merged[edge_type][key] = torch.cat(tensors, dim=1 if key == 'edge_index' else 0)
    
    return merged

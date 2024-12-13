"""NetworkX-based graph storage implementation"""
import os
import json
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from node2vec import Node2Vec
from .base import BaseGraphStorage, StorageConfig

class NetworkXStorage(BaseGraphStorage):
    """NetworkX-based graph storage"""
    
    def __init__(
        self,
        config: StorageConfig,
        node2vec_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        
        # Initialize storage paths
        self._graph_file = os.path.join(
            config.working_dir,
            f"graph_store_{config.namespace}.graphml"
        )
        self._embeddings_file = os.path.join(
            config.working_dir,
            f"graph_store_{config.namespace}_embeddings.npz"
        )
        
        # Initialize graph
        self._graph = self._load_graph()
        
        # Node2Vec parameters
        self.node2vec_params = node2vec_params or {
            'dimensions': 64,
            'walk_length': 30,
            'num_walks': 200,
            'workers': 4,
            'window': 10,
            'min_count': 1,
            'batch_words': 4
        }
        
        # Node embeddings
        self._node_embeddings: Optional[Dict[str, np.ndarray]] = None
        if os.path.exists(self._embeddings_file):
            self._load_embeddings()
    
    def _load_graph(self) -> nx.Graph:
        """Load graph from file"""
        if os.path.exists(self._graph_file):
            return nx.read_graphml(self._graph_file)
        return nx.Graph()
    
    def _save_graph(self):
        """Save graph to file"""
        # Create directory if needed
        os.makedirs(os.path.dirname(self._graph_file), exist_ok=True)
        
        # Save graph
        nx.write_graphml(self._graph, self._graph_file)
    
    def _load_embeddings(self):
        """Load node embeddings"""
        data = np.load(self._embeddings_file)
        self._node_embeddings = {
            node: embedding for node, embedding in zip(data['nodes'], data['embeddings'])
        }
    
    def _save_embeddings(self):
        """Save node embeddings"""
        if self._node_embeddings:
            nodes = list(self._node_embeddings.keys())
            embeddings = np.stack(list(self._node_embeddings.values()))
            np.savez(
                self._embeddings_file,
                nodes=nodes,
                embeddings=embeddings
            )
    
    async def index_done_callback(self):
        """Save data after indexing"""
        self._save_graph()
        if self._node_embeddings:
            self._save_embeddings()
    
    async def has_node(self, node_id: str) -> bool:
        """Check if node exists"""
        return self._graph.has_node(node_id)
    
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if edge exists"""
        return self._graph.has_edge(source_node_id, target_node_id)
    
    async def node_degree(self, node_id: str) -> int:
        """Get node degree"""
        return self._graph.degree(node_id)
    
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get total degree of edge endpoints"""
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data"""
        return dict(self._graph.nodes[node_id]) if self._graph.has_node(node_id) else None
    
    async def get_edge(
        self,
        source_node_id: str,
        target_node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get edge data"""
        return (
            dict(self._graph.edges[source_node_id, target_node_id])
            if self._graph.has_edge(source_node_id, target_node_id)
            else None
        )
    
    async def get_node_edges(
        self,
        source_node_id: str
    ) -> Optional[List[Tuple[str, str]]]:
        """Get edges connected to node"""
        if not self._graph.has_node(source_node_id):
            return None
        return list(self._graph.edges(source_node_id))
    
    async def upsert_node(
        self,
        node_id: str,
        node_data: Dict[str, Any]
    ):
        """Insert or update node"""
        self._graph.add_node(node_id, **node_data)
        # Invalidate embeddings
        self._node_embeddings = None
    
    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: Dict[str, Any]
    ):
        """Insert or update edge"""
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)
        # Invalidate embeddings
        self._node_embeddings = None
    
    async def delete_node(self, node_id: str):
        """Delete node"""
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            # Invalidate embeddings
            self._node_embeddings = None
    
    async def embed_nodes(
        self,
        algorithm: str = "node2vec"
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate node embeddings"""
        if algorithm != "node2vec":
            raise ValueError(f"Unsupported embedding algorithm: {algorithm}")
        
        # Check if we have cached embeddings
        if self._node_embeddings is not None:
            nodes = list(self._node_embeddings.keys())
            embeddings = np.stack(list(self._node_embeddings.values()))
            return embeddings, nodes
        
        # Initialize Node2Vec model
        node2vec = Node2Vec(
            self._graph,
            dimensions=self.node2vec_params['dimensions'],
            walk_length=self.node2vec_params['walk_length'],
            num_walks=self.node2vec_params['num_walks'],
            workers=self.node2vec_params['workers']
        )
        
        # Train embeddings
        model = node2vec.fit(
            window=self.node2vec_params['window'],
            min_count=self.node2vec_params['min_count'],
            batch_words=self.node2vec_params['batch_words']
        )
        
        # Get embeddings for all nodes
        nodes = list(self._graph.nodes())
        embeddings = np.stack([model.wv[node] for node in nodes])
        
        # Cache embeddings
        self._node_embeddings = {
            node: embedding for node, embedding in zip(nodes, embeddings)
        }
        self._save_embeddings()
        
        return embeddings, nodes
    
    def get_largest_connected_component(self) -> nx.Graph:
        """Get largest connected component of graph"""
        if not self._graph.nodes():
            return self._graph
            
        # Find largest component
        largest_cc = max(nx.connected_components(self._graph), key=len)
        
        # Create subgraph
        return self._graph.subgraph(largest_cc).copy()
    
    def get_subgraph(
        self,
        nodes: List[str],
        n_hops: int = 1
    ) -> nx.Graph:
        """Get subgraph around specified nodes"""
        # Start with initial nodes
        subgraph_nodes = set(nodes)
        
        # Add n-hop neighbors
        current_nodes = set(nodes)
        for _ in range(n_hops):
            next_nodes = set()
            for node in current_nodes:
                if self._graph.has_node(node):
                    next_nodes.update(self._graph.neighbors(node))
            current_nodes = next_nodes - subgraph_nodes
            subgraph_nodes.update(current_nodes)
        
        # Create subgraph
        return self._graph.subgraph(subgraph_nodes).copy()

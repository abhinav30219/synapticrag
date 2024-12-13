"""Build knowledge graph from documents"""
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import spacy
from langchain_openai import OpenAIEmbeddings
from ..config import GraphConfig

@dataclass
class Entity:
    """Container for entities"""
    name: str
    type: str
    description: str
    source_id: str
    embedding: Optional[torch.Tensor] = None

@dataclass
class Relationship:
    """Container for relationships"""
    source: str
    target: str
    type: str
    description: str
    keywords: List[str]
    weight: float
    source_id: str
    embedding: Optional[torch.Tensor] = None

class GraphBuilder:
    """Builds knowledge graph from documents"""
    
    def __init__(
        self,
        config: Optional[GraphConfig] = None,
        embedding_model_name: str = "text-embedding-3-large",
        device: Optional[str] = None
    ):
        self.config = config or GraphConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        
        # Initialize storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.nx_graph = nx.Graph()
        self.pyg_graph: Optional[HeteroData] = None
        self.node_to_idx: Dict[str, Dict[str, int]] = {}
        
        # Node2Vec parameters
        self.node2vec_embeddings: Optional[Dict[str, torch.Tensor]] = None

    @property
    def embeddings(self):
        """Property to access the embedding model"""
        return self.embedding_model
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        source_id: str
    ) -> None:
        """Add entity to graph"""
        if not name or not entity_type:
            raise ValueError("Entity name and type cannot be empty")
            
        # Generate embedding
        embedding = torch.tensor(self.embedding_model.embed_query(description))
        
        # Create entity
        entity = Entity(
            name=name,
            type=entity_type,
            description=description,
            source_id=source_id,
            embedding=embedding
        )
        
        # Add to storage
        self.entities[name] = entity
        
        # Update NetworkX graph
        self.nx_graph.add_node(
            name,
            type=entity_type,
            description=description,
            source_id=source_id,
            embedding=embedding
        )
    
    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        description: str,
        keywords: List[str],
        weight: float,
        source_id: str
    ) -> None:
        """Add relationship to graph"""
        # Verify entities exist
        if source not in self.entities:
            raise ValueError(f"Source entity {source} not found")
        if target not in self.entities:
            raise ValueError(f"Target entity {target} not found")
        
        # Generate embedding
        embedding = torch.tensor(self.embedding_model.embed_query(description))
        
        # Create relationship
        relationship = Relationship(
            source=source,
            target=target,
            type=rel_type,
            description=description,
            keywords=keywords,
            weight=weight,
            source_id=source_id,
            embedding=embedding
        )
        
        # Add to storage
        self.relationships.append(relationship)
        
        # Update NetworkX graph
        self.nx_graph.add_edge(
            source,
            target,
            type=rel_type,
            description=description,
            keywords=keywords,
            weight=weight,
            source_id=source_id,
            embedding=embedding
        )
    
    def build_pyg_graph(self) -> HeteroData:
        """Convert NetworkX graph to PyG HeteroData"""
        if not self.entities:
            raise ValueError("No entities in graph")
            
        # Create PyG graph
        data = HeteroData()
        
        # Group entities by type
        entities_by_type: Dict[str, List[Entity]] = {}
        for entity in self.entities.values():
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)
        
        # Add nodes
        self.node_to_idx = {}
        for entity_type, entities in entities_by_type.items():
            # Create node mapping
            self.node_to_idx[entity_type] = {
                entity.name: i for i, entity in enumerate(entities)
            }
            
            # Create node features
            data[entity_type].x = torch.stack([
                entity.embedding for entity in entities
            ]).to(self.device)
            
            # Add metadata
            data[entity_type].metadata = [
                {"name": entity.name, "source_id": entity.source_id}
                for entity in entities
            ]
        
        # Group relationships by type
        edges_by_type: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
        for rel in self.relationships:
            src_type = self.entities[rel.source].type
            dst_type = self.entities[rel.target].type
            edge_type = (src_type, rel.type, dst_type)
            
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
                
            src_idx = self.node_to_idx[src_type][rel.source]
            dst_idx = self.node_to_idx[dst_type][rel.target]
            edges_by_type[edge_type].append((src_idx, dst_idx))
        
        # Add edges
        for edge_type, edges in edges_by_type.items():
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
                data[edge_type].edge_index = edge_index
        
        self.pyg_graph = data
        return data

    def compute_node2vec_embeddings(
        self,
        embedding_dim: int = 64,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        epochs: int = 100,
        batch_size: int = 128
    ) -> Dict[str, torch.Tensor]:
        """Compute Node2Vec embeddings for the graph"""
        if not self.pyg_graph:
            raise ValueError("PyG graph not built. Call build_pyg_graph() first.")
            
        # Use entity embeddings
        self.node2vec_embeddings = {}
        for node_type in self.pyg_graph.node_types:
            self.node2vec_embeddings[node_type] = self.pyg_graph[node_type].x
        return self.node2vec_embeddings
    
    def save(self, path: str) -> None:
        """Save graph state"""
        # Move tensors to CPU before saving
        entities = {
            name: Entity(
                name=entity.name,
                type=entity.type,
                description=entity.description,
                source_id=entity.source_id,
                embedding=entity.embedding.cpu() if entity.embedding is not None else None
            )
            for name, entity in self.entities.items()
        }
        
        relationships = [
            Relationship(
                source=rel.source,
                target=rel.target,
                type=rel.type,
                description=rel.description,
                keywords=rel.keywords,
                weight=rel.weight,
                source_id=rel.source_id,
                embedding=rel.embedding.cpu() if rel.embedding is not None else None
            )
            for rel in self.relationships
        ]
        
        # Include node2vec embeddings in save state
        node2vec_embeddings_cpu = None
        if self.node2vec_embeddings:
            node2vec_embeddings_cpu = {
                k: v.cpu() for k, v in self.node2vec_embeddings.items()
            }
        
        save_dict = {
            'entities': entities,
            'relationships': relationships,
            'nx_graph': self.nx_graph,
            'pyg_graph': self.pyg_graph,
            'node_to_idx': self.node_to_idx,
            'node2vec_embeddings': node2vec_embeddings_cpu
        }
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load graph state"""
        load_dict = torch.load(path)
        
        # Move tensors to device
        self.entities = {
            name: Entity(
                name=entity.name,
                type=entity.type,
                description=entity.description,
                source_id=entity.source_id,
                embedding=entity.embedding.to(self.device) if entity.embedding is not None else None
            )
            for name, entity in load_dict['entities'].items()
        }
        
        self.relationships = [
            Relationship(
                source=rel.source,
                target=rel.target,
                type=rel.type,
                description=rel.description,
                keywords=rel.keywords,
                weight=rel.weight,
                source_id=rel.source_id,
                embedding=rel.embedding.to(self.device) if rel.embedding is not None else None
            )
            for rel in load_dict['relationships']
        ]
        
        self.nx_graph = load_dict['nx_graph']
        self.pyg_graph = load_dict['pyg_graph']
        self.node_to_idx = load_dict['node_to_idx']
        
        # Load node2vec embeddings
        if 'node2vec_embeddings' in load_dict and load_dict['node2vec_embeddings']:
            self.node2vec_embeddings = {
                k: v.to(self.device) for k, v in load_dict['node2vec_embeddings'].items()
            }

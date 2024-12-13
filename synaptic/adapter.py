"""SynapticRAG adapter combining memory and graph components"""
import json
import torch
import logging
from typing import Dict, List, Optional, Union, Any, Set, Literal
from dataclasses import dataclass
import spacy
import tiktoken
from langchain_openai import OpenAIEmbeddings
from .config import SynapticConfig
from .memory.memory_model import MemoryModel
from .memory.clue_generator import ClueGenerator, Clue
from .graph.build_graph import GraphBuilder
from .graph.retrieval import HybridRetriever
from .graph.pyG_utils import HeteroGNN
from .utils import cos_sim
from .storage.base import StorageConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Container for generation results"""
    response: str
    sources: List[str]
    metadata: Optional[Dict] = None

class SynapticRAG:
    """Retrieval-augmented generation with memory and graph components"""
    
    def __init__(
        self,
        config: Optional[SynapticConfig] = None,
        device: Optional[str] = None
    ):
        self.config = config or SynapticConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SynapticRAG with device: {self.device}")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=self.config.retriever.model_name_or_path)
        
        # Initialize memory components
        self.memory_model = MemoryModel(
            model_name_or_path=self.config.memory.model_name_or_path,
            device=self.device
        )
        
        self.clue_generator = ClueGenerator(
            llm=self.memory_model,
            max_clues=self.config.memory.max_clues,
            min_score=self.config.memory.min_clue_score,
            temperature=self.config.memory.temperature,
            max_tokens=self.config.memory.max_tokens
        )
        
        # Initialize graph components
        self.graph_builder = GraphBuilder(
            config=self.config.graph,
            device=self.device
        )
        
        # Initialize hybrid retriever
        self.retriever = HybridRetriever(
            embedding_model_name=self.config.retriever.model_name_or_path,
            top_k=self.config.retriever.hits,
            device=self.device,
            memory_weight=self.config.memory.memory_weight,
            vector_weight=self.config.retriever.vector_weight,
            graph_weight=self.config.retriever.graph_weight
        )
        
        # Initialize GNN
        self.gnn = None
        
        # Initialize NLP and tokenizer
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Storage
        self.documents: Dict[str, str] = {}
        self._processed_docs: Set[str] = set()
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        
        # Initialize storage components
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage components"""
        # Create storage configs with namespaces from main config
        kv_config = StorageConfig(
            working_dir=self.config.storage.working_dir,
            cache_dir=self.config.storage.cache_dir,
            namespace=self.config.storage.kv_namespace
        )
        
        vector_config = StorageConfig(
            working_dir=self.config.storage.working_dir,
            cache_dir=self.config.storage.cache_dir,
            namespace=self.config.storage.vector_namespace
        )
        
        graph_config = StorageConfig(
            working_dir=self.config.storage.working_dir,
            cache_dir=self.config.storage.cache_dir,
            namespace=self.config.storage.graph_namespace
        )
        
        # Initialize KV storage
        self.kv_storage = self.config.storage.kv_storage_cls(config=kv_config)
        
        # Initialize vector storage with embeddings
        self.vector_storage = self.config.storage.vector_storage_cls(
            config=vector_config,
            embeddings=self.embeddings,
            dimension=1536  # OpenAI embedding dimension
        )
        
        # Initialize graph storage with embeddings
        self.graph_storage = self.config.storage.graph_storage_cls(config=graph_config)
    
    def add_documents(self, documents: Union[str, List[str]]) -> None:
        """Process and add documents to both memory and graph"""
        if isinstance(documents, str):
            documents = [documents]
        
        # Process each document
        for doc in documents:
            logger.info(f"Processing document of length: {len(doc)}")
            
            # Skip if document is already processed
            doc_hash = hash(doc.strip())
            if doc_hash in self._processed_docs:
                logger.info("Document already processed, skipping...")
                continue
            self._processed_docs.add(doc_hash)
            
            # Generate document ID
            doc_id = f"doc_{len(self.documents)}"
            self.documents[doc_id] = doc
            
            try:
                # Add to memory with metadata
                logger.info("Adding document to memory...")
                memory_state = self.memory_model.memorize(doc, {"source_id": doc_id})
                logger.info(f"Memory state: tokens shape={memory_state.memory_tokens.shape}")
                
                # Generate clues for entity extraction
                logger.info("Generating clues for entity extraction...")
                clues = self.clue_generator.generate_clues(
                    doc,
                    memory_state={"memory_embeddings": memory_state.memory_tokens}
                )
                logger.info(f"Generated {len(clues)} clues for entity extraction")
                
                # Extract entities and relationships using clues
                logger.info("Extracting entities...")
                entities = self._extract_entities(doc, clues)
                logger.info(f"Extracted {len(entities)} entities")
                
                # Add entities to graph
                for entity in entities:
                    # Adjust entity type based on clues
                    if clues:  # Only check clues if list is not empty
                        for clue in clues:
                            clue_text = clue if isinstance(clue, str) else clue.text
                            if clue_text.lower() in entity["name"].lower():
                                entity["type"] = "KEY_CONCEPT"
                                break
                    
                    entity["source_id"] = doc_id
                    self.graph_builder.add_entity(
                        name=entity["name"],
                        entity_type=entity["type"],
                        description=entity["description"],
                        source_id=doc_id
                    )
                    logger.info(f"Added entity: {entity['name']} ({entity['type']})")
                
                # Extract relationships between entities
                logger.info("Extracting relationships...")
                relationships = self._extract_relationships(doc, entities)
                logger.info(f"Extracted {len(relationships)} relationships")
                
                # Add relationships to graph
                for rel in relationships:
                    rel["source_id"] = doc_id
                    self.graph_builder.add_relationship(
                        source=rel["source"],
                        target=rel["target"],
                        rel_type=rel["type"],
                        description=rel["description"],
                        keywords=rel["keywords"],
                        weight=rel["weight"],
                        source_id=doc_id
                    )
                    logger.info(f"Added relationship: {rel['source']} -{rel['type']}-> {rel['target']}")
                
                # Add chunks to retriever
                logger.info("Creating chunks...")
                chunks = self._chunk_document(doc)
                logger.info(f"Created {len(chunks)} chunks")
                chunk_dict = {
                    f"{doc_id}_chunk_{i}": chunk
                    for i, chunk in enumerate(chunks)
                }
                self.retriever.add_chunks(chunk_dict)
                
                # Build/update PyG graph
                if self.graph_builder.entities:
                    logger.info("Building PyG graph...")
                    graph = self.graph_builder.build_pyg_graph()
                    logger.info(f"Graph metadata: {graph.metadata()}")
                    
                    # Initialize/update GNN if needed
                    if self.gnn is None:
                        logger.info("Initializing GNN...")
                        self.gnn = HeteroGNN(
                            hidden_channels=self.config.graph.gnn_hidden_channels,
                            num_layers=self.config.graph.gnn_num_layers,
                            metadata=graph.metadata(),
                            dropout=self.config.graph.dropout
                        ).to(self.device)
                    
                    # Compute Node2Vec embeddings
                    logger.info("Computing Node2Vec embeddings...")
                    self.graph_builder.compute_node2vec_embeddings()
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue
    
    def query(
        self,
        query: str,
        return_sources: bool = True,
        retrieval_mode: str = "hybrid"
    ) -> GenerationResult:
        """Process query and generate response"""
        if not query or not isinstance(query, str) or not query.strip():
            return GenerationResult(response="", sources=[], metadata={})
            
        logger.info(f"Processing query: {query}")
        
        try:
            # Generate clues from memory and query
            logger.info("Generating clues...")
            query_clues = self.clue_generator.generate_clues(query)
            memory_clues = []
            if self.memory_model.memory_state:
                memory_clues = self.clue_generator.generate_clues(
                    query,
                    memory_state={"memory_embeddings": self.memory_model.memory_state.memory_tokens}
                )
            clues = query_clues + memory_clues
            
            # Separate high and low level clues
            high_level_clues = [c for c in clues if isinstance(c, Clue) and c.level == "high"]
            low_level_clues = [c for c in clues if isinstance(c, Clue) and c.level == "low"]
            logger.info(f"Generated {len(high_level_clues)} high-level and {len(low_level_clues)} low-level clues")
            
            # Get memory recall
            logger.info("Getting memory recall...")
            memory_response = self.memory_model.recall(query)
            if memory_response:
                logger.info("Memory response received")
                logger.info(f"Memory response length: {len(memory_response)}")
            
            # Get retrieval results based on mode
            results = []
            if retrieval_mode == "hybrid" and self.graph_builder.entities:
                logger.info("Performing hybrid retrieval...")
                # Use both high and low level clues for hybrid retrieval
                all_clue_texts = [c.text for c in clues]
                results = self.retriever.hybrid_retrieve(
                    query,
                    clues=all_clue_texts,
                    graph=self.graph_builder.pyg_graph,
                    entity_embeddings=self.graph_builder.node2vec_embeddings,
                    memory_response=memory_response if memory_response else None
                )
            elif retrieval_mode == "memory":
                logger.info("Performing memory-based retrieval...")
                if memory_response:
                    # Use low level clues for memory retrieval
                    low_level_texts = [c.text for c in low_level_clues]
                    results = self.retriever.retrieve_by_query(
                        query,
                        clues=low_level_texts,
                        memory_response=memory_response
                    )
            elif retrieval_mode == "graph" and self.graph_builder.entities:
                logger.info("Performing graph-based retrieval...")
                # Use high level clues for graph retrieval
                high_level_texts = [c.text for c in high_level_clues]
                results = self.retriever.retrieve_by_graph(
                    query,
                    self.graph_builder.pyg_graph,
                    self.graph_builder.node2vec_embeddings,
                    clues=high_level_texts
                )
            elif retrieval_mode == "vector":
                logger.info("Performing vector retrieval...")
                # Use all clues for vector retrieval
                all_clue_texts = [c.text for c in clues]
                results = self.retriever.retrieve_by_query(
                    query,
                    clues=all_clue_texts
                )
            
            logger.info(f"Retrieved {len(results)} results")
            
            # Extract sources and scores
            sources = [result.source_id for result in results]
            scores = [result.score for result in results]
            
            # Generate response
            response = ""
            if results or memory_response:
                # Combine retrieved chunks with memory recall
                relevant_chunks = []
                
                # Add memory recall if available
                if memory_response:
                    relevant_chunks.append("Memory Context:")
                    relevant_chunks.append(memory_response)
                
                # Add retrieved chunks
                if results:
                    relevant_chunks.append("\nRetrieved Context:")
                    for result in results:
                        relevant_chunks.append(result.text)
                
                # Generate response using combined context
                logger.info("Generating response...")
                context = "\n\n".join(relevant_chunks)
                logger.info(f"Combined context length: {len(context)}")
                response = self.memory_model.generate_response(query, context)
                logger.info(f"Generated response length: {len(response)}")
            
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)
            
            # Include structured clues in metadata
            clue_metadata = {
                "high_level_clues": [{"text": c.text, "score": c.score} for c in high_level_clues],
                "low_level_clues": [{"text": c.text, "score": c.score} for c in low_level_clues]
            }
            
            return GenerationResult(
                response=response,
                sources=sources if return_sources else [],
                metadata={
                    "clues": clue_metadata,
                    "retrieval_scores": scores,
                    "memory_recall": memory_response is not None,
                    "memory_score": scores[0] if scores else 0.0,
                    "vector_score": scores[1] if len(scores) > 1 else 0.0,
                    "graph_score": scores[2] if len(scores) > 2 else 0.0
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return GenerationResult(
                response="Error processing query",
                sources=[],
                metadata={"error": str(e)}
            )
    
    def save(self, path: str) -> None:
        """Save RAG state"""
        save_dict = {
            'documents': self.documents,
            'processed_docs': self._processed_docs,
            'memory_state': self.memory_model.memory_state,
            'graph_builder_state': {
                'entities': self.graph_builder.entities,
                'relationships': self.graph_builder.relationships,
                'nx_graph': self.graph_builder.nx_graph,
                'pyg_graph': self.graph_builder.pyg_graph,
                'node_to_idx': self.graph_builder.node_to_idx,
                'node2vec_embeddings': self.graph_builder.node2vec_embeddings
            },
            'retriever_state': {
                'chunks': self.retriever.chunks,
                'chunk_embeddings': {k: v.cpu() for k, v in self.retriever.chunk_embeddings.items()}
            },
            'gnn_state': self.gnn.state_dict() if self.gnn else None,
            'embedding_cache': {k: v.cpu() for k, v in self._embedding_cache.items()}
        }
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load RAG state"""
        load_dict = torch.load(path)
        
        # Load documents and processed docs
        self.documents = load_dict['documents']
        self._processed_docs = load_dict['processed_docs']
        
        # Load memory state
        self.memory_model.memory_state = load_dict['memory_state']
        
        # Load graph builder state
        graph_state = load_dict['graph_builder_state']
        self.graph_builder.entities = graph_state['entities']
        self.graph_builder.relationships = graph_state['relationships']
        self.graph_builder.nx_graph = graph_state['nx_graph']
        self.graph_builder.pyg_graph = graph_state['pyg_graph']
        self.graph_builder.node_to_idx = graph_state['node_to_idx']
        self.graph_builder.node2vec_embeddings = graph_state['node2vec_embeddings']
        
        # Load retriever state
        retriever_state = load_dict['retriever_state']
        self.retriever.chunks = retriever_state['chunks']
        self.retriever.chunk_embeddings = {
            k: v.to(self.device) for k, v in retriever_state['chunk_embeddings'].items()
        }
        
        # Load GNN state if available
        if load_dict['gnn_state'] and self.gnn:
            self.gnn.load_state_dict(load_dict['gnn_state'])
            
        # Load embedding cache
        self._embedding_cache = {
            k: v.to(self.device) for k, v in load_dict['embedding_cache'].items()
        }
    
    def clear(self) -> None:
        """Clear all state"""
        self.documents.clear()
        self._processed_docs.clear()
        self._embedding_cache.clear()
        self.memory_model.memory_state = None
        self.graph_builder = GraphBuilder(
            config=self.config.graph,
            device=self.device
        )
        self.retriever = HybridRetriever(
            embedding_model_name=self.config.retriever.model_name_or_path,
            top_k=self.config.retriever.hits,
            device=self.device,
            memory_weight=self.config.memory.memory_weight,
            vector_weight=self.config.retriever.vector_weight,
            graph_weight=self.config.retriever.graph_weight
        )
        self.gnn = None
    
    def _extract_entities(self, text: str, clues: List[Any]) -> List[Dict]:
        """Extract entities from text using clues and spaCy"""
        entities = []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                # Map spaCy entity types to our types
                type_map = {
                    "PERSON": "PERSON",
                    "ORG": "ORG",
                    "GPE": "LOCATION",
                    "LOC": "LOCATION",
                    "PRODUCT": "PRODUCT",
                    "EVENT": "EVENT",
                    "WORK_OF_ART": "WORK",
                    "LAW": "LAW",
                    "LANGUAGE": "LANGUAGE",
                    "DATE": "DATE",
                    "TIME": "TIME",
                    "MONEY": "MONEY",
                    "QUANTITY": "QUANTITY",
                    "ORDINAL": "ORDINAL",
                    "CARDINAL": "CARDINAL"
                }
                
                entity_type = type_map.get(ent.label_, "CONCEPT")
                
                # Add entity
                entities.append({
                    "name": ent.text,
                    "type": entity_type,
                    "description": text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)],
                    "source_id": ""  # Will be set by caller
                })
            
            # Extract concepts from noun chunks and clues
            seen_concepts = set()
            for chunk in doc.noun_chunks:
                if not any(ent.text == chunk.text for ent in doc.ents):
                    # Check if concept appears in clues
                    is_key_concept = False
                    if clues:  # Only check clues if list is not empty
                        for clue in clues:
                            clue_text = clue if isinstance(clue, str) else clue.text
                            if clue_text.lower() in chunk.text.lower():
                                is_key_concept = True
                                break
                    
                    concept_type = "KEY_CONCEPT" if is_key_concept else "CONCEPT"
                    
                    if chunk.text not in seen_concepts:
                        entities.append({
                            "name": chunk.text,
                            "type": concept_type,
                            "description": text[max(0, chunk.start_char-50):min(len(text), chunk.end_char+50)],
                            "source_id": ""  # Will be set by caller
                        })
                        seen_concepts.add(chunk.text)
            
            # Extract additional concepts from clues
            if clues:  # Only process clues if list is not empty
                for clue in clues:
                    clue_text = clue if isinstance(clue, str) else clue.text
                    clue_doc = self.nlp(clue_text)
                    for chunk in clue_doc.noun_chunks:
                        if chunk.text not in seen_concepts:
                            entities.append({
                                "name": chunk.text,
                                "type": "KEY_CONCEPT",
                                "description": clue_text,
                                "source_id": ""  # Will be set by caller
                            })
                            seen_concepts.add(chunk.text)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            logger.error(f"Clues: {clues}")  # Log clues for debugging
        
        return entities
    
    def _extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []
        
        try:
            # Create entity lookup
            entity_names = {entity["name"].lower(): entity for entity in entities}
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract subject-verb-object relationships
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":
                        # Find subject
                        subj = None
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                subj = child
                                break
                        
                        # Find object
                        obj = None
                        for child in token.children:
                            if child.dep_ in ["dobj", "pobj"]:
                                obj = child
                                break
                        
                        if subj and obj:
                            # Check if subject and object are known entities
                            subj_text = subj.text.lower()
                            obj_text = obj.text.lower()
                            
                            if subj_text in entity_names and obj_text in entity_names:
                                # Extract keywords from the relationship
                                keywords = [token.text]
                                for child in token.children:
                                    if child.dep_ in ["advmod", "amod"]:
                                        keywords.append(child.text)
                                
                                relationships.append({
                                    "source": entity_names[subj_text]["name"],
                                    "target": entity_names[obj_text]["name"],
                                    "type": token.lemma_.upper(),
                                    "description": sent.text,
                                    "keywords": keywords,
                                    "weight": 1.0,
                                    "source_id": ""  # Will be set by caller
                                })
                                
        except Exception as e:
            logger.error(f"Error extracting relationships: {str(e)}")
        
        return relationships
    
    def _chunk_document(self, text: str) -> List[str]:
        """Split document into chunks using LightRAG's approach"""
        try:
            # Encode text into tokens
            tokens = self.tokenizer.encode(text)
            chunk_size = self.config.graph.chunk_token_size
            overlap_size = self.config.graph.chunk_overlap_token_size
            
            # Create chunks with overlap
            chunks = []
            for i in range(0, len(tokens), chunk_size - overlap_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            return []

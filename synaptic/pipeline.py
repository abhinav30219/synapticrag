"""Pipeline for document processing and querying"""
import os
import torch
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from .config import SynapticConfig
from .adapter import SynapticRAG, GenerationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Container for document processing results"""
    num_documents: int
    num_entities: int
    num_relationships: int
    memory_size: int
    graph_size: Dict[str, int]

class SynapticPipeline:
    """Pipeline for document processing and querying"""
    
    def __init__(self, config: Optional[SynapticConfig] = None):
        self.config = config or SynapticConfig()
        logger.info(f"Initializing SynapticPipeline with config: {self.config}")
        
        # Create working directory if needed
        if not os.path.exists(self.config.storage.working_dir):
            logger.info(f"Creating working directory: {self.config.storage.working_dir}")
            os.makedirs(self.config.storage.working_dir)
        
        # Initialize RAG
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SynapticRAG with device: {device}")
        self.rag = SynapticRAG(config=self.config, device=device)
    
    def process_documents(
        self,
        documents: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> ProcessingResult:
        """Process and add documents to RAG system"""
        if isinstance(documents, str):
            documents = [documents]
            logger.info("Converting single document to list")
            
        if not documents or not any(doc.strip() for doc in documents):
            raise ValueError("Documents must not be empty")
            
        logger.info(f"Processing {len(documents)} documents")
        
        # Get initial state
        initial_entities = len(self.rag.graph_builder.entities)
        initial_relationships = len(self.rag.graph_builder.relationships)
        logger.info(f"Initial state - Entities: {initial_entities}, Relationships: {initial_relationships}")
        
        # Process documents in batches
        batch_size = batch_size or self.config.batch_size
        processed_docs = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            # Filter out empty documents
            batch = [doc for doc in batch if doc.strip()]
            if batch:
                logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} documents")
                # Process batch
                self.rag.add_documents(batch)
                processed_docs += len(batch)
                logger.info(f"Processed {processed_docs} documents so far")
        
        # Calculate changes
        final_entities = len(self.rag.graph_builder.entities)
        final_relationships = len(self.rag.graph_builder.relationships)
        logger.info(f"Final state - Entities: {final_entities}, Relationships: {final_relationships}")
        
        # Calculate memory size from memory state
        memory_size = 0
        if self.rag.memory_model.memory_state is not None:
            # Count number of memory tokens
            memory_size = self.rag.memory_model.memory_state.memory_tokens.size(1)
            logger.info(f"Memory size: {memory_size} tokens")
        
        # Get graph size
        graph_size = {
            "nodes": final_entities,
            "edges": final_relationships
        }
        logger.info(f"Graph size: {graph_size}")
        
        # Log entity and relationship details
        logger.info("Entity details:")
        for entity_name, entity in self.rag.graph_builder.entities.items():
            logger.info(f"- {entity_name} ({entity.type}) from {entity.source_id}")
        
        logger.info("Relationship details:")
        for rel in self.rag.graph_builder.relationships:
            logger.info(f"- {rel.source} -{rel.type}-> {rel.target}")
        
        result = ProcessingResult(
            num_documents=processed_docs,
            num_entities=final_entities - initial_entities,
            num_relationships=final_relationships - initial_relationships,
            memory_size=memory_size,
            graph_size=graph_size
        )
        logger.info(f"Processing complete: {result}")
        return result
    
    def query(
        self,
        query: str,
        return_sources: bool = True,
        retrieval_mode: str = "hybrid"
    ) -> GenerationResult:
        """Process query and generate response"""
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty query received")
            return GenerationResult(response="", sources=[], metadata={})
            
        # Check if we have any documents processed
        if not self.rag.documents:
            logger.warning("No documents processed yet")
            return GenerationResult(response="", sources=[], metadata={})
            
        logger.info(f"Processing query: {query}")
        logger.info(f"Document count: {len(self.rag.documents)}")
        logger.info(f"Entity count: {len(self.rag.graph_builder.entities)}")
        logger.info(f"Memory state: {self.rag.memory_model.memory_state is not None}")
        logger.info(f"Retrieval mode: {retrieval_mode}")
        
        result = self.rag.query(
            query, 
            return_sources=return_sources,
            retrieval_mode=retrieval_mode
        )
        
        # Log query results
        logger.info("Query results:")
        if result.sources:
            logger.info(f"Sources: {result.sources}")
        if result.metadata:
            if 'clues' in result.metadata:
                logger.info(f"Generated clues: {result.metadata['clues']}")
            if 'retrieval_scores' in result.metadata:
                logger.info(f"Retrieval scores: {result.metadata['retrieval_scores']}")
            if 'memory_recall' in result.metadata:
                logger.info("Memory was used in generating the response")
        
        # Ensure response is a string
        if result.response is None:
            result.response = ""
        elif not isinstance(result.response, str):
            result.response = str(result.response)
        
        logger.info(f"Response length: {len(result.response)}")
        return result
    
    def batch_query(
        self,
        queries: List[str],
        batch_size: Optional[int] = None,
        return_sources: bool = True,
        retrieval_mode: str = "hybrid"
    ) -> List[GenerationResult]:
        """Process multiple queries in batches"""
        if not queries:
            logger.warning("Empty queries list received")
            return []
            
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        # Filter out empty queries
        queries = [q for q in queries if q.strip()]
        if not queries:
            logger.warning("No valid queries after filtering")
            return []
            
        logger.info(f"Processing {len(queries)} queries in batches")
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} queries")
            batch_results = [
                self.query(
                    q, 
                    return_sources=return_sources,
                    retrieval_mode=retrieval_mode
                )
                for q in batch
            ]
            results.extend(batch_results)
            logger.info(f"Processed {len(results)} queries so far")
        
        return results
    
    def save(self, path: str) -> None:
        """Save pipeline state"""
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving pipeline state to {path}")
        
        # Save RAG state
        self.rag.save(path)
        logger.info("Pipeline state saved successfully")
    
    def load(self, path: str) -> None:
        """Load pipeline state"""
        if not os.path.exists(path):
            raise ValueError(f"State file not found: {path}")
            
        logger.info(f"Loading pipeline state from {path}")
        # Load RAG state
        self.rag.load(path)
        logger.info("Pipeline state loaded successfully")
        
        # Log loaded state
        logger.info(f"Loaded {len(self.rag.documents)} documents")
        logger.info(f"Loaded {len(self.rag.graph_builder.entities)} entities")
        logger.info(f"Loaded {len(self.rag.graph_builder.relationships)} relationships")
        if self.rag.memory_model.memory_state:
            logger.info(f"Loaded memory state with {self.rag.memory_model.memory_state.memory_tokens.size(1)} tokens")
    
    def clear(self) -> None:
        """Clear all state"""
        logger.info("Clearing pipeline state")
        self.rag.clear()
        logger.info("Pipeline state cleared")

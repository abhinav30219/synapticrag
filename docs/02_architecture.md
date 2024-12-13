# Architecture

## System Overview

SynapticRAG's architecture is designed around four core principles:
1. Hybrid knowledge processing
2. Multi-modal understanding
3. Dynamic memory management
4. Flexible storage integration

![SynapticRAG Architecture](images/synapticrag_architecture.png)

## Core Components

### 1. Text Processing Pipeline

#### a. Hybrid Text Splitter
- **Semantic Splitting**
  - Uses GPT-3.5-turbo tokenizer
  - Language-aware chunk sizes (512/2048 tokens)
  - Preserves semantic coherence
- **Token-based Refinement**
  - cl100k_base tokenizer
  - 1200 token chunks with 100 token overlap
  - Maintains token-level context

#### b. Entity Extraction
- Spacy-based named entity recognition
- Memory-guided entity identification
- Relationship extraction from context

### 2. Multi-modal Processing

#### a. Image Processing
- CLIP model integration (clip-vit-base-patch32)
- PDF image extraction via PyMuPDF
- RGBA to RGB conversion handling

#### b. Unified Embedding Space
- OpenAI embeddings for text (1536D)
- CLIP embeddings resized to match (1536D)
- Linear interpolation for dimension matching

### 3. Knowledge Graph

#### a. Graph Structure
- NetworkX-based implementation
- Heterogeneous node types
- Weighted relationships
- Dynamic updates

#### b. Node2Vec Integration
- Graph embeddings for structural context
- Configurable walk parameters
- Dimension alignment with text/image embeddings

### 4. Memory Management

#### a. Memory Model
- Based on MemoRAG's approach
- Dynamic memory consolidation
- Context-aware memory updates

#### b. Clue Generation
- Memory-guided query enhancement
- Surrogate query generation
- Context refinement

### 5. Storage Layer

#### a. Vector Stores
- FAISS implementation
- Separate indices for text and images
- Unified similarity search

#### b. Graph Storage
- NetworkX persistence
- Relationship caching
- Efficient traversal support

#### c. Key-Value Storage
- JSON-based implementation
- Metadata management
- Cache control

## Integration Flow

1. **Document Ingestion**
   ```
   Document → Text Chunks + Images → Embeddings + Graph Nodes
   ```

2. **Query Processing**
   ```
   Query → Memory Lookup → Graph Traversal → Hybrid Retrieval
   ```

3. **Response Generation**
   ```
   Retrieved Context → Memory Integration → LLM Generation
   ```

## Design Considerations

### 1. Scalability
- Modular component design
- Efficient storage backends
- Batch processing support

### 2. Extensibility
- Plugin architecture for models
- Custom storage backends
- Flexible embedding options

### 3. Performance
- Optimized embedding caching
- Efficient graph traversal
- Smart memory management

### 4. Reliability
- Error handling for corrupted images
- Fallback mechanisms
- Consistent state management

## Implementation Stack

### Core Technologies
- PyTorch for deep learning
- FAISS for vector search
- NetworkX for graph operations
- LangChain for LLM integration

### External Services
- OpenAI API for embeddings
- Claude API for text generation
- CLIP for image understanding

### Development Tools
- Streamlit for demo interface
- PyTest for testing
- GitHub Actions for CI/CD

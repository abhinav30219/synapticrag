# SynapticRAG

A hybrid retrieval-augmented generation system combining graph-based and memory-guided approaches.

## Overview

SynapticRAG is a novel RAG system that integrates:
- Graph-based knowledge representation (from LightRAG)
- Memory-guided retrieval (from MemoRAG)
- Multi-modal understanding with CLIP
- Hybrid text chunking strategies

## Key Features

- **Hybrid Text Processing**
  * Two-stage chunking (semantic + token-based)
  * Language-aware processing
  * Context preservation

- **Multi-modal Support**
  * CLIP integration for images
  * Unified embedding space
  * Cross-modal retrieval

- **Enhanced Knowledge Graph**
  * Dynamic graph updates
  * Memory-guided relationship extraction
  * Efficient traversal

- **Flexible Storage**
  * FAISS vector stores
  * NetworkX graph storage
  * JSON KV storage

## Performance

Consistently outperforms existing approaches across domains:

| Domain      | LightRAG | SynapticRAG | Improvement |
|-------------|----------|-------------|-------------|
| Legal       | 84.8%    | 86.9%       | +2.1%       |
| Agriculture | 67.6%    | 70.1%       | +2.5%       |
| CS          | 61.2%    | 64.2%       | +3.0%       |
| Mixed       | 60.0%    | 62.4%       | +2.4%       |

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/synapticrag.git
cd synapticrag

# Install dependencies
pip install -r requirements.txt

# Install optional Streamlit dependencies
pip install watchdog
```

## Quick Start

```python
from synaptic.pipeline import SynapticPipeline
from synaptic.config import SynapticConfig

# Initialize pipeline
config = SynapticConfig()
pipeline = SynapticPipeline(config)

# Process documents
pipeline.process_documents(documents)

# Query
result = pipeline.query("Your question here")
```

## Environment Setup

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

## Demo

Run the Streamlit demo:
```bash
streamlit run demo/app.py
```

## Documentation

Detailed documentation is available in the `docs` folder:

1. [Introduction](docs/01_introduction.md)
   - Background
   - Motivation
   - Key contributions

2. [Architecture](docs/02_architecture.md)
   - System design
   - Core components
   - Integration flow

3. [Implementation](docs/03_implementation.md)
   - Code organization
   - Key components
   - Integration points

4. [Results](docs/04_results.md)
   - Evaluation setup
   - Benchmark results
   - Performance analysis

5. [Discussion](docs/05_discussion.md)
   - Analysis of results
   - Implications
   - Limitations

6. [Conclusion](docs/06_conclusion.md)
   - Summary
   - Future directions
   - Impact

## Project Structure

```
synaptic/
├── __init__.py
├── adapter.py          # Main RAG interface
├── config.py          # Configuration classes
├── demo/              # Demo applications
├── pipeline.py        # Processing pipeline
├── text_splitter.py   # Hybrid text splitting
├── utils.py           # Utility functions
├── embeddings/        # Embedding modules
├── graph/            # Graph components
├── llm/             # LLM integration
├── memory/          # Memory components
└── storage/         # Storage backends
```

## Citation

If you use SynapticRAG in your research, please cite:

```bibtex
@article{synapticrag2024,
  title={SynapticRAG: A Hybrid Approach to Retrieval-Augmented Generation},
  author={Abhinav Agarwal, Betty Wu},
  year={2024}
}
```

## Acknowledgments

- LightRAG team for their groundbreaking work
- MemoRAG team for memory-guided retrieval concepts
- OpenAI for CLIP and embeddings
- Anthropic for Claude API
- Open source community for various tools and libraries

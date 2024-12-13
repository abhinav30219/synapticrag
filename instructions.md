# SynapticRAG Instructions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/synapticrag.git
cd synapticrag
```

2. Create and activate virtual environment:
```bash
python -m venv synapticenv
source synapticenv/bin/activate
```

3. Install dependencies (M1/M2 Mac):
```bash
# Update pip first
pip install --upgrade pip

# Install PyTorch
pip install torch

# Install PyTorch Geometric dependencies one by one
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install the package in development mode
pip install -e .
```

For other systems:
```bash
pip install -e .
```

4. Configure environment variables:
- Copy `.env.example` to `.env`
- Add your API keys to the `.env` file:
  ```
  ANTHROPIC_API_KEY=your_api_key_here
  ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
  OPENROUTER_API_KEY=your_openrouter_key_here
  ```

## Development Setup

For development, install additional dependencies:
```bash
pip install -e ".[dev]"
```

This will install:
- pytest for testing
- pytest-cov for coverage reports
- black for code formatting
- isort for import sorting
- flake8 for linting

## Running the Program

### Using the CLI

1. Process documents:
```bash
python main.py process --input /path/to/documents --working-dir ./workspace
```

2. Query the system:
```bash
python main.py query --query "Your question here" --working-dir ./workspace
```

3. Interactive mode:
```bash
python main.py interactive --working-dir ./workspace
```

### Using the Python API

```python
from synaptic.pipeline import SynapticPipeline
from synaptic.config import SynapticConfig

# Initialize pipeline
config = SynapticConfig()
pipeline = SynapticPipeline(config=config)

# Process documents
documents = ["doc1.txt", "doc2.txt"]
pipeline.process_documents(documents)

# Query
result = pipeline.query("Your question here")
print(result.response)
```

## Example Notebooks

The `examples/` directory contains Jupyter notebooks demonstrating various use cases:

1. `basic_usage.ipynb`: Getting started with SynapticRAG
2. `memory_analysis.ipynb`: Exploring memory-based retrieval
3. `graph_analysis.ipynb`: Visualizing and analyzing knowledge graphs
4. `complex_queries.ipynb`: Handling ambiguous and complex queries

To run the notebooks:
```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook
```

## Configuration

### Memory Settings

- `memory_model`: Using Claude 3.5 Sonnet (default: "claude-3-5-sonnet-20241022")
- `beacon_ratio`: Memory compression ratio (default: 4)
- `max_memory_blocks`: Maximum memory blocks to maintain (default: 100)

### Graph Settings

- `chunk_token_size`: Size of text chunks for graph construction (default: 1200)
- `gnn_hidden_channels`: Hidden dimensions for GNN (default: 256)
- `node2vec_params`: Parameters for node embeddings

### Retrieval Settings

- `retriever_model`: Embedding model for retrieval (default: "BAAI/bge-large-en-v1.5")
- `top_k`: Number of results to retrieve (default: 3)

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=synaptic tests/

# Run specific test files
pytest tests/test_memory.py
pytest tests/test_graph.py
pytest tests/test_retrieval.py
```

## Code Quality

Format code:
```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8
```

## Troubleshooting

1. Installation Issues on M1/M2 Mac:
- Make sure to install dependencies in the exact order shown in setup steps
- If PyTorch Geometric installation fails, try installing each package separately
- Use CPU versions of packages as CUDA is not supported on M1/M2 Macs
- Make sure you're using Python 3.8 or higher

2. Import Errors:
- Make sure you've installed the package in development mode with `pip install -e .`
- Check that you're running Python from the correct virtual environment
- Verify the package structure is correct with all __init__.py files

3. Memory Issues:
- Reduce `chunk_token_size` if processing large documents
- Enable 4-bit quantization with `--no-4bit false`
- Monitor memory usage with Activity Monitor

4. Graph Issues:
- Reduce `gnn_hidden_channels` for large graphs
- Adjust `node2vec_params` for better embeddings
- Use smaller batch sizes for processing

5. API Issues:
- Verify API keys are correctly set in .env file
- Check internet connection for API calls
- Monitor API rate limits and usage

6. Retrieval Issues:
- Increase `top_k` for more comprehensive results
- Adjust `memory_threshold` for stricter/looser matching
- Check query formatting and length

## Common Error Solutions

1. ModuleNotFoundError: No module named 'synaptic':
```bash
# Make sure you're in the project root directory
cd /path/to/synapticrag

# Install package in development mode
pip install -e .
```

2. PyTorch Geometric Installation Errors:
```bash
# Try installing with specific versions
pip install torch==2.0.0
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

3. Memory Errors:
```bash
# Reduce memory usage in config.py
chunk_token_size=800
gnn_hidden_channels=128
```

4. API Rate Limits:
```python
# Add delay between API calls
import time
time.sleep(1)  # 1 second delay
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests and linting
```bash
pytest
black .
isort .
flake8
```
5. Submit pull request

For detailed contribution guidelines, see CONTRIBUTING.md.

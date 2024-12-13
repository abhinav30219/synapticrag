import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
from synaptic.pipeline import SynapticPipeline
from synaptic.config import SynapticConfig, MemoryConfig, GraphConfig, RetrieverConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_documents(path: str) -> List[str]:
    """Load documents from a file or directory"""
    path = Path(path)
    
    if path.is_file():
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return list(data.values())
        else:
            with open(path) as f:
                return [f.read()]
    
    elif path.is_dir():
        documents = []
        for file_path in path.glob('**/*'):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.json']:
                try:
                    if file_path.suffix == '.json':
                        with open(file_path) as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                documents.extend(data)
                            elif isinstance(data, dict):
                                documents.extend(data.values())
                            else:
                                documents.append(json.dumps(data))
                    else:
                        with open(file_path) as f:
                            documents.append(f.read())
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
        return documents
    
    raise ValueError(f"Invalid path: {path}")

def create_config(args) -> SynapticConfig:
    """Create configuration from command line arguments"""
    return SynapticConfig(
        working_dir=args.working_dir,
        
        memory=MemoryConfig(
            model_name_or_path=args.memory_model,
            load_in_4bit=not args.no_4bit,
            enable_flash_attn=not args.no_flash_attn
        ),
        
        graph=GraphConfig(
            chunk_token_size=args.chunk_size,
            gnn_hidden_channels=args.gnn_hidden_size,
            gnn_num_layers=args.gnn_num_layers
        ),
        
        retriever=RetrieverConfig(
            model_name_or_path=args.retriever_model,
            hits=args.top_k
        )
    )

def main():
    parser = argparse.ArgumentParser(description="SynapticRAG CLI")
    
    # Mode selection
    parser.add_argument('mode', choices=['process', 'query', 'interactive'],
                       help="Operation mode")
    
    # Input/output options
    parser.add_argument('--input', '-i', type=str,
                       help="Input file or directory path")
    parser.add_argument('--query', '-q', type=str,
                       help="Query string (for query mode)")
    parser.add_argument('--output', '-o', type=str,
                       help="Output file path")
    parser.add_argument('--working-dir', type=str, default='./synaptic_workspace',
                       help="Working directory for cache and storage")
    
    # Model configuration
    parser.add_argument('--memory-model', type=str,
                       default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="Memory model name or path")
    parser.add_argument('--retriever-model', type=str,
                       default="BAAI/bge-large-en-v1.5",
                       help="Retriever model name or path")
    
    # Processing options
    parser.add_argument('--chunk-size', type=int, default=1200,
                       help="Token size for text chunks")
    parser.add_argument('--gnn-hidden-size', type=int, default=256,
                       help="Hidden size for GNN layers")
    parser.add_argument('--gnn-num-layers', type=int, default=3,
                       help="Number of GNN layers")
    parser.add_argument('--top-k', type=int, default=3,
                       help="Number of top results to retrieve")
    
    # Performance options
    parser.add_argument('--no-4bit', action='store_true',
                       help="Disable 4-bit quantization")
    parser.add_argument('--no-flash-attn', action='store_true',
                       help="Disable flash attention")
    parser.add_argument('--batch-size', type=int, default=1,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create pipeline
    config = create_config(args)
    pipeline = SynapticPipeline(config=config)
    
    try:
        if args.mode == 'process':
            if not args.input:
                raise ValueError("Input path required for process mode")
                
            # Load and process documents
            documents = load_documents(args.input)
            stats = pipeline.process_documents(
                documents,
                batch_size=args.batch_size,
                show_progress=True
            )
            
            # Save pipeline state
            pipeline.save()
            
            # Output statistics
            stats_dict = {
                "num_documents": stats.num_documents,
                "num_entities": stats.num_entities,
                "num_relationships": stats.num_relationships,
                "memory_size": stats.memory_size,
                "graph_size": stats.graph_size
            }
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(stats_dict, f, indent=2)
            else:
                print(json.dumps(stats_dict, indent=2))
                
        elif args.mode == 'query':
            if not args.query:
                raise ValueError("Query string required for query mode")
                
            # Load existing pipeline state if available
            try:
                pipeline.load()
            except FileNotFoundError:
                logger.warning("No existing pipeline state found. "
                             "Please process documents first.")
                return
                
            # Process query
            result = pipeline.query(args.query, return_sources=True)
            
            # Format output
            output_dict = {
                "response": result.response,
                "sources": result.sources,
                "metadata": result.metadata
            }
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output_dict, f, indent=2)
            else:
                print(json.dumps(output_dict, indent=2))
                
        elif args.mode == 'interactive':
            # Load existing pipeline state if available
            try:
                pipeline.load()
            except FileNotFoundError:
                logger.warning("No existing pipeline state found. "
                             "Please process documents first.")
                return
                
            print("Enter your queries (type 'exit' to quit):")
            while True:
                query = input("\nQuery: ").strip()
                if query.lower() == 'exit':
                    break
                    
                result = pipeline.query(query, return_sources=True)
                print("\nResponse:", result.response)
                print("\nSources:", result.sources)
                
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()

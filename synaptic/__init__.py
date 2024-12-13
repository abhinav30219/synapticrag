"""SynapticRAG: A hybrid RAG system combining LightRAG and MemoRAG approaches"""

# Version
__version__ = "0.1.0"

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
try:
    from .adapter import SynapticRAG, GenerationResult
    from .pipeline import SynapticPipeline, ProcessingResult
    from .config import SynapticConfig, GraphConfig, MemoryConfig, RetrieverConfig
    logger.info("Successfully imported all components")
except Exception as e:
    logger.error(f"Failed to import components: {str(e)}")
    raise

__all__ = [
    'SynapticRAG',
    'GenerationResult',
    'SynapticPipeline',
    'ProcessingResult',
    'SynapticConfig',
    'GraphConfig',
    'MemoryConfig',
    'RetrieverConfig'
]

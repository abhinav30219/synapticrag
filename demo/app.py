"""Streamlit demo for SynapticRAG with multimodal support"""
import os
import streamlit as st
import tempfile
from typing import Optional, Tuple, List
from pathlib import Path
import PyPDF2
import docx
import fitz  # PyMuPDF
from PIL import Image
import io
import torch
import numpy as np
import logging
from transformers import CLIPProcessor, CLIPModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from synaptic.config import SynapticConfig, MemoryConfig, RetrieverConfig, GraphConfig, StorageConfig
from synaptic.pipeline import SynapticPipeline
from synaptic.text_splitter import HybridTextSplitter
from synaptic.storage.kv_storage import JsonKVStorage
from synaptic.storage.vector_storage import FaissVectorStorage
from synaptic.storage.graph_storage import NetworkXStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache CLIP model loading
@st.cache_resource
def load_clip_model():
    """Load CLIP model for image embeddings"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_image_embedding(image: Image, model: CLIPModel, processor: CLIPProcessor) -> torch.Tensor:
    """Get CLIP embeddings for image"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.squeeze()

def extract_from_pdf(file_path: str) -> Tuple[str, List[Image.Image]]:
    """Extract text and images from PDF"""
    text_chunks = []
    images = []
    
    # Open PDF
    doc = fitz.open(file_path)
    
    # Extract text and images from each page
    for page in doc:
        # Extract text
        text = page.get_text()
        if text.strip():  # Only add non-empty text
            text_chunks.append(text)
        
        # Extract images
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # Convert RGBA to RGB if needed
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                images.append(image)
            except Exception as e:
                st.warning(f"Failed to process an image: {str(e)}")
                continue
    
    # Combine text chunks into single string
    full_text = "\n\n".join(text_chunks)
    return full_text, images

def read_docx(file_path: str) -> str:
    """Read text from DOCX file"""
    doc = docx.Document(file_path)
    text_chunks = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return "\n\n".join(text_chunks)

def read_txt(file_path: str) -> str:
    """Read text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def process_uploaded_file(uploaded_file) -> Tuple[Optional[str], Optional[List[Image.Image]]]:
    """Process uploaded file and return text and images"""
    if uploaded_file is None:
        return None, None
        
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
    
    try:
        # Get file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        # Process based on file type
        if file_ext == '.pdf':
            return extract_from_pdf(file_path)
        elif file_ext == '.docx':
            return read_docx(file_path), []
        elif file_ext == '.txt':
            return read_txt(file_path), []
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return None, None
    finally:
        # Clean up temporary file
        os.unlink(file_path)

def initialize_rag() -> SynapticPipeline:
    """Initialize RAG pipeline"""
    # Create storage config with storage classes and namespaces
    storage_config = StorageConfig(
        working_dir="./workspace",
        cache_dir="./cache",
        kv_storage_cls=JsonKVStorage,
        vector_storage_cls=FaissVectorStorage,
        graph_storage_cls=NetworkXStorage,
        kv_namespace="kv_storage",
        vector_namespace="vector_storage",
        graph_namespace="graph_storage",
        enable_compression=True,
        compression_level=3,
        max_cache_size=1000,
        cache_ttl=3600
    )
    
    # Create memory config
    memory_config = MemoryConfig(
        model_name_or_path="claude-3-5-sonnet-20241022",
        memory_weight=0.4,
        clue_boost=0.2,
        min_memory_relevance=0.3
    )
    
    # Create retriever config
    retriever_config = RetrieverConfig(
        model_name_or_path="text-embedding-3-large",
        vector_weight=0.3,
        graph_weight=0.3,
        enable_reranking=True
    )
    
    # Create graph config
    graph_config = GraphConfig(
        entity_confidence_threshold=0.6,
        relationship_confidence_threshold=0.7,
        enable_node2vec=True
    )
    
    # Create main config
    config = SynapticConfig(
        memory=memory_config,
        retriever=retriever_config,
        graph=graph_config,
        storage=storage_config
    )
    
    return SynapticPipeline(config)

class PrecomputedImageEmbeddings(Embeddings):
    """Custom embedding class for pre-computed image embeddings"""
    def __init__(self, embeddings_dict: dict):
        self.embeddings_dict = embeddings_dict
        self._openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return pre-computed embeddings for documents"""
        return [self.embeddings_dict[text] for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using OpenAI embeddings"""
        query_embedding = self._openai_embeddings.embed_query(text)
        # Ensure query embedding has same dimension as document embeddings
        sample_dim = len(next(iter(self.embeddings_dict.values())))
        if len(query_embedding) != sample_dim:
            # Resize query embedding if dimensions don't match
            query_embedding = np.array(query_embedding)
            query_embedding = torch.nn.functional.interpolate(
                torch.from_numpy(query_embedding).view(1, 1, -1),
                size=sample_dim,
                mode='linear'
            ).squeeze().numpy().tolist()
        return query_embedding

def main():
    st.title("SynapticRAG Demo")
    
    # Load models
    clip_model, clip_processor = load_clip_model()
    text_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Initialize text splitter
    text_splitter = HybridTextSplitter(
        chunk_token_size=1200,
        chunk_overlap_tokens=100,
        semantic_chunk_size=512,
        tiktoken_model="cl100k_base"
    )
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = initialize_rag()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Memory settings
    st.sidebar.subheader("Memory Settings")
    memory_weight = st.sidebar.slider("Memory Weight", 0.0, 1.0, 0.4)
    clue_boost = st.sidebar.slider("Clue Boost", 0.0, 1.0, 0.2)
    
    # Retrieval settings
    st.sidebar.subheader("Retrieval Settings")
    vector_weight = st.sidebar.slider("Vector Weight", 0.0, 1.0, 0.3)
    graph_weight = st.sidebar.slider("Graph Weight", 0.0, 1.0, 0.3)
    enable_reranking = st.sidebar.checkbox("Enable Reranking", True)
    
    # Update configuration
    rag = st.session_state.rag
    rag.config.memory.memory_weight = memory_weight
    rag.config.memory.clue_boost = clue_boost
    rag.config.retriever.vector_weight = vector_weight
    rag.config.retriever.graph_weight = graph_weight
    rag.config.retriever.enable_reranking = enable_reranking
    
    # File upload
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Upload a PDF, DOCX, or TXT file to analyze"
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Process file
            text, images = process_uploaded_file(uploaded_file)
            
            if text:
                logger.info(f"Processing text of length: {len(text)}")
                
                # Process with SynapticRAG
                result = st.session_state.rag.process_documents(text)
                
                # Show memory state
                st.subheader("Memory State")
                memory_state = st.session_state.rag.rag.memory_model.memory_state
                if memory_state:
                    st.metric("Memory Tokens", memory_state.memory_tokens.shape[1])
                    if memory_state.metadata:
                        st.json(memory_state.metadata)
                
                # Show graph state
                st.subheader("Knowledge Graph")
                graph_builder = st.session_state.rag.rag.graph_builder
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entities", len(graph_builder.entities))
                with col2:
                    st.metric("Relationships", len(graph_builder.relationships))
                
                # Show entity details
                with st.expander("View Entities"):
                    for entity_name, entity in graph_builder.entities.items():
                        st.text(f"{entity_name} ({entity.type})")
                        st.text(f"Source: {entity.source_id}")
                        st.text("---")
                
                # Process images if present
                if images:
                    st.subheader("Extracted Images")
                    embeddings_dict = {}
                    
                    # Display images and get embeddings
                    for i, image in enumerate(images):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(image, use_container_width=True, caption=f"Image {i+1}")
                        with col2:
                            # Get CLIP embedding and convert to numpy array
                            embedding = get_image_embedding(image, clip_model, clip_processor)
                            # Resize to match OpenAI embedding dimension (1536)
                            embedding = torch.nn.functional.interpolate(
                                embedding.unsqueeze(0).unsqueeze(0),
                                size=1536,
                                mode='linear'
                            ).squeeze().numpy()
                            
                            # Store in dictionary
                            image_key = f"image_{i}"
                            embeddings_dict[image_key] = embedding.tolist()
                            st.write(f"Generated embedding for Image {i+1}")
                    
                    # Store images in session state
                    st.session_state.images = images
                    
                    # Create custom embeddings
                    image_embeddings = PrecomputedImageEmbeddings(embeddings_dict)
                    
                    # Create image vector store with pre-computed embeddings
                    image_texts = list(embeddings_dict.keys())
                    image_vectorstore = FAISS.from_texts(
                        texts=image_texts,
                        embedding=image_embeddings,
                        metadatas=[{"type": "image", "index": i} for i in range(len(images))]
                    )
                    
                    # Store vector store in session state
                    st.session_state.image_vectorstore = image_vectorstore
                
                # Show stats
                st.success("Document processed successfully!")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Text Length", len(text))
                with col2:
                    st.metric("Images", len(images) if images else 0)
                with col3:
                    st.metric("Entities", result.num_entities)
                with col4:
                    st.metric("Relationships", result.num_relationships)
    
    # Query interface
    st.header("Ask Questions")
    
    # Query settings
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["hybrid", "memory", "graph", "vector"],
        help="Choose how to retrieve information"
    )
    
    # Show current graph state
    st.subheader("Knowledge Graph State")
    graph_builder = st.session_state.rag.rag.graph_builder
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Entities", len(graph_builder.entities))
    with col2:
        st.metric("Relationships", len(graph_builder.relationships))
    
    query = st.text_input("Enter your question")
    
    if query:
        with st.spinner("Generating answer..."):
            logger.info(f"Processing query: {query}")
            
            # Get response from SynapticRAG
            result = st.session_state.rag.query(query, retrieval_mode=retrieval_mode)
            
            # Show answer
            st.markdown("### Answer")
            st.write(result.response)
            
            # Show retrieval details
            st.subheader("Retrieval Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'memory_score' in result.metadata:
                    st.metric("Memory Score", f"{result.metadata['memory_score']:.3f}")
            with col2:
                if 'vector_score' in result.metadata:
                    st.metric("Vector Score", f"{result.metadata['vector_score']:.3f}")
            with col3:
                if 'graph_score' in result.metadata:
                    st.metric("Graph Score", f"{result.metadata['graph_score']:.3f}")
            
            # Show clues
            if 'clues' in result.metadata and result.metadata['clues']:
                st.subheader("Generated Clues")
                for i, clue in enumerate(result.metadata['clues'], 1):
                    if isinstance(clue, dict):
                        st.text(f"{i}. {clue['text']} ({clue['level']}) - Score: {clue['score']:.2f}")
                    else:
                        st.text(f"{i}. {clue}")
            
            # Show sources
            if result.sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(result.sources, 1):
                        st.text(f"Source {i}: {source}")
            
            # Show memory recall
            if 'memory_recall' in result.metadata:
                with st.expander("View Memory Recall"):
                    st.markdown("**Memory Context:**")
                    st.text(result.metadata['memory_recall'])
            
            # Show entity details
            with st.expander("View Current Entities"):
                for entity_name, entity in graph_builder.entities.items():
                    st.text(f"{entity_name} ({entity.type})")
                    st.text(f"Source: {entity.source_id}")
                    st.text("---")
            
            # Show relationship details
            with st.expander("View Current Relationships"):
                for rel in graph_builder.relationships:
                    st.text(f"{rel.source} -{rel.type}-> {rel.target}")
                    st.text(f"Source: {rel.source_id}")
                    st.text("---")
            
            # Show related images
            if 'images' in st.session_state and query:
                st.subheader("Related Images")
                images = st.session_state.images
                image_vectorstore = st.session_state.image_vectorstore
                
                # Find similar images using the custom embeddings
                image_results = image_vectorstore.similarity_search(query, k=2)
                
                # Display similar images
                for img_result in image_results:
                    img_index = int(img_result.page_content.split('_')[1])
                    st.image(images[img_index], caption=f"Image {img_index + 1}", use_container_width=True)

if __name__ == "__main__":
    main()

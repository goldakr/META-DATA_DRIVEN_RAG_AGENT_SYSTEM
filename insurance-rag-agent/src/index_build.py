from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import SETTINGS
from ingest import ingest_directory

def build_index(data_dir: Path, storage_dir: Path):
    """Build and persist a VectorStoreIndex using ChromaDB from all PDFs under data_dir."""
    # Prepare nodes with anchors/metadata and table registry
    result = ingest_directory(data_dir, storage_dir)
    nodes = result["nodes"]

    # Configure global settings
    Settings.llm = OpenAI(model=SETTINGS.chat_model)
    Settings.embed_model = OpenAIEmbedding(model=SETTINGS.embed_model)

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(storage_dir / "chroma_db"))
    
    # Create ChromaDB collection
    chroma_collection = chroma_client.get_or_create_collection(
        name=SETTINGS.chroma_collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    # Create ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context with ChromaDB
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index with ChromaDB
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    # Persist the index to storage
    index.storage_context.persist(persist_dir=storage_dir)
    
    return {"count": len(nodes)}

def load_or_build(storage_dir: Path):
    """Load existing ChromaDB index or return None if not found."""
    chroma_db_path = storage_dir / "chroma_db"
    
    if chroma_db_path.exists():
        try:
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
            
            # Get existing collection
            chroma_collection = chroma_client.get_collection(
                name=SETTINGS.chroma_collection_name
            )
            
            # Create ChromaVectorStore
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create storage context with ChromaDB
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load index from ChromaDB
            return load_index_from_storage(storage_context)
        except Exception as e:
            print(f"Warning: Could not load existing ChromaDB index: {e}")
            return None
    return None

from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Settings:
    # Models
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Retrieval sizes
    dense_k: int = 10           # dense top-k
    sparse_k: int = 10          # BM25 top-k
    candidate_k: int = 40       # pool before rerank (e.g., 20-40)
    rerank_top_n: int = 8       # final chunks after rerank

    # Chunk budget (whichever is smaller): 5% of doc or 10 chunks
    max_chunks: int = 10
    chunk_budget_pct: float = 0.05

    # Storage
    storage_dir: Path = Path("./storage")
    tables_dir: Path = storage_dir / "tables"
    table_registry: Path = tables_dir / "registry.json"
    
    # ChromaDB settings
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "insurance_rag_collection")
    chroma_db_path: Path = storage_dir / "chroma_db"

    # General
    seed: int = 42

SETTINGS = Settings()

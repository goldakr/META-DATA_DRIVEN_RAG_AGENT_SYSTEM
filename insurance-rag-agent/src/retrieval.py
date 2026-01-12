from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
import math

from llama_index.core import load_index_from_storage, StorageContext, VectorStoreIndex  # pyright: ignore[reportMissingImports]
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.retrievers.bm25 import BM25Retriever  # from llama-index-retrievers-bm25

from config import SETTINGS

@dataclass
class HybridResult:
    candidates: List[NodeWithScore]
    reranked: List[NodeWithScore]

def _load_index(storage_dir: Path) -> VectorStoreIndex:
    """Load VectorStoreIndex from ChromaDB storage."""
    chroma_db_path = storage_dir / "chroma_db"
    
    if not chroma_db_path.exists():
        raise FileNotFoundError(f"ChromaDB not found at {chroma_db_path}. Please build the index first.")
    
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
    
    # Set the embedding model for queries
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import Settings
    Settings.embed_model = OpenAIEmbedding(model=SETTINGS.embed_model)
    
    # Create index from ChromaDB collection (since ChromaDB stores vectors directly)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

def hybrid_retrieve(
    query: str,
    storage_dir: Path,
    filters: Optional[MetadataFilters] = None,
) -> HybridResult:
    """Dense (vector) + sparse (BM25) retrieval, fuse, LLM rerank to top-n.
       Enforces chunk budget (~5% per doc, max 10 chunks) before returning."""
    index = _load_index(storage_dir)

    # Dense retriever
    dense = index.as_retriever(similarity_top_k=SETTINGS.candidate_k, filters=filters)

    # Sparse BM25 retriever built on the same nodes
    # Note: For ChromaDB, we need to get nodes from the docstore
    bm25 = None
    try:
        # Get all nodes from the index for BM25 initialization
        # Try different ways to access the nodes
        all_nodes = []
        
        # Method 1: Try docstore
        if hasattr(index, 'docstore') and index.docstore:
            all_nodes = list(index.docstore.docs.values())
        
        # Method 2: If docstore is empty, try to get nodes from the vector store
        if not all_nodes and hasattr(index, 'vector_store'):
            try:
                # Get nodes from ChromaDB collection
                chroma_collection = index.vector_store._collection
                results = chroma_collection.get(include=['metadatas', 'documents'])
                if results['documents']:
                    from llama_index.core.schema import TextNode
                    for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
                        if doc and doc.strip():  # Only include non-empty documents
                            node = TextNode(text=doc, metadata=meta)
                            all_nodes.append(node)
            except Exception as e:
                print(f"Warning: Could not get nodes from vector store: {e}")
        
        if all_nodes:
            # Create BM25 retriever with the actual nodes
            bm25 = BM25Retriever.from_defaults(
                nodes=all_nodes, 
                similarity_top_k=SETTINGS.candidate_k
            )
            print(f"BM25 retriever initialized with {len(all_nodes)} nodes")
        else:
            print("Warning: No documents found for BM25 initialization - using dense-only retrieval")
    except Exception as e:
        print(f"Warning: BM25 retriever failed: {e}")
        # Fallback to dense-only retrieval
        bm25 = None

    dense_nodes = dense.retrieve(query)
    sparse_nodes = bm25.retrieve(query) if bm25 else []

    # Reciprocal Rank Fusion (RRF) fusion
    def rrf_score(rank: int, k: float = 60.0) -> float:
        return 1.0 / (k + rank)

    scores: Dict[str, float] = {}
    ranked: List[NodeWithScore] = []
    for rank, n in enumerate(dense_nodes, start=1):
        scores[n.node.node_id] = scores.get(n.node.node_id, 0.0) + rrf_score(rank)
    for rank, n in enumerate(sparse_nodes, start=1):
        scores[n.node.node_id] = scores.get(n.node.node_id, 0.0) + rrf_score(rank)
    # Map back to unique NodeWithScore
    by_id: Dict[str, NodeWithScore] = {n.node.node_id: n for n in dense_nodes + sparse_nodes}
    fused = sorted(by_id.values(), key=lambda n: scores.get(n.node.node_id, 0.0), reverse=True)

    # LLM rerank for precision
    reranker = LLMRerank(top_n=SETTINGS.rerank_top_n, choice_batch_size=8, llm=OpenAI(model=SETTINGS.chat_model))
    reranked = reranker.postprocess_nodes(fused, query_str=query)

    # enforce chunk budget: max 10 chunks AND ≤ 5% per document
    final_nodes = _apply_chunk_budget(reranked, max_chunks=SETTINGS.max_chunks, pct=SETTINGS.chunk_budget_pct)

    return HybridResult(candidates=fused, reranked=final_nodes)

def _apply_chunk_budget(nodes: List[NodeWithScore], max_chunks: int, pct: float) -> List[NodeWithScore]:
    from collections import defaultdict
    per_file = defaultdict(list)
    for n in nodes:
        per_file[n.node.metadata.get("FileName", "unknown")].append(n)

    allowed: List[NodeWithScore] = []
    # allow up to ceil(pct * total nodes for that file) per file
    for items in per_file.values():
        total = len(items)
        cap = max(1, math.ceil(total * pct))
        allowed.extend(items[:cap])

    # Global cap
    return allowed[:max_chunks]

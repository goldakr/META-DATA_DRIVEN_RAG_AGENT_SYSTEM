
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import re

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import SimpleDirectoryReader, Document
import pdfplumber
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from utils import Anchor, make_node, extract_incident_fields, summarize_table, keywords, ensure_json, save_registry
from config import SETTINGS


def find_pdfs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.pdf") if p.is_file()]


def extract_tables_for_pdf(pdf_path: Path, table_out_dir: Path, pages: str = "all") -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        table_out_dir: Directory to save extracted CSV files
        pages: Pages to process ("all" or specific page numbers)
        
    Returns:
        List of dictionaries containing table metadata
    """
    return extract_tables_with_pdfplumber(pdf_path, table_out_dir, pages)

def extract_tables_with_pdfplumber(pdf_path: Path, table_out_dir: Path, pages: str = "all") -> List[Dict[str, Any]]:
    """
    Use pdfplumber to read tables. For each page, attempt to extract grid-based tables.
    Save each table to CSV and register with TableId + brief summary.
    """
    entries: List[Dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p_idx, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables, start=1):
                # Clean rows and normalize lengths
                rows = []
                max_len = 0
                for row in table:
                    if row is None: 
                        continue
                    row = [("" if c is None else str(c).strip()) for c in row]
                    max_len = max(max_len, len(row))
                    rows.append(row)
                rows = [r + [""]*(max_len-len(r)) for r in rows] if rows else []
                import pandas as pd
                df = pd.DataFrame(rows)
                # If first row looks like headers, set them
                if len(df) > 1 and all(isinstance(x, str) and len(x) < 64 for x in df.iloc[0].tolist()):
                    df.columns = [c if c not in [None, ""] else f"col_{i+1}" for i, c in enumerate(df.iloc[0].tolist())]
                    df = df.iloc[1:].reset_index(drop=True)
                table_id = f"Table-{pdf_path.stem}-{p_idx}-{t_idx}"
                csv_path = table_out_dir / f"{table_id}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8")
                from .utils import summarize_table
                entries.append({
                    "table_id": table_id,
                    "page": p_idx,
                    "csv": str(csv_path),
                    "file_name": pdf_path.name,
                    "method": "pdfplumber",
                    "summary": summarize_table(df)
                })
    return entries

def chunk_pdf_text(pdf_path: Path) -> List[TextNode]:
    """Load text with SimpleDirectoryReader and chunk by sentences, attaching anchors/metadata."""
    docs: List[Document] = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
    splitter = SentenceSplitter(chunk_size=450, chunk_overlap=60)  # ~250-500 tokens target
    nodes: List[TextNode] = []
    for d in docs:
        base_meta = extract_incident_fields(d.text)
        # naive page tag if present in metadata
        page = int(d.metadata.get("page_label", d.metadata.get("page", 1)) or 1)
        chunks = splitter.split_text(d.text)
        for idx, chunk in enumerate(chunks, start=1):
            anchor = Anchor(page=page, section_type=_infer_section_type(chunk))
            extra = dict(base_meta)
            extra["Keywords"] = keywords(chunk)
            nodes.append(make_node(chunk, pdf_path.name, anchor, extra=extra))
    return nodes

def _infer_section_type(text: str) -> str:
    t = text.lower()
    if "timeline" in t or re.search(r"\b\d{1,2}:\d{2}\b", t):
        return "Timeline"
    if "conclusion" in t or "summary" in t:
        return "Summary"
    if "policy" in t:
        return "Policy"
    if "report" in t:
        return "Report"
    return "Body"

def ingest_directory(data_dir: Path, storage_dir: Path) -> Dict[str, Any]:
    """
    End-to-end ingestion: extract text chunks + table CSVs, persist table registry.
    
    Args:
        data_dir: Directory containing PDF files
        storage_dir: Directory to store extracted data
    """
    storage_dir.mkdir(parents=True, exist_ok=True)
    table_dir = storage_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table_registry = ensure_json(storage_dir / "tables" / "registry.json")
    all_nodes: List[TextNode] = []

    for pdf in find_pdfs(data_dir):
        # text chunks
        all_nodes.extend(chunk_pdf_text(pdf))
        # tables
        entries = extract_tables_for_pdf(pdf, table_dir, pages="all")
        for e in entries:
            table_registry[e["table_id"]] = e

    save_registry(storage_dir / "tables" / "registry.json", table_registry)
    return {"nodes": all_nodes, "table_registry": table_registry}

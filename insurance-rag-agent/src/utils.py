from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import re, json
import pandas as pd

from llama_index.core.schema import TextNode


MANDATORY_ANCHORS = ("PageNumber", "SectionType")

@dataclass
class Anchor:
    page: int
    section_type: str  # e.g., Summary, Timeline, Table, Figure, Analysis, Conclusion
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    position: Optional[str] = None  # row/col or heading path

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_incident_fields(text: str) -> Dict[str, Any]:
    """Heuristic extraction of IncidentType/IncidentDate/ClientId/CaseId from raw text."""
    incident_type = None
    m = re.search(r"(accident|collision|burglary|theft|fire|vandalism)", text, re.I)
    if m:
        incident_type = m.group(1).lower()

    incident_date = None
    m = re.search(r"(?:Incident|Accident|Burglary)\s*Date[:\s]+([0-9]{1,2}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})", text, re.I)
    if m:
        incident_date = m.group(1)

    client_id = None
    m = re.search(r"(Client(?:\s*ID)?|Policyholder\s*ID)\s*[:#]\s*([A-Za-z0-9\-]+)", text, re.I)
    if m:
        client_id = m.group(2)

    case_id = None
    m = re.search(r"(Case(?:\s*ID)?|Claim\s*#?)\s*[:#]\s*([A-Za-z0-9\-]+)", text, re.I)
    if m:
        case_id = m.group(2)

    return {
        "IncidentType": incident_type,
        "IncidentDate": incident_date,
        "ClientId": client_id,
        "CaseId": case_id,
    }

def summarize_table(df: pd.DataFrame, max_cols: int = 6, max_rows: int = 6) -> str:
    """Generate a brief textual description of a table for better semantic retrieval."""
    preview = df.iloc[:max_rows, :max_cols].to_csv(index=False)
    shape = f"{df.shape[0]}x{df.shape[1]}"
    return f"Table with shape {shape}. Preview (CSV):\n{preview}"

def make_node(
    text: str,
    file_name: str,
    anchor: Anchor,
    extra: Optional[Dict[str, Any]] = None,
) -> TextNode:
    metadata: Dict[str, Any] = {
        "FileName": file_name,
        "PageNumber": anchor.page,
        "SectionType": anchor.section_type,
    }
    if anchor.table_id:
        metadata["TableId"] = anchor.table_id
    if anchor.figure_id:
        metadata["FigureId"] = anchor.figure_id
    if anchor.position:
        metadata["Position"] = anchor.position
    if extra:
        # Convert Keywords list to string for ChromaDB compatibility
        if "Keywords" in extra and isinstance(extra["Keywords"], list):
            extra["Keywords"] = ", ".join(extra["Keywords"])
        metadata.update(extra)

    node = TextNode(text=text, metadata=metadata)
    return node

def keywords(text: str, top_k: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-/]+", text.lower())
    # drop very short and common terms
    stop = set("the a an and or of for to in on at by from with without be is are was were".split())
    words = [w for w in words if len(w) > 2 and w not in stop]
    # crude tf
    from collections import Counter
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_k)]

def ensure_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_registry(path: Path, registry: Dict[str, Any]) -> None:
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

@dataclass
class AgentAnswer:
    text: str
    anchors: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]

def _format_anchor(n) -> Dict[str, Any]:
    """Format a NodeWithScore object into a dictionary for anchor display."""
    if hasattr(n, 'node'):
        # NodeWithScore object
        md = n.node.metadata
        score = getattr(n, "score", None)
    else:
        # TextNode object
        md = n.metadata
        score = getattr(n, "score", None)
    
    return {
        "FileName": md.get("FileName"),
        "PageNumber": md.get("PageNumber"),
        "SectionType": md.get("SectionType"),
        "TableId": md.get("TableId"),
        "FigureId": md.get("FigureId"),
        "Score": score,
    }

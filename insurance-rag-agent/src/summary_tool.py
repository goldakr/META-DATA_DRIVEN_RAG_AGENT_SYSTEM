from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.openai import OpenAI

from config import SETTINGS
from retrieval import hybrid_retrieve
from utils import _format_anchor, AgentAnswer


def summary_tool(query: str, storage_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Summarize selected parts of retrieved info using map-reduce approach. 
    Use this tool when you need to provide a comprehensive summary of information related to the query."""
    print("From summary_agent_tool (map-reduce).")
    storage_path = Path(storage_dir)
    
    # Build metadata filters if provided
    filters = None
    if metadata:
        filt_list = []
        for k, v in metadata.items():
            if v is None:
                continue
            filt_list.append(ExactMatchFilter(key=k, value=v))
        if filt_list:
            filters = MetadataFilters(filters=filt_list)
    
    hr = hybrid_retrieve(query, storage_path, filters)
    
    # Handle both NodeWithScore and TextNode objects from reranker
    nodes_for_processing = []
    for n in hr.reranked:
        if hasattr(n, 'node'):
            # NodeWithScore object
            nodes_for_processing.append(n)
        else:
            # TextNode object - create a NodeWithScore wrapper
            from llama_index.core.schema import NodeWithScore
            mock_nws = NodeWithScore(node=n, score=getattr(n, 'score', 0.0))
            nodes_for_processing.append(mock_nws)
    
    # Map-Reduce Summary Implementation
    llm = OpenAI(model=SETTINGS.chat_model)
    
    # MAP PHASE: Process each node individually
    map_prompt = f"""
    You are a helpful assistant that summarizes information related to a specific query.
    
    Query: {query}
    
    Please provide a concise summary of the following text that is relevant to the query.
    Focus on key information, facts, and details that directly relate to the query.
    Be inclusive - if the text contains any information that could be related to the query (even tangentially), include it.
    Only exclude text if it is completely unrelated to the query topic.
    
    IMPORTANT: When summarizing, be precise about different people mentioned. Do not conflate or confuse different individuals. If the text mentions multiple people, clearly distinguish between them in your summary.
    
    Text to summarize:
    {{text}}
    
    Summary:
    """
    map_summaries = []
    for n in nodes_for_processing[:10]:  # Limit to top 10 nodes for efficiency
        if hasattr(n, 'node'):
            text = n.node.get_content()
        else:
            text = n.get_content()
        
        # Truncate text if too long for individual processing
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        map_prompt_filled = map_prompt.format(text=text)
        response = llm.complete(map_prompt_filled)
        map_summary = getattr(response, "text", str(response)).strip()
        
        # Debug: Print what we're getting from the map phase
        print(f"Map phase result: {map_summary[:100]}...")
        
        # Only include summaries that are relevant
        if "Not directly relevant" not in map_summary and "not relevant" not in map_summary.lower():
            map_summaries.append(map_summary)
    
    # REDUCE PHASE: Combine all map summaries into final summary
    if not map_summaries:
        final_summary = "No relevant information found for the given query."
    else:
        reduce_prompt = f"""
        You are a helpful assistant that creates a comprehensive summary by combining multiple partial summaries.
        
        Query: {query}
        
        Please create a coherent, comprehensive summary by combining the following partial summaries.
        Remove redundancy, organize information logically, and ensure the final summary directly addresses the query.
        
        IMPORTANT: Be very careful to distinguish between different people mentioned in the summaries. Do not conflate or confuse different individuals. If multiple people are mentioned, clearly identify who is who and what role each person played.
        
        Partial summaries to combine:
        {chr(10).join([f"{i+1}. {summary}" for i, summary in enumerate(map_summaries)])}
        
        Comprehensive Summary:
        """
        
        reduce_response = llm.complete(reduce_prompt)
        final_summary = getattr(reduce_response, "text", str(reduce_response)).strip()
    
    # Format anchors, handling both object types
    anchors = []
    for n in hr.reranked:
        if hasattr(n, 'node'):
            # NodeWithScore object
            anchors.append(_format_anchor(n))
        else:
            # TextNode object - create a proper NodeWithScore for _format_anchor
            from llama_index.core.schema import NodeWithScore
            mock_nws = NodeWithScore(node=n, score=getattr(n, 'score', 0.0))
            anchors.append(_format_anchor(mock_nws))
    
    result = AgentAnswer(text=final_summary, anchors=anchors, tables=[])
    return json.dumps({
        "text": result.text,
        "anchors": result.anchors,
        "tables": result.tables
    })

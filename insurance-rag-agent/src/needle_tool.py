from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI

from config import SETTINGS
from retrieval import hybrid_retrieve
from utils import _format_anchor, AgentAnswer


def needle_tool(query: str, storage_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Return the most precise paragraph/anchor. Use this tool when you need exact quotes, specific page references, or precise information from documents."""
    print("From needle_agent_tool.")
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
    
    # Use compact mode for concise responses
    synth = get_response_synthesizer(response_mode="compact", use_async=False, llm=OpenAI(model=SETTINGS.chat_model))
    
    # Handle both NodeWithScore and TextNode objects from reranker
    nodes_for_synthesis = []
    for n in hr.reranked:
        if hasattr(n, 'node'):
            # NodeWithScore object - pass the whole object to synthesizer
            nodes_for_synthesis.append(n)
        else:
            # TextNode object (from LLM reranker) - create a NodeWithScore wrapper
            from llama_index.core.schema import NodeWithScore
            mock_nws = NodeWithScore(node=n, score=getattr(n, 'score', 0.0))
            nodes_for_synthesis.append(mock_nws)
    
    if not nodes_for_synthesis:
        answer_text = "Empty Response"
    else:
        try:
            # Create a custom synthesis prompt that emphasizes direct quoting and faithfulness
            synthesis_prompt = f"""Based on the provided context documents, provide a direct answer to the question. Use the exact wording from the sources when possible:

                                Question: {query}

                                Instructions:
                                - Use the exact text from the context documents
                                - Do not add any information not present in the sources
                                - If you must paraphrase, stay as close to the original wording as possible
                                - If the answer is not available in the context, say "Information not available in the provided documents"

                                Answer:"""
            
            answer = synth.synthesize(synthesis_prompt, nodes=nodes_for_synthesis)
            answer_text = answer.response if answer and hasattr(answer, 'response') else "No response generated"
        except Exception as e:
            print(f"Error in synthesis: {e}")
            answer_text = f"Error in synthesis: {str(e)}"
    
    # Format anchors, handling both object types
    anchors = []
    for n in hr.reranked:
        if hasattr(n, 'node'):
            # NodeWithScore object
            anchors.append(_format_anchor(n))
        else:
            mock_nws = NodeWithScore(node=n, score=getattr(n, 'score', 0.0))
            anchors.append(_format_anchor(mock_nws))
    
    result = AgentAnswer(text=answer_text, anchors=anchors, tables=[])
    return json.dumps({
        "text": result.text,
        "anchors": result.anchors,
        "tables": result.tables
    })

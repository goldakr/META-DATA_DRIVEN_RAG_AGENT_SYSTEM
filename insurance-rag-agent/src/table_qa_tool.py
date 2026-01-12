from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.openai import OpenAI

from config import SETTINGS
from retrieval import hybrid_retrieve
from utils import ensure_json, _format_anchor, AgentAnswer


def table_qa_tool(query: str, storage_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Handle tabular queries; first searches extracted tables directly, then falls back to hybrid_retrieve if no relevant information found."""
    print("From table_qa_agent_tool.")
    storage_path = Path(storage_dir)
    
    # Step 1: First try to answer using extracted tables directly
    registry = ensure_json(storage_path / "tables" / "registry.json")
    
    # Load all available tables
    all_tables = []
    for table_id, table_info in registry.items():
        csv_path = table_info["csv"]
        try:
            df = pd.read_csv(csv_path)
            all_tables.append({
                "table_id": table_id,
                "df": df,
                "info": table_info,
                "csv_path": csv_path
            })
        except Exception as e:
            print(f"Error loading table {table_id}: {e}")
            continue
    
    # Step 2: Use LLM to determine which tables are relevant to the query
    llm = OpenAI(model=SETTINGS.chat_model)
    
    # Create table summaries for LLM to analyze
    table_summaries = []
    for table in all_tables:
        summary = f"Table ID: {table['table_id']}\n"
        summary += f"File: {table['info']['file_name']}\n"
        summary += f"Columns: {', '.join(table['df'].columns.tolist())}\n"
        summary += f"Rows: {len(table['df'])}\n"
        summary += f"Preview: {table['df'].head(3).to_csv(index=False)[:500]}\n"
        table_summaries.append(summary)
    
    # Ask LLM which tables are relevant
    relevance_prompt = f"""Given this query: "{query}"

                        Here are available tables:
                        {chr(10).join(table_summaries)}

                        Which table IDs are most relevant to answering this query?
                        Respond with only the table IDs separated by commas, or "none" if no tables are relevant.
                        Relevant table IDs:"""
    
    try:
        relevance_response = llm.complete(relevance_prompt)
        relevant_table_ids = [tid.strip() for tid in relevance_response.text.strip().split(',') if tid.strip() and tid.strip() != "none"]
    except Exception as e:
        print(f"Error in relevance analysis: {e}")
        relevant_table_ids = []
    
    # Step 3: If we found relevant tables, try to answer using them
    if relevant_table_ids:
        relevant_tables = [t for t in all_tables if t['table_id'] in relevant_table_ids]
        
        # Create detailed context from relevant tables
        table_contexts = []
        referenced = []
        
        for table in relevant_tables:
            table_context = f"Table: {table['table_id']} (from {table['info']['file_name']})\n"
            table_context += f"Columns: {', '.join(table['df'].columns.tolist())}\n"
            table_context += f"Data:\n{table['df'].to_csv(index=False)}\n"
            table_contexts.append(table_context)
            referenced.append({
                "TableId": table['table_id'], 
                "csv": table['csv_path'],
                "file_name": table['info']['file_name']
            })
        
        # Ask LLM to answer using the relevant tables with faithfulness focus
        answer_prompt = f"""Question: {query}

                            Here are the relevant tables:
                            {chr(10).join(table_contexts)}

                            CRITICAL INSTRUCTIONS:
                            - Answer the question using only the data provided in the tables above
                            - Base your answer exclusively on the table data - do not add external information
                            - Quote exact values from the tables when possible
                            - If the answer is not available in the tables, say "The information is not available in the provided tables."
                            - Do not make assumptions or inferences beyond what is explicitly shown in the data

                            Answer:"""
        
        try:
            answer_response = llm.complete(answer_prompt)
            answer_text = answer_response.text.strip()
            
            # Check if the answer indicates no relevant information
            if "not available" in answer_text.lower() or "no information" in answer_text.lower():
                print("No relevant information found in tables, falling back to hybrid_retrieve...")
                return _fallback_to_hybrid_retrieve(query, storage_path, metadata)
            else:
                # Create mock anchors for table references
                anchors = []
                for table in relevant_tables:
                    anchors.append({
                        "FileName": table['info']['file_name'],
                        "PageNumber": table['info'].get('page', 'Unknown'),
                        "SectionType": "Table",
                        "TableId": table['table_id'],
                        "FigureId": None,
                        "Score": 10.0
                    })
                
                result = AgentAnswer(text=answer_text, anchors=anchors, tables=referenced)
                return json.dumps({
                    "text": result.text,
                    "anchors": result.anchors,
                    "tables": result.tables
                })
                
        except Exception as e:
            print(f"Error in table-based answer: {e}")
            return _fallback_to_hybrid_retrieve(query, storage_path, metadata)
    
    # Step 4: Fallback to hybrid_retrieve if no relevant tables found
    print("No relevant tables found, falling back to hybrid_retrieve...")
    return _fallback_to_hybrid_retrieve(query, storage_path, metadata)


def _fallback_to_hybrid_retrieve(query: str, storage_path: Path, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Fallback method using hybrid_retrieve when tables don't contain relevant information."""
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
    # find any nodes that reference a table
    table_nodes = []
    for n in hr.candidates:
        if hasattr(n, 'node'):
            # NodeWithScore object
            if n.node.metadata.get("SectionType") == "Table" or n.node.metadata.get("TableId"):
                table_nodes.append(n)
        else:
            # TextNode object
            if n.metadata.get("SectionType") == "Table" or n.metadata.get("TableId"):
                table_nodes.append(n)
    
    registry = ensure_json(storage_path / "tables" / "registry.json")

    dfs = []
    referenced = []
    for n in table_nodes[:5]:
        if hasattr(n, 'node'):
            # NodeWithScore object
            tid = n.node.metadata.get("TableId")
        else:
            # TextNode object
            tid = n.metadata.get("TableId")
            
        if tid and tid in registry:
            csv_path = registry[tid]["csv"]
            try:
                df = pd.read_csv(csv_path)
                dfs.append(df)
                referenced.append({"TableId": tid, "csv": csv_path})
            except Exception:
                pass

    # faithful LLM synthesis over small table previews
    previews = []
    for df in dfs:
        previews.append(df.to_csv(index=False)[:2000])  # limit preview
    context = "\n\n".join(previews) if previews else "No tables found."
    
    prompt = f"""Question: {query}

                Here are CSV previews of possibly relevant tables: {context}

                CRITICAL INSTRUCTIONS:
                - Answer using only the data provided in the table previews above
                - Base your answer exclusively on the table data - do not add external information
                - Quote exact values from the tables when possible
                - If the answer is not available in the tables, say "The information is not available in the provided tables."
                - Do not make assumptions or inferences beyond what is explicitly shown in the data
                - Reference TableIds when citing specific data

                Answer:"""
    
    # Create a custom LLM with faithfulness-focused instructions
    faithful_llm = OpenAI(
        model=SETTINGS.chat_model,
        system_prompt="""You are a precise information retrieval assistant. Your task is to provide accurate, faithful answers based ONLY on the provided source documents. 

                    CRITICAL RULES:
                    - Base your answer exclusively on the information provided in the source documents
                    - Do not add information not present in the sources
                    - Do not make assumptions or inferences beyond what is explicitly stated
                    - If information is not available in the sources, clearly state this
                    - Quote directly from the sources when possible
                    - Maintain the exact meaning and context from the original documents
                    - Do not paraphrase in ways that change the original meaning

                    Your response should be a direct, faithful representation of the source material."""
    )
    resp = faithful_llm.complete(prompt)  # simple completion-style call from LlamaIndex OpenAI wrapper
    text = getattr(resp, "text", str(resp))

    anchors = [_format_anchor(n) for n in table_nodes[:3]]
    result = AgentAnswer(text=text, anchors=anchors, tables=referenced)
    return json.dumps({
        "text": result.text,
        "anchors": result.anchors,
        "tables": result.tables
    })
from __future__ import annotations
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import pandas as pd
import numpy as np

from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.llms.openai import OpenAI

from config import SETTINGS
from retrieval import hybrid_retrieve
from utils import _format_anchor, AgentAnswer, ensure_json


def calculate_correlation_matrix(tables: List[Dict[str, Any]]) -> str:
    """Calculate correlation matrix for numerical columns across multiple tables."""
    try:
        # Combine all numerical data from all tables
        all_numerical_data = []
        table_info = []
        
        for table in tables:
            df = table['df']
            table_id = table['table_id']
            
            # Select only numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numerical_cols:
                # Add table prefix to column names to avoid conflicts
                df_numerical = df[numerical_cols].copy()
                df_numerical.columns = [f"{table_id}_{col}" for col in numerical_cols]
                
                all_numerical_data.append(df_numerical)
                table_info.append({
                    'table_id': table_id,
                    'columns': numerical_cols,
                    'prefixed_columns': df_numerical.columns.tolist()
                })
        
        if not all_numerical_data:
            return "No numerical data found for correlation analysis."
        
        # Combine all numerical data
        combined_df = pd.concat(all_numerical_data, axis=1, sort=False)
        
        # Calculate correlation matrix
        correlation_matrix = combined_df.corr()
        
        # Format the correlation matrix for display
        result = "CORRELATION MATRIX ANALYSIS:\n"
        result += "=" * 50 + "\n\n"
        
        # Show correlation matrix
        result += "Correlation Matrix:\n"
        result += correlation_matrix.to_string() + "\n\n"
        
        # Find strong correlations (|r| > 0.7)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if not np.isnan(corr_value) and abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            result += "Strong Correlations (|r| > 0.7):\n"
            for corr in strong_correlations:
                result += f"  {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}\n"
        else:
            result += "No strong correlations found (|r| > 0.7).\n"
        
        # Add table information
        result += "\nTable Information:\n"
        for info in table_info:
            result += f"  {info['table_id']}: {', '.join(info['columns'])}\n"
        
        return result
        
    except Exception as e:
        return f"Error calculating correlation matrix: {str(e)}"


def statistics_tool(query: str, storage_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Statistics tool for comparing and analyzing data across multiple tables. 
    Use this tool when the query requires comparison, analysis, or statistics from multiple tables.
    This tool leverages table_qa_tool functionality for comprehensive multi-table analysis."""
    print("From statistics_agent_tool.")
    storage_path = Path(storage_dir)
    
    # Step 1: Load all available tables for multi-table analysis
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
    
    if not all_tables:
        print("No tables available for statistics analysis, falling back to hybrid_retrieve...")
        return _fallback_to_hybrid_retrieve(query, storage_path, metadata)
    
    # Step 2: Use LLM to determine which tables are relevant for statistical analysis
    llm = OpenAI(model=SETTINGS.chat_model)
    
    # Create detailed table summaries for multi-table analysis
    table_summaries = []
    for table in all_tables:
        summary = f"Table ID: {table['table_id']}\n"
        summary += f"File: {table['info']['file_name']}\n"
        summary += f"Columns: {', '.join(table['df'].columns.tolist())}\n"
        summary += f"Rows: {len(table['df'])}\n"
        summary += f"Data Types: {dict(table['df'].dtypes)}\n"
        summary += f"Preview:\n{table['df'].head(3).to_csv(index=False)}\n"
        table_summaries.append(summary)
    
    # Ask LLM which tables are relevant for statistical analysis
    relevance_prompt = f"""Given this query: "{query}"

                        Here are available tables for statistical analysis:
                        {chr(10).join(table_summaries)}

                        Which table IDs are relevant for answering this statistical/comparative query? 

                        IMPORTANT: Be INCLUSIVE in your selection. Consider ANY table that might contain relevant data, including:
                        - Payment reports (for settlement amounts, payment patterns, totals)
                        - Insurance policies (for deductibles, coverage limits, premiums)
                        - Accident cases (for settlement data, case information)

                        For queries about:
                        - "settlement amounts" → include ALL payment report tables
                        - "payment patterns" → include ALL payment report tables  
                        - "deductibles" → include ALL insurance policy tables (Table-*Insurance_Policy*)
                        - "coverage" → include ALL insurance policy tables (Table-*Insurance_Policy*)
                        - "premiums" → include ALL insurance policy tables (Table-*Insurance_Policy*)
                        - "totals" or "sums" → include ALL relevant tables with numerical data
                        - "highest", "maximum", "lowest", "minimum", "correlation" → include ALL relevant tables

                        Respond with ALL relevant table IDs separated by commas. Only respond "none" if absolutely no tables contain any relevant data.

                        Relevant table IDs:"""
    
    try:
        relevance_response = llm.complete(relevance_prompt)
        relevant_table_ids = [tid.strip() for tid in relevance_response.text.strip().split(',') if tid.strip() and tid.strip() != "none"]
        
        # If LLM returns "none" but query contains statistical keywords, include all tables
        query_lower = query.lower()
        statistical_keywords = [
            'average', 'total', 'sum', 'compare', 'analysis', 'pattern', 'statistic', 
            'settlement', 'payment', 'deductible', 'coverage', 'premium', 'amount', 'correlation',
            'highest', 'maximum', 'lowest', 'minimum', 'across', 'between', 'difference'
        ]
        has_keywords = any(keyword in query_lower for keyword in statistical_keywords)
        
        if not relevant_table_ids and has_keywords:
            print("LLM returned no relevant tables, but query contains statistical keywords. Including all tables for analysis.")
            relevant_table_ids = [table['table_id'] for table in all_tables]
            
    except Exception as e:
        print(f"Error in relevance analysis: {e}")
        relevant_table_ids = []
    
    # Step 3: If we found relevant tables, perform statistical analysis
    if relevant_table_ids and len(relevant_table_ids) > 1:
        # Multi-table analysis for statistics
        relevant_tables = [t for t in all_tables if t['table_id'] in relevant_table_ids]
        
        # Create comprehensive context from all relevant tables
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
        
        # Check if query is about correlation
        query_lower = query.lower()
        correlation_keywords = ['correlation', 'correlate', 'relationship', 'association', 'related', 'link']
        is_correlation_query = any(keyword in query_lower for keyword in correlation_keywords)
        
        # Calculate correlation matrix only for correlation queries
        correlation_analysis = ""
        if is_correlation_query:
            correlation_analysis = calculate_correlation_matrix(relevant_tables)
        
        # Ask LLM to perform statistical analysis across tables
        analysis_prompt = f"""Question: {query}

                            Here are the relevant tables for statistical analysis:
                            {chr(10).join(table_contexts)}

                            {correlation_analysis}

                            Perform a comprehensive statistical analysis and comparison across these tables.
                            IMPORTANT: Base your analysis only on the data provided in the tables above. Do not make assumptions or add information not present in the data.

                            Provide:
                            1. Key statistics and metrics (use exact numbers from the tables)
                            2. Comparisons between different data sets (based on actual table data)
                            3. Trends and patterns (derived from the provided data)
                            4. Summary insights (grounded in the table information)"""

        # Add correlation analysis instruction only for correlation queries
        if is_correlation_query:
            analysis_prompt += """
                            5. Analysis of the correlation matrix provided above (based on actual table dataa)"""
        else:
            analysis_prompt += """
                            5. Additional insights and recommendations based on the data (derived from the provided data)"""

        analysis_prompt += """

                            IMPORTANT: Only say "NO_DATA_AVAILABLE" if absolutely no relevant information can be found in any of the provided tables. 
                            Otherwise, provide a complete analysis using the available data.

                            Statistical Analysis:"""
        
        # Create a custom LLM with faithfulness-focused instructions
        faithful_llm = OpenAI(
            model=SETTINGS.chat_model,
            system_prompt="""You are a precise information retrieval assistant. Your task is to provide accurate, faithful answers based ONLY on the provided source documents. 

                            CRITICAL RULES:
                            1. Base your answer exclusively on the information provided in the source documents
                            2. Do not add information not present in the sources
                            3. Do not make assumptions or inferences beyond what is explicitly stated
                            4. If information is not available in the sources, clearly state this
                            5. Quote directly from the sources when possible
                            6. Maintain the exact meaning and context from the original documents
                            7. Do not paraphrase in ways that change the original meaning

                            Your response should be a direct, faithful representation of the source material."""
        )
        
        try:
            analysis_response = faithful_llm.complete(analysis_prompt)
            analysis_text = analysis_response.text.strip()
            
            # Check if the answer indicates no relevant information
            if "no_data_available" in analysis_text.lower():
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
                
                result = AgentAnswer(text=analysis_text, anchors=anchors, tables=referenced)
                return json.dumps({
                    "text": result.text,
                    "anchors": result.anchors,
                    "tables": result.tables
                })
                
        except Exception as e:
            print(f"Error in statistical analysis: {e}")
            return _fallback_to_hybrid_retrieve(query, storage_path, metadata)
    
    elif relevant_table_ids and len(relevant_table_ids) == 1:
        # Single table analysis - delegate to table_qa_tool
        print("Single table found, delegating to table_qa_tool...")
        from .table_qa_tool import table_qa_tool
        return table_qa_tool(query, storage_dir, metadata)
    
    else:
        # No relevant tables found, fall back to hybrid_retrieve
        print("No relevant tables found for statistical analysis, falling back to hybrid_retrieve...")
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
    
    # retrieve policy-relevant chunks
    policy_filters = filters
    hr = hybrid_retrieve(query, storage_path, policy_filters)

    # build structured mapping via prompt
    context_parts = []
    for n in hr.reranked:
        if hasattr(n, 'node'):
            # NodeWithScore object
            context_parts.append(n.node.get_content()[:1200])
        else:
            # TextNode object
            context_parts.append(n.get_content()[:1200])
    context = "\n\n".join(context_parts)
    prompt = (
        "You are an insurance policy matching assistant.\n"
        "Given the user query and policy/incident excerpts, map incident components to policy clauses.\n"
        "Return a concise JSON with keys: incident_components[], matched_policy_sections[], coverage_notes.\n\n"
        f"Query: {query}\n"
        f"Excerpts:\n{context}\n\n"
        "Output JSON:"
    )
    llm = OpenAI(model=SETTINGS.chat_model)
    resp = llm.complete(prompt)
    text = getattr(resp, "text", str(resp))
    result = AgentAnswer(text=text, anchors=[_format_anchor(n) for n in hr.reranked], tables=[])
    return json.dumps({
        "text": result.text,
        "anchors": result.anchors,
        "tables": result.tables
    })

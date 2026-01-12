from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json

from llama_index.llms.openai import OpenAI

from config import SETTINGS
from needle_tool import needle_tool
from summary_tool import summary_tool
from table_qa_tool import table_qa_tool
from statistics_tool import statistics_tool
from utils import AgentAnswer


def _rule_route(q: str) -> str:
    """Route query to appropriate tool using LLM-based decision making."""
    
    # Initialize LLM
    llm = OpenAI(model=SETTINGS.chat_model)
    
    # Define available tools
    tools = {
        "summary": "summarize_pdf_refine - Use for general summaries, overviews, or broad questions about the documents",
        "needle": "QnA_RAG - Use for specific questions requiring exact quotes, precise information, or detailed answers",
        "table_qa": "table_qa_tool - Use for questions about tables, data analysis, calculations, or structured data",
        "statistics": "statistics_tool - Use for questions about coverage, deductibles, liability, exclusions, or policy statistics"
    }
    
    # Create routing prompt
    routing_prompt = f"""You are a query router for an insurance RAG system. Analyze the input question and determine which tool is most appropriate.

                    Available tools:
                    {json.dumps(tools, indent=2)}

                    Routing rules:
                    - If the question asks for a general summary, overview, or broad understanding of documents, use "summary"
                    - If the question contains words like "summarize", "summary", "what happened with", "what happened to", "tell me about", "common", "findings", "conclusion", "overview", "general", "broad", use "summary"
                    - If the question asks for specific details, exact quotes, precise information, or detailed answers that should be found in the text, use "needle" 
                    - If the question asks about tables, data analysis, calculations, sums, averages, or structured data from tables, use "table_qa"
                    - If the question asks about coverage, deductibles, liability, exclusions, or analytics across multiple cases, use "statistics"
                    - If the question contains words like "statistic", "compare", "pattern", "highest", "lowest", "difference", "across", "between", "correlation", "correlate", "relationship", "association", "related", "link", use "statistics"

                    SUMMARY TOOL should be used for:
                    - Questions asking for general overviews or summaries of cases, accidents, or events
                    - Questions like "What happened with [person]?", "Tell me about [person]"
                    - Questions asking for broad understanding of incidents or cases
                    - Questions about general findings, conclusions, or overviews

                    NEEDLE TOOL should be used for:
                    - Questions asking for specific information that should be found in the text documents
                    - Questions about percentages, ratios, or specific facts mentioned in the documents
                    - Questions asking "how much percent", "what percentage", "what ratio"
                    - Questions about specific details, events, or facts from the case documents
                    - Questions requiring exact quotes or precise information from the text

                    TABLE_QA TOOL should be used for:
                    - Questions about payment amounts, dates, descriptions from payment reports
                    - Questions asking for calculations, sums, totals from tabular data
                    - Questions about specific dollar amounts, dates, or financial data from tables
                    - Questions containing words like "liability", "limit", "coverage", "premium", "bodily ", "injury" "property", "damage", "deductible", "policy amount"

                    STATISTICS_TOOL should be used for:
                    - Questions about coverage, deductibles, liability, exclusions, reports or policies
                    - Questions about financial data, amounts, dates, or totals from tables
                    - Questions about specific dollar amounts, dates, or financial data from tables
                    - Questions containing words like "statistic", "compare", "pattern", "highest", "lowest", "difference", "across", "between", "correlation", "correlate", "relationship", "association", "related", "link"

                    Examples:
                    - "Please, summarize Petr Petrov Case" → summary
                    - "Give me a summary of the accident" → summary
                    - "What happened with Alex Jones?" → summary
                    - "Tell me about the accident" → summary
                    - "What are the common findings?" → summary
                    - "What vehicles were involved in the accident?" → needle
                    - "How much percent of expenses were retrieved?" → needle
                    - "What percentage of projected expenses?" → needle
                    - "What was the weather like at the time of the accident?" → needle
                    - "How much was the total settlement?" → table_qa
                    - "What is Alex Jones' total policy premium?" → table_qa
                    - "What is the property damage liability limit?" → table_qa
                    - "What is the bodily injury liability limit?" → table_qa
                    - "Compare coverage limits between policies" → statistics

                    Input question: "{q}"

                    Please provide your reasoning in the following format:

                    THOUGHT: [Your analysis of the question and why you chose this tool]
                    TOOL: [tool name]

                    Respond with the tool name and your reasoning."""

    try:
        # Get LLM response
        response = llm.complete(routing_prompt)
        response_text = response.text.strip()
        
        # Parse the response to extract thought and tool
        lines = response_text.split('\n')
        thought = ""
        tool_name = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("THOUGHT:"):
                thought = line.replace("THOUGHT:", "").strip()
            elif line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip().lower()
        
        # Print the chain of thoughts
        print(f"[ROUTING] Decision:")
        print(f"   Thought: {thought}")
        print(f"   Selected Tool: {tool_name}")
        
        # Map LLM tool names to actual tool names
        tool_mapping = {
            "table_qa_tool": "table_qa",
            "statistics_tool": "statistics", 
            "summary_tool": "summary",
            "needle_tool": "needle"
        }
        
        # Apply mapping if needed
        if tool_name in tool_mapping:
            tool_name = tool_mapping[tool_name]
        
        # Validate response
        if tool_name in tools:
            return tool_name
        else:
            print(f"[WARNING] Invalid tool name '{tool_name}', falling back to keyword-based routing")
            # Fallback to keyword-based routing if LLM response is invalid
            return _fallback_route(q)
            
    except Exception as e:
        print(f"[ERROR] LLM routing failed: {e}")
        # Fallback to keyword-based routing
        return _fallback_route(q)


def _fallback_route(q: str) -> str:
    """Fallback keyword-based routing when LLM routing fails."""
    print(f"[FALLBACK] Using keyword-based routing for: '{q}'")
    t = q.lower()
    if any(k in t for k in ["table", "row", "column", "sum", "avg", "total", "count"]):
        print(f"   -> table_qa (keyword match)")
        return "table_qa"
    if any(k in t for k in ['average', 'total', 'sum', 'compare', 'analysis', 'pattern', 'statistic', 
                            'settlement', 'payment', 'deductible', 'coverage', 'premium', 'amount',
                            'correlation', 'correlate', 'relationship', 'association', 'related', 'link'
                            'highest', 'maximum', 'lowest', 'minimum', 'across', 'between', 'difference']):
        print(f"   -> statistics (keyword match)")
        return "statistics"
    if any(k in t for k in ["exact", "page", "figure", "anchor", "quote"]):
        print(f"   -> needle (keyword match)")
        return "needle"
    print(f"   -> summary (default)")
    return "summary"


def route_and_answer(query: str, storage_dir: Path, metadata: Optional[Dict[str, Any]] = None) -> AgentAnswer:
    """Route query to appropriate agent tool and return AgentAnswer."""
    route = _rule_route(query)
    try:
        if route == "table_qa":
            result = table_qa_tool(query, str(storage_dir), metadata)
            return AgentAnswer(**json.loads(result))
        if route == "statistics":
            result = statistics_tool(query, str(storage_dir), metadata)
            return AgentAnswer(**json.loads(result))
        if route == "needle":
            result = needle_tool(query, str(storage_dir), metadata)
            return AgentAnswer(**json.loads(result))
        result = summary_tool(query, str(storage_dir), metadata)
        return AgentAnswer(**json.loads(result))
    except Exception as e:
        print(f"Error in route_and_answer: {e}")
        return AgentAnswer(text=f"Error: {str(e)}", anchors=[], tables=[])

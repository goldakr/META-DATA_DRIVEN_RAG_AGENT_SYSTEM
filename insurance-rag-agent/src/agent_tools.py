from __future__ import annotations
from typing import List
from llama_index.core.tools import FunctionTool

from summary_tool import summary_tool
from needle_tool import needle_tool
from table_qa_tool import table_qa_tool
from statistics_tool import statistics_tool


def create_agent_tools(storage_dir: str) -> List[FunctionTool]:
    """Create the four agent tools for use with an agent."""
    tools = [
        FunctionTool.from_defaults(
            fn=summary_tool,
            name="summary_tool",
            description="Summarize selected parts of retrieved info. Use this tool when you need to provide a comprehensive summary of information related to the query."
        ),
        FunctionTool.from_defaults(
            fn=needle_tool,
            name="needle_tool", 
            description="Return the most precise paragraph/anchor. Use this tool when you need exact quotes, specific page references, or precise information from documents."
        ),
        FunctionTool.from_defaults(
            fn=table_qa_tool,
            name="table_qa_tool",
            description="Handle tabular queries; returns a compact answer plus any referenced table IDs. Use this tool when the query involves tables, data analysis, calculations, or structured data."
        ),
        FunctionTool.from_defaults(
            fn=statistics_tool,
            name="statistics_tool",
            description="Match policy sections to incident components; returns mapping. Use this tool when the query involves insurance policies, coverage analysis, deductibles, or policy matching."
        )
    ]
    return tools
from __future__ import annotations
from pathlib import Path
import json
import typer
import os
from rich import print as rprint
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Make sure to set OPENAI_API_KEY environment variable manually")

# Ensure API key is properly set
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    api_key = api_key.strip()
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"API key loaded successfully")
else:
    print("Warning: No OPENAI_API_KEY found in environment variables")

from index_build import build_index
from agent import route_and_answer

app = typer.Typer(add_completion=False)

@app.command()
def build(data_dir: str = "./data", storage: str = "./storage"):
    """Ingest PDFs from --data-dir, extract tables, build vector index and persist to --storage."""
    res = build_index(Path(data_dir), Path(storage))
    rprint(f"[bold green]Indexed {res['count']} chunks[/] into {storage}")

@app.command()
def ask(query: str, storage: str = "./storage", metadata: str = "{}"):
    """Ask a question. Optional --metadata='{\"ClientId\":\"CASE-123\"}' to filter."""
    meta = json.loads(metadata)
    ans = route_and_answer(query, Path(storage), meta)
    rprint("\n[bold cyan]Answer[/]:")
    rprint(ans.text)
    if ans.tables:
        rprint("\n[bold magenta]Tables[/]:")
        rprint(ans.tables)
    if ans.anchors:
        rprint("\n[bold yellow]Anchors[/]:")
        rprint(ans.anchors)

@app.command()
def chat(storage: str = "./storage", metadata: str = "{}"):
    """Start an interactive chatbot session. Type 'exit' or 'quit' to stop."""
    meta = json.loads(metadata)
    rprint("[bold green]Insurance RAG Agent Chat Started![/]")
    rprint("[dim]Type 'exit' or 'quit' to stop the session.[/]\n")
    
    while True:
        try:
            query = input("Ask a question: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                rprint("[bold red]Goodbye![/]")
                break
                
            if not query:
                rprint("[dim]Please enter a question.[/]")
                continue
                
            rprint(f"\n[dim]Processing: {query}[/]")
            ans = route_and_answer(query, Path(storage), meta)
            
            rprint("\n[bold cyan]Answer[/]:")
            rprint(ans.text)
            if ans.tables:
                rprint("\n[bold magenta]Tables[/]:")
                rprint(ans.tables)
            if ans.anchors:
                rprint("\n[bold yellow]Anchors[/]:")
                rprint(ans.anchors)
            rprint("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            rprint("\n[bold red]Goodbye![/]")
            break
        except Exception as e:
            rprint(f"[bold red]Error: {e}[/]")
            rprint("Please try again.\n")

if __name__ == "__main__":
    app()

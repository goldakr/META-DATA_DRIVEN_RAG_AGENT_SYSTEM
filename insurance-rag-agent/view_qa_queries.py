"""
Script to view all queries in the RAGAS evaluation QA file.
Usage: python view_qa_queries.py [qa_file]
"""

import json
import sys
from pathlib import Path

def view_qa_queries(qa_file: str = "ragas_qa.jsonl"):
    """View all queries in the QA file."""
    
    qa_path = Path(qa_file)
    
    if not qa_path.exists():
        print(f"❌ QA file not found: {qa_file}")
        return
    
    print(f"📋 Queries in {qa_file}:")
    print("=" * 80)
    
    with open(qa_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    qa_entry = json.loads(line.strip())
                    print(f"\n🔍 Query {i}:")
                    print(f"   Question: {qa_entry['question']}")
                    print(f"   Ground Truth: {qa_entry['ground_truth']}")
                    print(f"   Metadata: {qa_entry.get('metadata', {})}")
                    print("-" * 40)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing line {i}: {e}")
    
    print(f"\n✅ Total queries: {i}")

def main():
    qa_file = sys.argv[1] if len(sys.argv) > 1 else "ragas_qa.jsonl"
    view_qa_queries(qa_file)

if __name__ == "__main__":
    main()


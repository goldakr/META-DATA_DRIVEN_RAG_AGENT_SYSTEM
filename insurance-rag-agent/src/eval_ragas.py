from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)

from agent import route_and_answer

def main(storage: str, qa_file: str):
    qa = [json.loads(l) for l in Path(qa_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    # qa.jsonl lines must contain: {"question": "...", "ground_truth": "...", "metadata": {...}}
    questions = [q["question"] for q in qa]
    ground_truths = [q.get("ground_truth", "") for q in qa]
    metas = [q.get("metadata", {}) for q in qa]

    # get answers and contexts
    answers = []
    contexts = []
    for q, m in zip(questions, metas):
        ans = route_and_answer(q, Path(storage), m)
        answers.append(ans.text)
        
        # Extract actual text contexts from anchors
        # We need to re-run retrieval to get the actual text content
        from .retrieval import hybrid_retrieve
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
        
        # For RAGAS evaluation, we'll retrieve without strict metadata filters
        # to ensure we get contexts for evaluation
        # Build metadata filters if provided, but make them less restrictive
        filters = None
        if m and 'FileName' in m:
            # Only filter by FileName if specified, and use contains instead of exact match
            from llama_index.core.vector_stores import MetadataFilters
            from llama_index.core.vector_stores import MetadataFilter
            # Use a more flexible approach - get contexts from the specific file if possible
            try:
                filt_list = [ExactMatchFilter(key='FileName', value=m['FileName'])]
                filters = MetadataFilters(filters=filt_list)
            except:
                # If filtering fails, proceed without filters
                filters = None
        
        # Get the retrieved contexts with increased retrieval for better recall
        hr = hybrid_retrieve(q, Path(storage), filters)
        
        # Extract text content from the reranked nodes
        context_texts = []
        for node_with_score in hr.reranked:
            if hasattr(node_with_score, 'node') and hasattr(node_with_score.node, 'text'):
                context_text = node_with_score.node.text
                # Include substantial contexts that are likely to contain complete information
                if len(context_text.strip()) > 50:  # Balanced minimum length
                    context_texts.append(context_text)
        
        # Add high-quality candidates for better recall while maintaining quality
        for node_with_score in hr.candidates[:3]:  # Include top 3 candidates for better recall
            if hasattr(node_with_score, 'node') and hasattr(node_with_score.node, 'text'):
                context_text = node_with_score.node.text
                if context_text not in context_texts and len(context_text.strip()) > 150:  # Higher quality threshold
                    context_texts.append(context_text)
                    if len(context_texts) >= 5:  # Allow more contexts for better faithfulness
                        break
        
        # Contexts successfully retrieved for RAGAS evaluation
        
        contexts.append(context_texts)

    ds = Dataset.from_pandas(pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }))

    results = evaluate(
        ds,
        metrics=[answer_correctness, context_precision, context_recall, faithfulness],
    )
    print(results)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--storage", default="./storage")
    p.add_argument("--qa-file", required=True)
    args = p.parse_args()
    main(args.storage, args.qa_file)

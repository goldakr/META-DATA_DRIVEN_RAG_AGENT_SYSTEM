# Insurance RAG Agent (Hybrid Retrieval + Reranker + Metadata Anchors)

This project creates a Retrieval-Augmented Generation (RAG) agent tailored for **insurance incident reports** (accidents / break‑ins) with **PDFs containing text, tables and diagrams**. It uses:
- **LlamaIndex** for ingestion, chunking and querying
- **Hybrid retrieval**: dense (OpenAI embeddings) **+** sparse (BM25)
- **Reranking**: FlashRank (cross‑encoder) by default, with **LLM-based rerank** fallback
- **Table extraction** from PDFs via **`PDFTableReader`** (Camelot)
- **Rich metadata & anchors** on every chunk (e.g., `PageNumber`, `SectionType`, `TableId/FigureId`, `IncidentDate/Type`, `ClientId/CaseId`)
- **Agent tools**: Router → (Summary / Needle / Table‑QA / Statistics)
- **Chunk budget**: ~5% of document OR max 10 chunks passed to the LLM

> Place your source PDFs under: `./data/` (you may create subfolders per category/case).

## Quick start (Windows-friendly)

1. **Python ** (recommended) — create & activate a virtualenv.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Copy `.env.example` → `.env` and set your keys (at least `OPENAI_API_KEY`).  
4. **Install Ghostscript** (required by Camelot for robust table extraction):  
   - Download from the official site and ensure `gswin64c.exe` is on your PATH.
5. Build the index (ingests PDFs, extracts tables/anchors/metadata):

    ```bash
    python src/main.py build
    ```

    Or with custom paths:

    ```bash
    python src/main.py build --data-dir ./data --storage ./storage
    ```

6. Ask questions:

    **Single question:**
    ```bash
    python src/main.py ask "your question"
    ```

    Or with custom storage and metadata:
    ```bash
    python src/main.py ask "your question" --storage ./storage --metadata '{"ClientId":"CASE-123"}'
    ```

    **Interactive chat session:**
    ```bash
    python src/main.py chat
    ```

    Or with custom storage and metadata:
    ```bash
    python src/main.py chat --storage ./storage --metadata '{"ClientId":"CASE-123"}'
    ```

### Data layout

```
project/
├── data/
│   ├── case-001/
│   │   ├── incident_report.pdf
│   │   └── policy_ABC.pdf
|   |   └── payment_report.pdf
│   └── case-002/
│   |   └── incident_report.pdf
|   |   └── policy_ABC.pdf
|   |   └── payment_report.pdf
│   └── case-002/
├── storage/
│   └── tables/   # extracted CSVs + registry.json
└── src/...
```

### Notes
- The BM25 retriever and `PDFTableReader` are provided by LlamaIndex integrations.  
- By default we persist the LlamaIndex vector store locally; you can switch to Chroma/Qdrant/Milvus if you need native metadata filtering/scalability.

## Evaluation (RAGAS)

Prepare a small Q/A dataset and run:

```bash
python -m src.eval_ragas --storage ./storage --qa-file ./eval/qa.jsonl
```

The script computes: **answer correctness, context precision/recall, faithfulness**, and **table‑QA accuracy** (where applicable).

## Troubleshooting

- **pdfplumber relies on pdfminer.six; no Ghostscript required**. Install it and ensure it’s on PATH.  
- If `flashrank` downloads a model the first time, it may take a minute. It runs on CPU by default.  
- LlamaIndex API moves fast. If you see import errors, try upgrading the corresponding `llama-index-*` integration listed in `requirements.txt`.

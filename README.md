# Insurance RAG Agent (Hybrid Retrieval + Reranker + Metadata Anchors)

This project creates a Retrieval-Augmented Generation (RAG) agent tailored for **insurance incident reports** (accidents / break‑ins) with **PDFs containing text, tables and diagrams**. It uses:
- **LlamaIndex** for ingestion, chunking and querying
- **Hybrid retrieval**: dense (OpenAI embeddings) **+** sparse (BM25)
- **Reranking**: LLM-based reranking for optimal result ordering
- **Table extraction** from PDFs via **pdfplumber** for robust table extraction
- **ChromaDB** for vector storage with persistent storage
- **Rich metadata & anchors** on every chunk (e.g., `PageNumber`, `SectionType`, `TableId/FigureId`, `IncidentDate/Type`, `ClientId/CaseId`)
- **Agent tools**: Router → (Summary / Needle / Table‑QA / Statistics)
- **Chunk budget**: ~5% of document OR max 10 chunks passed to the LLM

> Place your source PDFs under: `./data/` (you may create subfolders per category/case).

## Quick start (Windows-friendly)

1. **Python 3.8+** (recommended) — create & activate a virtualenv.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and set your keys (at least `OPENAI_API_KEY`):
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```
   
4. Build the index (ingests PDFs, extracts tables/anchors/metadata):

    ```bash
    python -m src.main build
    ```

    Or with custom paths:

    ```bash
    python -m src.main build --data-dir ./data --storage ./storage
    ```

5. Ask questions:

    **Single question:**
    ```bash
    python -m src.main ask "your question"
    ```

    Or with custom storage and metadata:
    ```bash
    python -m src.main ask "your question" --storage ./storage --metadata '{"ClientId":"CASE-123"}'
    ```

    **Interactive chat session:**
    ```bash
    python -m src.main chat
    ```

    Or with custom storage and metadata:
    ```bash
    python -m src.main chat --storage ./storage --metadata '{"ClientId":"CASE-123"}'
    ```

## Agent Tools

The system uses an intelligent router that directs queries to one of four specialized tools:

### 1. Summary Tool
- **Purpose**: Provides comprehensive overviews and general summaries
- **Use Cases**: Case overviews, incident summaries, broad understanding of events
- **Example**: "Please, summarize Petr Petrov Case" or "What happened with Alex Jones?"

### 2. Needle Tool
- **Purpose**: Delivers precise, exact information with high fidelity
- **Use Cases**: Specific facts, exact quotes, precise percentages, detailed information
- **Example**: "What vehicles were involved in the accident?" or "How much percent of expenses were retrieved?"

### 3. Table QA Tool
- **Purpose**: Handles structured data queries and tabular analysis
- **Use Cases**: Payment amounts, policy details, financial calculations, structured data queries
- **Example**: "How much was the total settlement?" or "What is the property damage liability limit?"

### 4. Statistics Tool
- **Purpose**: Performs advanced statistical analysis and cross-table comparisons
- **Use Cases**: Coverage analysis, deductibles comparison, pattern recognition, correlation analysis
- **Example**: "Compare coverage limits between policies" or "What is the correlation between payment amounts?"

## Data layout

```
project/
├── data/
│   ├── Alex_Jones_Accident_Report.pdf
│   ├── Alex_Jones_Insurance_Policy.pdf
│   ├── Alex_Jones_Payment_Report.pdf
│   ├── Maria_Petrov_Accident_Case.pdf
│   ├── Maria_Petrov_Payment_Report.pdf
│   ├── Petr_Petrov_Accident_Case.pdf
│   ├── Petr_Petrov_Insurance_Policy_Honda.pdf
│   ├── Petr_Petrov_Insurance_Policy_Toyota.pdf
│   └── Petr_Petrov_Payment_Report.pdf
├── storage/
│   ├── tables/           # extracted CSVs + registry.json
│   ├── chroma_db/        # ChromaDB vector database (not tracked in git)
│   ├── docstore.json     # document store
│   ├── graph_store.json  # graph store
│   ├── index_store.json  # index store
│   └── image__vector_store.json  # image vector store
├── src/                  # source code
├── tests/                # test files
├── output/               # output files and evaluation results
├── requirements.txt      # Python dependencies
└── README.md             # this file
```

## Additional Scripts

### Rebuild Tables
Rebuild the table registry from PDF files:

```bash
python rebuild_tables.py
```

Or with specific extraction method:
```bash
python rebuild_tables.py --method pdfplumber
```

### View QA Queries
View and test QA queries from evaluation datasets:

```bash
python view_qa_queries.py
```

### Print Chunks and Tables
Extract and display document chunks and tables:

```bash
python print_chunks_and_tables.py
```

## Evaluation (RAGAS)

Prepare a small Q/A dataset in JSONL format and run:

```bash
python -m src.eval_ragas --storage ./storage --qa-file ./ragas_eval_qa.jsonl
```

The script computes: **answer correctness, context precision/recall, faithfulness**, and **table‑QA accuracy** (where applicable).

### Table QA Evaluation

Run specialized table QA evaluation:

```bash
python run_table_qa_eval.py
```

## Technical Details

### Retrieval System
- **Dense Retrieval**: OpenAI embeddings (`text-embedding-3-small` by default)
- **Sparse Retrieval**: BM25 for keyword-based search
- **Reranking**: LLM-based reranking using OpenAI models
- **Vector Store**: ChromaDB with persistent storage
- **Metadata Filtering**: Support for filtering by document metadata

### Table Extraction
- **Primary Method**: pdfplumber for robust table extraction
- **Storage**: Tables are extracted as CSV files in `storage/tables/`
- **Registry**: Table metadata stored in `storage/tables/registry.json`
- Each table includes: table ID, page number, CSV path, file name, method, and summary

### Configuration
Default settings can be found in `src/config.py`:
- Chat model: `gpt-4o-mini` (configurable via `OPENAI_CHAT_MODEL`)
- Embedding model: `text-embedding-3-small` (configurable via `OPENAI_EMBED_MODEL`)
- Dense top-k: 10
- Sparse top-k: 10
- Candidate pool: 40
- Rerank top-n: 8
- Max chunks: 10
- Chunk budget: 5% of document

## Notes
- The BM25 retriever is provided by `llama-index-retrievers-bm25`
- ChromaDB is used for vector storage with persistent storage in `storage/chroma_db/`
- The `storage/chroma_db/` directory is not tracked in git (see `.gitignore`)
- Table extraction uses pdfplumber which relies on pdfminer.six
- All tools use LLM-based routing for intelligent query classification
- The system supports metadata filtering for precise document retrieval

## Troubleshooting

- **PDF processing**: pdfplumber relies on pdfminer.six; no Ghostscript required
- **API errors**: Ensure your `OPENAI_API_KEY` is set correctly in the `.env` file
- **ChromaDB errors**: If you encounter ChromaDB errors, try rebuilding the index
- **Import errors**: LlamaIndex API moves fast. If you see import errors, try upgrading the corresponding `llama-index-*` integration listed in `requirements.txt`
- **Table extraction issues**: If tables are not being extracted correctly, try using the `rebuild_tables.py` script with different extraction methods

## Testing

Run tests with:

```bash
pytest tests/
```

The test suite includes integration tests for ingestion and table extraction.

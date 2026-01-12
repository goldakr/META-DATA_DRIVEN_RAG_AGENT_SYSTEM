# Insurance RAG Agent - Project Description

## Overview

The **Insurance RAG Agent** is a sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for processing and analyzing insurance-related documents. The system combines advanced document processing, intelligent retrieval mechanisms, and specialized agent tools to provide comprehensive answers to complex insurance queries.

## Core Architecture

### 1. Document Processing Pipeline
- **PDF Ingestion**: Processes insurance documents including accident reports, policy documents, and payment reports
- **Table Extraction**: Uses Camelot library for robust table extraction from PDFs with high accuracy
- **Multi-modal Processing**: Handles text, tables, and diagrams within insurance documents
- **Metadata Enrichment**: Automatically extracts and stores rich metadata including page numbers, section types, table IDs, and case information

### 2. Hybrid Retrieval System
- **Dense Retrieval**: OpenAI embeddings for semantic similarity search
- **Sparse Retrieval**: BM25 for keyword-based search
- **Reranking**: FlashRank cross-encoder with LLM-based fallback for optimal result ranking
- **Metadata Filtering**: Advanced filtering capabilities for precise document retrieval

### 3. Intelligent Agent Tools

#### Summary Tool
- **Purpose**: Provides comprehensive overviews and general summaries
- **Use Cases**: Case overviews, incident summaries, broad understanding of events
- **Method**: Map-reduce approach for processing multiple document chunks
- **Response Mode**: Comprehensive summaries with source attribution

#### Needle Tool
- **Purpose**: Delivers precise, exact information with high fidelity
- **Use Cases**: Specific facts, exact quotes, precise percentages, detailed information
- **Method**: Compact response mode for concise, accurate answers
- **Response Mode**: Direct quotes and precise information with source anchors

#### Table QA Tool
- **Purpose**: Handles structured data queries and tabular analysis
- **Use Cases**: Payment amounts, policy details, financial calculations, structured data queries
- **Method**: Direct table analysis with LLM-powered query understanding
- **Features**: Multi-table support, calculation capabilities, structured data processing

#### Statistics Tool
- **Purpose**: Performs advanced statistical analysis and cross-table comparisons
- **Use Cases**: Coverage analysis, deductibles comparison, pattern recognition, correlation analysis
- **Method**: Multi-table statistical analysis with correlation matrix support
- **Features**: 
  - Correlation matrix calculation using NumPy
  - Cross-table statistical comparisons
  - Pattern recognition and trend analysis
  - Conditional correlation analysis (only for correlation-specific queries)

### 4. Intelligent Routing System
- **LLM-based Routing**: Uses OpenAI models for intelligent query classification
- **Fallback Mechanism**: Keyword-based routing when LLM routing fails
- **Tool Selection**: Automatically routes queries to the most appropriate tool
- **Context Awareness**: Considers query intent and content type for optimal routing

## Key Features

### Advanced Document Processing
- **Multi-format Support**: Handles PDFs with complex layouts including tables and diagrams
- **Table Extraction**: High-accuracy table extraction using Camelot with Ghostscript support
- **Metadata Preservation**: Maintains document structure and metadata throughout processing
- **Chunk Optimization**: Intelligent chunking with budget constraints (5% of document or max 10 chunks)

### Statistical Analysis Capabilities
- **Correlation Analysis**: NumPy-powered correlation matrix calculation for numerical data
- **Multi-table Processing**: Combines data from multiple tables for comprehensive analysis
- **Pattern Recognition**: Identifies trends and patterns across insurance data
- **Comparative Analysis**: Cross-table comparisons and statistical insights

### Evaluation Framework
- **RAGAS Integration**: Comprehensive evaluation using RAGAS metrics
- **Answer Correctness**: Measures accuracy of generated responses
- **Context Precision/Recall**: Evaluates retrieval quality
- **Faithfulness**: Ensures responses are grounded in source documents
- **Table QA Accuracy**: Specialized evaluation for tabular data queries

## Technical Stack

### Core Technologies
- **LlamaIndex**: Document processing, chunking, and querying framework
- **OpenAI**: Language models for chat, embeddings, and routing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and correlation analysis
- **Camelot**: PDF table extraction with high accuracy

### Retrieval Components
- **BM25**: Sparse retrieval for keyword-based search
- **FlashRank**: Cross-encoder reranking for optimal result ordering
- **ChromaDB**: Vector storage and metadata management
- **Hybrid Retrieval**: Combines dense and sparse retrieval methods

### Evaluation Tools
- **RAGAS**: Comprehensive RAG evaluation framework
- **Custom Metrics**: Specialized evaluation for insurance domain
- **Table QA Evaluation**: Specific evaluation for tabular data processing

## Data Structure

### Document Types
- **Accident Reports**: Detailed incident documentation with tables and diagrams
- **Insurance Policies**: Policy documents with coverage details and terms
- **Payment Reports**: Financial data with settlement amounts and payment details

### Storage Organization
```
storage/
├── tables/                    # Extracted CSV files
│   ├── registry.json         # Table metadata registry
│   └── Table-*.csv           # Individual table files
├── chroma_db/                # Vector database
└── *.json                    # Index and metadata files
```

## Use Cases

### Insurance Claims Processing
- **Incident Analysis**: Comprehensive analysis of accident reports
- **Policy Verification**: Cross-referencing claims with policy terms
- **Settlement Calculation**: Financial analysis and payment verification

### Risk Assessment
- **Pattern Recognition**: Identifying trends in insurance data
- **Correlation Analysis**: Understanding relationships between variables
- **Comparative Analysis**: Cross-case and cross-policy comparisons

### Compliance and Auditing
- **Document Review**: Systematic analysis of insurance documentation
- **Data Validation**: Verification of tabular data accuracy
- **Report Generation**: Automated summary and analysis reports

## Performance Characteristics

### Retrieval Efficiency
- **Hybrid Approach**: Combines semantic and keyword search for optimal results
- **Reranking**: Advanced reranking ensures most relevant results are prioritized
- **Metadata Filtering**: Precise filtering reduces noise and improves accuracy

### Response Quality
- **Tool Specialization**: Each tool optimized for specific query types
- **Source Attribution**: All responses include proper source references
- **Faithfulness**: Responses grounded in source documents with minimal hallucination

### Scalability
- **Modular Design**: Easy to extend with additional tools and capabilities
- **Efficient Processing**: Optimized for handling large document collections
- **Caching**: Intelligent caching reduces redundant processing

## Future Enhancements

### Planned Features
- **Multi-language Support**: Processing documents in multiple languages
- **Advanced Analytics**: Machine learning-based pattern recognition
- **Real-time Processing**: Live document ingestion and analysis
- **API Integration**: RESTful API for external system integration

### Extensibility
- **Plugin Architecture**: Easy addition of new analysis tools
- **Custom Metrics**: Domain-specific evaluation metrics
- **Integration Hooks**: Seamless integration with existing insurance systems

## Conclusion

The Insurance RAG Agent represents a comprehensive solution for insurance document processing and analysis. By combining advanced retrieval techniques, specialized agent tools, and robust evaluation frameworks, it provides accurate, reliable, and comprehensive answers to complex insurance queries while maintaining high fidelity to source documents.

The system's modular architecture and specialized tools make it particularly well-suited for insurance professionals who need to quickly analyze complex documents, extract precise information, and perform statistical analysis across multiple data sources.


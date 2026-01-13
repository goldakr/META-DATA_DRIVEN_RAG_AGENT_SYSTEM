# LinkedIn Post - Insurance RAG Agent Project

🚀 Excited to share my latest project: A Meta-data Driven RAG Agent System for Insurance Document Analysis!

This project was developed as my final project for the "AI for Developers" course at The Institute AI, Ben-Gurion University of the Negev, Israel.

I've built an intelligent Retrieval-Augmented Generation (RAG) system specifically designed for processing insurance incident reports, policy documents, and payment records. The system combines advanced document processing with AI-powered analysis to extract precise information from complex PDFs.

**Key Features:**
✨ Hybrid Retrieval System - Combines semantic search (OpenAI embeddings) with keyword-based BM25 search for optimal results

🧠 Intelligent Agent Router - Automatically routes queries to 4 specialized tools:
• Summary Tool - For comprehensive case overviews
• Needle Tool - For precise, exact information extraction
• Table QA Tool - For structured data analysis
• Statistics Tool - For advanced statistical comparisons and pattern recognition

📊 Advanced Document Processing - Handles PDFs with text, tables, and diagrams, using pdfplumber for robust table extraction

🔍 Smart Reranking - LLM-based reranking ensures the most relevant information is prioritized

💾 Persistent Storage - Uses ChromaDB for efficient vector storage and retrieval

The system uses LlamaIndex for document processing, OpenAI for embeddings and language models, and includes comprehensive evaluation metrics using RAGAS. It's designed to handle complex insurance queries with high accuracy while maintaining source attribution through rich metadata anchors.

This project demonstrates how RAG systems can be specialized for domain-specific applications, making document analysis faster, more accurate, and more reliable.

#AI #RAG #LlamaIndex #OpenAI #MachineLearning #InsuranceTech #DocumentAI #LLM #VectorDatabase #DataScience #Python #NLP #RetrievalAugmentedGeneration

# 🔬 RAGForge — Research Paper Q&A

> **Production-grade Retrieval-Augmented Generation system for academic PDF question answering.**

RAGForge lets you upload research papers (PDFs) and ask questions about them.
It implements a complete 13-component RAG pipeline with intelligent PDF extraction,
hybrid retrieval, cross-encoder reranking, LLM-powered guardrails, and an agentic
routing layer — all served through a streaming FastAPI backend and a polished
Streamlit frontend.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAGForge Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PDF Upload ──→ Extraction Pipeline (marker-pdf / pdfplumber /          │
│                 PyMuPDF / Tesseract + quality gate + 6-step cleaner)    │
│                     │                                                   │
│                     ▼                                                   │
│  C1: Smart Chunking (Recursive + Semantic + Context Overlap)           │
│                     │                                                   │
│                     ▼                                                   │
│  C2: BGE Embeddings (bge-small-en-v1.5)                                │
│                     │                                                   │
│                     ▼                                                   │
│  C3: FAISS Vector Store + BM25 Index                                   │
│                     │                                                   │
│                     ▼                                                   │
│  ┌──────── RETRIEVAL PIPELINE ────────┐                                │
│  │ C5: Query Rewriting (Expand/HyDE)  │                                │
│  │ C4: Hybrid Retrieval (FAISS+BM25)  │                                │
│  │ C6: Multi-Vector (raw + summary)   │                                │
│  │ C7: Cross-Encoder Reranking        │                                │
│  │ C8: Context Compression            │                                │
│  └────────────────┬───────────────────┘                                │
│                   ▼                                                     │
│  C13: LangGraph Agentic RAG                                            │
│   (classify → rewrite → retrieve → rerank → compress → generate)       │
│                   │                                                     │
│                   ├──→ C10: Conversation Memory                        │
│                   ├──→ C11: Guardrails (Groundedness + NER + Confidence)│
│                   │                                                     │
│                   ▼                                                     │
│  C12: FastAPI + SSE Streaming  ◀──  Streamlit Frontend                 │
│                                                                         │
│  C9: RAGAS Evaluation (offline benchmark)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ragforge.git
cd ragforge

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for guardrails)
python -m spacy download en_core_web_sm
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env and add your keys:
#   GROQ_API_KEY=your-groq-api-key
#   OPENAI_API_KEY=your-openai-api-key  (optional fallback)
```

### 3. Install Tesseract (for scanned PDFs)

- **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- **Linux**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`

### 4. Ingest PDFs

```bash
# Place your PDF files in data/raw/
python scripts/ingest.py --input data/raw --verbose
```

### 5. Run the Server

```bash
# Terminal 1: Start FastAPI backend
python main.py

# Terminal 2: Start Streamlit frontend
streamlit run src/serving/streamlit_app.py
```

Open http://localhost:8501 in your browser and start asking questions!

---

## 🧩 Component Breakdown

| # | Component | Description |
|---|-----------|-------------|
| 0 | **Extraction Pipeline** | Multi-extractor PDF processing with marker-pdf → pdfplumber → PyMuPDF fallback chain, quality gates, and 6-step text cleaning |
| 1 | **Smart Chunking** | Three-strategy chunking: recursive splitting → semantic merging → context-aware overlap for optimal chunk boundaries |
| 2 | **BGE Embeddings** | BAAI/bge-small-en-v1.5 dense embeddings with automatic query prefix for asymmetric retrieval |
| 3 | **FAISS Vector Store** | Persisted FAISS index with BM25 co-located for hybrid search |
| 4 | **Hybrid Retrieval** | FAISS (dense) + BM25 (sparse) combined via Reciprocal Rank Fusion |
| 5 | **Query Rewriting** | LLM-powered query expansion, conversational reformulation, and HyDE (hypothetical document embeddings) |
| 6 | **Multi-Vector Retrieval** | Dual embeddings (raw + LLM summary) for improved recall |
| 7 | **Cross-Encoder Reranking** | ms-marco-MiniLM-L-6-v2 for fine-grained relevance scoring |
| 8 | **Context Compression** | LLM extracts relevant sentences + embedding-based deduplication |
| 9 | **RAGAS Evaluation** | Automated benchmarking with faithfulness, relevancy, precision, recall |
| 10 | **Conversation Memory** | Summary buffer memory (2000 tokens) for multi-turn Q&A |
| 11 | **Guardrails** | Three-layer safety: LLM groundedness, spaCy NER hallucination detection, confidence scoring |
| 12 | **SSE Streaming** | FastAPI with Server-Sent Events for real-time token streaming |
| 13 | **Agentic RAG** | LangGraph StateGraph with query classification and conditional routing |

---

## 🔄 Extraction Pipeline Detail

```
PDF
 │
 ▼
detect_pdf_type() ──→ "native" / "mixed" / "scanned"
 │
 ├─ native/mixed ──→ marker-pdf (primary, Markdown output)
 │                      │
 │                      ▼
 │                   quality_check()
 │                      ├─ PASS → 6-step cleaning → Documents
 │                      └─ FAIL → pdfplumber (fallback 1)
 │                                  │
 │                                  ▼
 │                               quality_check()
 │                                  ├─ PASS → cleaning
 │                                  └─ FAIL → PyMuPDF (fallback 2)
 │                                              │
 │                                              ▼
 │                                           quality_check()
 │                                              ├─ PASS → cleaning
 │                                              └─ FAIL → ExtractorError
 │
 └─ scanned ──→ Tesseract OCR (300 DPI)
                   │
                   ▼
                quality_check()
                   ├─ PASS → cleaning
                   └─ FAIL → retry at 400 DPI
                               │
                               ▼
                            quality_check()
                               ├─ PASS → cleaning
                               └─ FAIL → ExtractorError
```

---

## 📊 RAGAS Benchmark Results

Run the benchmark:
```bash
python scripts/evaluate.py --sample-size 50 --output logs/ragas_results.json
```

| Metric | Score |
|--------|-------|
| Faithfulness | _TBD_ |
| Answer Relevancy | _TBD_ |
| Context Precision | _TBD_ |
| Context Recall | _TBD_ |
| **Overall** | _TBD_ |

_(Fill in after running the evaluation on your document set)_

---

## 📁 Project Structure

```
ragforge/
├── config/
│   ├── __init__.py
│   └── settings.py                  # all config: paths, model names, thresholds
├── data/
│   ├── raw/                         # user drops PDFs here
│   ├── processed/                   # pickled clean Document objects
│   └── faiss_store/                 # persisted FAISS index
├── src/
│   ├── __init__.py
│   ├── extraction/                  # PDF → clean LangChain Documents
│   │   ├── __init__.py
│   │   ├── pdf_detector.py          # detect PDF type (native/scanned/mixed)
│   │   ├── extractors.py            # marker-pdf, pdfplumber, PyMuPDF, Tesseract
│   │   ├── quality.py               # quality gate + fallback chain
│   │   ├── cleaner.py               # 6-step cleaning pipeline
│   │   └── pipeline.py              # orchestrates the full extraction flow
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── smart_chunker.py         # 3-strategy chunking
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── bge_embedder.py          # BGE embedding wrapper
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── faiss_store.py           # FAISS + BM25 persistence
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_rewriter.py        # query expansion, HyDE, reformulation
│   │   ├── hybrid_retriever.py      # FAISS + BM25 + RRF
│   │   ├── multi_vector.py          # raw + summary embeddings
│   │   ├── reranker.py              # cross-encoder reranking
│   │   └── compressor.py            # context compression
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_factory.py           # Groq/OpenAI LLM with fallback
│   │   └── prompts.py               # all prompt templates
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py   # summary buffer memory
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── ragas_evaluator.py       # RAGAS scoring
│   ├── guardrails/
│   │   ├── __init__.py
│   │   └── guardrails.py            # groundedness + NER + confidence
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rag_agent.py             # LangGraph StateGraph
│   └── serving/
│       ├── __init__.py
│       ├── api.py                   # FastAPI with SSE streaming
│       └── streamlit_app.py         # Streamlit frontend
├── tests/
│   ├── __init__.py
│   ├── test_extraction/
│   │   ├── __init__.py
│   │   └── test_pipeline.py
│   ├── test_retrieval/
│   │   ├── __init__.py
│   │   └── test_hybrid.py
│   └── test_generation/
│       ├── __init__.py
│       └── test_guardrails.py
├── scripts/
│   ├── ingest.py                    # CLI: batch PDF ingestion
│   └── evaluate.py                  # CLI: RAGAS benchmark
├── notebooks/
│   └── exploration.ipynb            # debugging notebook
├── logs/                            # auto-created at runtime
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                          # entry point: starts FastAPI
```

---

## 🔌 API Reference

### `GET /health`
Health check.
```json
{"status": "healthy", "index_exists": true, "documents_count": 5}
```

### `POST /ingest`
Upload PDF files for processing.
- **Content-Type**: `multipart/form-data`
- **Body**: `files` — one or more PDF files
```json
{"message": "Ingestion successful", "files_processed": 2, "total_chunks": 148}
```

### `POST /query`
SSE streaming query endpoint.
- **Body**: `{"query": "What is attention?", "history": []}`
- **Response**: Server-Sent Events stream with events: `token`, `sources`, `guardrails`, `done`

### `POST /query/sync`
Non-streaming query (fallback).
```json
{"answer": "...", "sources": [...], "guardrail_result": {...}}
```

### `GET /documents`
List ingested documents.

### `DELETE /documents`
Clear all documents and indexes.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Orchestration | LangChain (latest) |
| Agent / Flow | LangGraph |
| LLM | Groq API (llama3-70b-8192) with OpenAI fallback |
| Embeddings | BAAI/bge-small-en-v1.5 via HuggingFace (local) |
| Vector Store | FAISS (local, CPU, persisted) |
| Keyword Search | BM25Retriever (langchain_community) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Backend | FastAPI with async SSE streaming |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| PDF Extraction | marker-pdf, pdfplumber, PyMuPDF, Tesseract |
| Text Fixing | ftfy |
| Guardrails | spaCy (NER) + LLM-based checking |

---

## 📝 License

This project is built for educational and portfolio purposes.

---

Built with ❤️ by RAGForge

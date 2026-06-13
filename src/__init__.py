"""
RAGForge — Research Paper Q&A with Retrieval-Augmented Generation.

Top-level package exposing the core modules:
    - ``src.extraction``   — PDF → clean LangChain Documents
    - ``src.chunking``     — smart document chunking
    - ``src.embeddings``   — BGE embedding wrapper
    - ``src.vector_store`` — FAISS persistence layer
    - ``src.retrieval``    — hybrid retrieval, reranking, compression
    - ``src.generation``   — LLM factory and prompt templates
    - ``src.memory``       — conversation memory management
    - ``src.evaluation``   — RAGAS evaluation
    - ``src.guardrails``   — groundedness and hallucination checks
    - ``src.agents``       — LangGraph agentic RAG
    - ``src.serving``      — FastAPI backend + Streamlit frontend
"""

"""
RAGForge — Entry Point
=======================

Starts the FastAPI server using uvicorn.

Usage::

    python main.py

The server runs on the host and port specified in ``config.settings``
(default: ``0.0.0.0:8000``).  Hot-reload is enabled for development.
"""

import uvicorn
from config.settings import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )

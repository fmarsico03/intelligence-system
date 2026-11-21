"""Arranque del módulo RAG: ingesta automática si el namespace de Pinecone está vacío.

`ingest()` ya es idempotente (no reindexa si el namespace ya tiene vectores), pero es síncrona y hace
llamadas de red (leer los PDFs, Pinecone): se corre en un thread aparte para no bloquear el event loop de
FastAPI. Así `docker compose up` alcanza como único comando, sin un paso manual de ingesta.
"""

from __future__ import annotations

import asyncio

from app.rag.ingest import ingest


async def ensure_index_ready() -> None:
    await asyncio.to_thread(ingest)

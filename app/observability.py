"""Observabilidad con LangSmith.

Qué queda trazado:
  * Cada ejecución del grafo (start o resume) es una traza raíz `agentops-job` con un `run_id` propio, guardado
    en el job (campo `trace.run_ids`). Adentro aparecen los nodos (supervisor, human_approval, researcher,
    analyst, synthesizer), los sub-agentes ReAct, cada tool y cada llamada a Gemini con sus tokens de entrada y
    salida (LangSmith calcula el costo con su tabla de precios por modelo).
  * Las llamadas al LLM están decoradas con `@traceable` (app/orchestrator/llm.py: span `llm_fallback`), que
    registra qué modelo respondió y cuántos intentos de fallback hicieron falta.
  * Metadata `thread_id=<job_id>`: LangSmith agrupa en la vista *Threads* la corrida inicial y su reanudación
    después del HITL, así el costo total del job se ve junto.

Variables: LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT (default "agentops-api"),
LANGSMITH_ENDPOINT (solo si usás la región EU).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from app.config import settings

log = logging.getLogger("intelligence.observability")

DEFAULT_PROJECT = "agentops-api"


def init_observability() -> dict[str, Any]:
    """Valida la configuración de LangSmith al arrancar y deja el proyecto por defecto."""
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if enabled and not os.getenv("LANGSMITH_API_KEY"):
        log.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: se desactiva el tracing.")
        os.environ["LANGSMITH_TRACING"] = "false"
        enabled = False
    status = {"provider": "langsmith", "enabled": enabled, "project": os.environ["LANGSMITH_PROJECT"]}
    log.info("Observabilidad: %s", status)
    return status


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").lower() == "true"


def run_config(job_id: str, query: str, action: str, max_steps: int) -> tuple[dict[str, Any], str]:
    """RunnableConfig para una ejecución del grafo: thread de Redis + metadata/tags para LangSmith."""
    run_id = uuid.uuid4()
    config = {
        "configurable": {"thread_id": job_id},
        "recursion_limit": settings.recursion_limit,
        "run_name": "agentops-job",
        "run_id": run_id,
        "tags": ["agentops-api", f"action:{action}"],
        "metadata": {
            "job_id": job_id,
            "thread_id": job_id,  # agrupa start + resume en la vista Threads
            "action": action,
            "max_steps": max_steps,
            "query": query[:300],
        },
    }
    return config, str(run_id)


async def flush() -> None:
    """Espera a que se envíen las trazas pendientes (sin bloquear el event loop)."""
    if not tracing_enabled():
        return
    from langchain_core.tracers.langchain import wait_for_all_tracers

    await asyncio.to_thread(wait_for_all_tracers)

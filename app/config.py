"""Configuración centralizada (variables de entorno, con defaults razonables para desarrollo local).

Las variables específicas del módulo RAG (Pinecone, embeddings, chunking) viven en `app/rag/config.py`:
ese módulo se usa de forma independiente (ingesta, evaluación) además de desde el agente "knowledge",
así que mantiene su propia validación de env vars requeridas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    key_prefix: str = os.getenv("REDIS_KEY_PREFIX", "intelligence")

    # Worker
    run_embedded_worker: bool = _bool("RUN_EMBEDDED_WORKER", True)
    worker_concurrency: int = int(os.getenv("WORKER_CONCURRENCY", "5"))
    job_timeout_seconds: int = int(os.getenv("JOB_TIMEOUT_SECONDS", "600"))
    job_ttl_seconds: int = int(os.getenv("JOB_TTL_SECONDS", str(7 * 24 * 3600)))

    # Grafo
    default_max_steps: int = int(os.getenv("DEFAULT_MAX_STEPS", "8"))
    recursion_limit: int = int(os.getenv("GRAPH_RECURSION_LIMIT", "40"))

    # HITL: créditos que se aprueban solos (0 => toda búsqueda paga requiere aprobación humana)
    hitl_auto_approve_max_credits: int = int(os.getenv("HITL_AUTO_APPROVE_MAX_CREDITS", "0"))

    # Tavily
    tavily_credit_budget: int = int(os.getenv("TAVILY_CREDIT_BUDGET", "1500"))
    tavily_max_searches_per_run: int = int(os.getenv("TAVILY_MAX_SEARCHES_PER_RUN", "4"))

    @property
    def web_search_enabled(self) -> bool:
        return bool(os.getenv("TAVILY_API_KEY"))

    def key(self, *parts: str) -> str:
        return ":".join([self.key_prefix, *parts])


settings = Settings()

"""Factory del LLM (Gemini) con fallback asíncrono entre modelos (reutilizado del Módulo 6).

Cambios respecto del M6:
  * `acall_with_fallback` es async (usa `await` + `asyncio.sleep`), así nunca bloquea el event loop.
  * Está decorada con `@traceable`: en LangSmith aparece un span `llm_fallback:<etiqueta>` que envuelve la
    llamada a Gemini y registra en metadata qué modelo respondió y cuántos intentos hicieron falta.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from collections.abc import Awaitable, Callable
from datetime import date
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.config import settings  # noqa: F401  (carga .env)

# Evita el warning "Both GOOGLE_API_KEY and GEMINI_API_KEY are set".
if os.getenv("GOOGLE_API_KEY"):
    os.environ.pop("GEMINI_API_KEY", None)

log = logging.getLogger("agentops.llm")
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_FALLBACKS = "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "504", "DEADLINE_EXCEEDED", "overloaded")

T = TypeVar("T")


def today() -> str:
    """Fecha actual (ISO). Se inyecta en los prompts para que los agentes evalúen recencia."""
    return date.today().isoformat()


def candidate_models() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    fallbacks = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",") if m.strip()]
    return list(dict.fromkeys([primary, *fallbacks]))


@lru_cache(maxsize=16)
def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model or candidate_models()[0],
        temperature=temperature,
        max_retries=1,  # fallar rápido y pasar al siguiente modelo
        timeout=90,
    )


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _trace_inputs(inputs: dict) -> dict:
    # La lambda no aporta nada a la traza: solo registramos la etiqueta y los modelos candidatos.
    return {"label": inputs.get("label"), "candidates": candidate_models()}


@traceable(run_type="chain", name="llm_fallback", process_inputs=_trace_inputs)
async def acall_with_fallback(fn: Callable[[str], Awaitable[T]], label: str = "llm", rounds: int = 2) -> T:
    """Ejecuta `await fn(model_name)` probando cada modelo candidato ante errores transitorios."""
    last_exc: Exception | None = None
    attempts = 0
    for attempt in range(rounds):
        for model in candidate_models():
            attempts += 1
            try:
                result = await fn(model)
                run = get_current_run_tree()
                if run is not None:
                    run.metadata.update({"model_used": model, "attempts": attempts, "agent": label})
                return result
            except Exception as exc:  # noqa: BLE001
                if not _is_transient(exc):
                    raise
                last_exc = exc
                log.warning("[%s] modelo %s no disponible (%s). Probando el siguiente...", label, model, str(exc)[:80])
        await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Ningún modelo Gemini disponible: {last_exc}") from last_exc


def message_text(message: BaseMessage) -> str:
    """Extrae texto plano de un mensaje (Gemini puede devolver content como lista de partes)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)

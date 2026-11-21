"""Herramientas del Agente de Investigación (M6), adaptadas a ejecución concurrente.

- `web_search`: búsqueda real con Tavily (async), protegida por un guardián de créditos.
- `mock_search`: búsqueda simulada sobre un corpus local (fallback sin key, sin créditos o sin aprobación).

Cambio clave respecto del M6: el guardián de créditos vivía en un archivo JSON con un contador por proceso,
que con 5 jobs concurrentes (o varios workers) se pisaría. Ahora vive en Redis:
  * `agentops:tavily:credits_used`   -> contador global (presupuesto de la cuenta).
  * `agentops:tavily:run:<job_id>`   -> búsquedas hechas por ese job (límite por ejecución).
  * `agentops:tavily:cache:<sha1>`   -> caché de resultados por consulta normalizada (0 créditos al repetir).
La reserva de créditos es un script Lua: chequeo + incremento atómicos, sin carreras entre jobs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.config import settings
from app.redis_client import get_redis

MOCK_CORPUS = Path(__file__).resolve().parent.parent / "data" / "mock_corpus.json"
CREDITS_PER_SEARCH = 1  # search_depth="basic"
CACHE_TTL_SECONDS = 24 * 3600

_RESERVE_LUA = """
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
local run = tonumber(redis.call('GET', KEYS[2]) or '0')
local cost = tonumber(ARGV[1])
if used + cost > tonumber(ARGV[2]) then return -1 end
if run >= tonumber(ARGV[3]) then return -2 end
redis.call('INCRBY', KEYS[1], cost)
redis.call('INCR', KEYS[2])
redis.call('EXPIRE', KEYS[2], 86400)
return used + cost
"""


class TavilyCreditGuard:
    """Presupuesto global + límite por job, ambos en Redis."""

    @property
    def budget(self) -> int:
        return settings.tavily_credit_budget

    @property
    def max_per_run(self) -> int:
        return settings.tavily_max_searches_per_run

    def _keys(self, job_id: str) -> list[str]:
        return [settings.key("tavily", "credits_used"), settings.key("tavily", "run", job_id)]

    async def credits_used(self) -> int:
        return int(await get_redis().get(settings.key("tavily", "credits_used")) or 0)

    async def remaining(self) -> int:
        return self.budget - await self.credits_used()

    async def try_consume(self, job_id: str, query: str) -> tuple[bool, str]:
        redis = get_redis()
        result = await redis.eval(_RESERVE_LUA, 2, *self._keys(job_id), CREDITS_PER_SEARCH, self.budget, self.max_per_run)
        if result == -1:
            return False, f"presupuesto global agotado ({self.budget} créditos)"
        if result == -2:
            return False, f"límite de {self.max_per_run} búsquedas por ejecución alcanzado"
        entry = {"ts": datetime.now().isoformat(timespec="seconds"), "job_id": job_id, "query": query}
        await redis.lpush(settings.key("tavily", "history"), json.dumps(entry, ensure_ascii=False))
        await redis.ltrim(settings.key("tavily", "history"), 0, 199)
        return True, "ok"


credit_guard = TavilyCreditGuard()


def _mock_lookup(query: str, k: int = 3) -> list[dict]:
    corpus = json.loads(MOCK_CORPUS.read_text(encoding="utf-8"))
    terms = {t for t in query.lower().split() if len(t) > 3}
    scored = []
    for doc in corpus:
        haystack = f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}".lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


def _format_results(results: list[dict], source: str) -> str:
    if not results:
        return f"[{source}] Sin resultados."
    lines = [f"[{source}] {len(results)} resultados:"]
    for i, r in enumerate(results, 1):
        content = (r.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {r.get('title', 'sin título')}\n   URL: {r.get('url', 'n/a')}\n   {content[:700]}")
    return "\n".join(lines)


def _job_id(config: RunnableConfig | None) -> str:
    configurable = (config or {}).get("configurable", {})
    return str(configurable.get("job_id") or configurable.get("thread_id") or "anon")


@tool
def mock_search(query: str) -> str:
    """Búsqueda simulada sobre un corpus local de documentos. Usar solo si web_search no está disponible."""
    return _format_results(_mock_lookup(query), "mock_search")


@tool
async def web_search(query: str, config: RunnableConfig) -> str:
    """Busca información actualizada en la web (Tavily). Devuelve títulos, URLs y extractos.

    Cada llamada consume créditos: formulá consultas precisas y evitá repetirlas.
    """
    if not settings.web_search_enabled:
        return "TAVILY_API_KEY no configurada -> " + mock_search.invoke({"query": query})

    redis = get_redis()
    cache_key = settings.key("tavily", "cache", hashlib.sha1(" ".join(query.lower().split()).encode()).hexdigest())
    if cached := await redis.get(cache_key):
        return cached + "\n(resultado en caché: 0 créditos)"

    ok, reason = await credit_guard.try_consume(_job_id(config), query)
    if not ok:
        return f"Búsqueda web bloqueada ({reason}) -> " + mock_search.invoke({"query": query})

    from langchain_tavily import TavilySearch

    tavily = TavilySearch(max_results=5, search_depth="basic", include_answer=False)
    try:
        response = await tavily.ainvoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        return f"Error en Tavily ({exc}) -> " + mock_search.invoke({"query": query})
    results = response.get("results", []) if isinstance(response, dict) else []
    formatted = _format_results(results, "tavily")
    await redis.set(cache_key, formatted, ex=CACHE_TTL_SECONDS)
    return formatted

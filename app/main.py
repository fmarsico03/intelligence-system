"""API REST (FastAPI) del orquestador multi-agente.

  POST /tasks                -> encola la tarea y devuelve job_id al instante (202). No ejecuta el grafo.
  GET  /tasks/{id}           -> estado del job (PENDING / RUNNING / AWAITING_APPROVAL / DONE / FAILED).
  GET  /tasks                -> últimos jobs.
  POST /tasks/{id}/approve   -> respuesta humana al pedido de aprobación (ver app/hitl.py).
  GET  /health               -> Redis, cola, worker y observabilidad.
  GET  /graph                -> diagrama Mermaid del grafo.

Ningún endpoint bloquea el event loop: solo hacen operaciones async contra Redis (redis.asyncio).
La ejecución del grafo ocurre en el worker (app/worker.py).
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app import jobs
from app.config import settings
from app.graph import build_graph, open_graph
from app.hitl import router as hitl_router
from app.jobs import TERMINAL, JobStatus
from app.observability import flush, init_observability
from app.redis_client import close_redis, get_redis
from app.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agentops.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.observability = init_observability()
    async with AsyncExitStack() as stack:
        await get_redis().ping()
        app.state.worker = None
        if settings.run_embedded_worker:
            graph = await open_graph(stack)
            app.state.worker = Worker(graph)
            await app.state.worker.start()
        try:
            yield
        finally:
            if app.state.worker:
                await app.state.worker.stop()
            await flush()
    await close_redis()


app = FastAPI(
    title="Intelligence System API",
    version="1.0.0",
    description="Orquestador multi-agente (LangGraph + Gemini) con RAG híbrido, cola en Redis, HITL y "
                 "trazas en LangSmith.",
    lifespan=lifespan,
)
app.include_router(hitl_router)


class TaskCreate(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, examples=["¿Cuál fue la inflación mensual de Argentina "
                                                                       "en los últimos 3 meses y su promedio?"])
    max_steps: int = Field(settings.default_max_steps, ge=2, le=15, description="Tope de pasos del supervisor")


def _links(job_id: str) -> dict[str, str]:
    return {"self": f"/tasks/{job_id}", "approve": f"/tasks/{job_id}/approve"}


@app.post("/tasks", status_code=202, tags=["tasks"])
async def create_task(body: TaskCreate) -> dict:
    """Crea el job en Redis (PENDING), lo encola y responde enseguida con su id."""
    job = await jobs.create_job(body.query, body.max_steps)
    return {"job_id": job["id"], "status": job["status"], "links": _links(job["id"])}


@app.get("/tasks/{job_id}", tags=["tasks"])
async def get_task(job_id: str, events: bool = Query(False, description="Incluir historial de transiciones")) -> dict:
    job = await jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} no encontrado")
    job["terminal"] = job["status"] in TERMINAL
    job["links"] = _links(job_id)
    if job["status"] == JobStatus.AWAITING_APPROVAL:
        job["next_action"] = f"POST /tasks/{job_id}/approve con {{'approved': true|false, 'reviewer', 'comment'}}"
    if events:
        job["events"] = await jobs.get_events(job_id)
    return job


@app.get("/tasks", tags=["tasks"])
async def list_tasks(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    keep = ("id", "status", "query", "created_at", "duration_seconds", "error")
    return [{k: j.get(k) for k in keep} for j in await jobs.list_jobs(limit)]


@app.get("/health", tags=["ops"])
async def health() -> dict:
    try:
        redis_ok = bool(await get_redis().ping())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"Redis no disponible: {exc}") from exc
    return {
        "status": "ok",
        "redis": redis_ok,
        "queue_length": await jobs.queue_length(),
        "worker": "embedded" if app.state.worker else "external",
        "worker_concurrency": settings.worker_concurrency if app.state.worker else None,
        "observability": app.state.observability,
        "web_search_enabled": settings.web_search_enabled,
    }


@app.get("/graph", tags=["ops"])
async def graph_diagram() -> dict:
    return {"mermaid": build_graph().get_graph().draw_mermaid()}

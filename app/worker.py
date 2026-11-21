"""Worker: consume la cola de Redis, ejecuta el grafo en segundo plano y actualiza el estado del job.

Dos modos (mismo código):
  * Embebido en la API (RUN_EMBEDDED_WORKER=true, default): N corrutinas dentro del event loop de uvicorn.
    El grafo es 100% async (ainvoke de Gemini/Tavily/Redis), así que no bloquea los endpoints.
  * Proceso separado: `python -m app.worker` (servicio `worker` en docker-compose). Escala horizontalmente:
    varios workers pueden consumir la misma cola; el "claim" atómico PENDING->RUNNING evita doble ejecución.

Manejo de errores (para que ningún cliente quede en polling infinito):
  * Cualquier excepción del grafo -> FAILED con `error` + `error_detail` (tipo y traceback recortado).
  * Timeout por ejecución (JOB_TIMEOUT_SECONDS) -> FAILED.
  * Worker detenido a mitad de un job -> FAILED.
  * Reaper periódico: un job RUNNING sin actividad por más de timeout+gracia (ej. el proceso murió) -> FAILED.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from contextlib import AsyncExitStack
from typing import Any

from langgraph.types import Command

from app import jobs
from app.config import settings
from app.graph import initial_state, open_graph
from app.jobs import JobStatus
from app.observability import flush, init_observability, run_config

log = logging.getLogger("agentops.worker")
REAPER_INTERVAL_SECONDS = 30
REAPER_GRACE_SECONDS = 60


class Worker:
    def __init__(self, graph: Any, concurrency: int = settings.worker_concurrency) -> None:
        self.graph = graph
        self.concurrency = concurrency
        self._tasks: list[asyncio.Task] = []
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------ ciclo de vida
    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._consume(i), name=f"worker-{i}") for i in range(self.concurrency)]
        self._tasks.append(asyncio.create_task(self._reaper(), name="reaper"))
        log.info("Worker iniciado con %d consumidores", self.concurrency)

    async def stop(self) -> None:
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _consume(self, idx: int) -> None:
        while not self._stopping.is_set():
            try:
                msg = await jobs.dequeue(timeout=1)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  (Redis caído: reintentar sin matar el consumidor)
                log.exception("[worker-%d] error leyendo la cola", idx)
                await asyncio.sleep(2)
                continue
            if msg:
                await self.process(msg)

    # ------------------------------------------------------------------ ejecución de un job
    async def process(self, msg: dict[str, Any]) -> None:
        job_id, action = msg["job_id"], msg.get("action", "start")
        job = await jobs.get_job(job_id)
        if job is None:
            log.warning("Job %s inexistente (¿expiró?), se descarta el mensaje", job_id)
            return

        # Claim atómico: solo un worker pasa de PENDING a RUNNING
        run_started = jobs.now()
        claimed = await jobs.transition(
            job_id,
            JobStatus.RUNNING,
            expected={JobStatus.PENDING},
            detail=f"worker tomó el job ({action})",
            run_started_at=run_started,
            **({"started_at": run_started} if action == "start" else {}),
        )
        if not claimed:
            log.info("Job %s no está PENDING (%s): se ignora el mensaje", job_id, job.get("status"))
            return

        await jobs.incr_runs(job_id)
        config, run_id = run_config(job_id, job["query"], action, job["max_steps"])
        trace = job.get("trace") or {"provider": "langsmith", "run_ids": []}
        trace["run_ids"].append(run_id)
        await jobs.update(job_id, trace=trace)

        graph_input = initial_state(job["query"], job["max_steps"]) if action == "start" else Command(resume=msg["payload"])
        processing = float(job.get("processing_seconds") or 0)
        try:
            await asyncio.wait_for(self.graph.ainvoke(graph_input, config), timeout=settings.job_timeout_seconds)
            snapshot = await self.graph.aget_state({"configurable": {"thread_id": job_id}})
            processing += jobs.now() - run_started

            if snapshot.interrupts:  # el nodo human_approval llamó a interrupt()
                await jobs.transition(
                    job_id,
                    JobStatus.AWAITING_APPROVAL,
                    expected={JobStatus.RUNNING},
                    detail="esperando aprobación humana (búsqueda paga)",
                    approval_request=snapshot.interrupts[0].value,
                    processing_seconds=round(processing, 3),
                )
                return

            values = snapshot.values
            finished = jobs.now()
            await jobs.transition(
                job_id,
                JobStatus.DONE,
                expected={JobStatus.RUNNING},
                detail="completado",
                result=values.get("final_answer", ""),
                decisions=values.get("decisions", []),
                contributions=[
                    {k: c[k] for k in ("agent", "iteration", "instruction", "tool_calls", "output")}
                    for c in values.get("contributions", [])
                ],
                search_approval=values.get("search_approval", ""),
                approval=values.get("approval_info") or None,
                finished_at=finished,
                duration_seconds=round(finished - job["created_at"], 3),
                processing_seconds=round(processing, 3),
            )
        except asyncio.TimeoutError:
            await self._fail(job_id, f"Timeout: la ejecución superó {settings.job_timeout_seconds}s", None)
        except asyncio.CancelledError:
            await self._fail(job_id, "El worker se detuvo mientras ejecutaba el job", None)
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Job %s falló", job_id)
            await self._fail(job_id, f"{type(exc).__name__}: {exc}", exc)

    async def _fail(self, job_id: str, error: str, exc: Exception | None) -> None:
        detail = None
        if exc is not None:
            detail = {"type": type(exc).__name__, "traceback": traceback.format_exc().splitlines()[-15:]}
        finished = jobs.now()
        job = await jobs.get_job(job_id) or {}
        await jobs.transition(
            job_id,
            JobStatus.FAILED,
            expected={JobStatus.RUNNING, JobStatus.PENDING},
            detail=error[:200],
            error=error[:2000],
            error_detail=detail,
            finished_at=finished,
            duration_seconds=round(finished - job.get("created_at", finished), 3),
        )

    # ------------------------------------------------------------------ reaper
    async def _reaper(self) -> None:
        limit = settings.job_timeout_seconds + REAPER_GRACE_SECONDS
        while not self._stopping.is_set():
            try:
                for job in await jobs.list_jobs(limit=200):
                    started = float(job.get("run_started_at") or 0)
                    if job["status"] == JobStatus.RUNNING and started and jobs.now() - started > limit:
                        await jobs.transition(
                            job["id"],
                            JobStatus.FAILED,
                            expected={JobStatus.RUNNING},
                            detail="reaper: sin actividad",
                            error=f"El job quedó RUNNING más de {limit}s (¿se cayó el worker?)",
                            finished_at=jobs.now(),
                        )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("Error en el reaper")
            await asyncio.sleep(REAPER_INTERVAL_SECONDS)


async def run_standalone() -> None:
    """Entry point del worker como proceso separado."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_observability()
    async with AsyncExitStack() as stack:
        graph = await open_graph(stack)
        worker = Worker(graph)
        await worker.start()
        try:
            await asyncio.Event().wait()
        finally:
            await worker.stop()
            await flush()


if __name__ == "__main__":
    try:
        asyncio.run(run_standalone())
    except KeyboardInterrupt:
        pass

"""5 escenarios concurrentes contra la API, pensados para cubrir las tres rutas del Supervisor
(knowledge / researcher+analyst / mixto) y ambos desenlaces del HITL (aprobar / rechazar).

1. Lanza 5 `POST /tasks` en paralelo y mide cuánto tarda la API en devolver el job_id (debe ser ~ms).
2. Hace polling de cada job. Si queda en AWAITING_APPROVAL, lo aprueba o rechaza según
   `APPROVAL_BY_INDEX` (alternado), simulando al revisor humano.
3. Imprime una tabla por job y las latencias p50/p95 end-to-end y de procesamiento.

El costo por ejecución y la latencia p95 "oficiales" se leen en el dashboard de LangSmith (ver README);
este script sirve para disparar la corrida y para contrastar los números (ver langsmith_report.py).

    python scripts/load_test.py                 # los 5 escenarios, HITL alternado aprobar/rechazar
    python scripts/load_test.py --approve-all    # fuerza aprobar el HITL en los 5 jobs
    python scripts/load_test.py --reject-all     # fuerza rechazar el HITL en los 5 jobs
    python scripts/load_test.py --manual         # no responde el HITL: queda para aprobarlo a mano
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

# Los 5 escenarios de la Fase 3 (ver README): cubren knowledge puro, knowledge+researcher, researcher+
# analyst con HITL aprobado, researcher+analyst con HITL rechazado, y un "No lo sé" del RAG con fallback
# al researcher.
QUERIES = [
    "What Python version does Google ADK require, and how do I run my agent as a REST API with "
    "`adk api_server`?",
    "Explicá brevemente, según la documentación de ADK, cómo se despliega un agente como servidor REST. "
    "Además, buscá cuál es el precio actual del dólar blue en Argentina.",
    "¿Cuál es la tasa de política monetaria actual del Banco Central de Chile? Calculá cuánto rinde "
    "1.000.000 CLP a esa tasa en un año con capitalización mensual.",
    "Buscá el precio actual del cobre por libra y analizá el sentimiento de los titulares recientes "
    "sobre el cobre.",
    "¿La documentación de ADK explica cómo integrar reconocimiento facial en los agentes? Si no está "
    "documentado, buscá en la web si existe alguna librería de Python para eso.",
]
# True = aprobar el HITL (búsqueda paga), False = rechazar (solo mock_search). Alternado por diseño para
# cubrir ambos desenlaces en la misma corrida.
APPROVAL_BY_INDEX = [True, True, True, False, True]
TERMINAL = {"DONE", "FAILED"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def pct(values: list[float], p: float) -> float:
    """Percentil con interpolación lineal (mismo criterio que numpy.percentile por defecto)."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


async def run_job(client: httpx.AsyncClient, idx: int, query: str, approve: bool, args) -> dict:
    t0 = time.perf_counter()
    r = await client.post("/tasks", json={"query": query, "max_steps": args.max_steps})
    r.raise_for_status()
    enqueue_ms = (time.perf_counter() - t0) * 1000
    job_id = r.json()["job_id"]
    print(f"[{idx}] encolado {job_id} en {enqueue_ms:.0f} ms")

    while True:
        job = (await client.get(f"/tasks/{job_id}")).json()
        status = job["status"]
        if status in TERMINAL:
            break
        if status == "AWAITING_APPROVAL" and not args.manual:
            req = job.get("approval_request", {})
            decision = {"approved": approve, "reviewer": "load-test",
                        "comment": f"auto ({req.get('estimated_credits')} créditos estimados)"}
            await client.post(f"/tasks/{job_id}/approve", json=decision)
            print(f"[{idx}] {'aprobado' if approve else 'rechazado'} (HITL)")
        if status == "AWAITING_APPROVAL" and args.manual:
            print(f"[{idx}] esperando aprobación manual: POST /tasks/{job_id}/approve")
            await asyncio.sleep(10)
        await asyncio.sleep(args.poll)

    total = time.perf_counter() - t0
    print(f"[{idx}] {status} en {total:.1f}s")
    return {
        "idx": idx,
        "job_id": job_id,
        "query": query,
        "status": status,
        "enqueue_ms": enqueue_ms,
        "total_s": total,
        "processing_s": job.get("processing_seconds"),
        "approval": job.get("search_approval"),
        "steps": len(job.get("decisions") or []),
        "error": job.get("error"),
        "run_ids": (job.get("trace") or {}).get("run_ids", []),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--poll", type=float, default=2.0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--approve-all", action="store_true", help="Aprobar el HITL en los 5 jobs")
    group.add_argument("--reject-all", action="store_true", help="Rechazar el HITL en los 5 jobs")
    group.add_argument("--manual", action="store_true", help="No responder el HITL (aprobación a mano)")
    parser.add_argument("--out", default="screenshots/load_test_results.json")
    args = parser.parse_args()

    if args.approve_all:
        approvals = [True] * len(QUERIES)
    elif args.reject_all:
        approvals = [False] * len(QUERIES)
    else:
        approvals = APPROVAL_BY_INDEX

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        (await client.get("/health")).raise_for_status()
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(run_job(client, i + 1, q, approvals[i], args) for i, q in enumerate(QUERIES))
        )
        wall = time.perf_counter() - t0

    print("\n" + "=" * 100)
    print(f"{'#':<3}{'job_id':<34}{'estado':<8}{'encolar':>9}{'total':>9}{'proceso':>9}  {'HITL':<13}{'pasos':>5}")
    for r in results:
        proc = f"{r['processing_s']:.1f}s" if r["processing_s"] else "-"
        print(f"{r['idx']:<3}{r['job_id']:<34}{r['status']:<8}{r['enqueue_ms']:>7.0f}ms{r['total_s']:>8.1f}s"
              f"{proc:>9}  {str(r['approval']):<13}{r['steps']:>5}")
        if r["error"]:
            print(f"   error: {r['error']}")

    ok = [r for r in results if r["status"] == "DONE"]
    totals = [r["total_s"] for r in ok]
    procs = [r["processing_s"] for r in ok if r["processing_s"]]
    enq = [r["enqueue_ms"] for r in results]
    summary = {
        "jobs": len(results),
        "done": len(ok),
        "failed": len(results) - len(ok),
        "wall_clock_s": round(wall, 2),
        "enqueue_ms_p95": round(pct(enq, 95), 1),
        "e2e_s_p50": round(pct(totals, 50), 2),
        "e2e_s_p95": round(pct(totals, 95), 2),
        "processing_s_p50": round(pct(procs, 50), 2),
        "processing_s_p95": round(pct(procs, 95), 2),
    }
    print("-" * 100)
    for k, v in summary.items():
        print(f"{k:>18}: {v}")
    print("\nLa suma de tiempos individuales vs. el wall-clock muestra que los jobs corrieron en paralelo:")
    print(f"   suma={sum(totals):.1f}s  wall-clock={wall:.1f}s")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    asyncio.run(main())

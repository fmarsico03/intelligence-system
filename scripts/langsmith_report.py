"""Lee de LangSmith las trazas de la última prueba de carga y resume costo, latencia y tokens por nodo.

Complementa las capturas del dashboard: toma los `run_ids` que guardó `load_test.py` y consulta la API de
LangSmith (los mismos datos que muestra la UI). Cada job tiene 2 trazas raíz cuando pasó por HITL:
`action:start` (hasta el interrupt, ~segundos) y `action:resume` (el trabajo real); si no hubo HITL hay una
sola traza. El costo del job es la suma de sus trazas.

    python scripts/langsmith_report.py                      # usa screenshots/load_test_results.json
    python scripts/langsmith_report.py --file otra_corrida.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()
warnings.filterwarnings("ignore", message=".*list_runs\\(\\) is deprecated.*")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else float("nan")
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="screenshots/load_test_results.json")
    args = parser.parse_args()

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    client = Client()

    per_job, root_latencies = [], []
    node_tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # prompt, completion, llm calls
    node_latency: dict[str, list[float]] = defaultdict(list)

    for job in data["results"]:
        cost, tokens, latency = 0.0, 0, 0.0
        for run in client.list_runs(run_ids=job["run_ids"]):
            run_latency = (run.end_time - run.start_time).total_seconds() if run.end_time else 0.0
            root_latencies.append(run_latency)
            latency += run_latency
            cost += float(run.total_cost or 0)
            tokens += int(run.total_tokens or 0)
            # Llamadas al LLM dentro de la traza, agrupadas por nodo del grafo (metadata langgraph_node)
            for llm in client.list_runs(trace_id=run.trace_id, run_type="llm"):
                # checkpoint_ns = "researcher:<id>|agent:<id>" -> se atribuye al nodo de primer nivel del grafo
                meta = (llm.extra or {}).get("metadata", {})
                ns = meta.get("langgraph_checkpoint_ns") or meta.get("langgraph_node") or "?"
                node = ns.split("|")[0].split(":")[0]
                agg = node_tokens[node]
                agg[0] += int(llm.prompt_tokens or 0)
                agg[1] += int(llm.completion_tokens or 0)
                agg[2] += 1
            for child in client.list_runs(trace_id=run.trace_id, run_type="chain"):
                meta = (child.extra or {}).get("metadata", {})
                top_level = "|" not in (meta.get("langgraph_checkpoint_ns") or "")
                if child.name == meta.get("langgraph_node") and top_level and child.end_time:
                    node_latency[child.name].append((child.end_time - child.start_time).total_seconds())
        per_job.append({"job": job["job_id"][:8], "status": job["status"], "cost": cost, "tokens": tokens,
                        "latency": latency})

    print(f"{'job':<10}{'estado':<8}{'costo USD':>12}{'tokens':>10}{'latencia':>10}")
    for j in per_job:
        print(f"{j['job']:<10}{j['status']:<8}{j['cost']:>12.6f}{j['tokens']:>10}{j['latency']:>9.1f}s")
    costs = [j["cost"] for j in per_job]
    lat = [j["latency"] for j in per_job]
    print("-" * 50)
    print(f"costo medio por ejecución: {statistics.fmean(costs):.6f} USD  (total {sum(costs):.6f})")
    print(f"latencia por job  p50={statistics.median(lat):.1f}s  p95={p95(lat):.1f}s")
    print(f"latencia trazas raíz (start+resume) p95={p95(root_latencies):.1f}s")

    print("\nTokens por nodo del grafo:")
    total = sum(a[0] + a[1] for a in node_tokens.values()) or 1
    for node, (pt, ct, calls) in sorted(node_tokens.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        print(f"  {node:<15} in={pt:>7} out={ct:>6} llamadas={calls:>3}  ({(pt + ct) / total:.0%} del total)")
    print("\nLatencia media por nodo:")
    for node, values in sorted(node_latency.items(), key=lambda kv: -statistics.fmean(kv[1])):
        print(f"  {node:<15} media={statistics.fmean(values):>6.1f}s  max={max(values):>6.1f}s  n={len(values)}")


if __name__ == "__main__":
    main()

"""Tests de integración SIN LLM: Redis real, AsyncRedisSaver real, interrupt/resume real y endpoints reales.

Los nodos de agentes se reemplazan por stubs deterministas para no gastar tokens ni créditos.
Requiere Redis 8 (o Redis Stack) en REDIS_URL; si no está, los tests se saltean.

    pytest -q
"""

from __future__ import annotations

import asyncio
import os
import uuid

os.environ["REDIS_KEY_PREFIX"] = f"agentops-test-{uuid.uuid4().hex[:6]}"
os.environ["LANGSMITH_TRACING"] = "false"

import httpx  # noqa: E402
import pytest  # noqa: E402
from contextlib import AsyncExitStack  # noqa: E402

import app.graph as graph_module  # noqa: E402
from app import jobs  # noqa: E402
from app.jobs import JobStatus  # noqa: E402
from app.main import app  # noqa: E402
from app.redis_client import close_redis, get_redis  # noqa: E402
from app.worker import Worker  # noqa: E402


def _redis_available() -> bool:
    async def ping():
        try:
            return await get_redis().ping()
        except Exception:  # noqa: BLE001
            return False
        finally:
            await close_redis()

    return asyncio.run(ping())


pytestmark = pytest.mark.skipif(not _redis_available(), reason="Redis no disponible")


# ------------------------------------------------------------------------------------- stubs
async def fake_supervisor(state):
    step = state.get("step_count", 0) + 1
    nxt = "researcher" if not state.get("research_findings") else "FINISH"
    return {"next_agent": nxt, "step_count": step, "current_instruction": "buscar datos",
            "decisions": [{"step": step, "next": nxt, "reasoning": "stub", "validation_passed": True, "forced": False}]}


async def fake_researcher(state, config):
    tools = "web_search" if state.get("search_approval") != "rejected" else "mock_search"
    return {"research_findings": f"hallazgos via {tools}",
            "contributions": [{"agent": "researcher", "iteration": 1, "instruction": "x", "output": tools,
                               "tool_calls": [tools]}]}


async def fake_synth(state):
    return {"final_answer": f"respuesta final ({state['research_findings']})", "task_completed": True}


async def exploding_supervisor(state):
    raise ValueError("boom: el LLM explotó")


async def _make_worker(stack: AsyncExitStack, monkeypatch, supervisor=fake_supervisor) -> Worker:
    monkeypatch.setattr(graph_module, "supervisor_node", supervisor)
    monkeypatch.setattr(graph_module, "research_node", fake_researcher)
    monkeypatch.setattr(graph_module, "synthesizer_node", fake_synth)
    graph = await graph_module.open_graph(stack)
    return Worker(graph, concurrency=1)


async def _drain(worker: Worker) -> None:
    while msg := await jobs.dequeue(timeout=1):
        await worker.process(msg)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# -------------------------------------------------------------------------------------- tests
def test_failed_state_on_exception(monkeypatch):
    async def scenario():
        async with AsyncExitStack() as stack, _client() as client:
            worker = await _make_worker(stack, monkeypatch, supervisor=exploding_supervisor)
            job_id = (await client.post("/tasks", json={"query": "consulta que falla"})).json()["job_id"]
            await _drain(worker)
            job = (await client.get(f"/tasks/{job_id}")).json()
            assert job["status"] == JobStatus.FAILED
            assert "boom" in job["error"]
            assert job["terminal"] is True
        await close_redis()

    asyncio.run(scenario())


def test_hitl_approve_flow(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")  # hace "crítica" la búsqueda

    async def scenario():
        async with AsyncExitStack() as stack, _client() as client:
            worker = await _make_worker(stack, monkeypatch)
            job_id = (await client.post("/tasks", json={"query": "inflación reciente"})).json()["job_id"]
            assert (await client.get(f"/tasks/{job_id}")).json()["status"] == JobStatus.PENDING

            await _drain(worker)
            job = (await client.get(f"/tasks/{job_id}")).json()
            assert job["status"] == JobStatus.AWAITING_APPROVAL
            assert job["approval_request"]["type"] == "paid_web_search"

            r = await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "reviewer": "franco"})
            assert r.status_code == 202
            # una segunda aprobación del mismo pedido se rechaza (transición atómica)
            assert (await client.post(f"/tasks/{job_id}/approve", json={"approved": True})).status_code == 409

            await _drain(worker)
            job = (await client.get(f"/tasks/{job_id}?events=true")).json()
            assert job["status"] == JobStatus.DONE
            assert job["search_approval"] == "approved"
            assert "web_search" in job["result"]
            assert job["runs"] == 2  # start + resume, mismo thread en Redis
            assert [e["status"] for e in job["events"]] == [
                "PENDING", "RUNNING", "AWAITING_APPROVAL", "PENDING", "RUNNING", "DONE"]
        await close_redis()

    asyncio.run(scenario())


def test_hitl_reject_uses_mock_only(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    async def scenario():
        async with AsyncExitStack() as stack, _client() as client:
            worker = await _make_worker(stack, monkeypatch)
            job_id = (await client.post("/tasks", json={"query": "inflación reciente"})).json()["job_id"]
            await _drain(worker)
            await client.post(f"/tasks/{job_id}/approve", json={"approved": False, "comment": "sin presupuesto"})
            await _drain(worker)
            job = (await client.get(f"/tasks/{job_id}")).json()
            assert job["status"] == JobStatus.DONE
            assert job["search_approval"] == "rejected"
            assert "mock_search" in job["result"]
        await close_redis()

    asyncio.run(scenario())


def test_no_approval_needed_without_tavily(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def scenario():
        async with AsyncExitStack() as stack, _client() as client:
            worker = await _make_worker(stack, monkeypatch)
            job_id = (await client.post("/tasks", json={"query": "sin búsqueda paga"})).json()["job_id"]
            await _drain(worker)
            job = (await client.get(f"/tasks/{job_id}")).json()
            assert job["status"] == JobStatus.DONE
            assert job["search_approval"] == "not_required"
        await close_redis()

    asyncio.run(scenario())


def test_approve_unknown_job_404():
    async def scenario():
        async with _client() as client:
            r = await client.post("/tasks/nope/approve", json={"approved": True})
            assert r.status_code == 404
        await close_redis()

    asyncio.run(scenario())

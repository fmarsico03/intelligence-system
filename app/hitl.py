"""Human-in-the-loop: aprobación obligatoria antes de gastar créditos en búsquedas pagas (Tavily).

¿Qué es "crítico"? La primera vez que el supervisor deriva al researcher con búsqueda web real habilitada
(TAVILY_API_KEY presente), el job puede consumir hasta TAVILY_MAX_SEARCHES_PER_RUN créditos de una cuenta
con presupuesto limitado. Si ese costo estimado supera HITL_AUTO_APPROVE_MAX_CREDITS (default 0 → siempre),
el grafo se detiene con `interrupt()`:

  1. El checkpoint queda persistido en Redis (AsyncRedisSaver) y el worker marca el job AWAITING_APPROVAL,
     exponiendo el pedido de aprobación en GET /tasks/{id}.
  2. Un humano responde con POST /tasks/{id}/approve {"approved": true|false, ...}.
  3. El job se reencola y el worker reanuda el MISMO thread con `Command(resume=decision)`.
  4. Si se aprobó, el researcher usa Tavily. Si se rechazó, el researcher se construye SIN `web_search`
     (lo impone el código, no el prompt) y trabaja solo con el corpus local.

La aprobación cubre todo el job: los refinamientos posteriores del researcher no vuelven a preguntar.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app import jobs
from app.config import settings
from app.jobs import JobStatus
from app.orchestrator.state import OrchestratorState
from app.orchestrator.tools import credit_guard
from app.orchestrator.tools.search_tools import CREDITS_PER_SEARCH


# --------------------------------------------------------------------------------------- criterio
def estimated_credits() -> int:
    return settings.tavily_max_searches_per_run * CREDITS_PER_SEARCH


def needs_approval_gate(state: OrchestratorState) -> bool:
    """La arista supervisor -> researcher pasa por el gate mientras la aprobación no esté decidida."""
    return state.get("search_approval", "pending") == "pending"


def is_critical() -> bool:
    return settings.web_search_enabled and estimated_credits() > settings.hitl_auto_approve_max_credits


# ------------------------------------------------------------------------------------------ nodo
async def human_approval_node(state: OrchestratorState) -> dict:
    """Nodo de aprobación humana. Al reanudar, LangGraph re-ejecuta el nodo y `interrupt` devuelve la decisión."""
    if not is_critical():
        reason = "sin TAVILY_API_KEY (solo mock_search)" if not settings.web_search_enabled else "costo bajo el umbral"
        return {
            "search_approval": "not_required",
            "approval_info": {"approved": True, "reviewer": "auto", "comment": reason},
        }

    request = {
        "type": "paid_web_search",
        "question": "¿Aprobás que el Agente de Investigación use búsqueda web paga (Tavily)?",
        "estimated_credits": estimated_credits(),
        "credits_remaining": await credit_guard.remaining(),
        "credit_budget": credit_guard.budget,
        "user_query": state["user_query"],
        "researcher_instruction": state.get("current_instruction", ""),
        "on_reject": "el researcher trabaja solo con mock_search (corpus local, sin costo)",
    }
    decision = interrupt(request) or {}

    approved = bool(decision.get("approved", False))
    info = {
        "approved": approved,
        "reviewer": str(decision.get("reviewer") or "anonymous"),
        "comment": str(decision.get("comment") or ""),
        "estimated_credits": request["estimated_credits"],
    }
    verdict = "APROBÓ" if approved else "RECHAZÓ"
    return {
        "search_approval": "approved" if approved else "rejected",
        "approval_info": info,
        "messages": [
            AIMessage(content=f"[human_approval] {info['reviewer']} {verdict} la búsqueda paga. {info['comment']}",
                      name="human_approval")
        ],
    }


# ---------------------------------------------------------------------------------------- endpoint
router = APIRouter(tags=["human-in-the-loop"])


class ApprovalDecision(BaseModel):
    approved: bool = Field(True, description="true = permitir búsqueda paga · false = solo corpus local")
    reviewer: str = Field("anonymous", max_length=100)
    comment: str = Field("", max_length=1000)


@router.post("/tasks/{job_id}/approve", status_code=202)
async def approve_task(job_id: str, decision: ApprovalDecision) -> dict:
    """Responde el pedido de aprobación de un job en AWAITING_APPROVAL y lo reencola para reanudarlo."""
    job = await jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} no encontrado")

    applied = await jobs.transition(
        job_id,
        JobStatus.PENDING,
        expected={JobStatus.AWAITING_APPROVAL},
        detail=f"{'aprobado' if decision.approved else 'rechazado'} por {decision.reviewer}",
        approval=decision.model_dump(),
    )
    if not applied:
        current = (await jobs.get_job(job_id) or {}).get("status")
        raise HTTPException(409, f"El job no está esperando aprobación (estado actual: {current})")

    await jobs.enqueue(job_id, "resume", decision.model_dump())
    return {"job_id": job_id, "status": JobStatus.PENDING, "approved": decision.approved}

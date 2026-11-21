"""Esquema de estado compartido del orquestador (M6) + campos de Human-in-the-loop (M7).

Hereda de MessagesState (campo `messages` con reducer add_messages). Todo este estado se persiste en
Redis en cada super-step mediante el checkpointer `AsyncRedisSaver` (ver app/graph.py).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

AgentName = Literal["researcher", "analyst"]
RouteName = Literal["researcher", "analyst", "FINISH"]
# "pending": todavía no se decidió · "not_required": no hay búsqueda paga (sin TAVILY_API_KEY o auto-aprobada)
SearchApproval = Literal["pending", "approved", "rejected", "not_required"]


class Contribution(TypedDict):
    agent: str
    iteration: int
    instruction: str
    output: str
    tool_calls: list[str]


class SupervisorDecisionLog(TypedDict):
    step: int
    next: str
    reasoning: str
    validation_passed: bool
    forced: bool


class ApprovalInfo(TypedDict, total=False):
    approved: bool
    reviewer: str
    comment: str
    estimated_credits: int


class OrchestratorState(MessagesState):
    user_query: str

    # Control de flujo
    next_agent: RouteName
    current_instruction: str
    validation_feedback: str
    step_count: int
    max_steps: int
    revision_counts: dict[str, int]

    # Resultados de cada dominio (última versión)
    research_findings: str
    analysis_results: str

    # Trazabilidad (append-only)
    contributions: Annotated[list[Contribution], operator.add]
    decisions: Annotated[list[SupervisorDecisionLog], operator.add]

    # Human-in-the-loop (M7): aprobación del gasto en búsquedas pagas
    search_approval: SearchApproval
    approval_info: ApprovalInfo

    # Cierre
    task_completed: bool
    final_answer: str

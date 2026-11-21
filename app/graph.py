"""Orquestador multi-agente del M6 compilado con checkpointer de Redis y un gate Human-in-the-loop.

    START -> supervisor -┬-> human_approval -> researcher -┐   (gate solo mientras la aprobación esté pendiente)
                         ├-> researcher ------------------─┤
                         ├-> analyst ----------------------┴-> supervisor (ciclo jerárquico)
                         └-> synthesizer -> END

El `thread_id` de LangGraph es el `job_id` de la API: cada job tiene su propia historia de checkpoints en
Redis, y por eso puede pausarse (interrupt) y reanudarse desde otro proceso o después de un reinicio.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.hitl import human_approval_node, needs_approval_gate
from app.orchestrator.agents import analyst_node, research_node, supervisor_node, synthesizer_node
from app.orchestrator.state import OrchestratorState

CHECKPOINT_TTL_MINUTES = settings.job_ttl_seconds // 60


def route_from_supervisor(
    state: OrchestratorState,
) -> Literal["human_approval", "researcher", "analyst", "synthesizer"]:
    nxt = state.get("next_agent", "FINISH")
    if nxt == "researcher":
        return "human_approval" if needs_approval_gate(state) else "researcher"
    if nxt == "analyst":
        return "analyst"
    return "synthesizer"


def build_graph(checkpointer=None):
    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("researcher", research_node)
    builder.add_node("analyst", analyst_node)
    # EXTENSION POINT: acá se agrega builder.add_node("knowledge", knowledge_node) + la rama correspondiente
    # en route_from_supervisor + builder.add_edge("knowledge", "supervisor") cuando se integre el RAG (fase
    # posterior).
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "human_approval": "human_approval",
            "researcher": "researcher",
            "analyst": "analyst",
            "synthesizer": "synthesizer",
        },
    )
    builder.add_edge("human_approval", "researcher")
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("synthesizer", END)
    return builder.compile(checkpointer=checkpointer)


async def open_graph(stack: AsyncExitStack):
    """Crea el AsyncRedisSaver (índices RediSearch incluidos) y compila el grafo. Se cierra con el stack."""
    saver = await stack.enter_async_context(
        AsyncRedisSaver.from_conn_string(settings.redis_url, ttl={"default_ttl": CHECKPOINT_TTL_MINUTES})
    )
    await saver.asetup()
    return build_graph(saver)


def initial_state(query: str, max_steps: int) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "next_agent": "researcher",
        "current_instruction": "",
        "validation_feedback": "",
        "step_count": 0,
        "max_steps": max_steps,
        "revision_counts": {},
        "research_findings": "",
        "analysis_results": "",
        "contributions": [],
        "decisions": [],
        "search_approval": "pending",
        "approval_info": {},
        "task_completed": False,
        "final_answer": "",
    }

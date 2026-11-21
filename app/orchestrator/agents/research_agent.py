"""Agente de Búsqueda / Investigación (M6), versión async.

Herramientas acotadas: `web_search` (Tavily) y `mock_search`. La decisión humana del nodo HITL se aplica
EN CÓDIGO: si la búsqueda paga fue rechazada, el agente se construye sin `web_search` (no depende del LLM).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from app.orchestrator.agents._common import contribution, summarize_agent_run
from app.orchestrator.llm import acall_with_fallback, get_llm, today
from app.orchestrator.state import OrchestratorState
from app.orchestrator.tools import mock_search, web_search

RESEARCH_PROMPT = """Sos el AGENTE DE INVESTIGACIÓN de un sistema multi-agente.
Tu único trabajo es conseguir información factual de fuentes externas para la tarea asignada.

Reglas:
- Usá `web_search` (si está disponible) con consultas precisas. Tenés un presupuesto de créditos: hacé como
  MÁXIMO 3 búsquedas y no repitas consultas equivalentes. Usá `mock_search` si web_search falla o no está.
- NO hagas cálculos ni interpretaciones: eso lo hace el Agente de Análisis.
- Si encontrás datos numéricos, transcribilos EXACTOS con su período/unidad.
- Si la información es insuficiente o contradictoria entre fuentes, decilo explícitamente.

Formato de salida (obligatorio):
## Hallazgos
- <dato concreto> (fuente: <URL>)
## Datos numéricos estructurados
<lista "etiqueta: valor unidad" en orden cronológico, o "N/A">
## Textos para análisis de sentimiento
<titulares/extractos textuales relevantes, uno por línea, o "N/A">
## Limitaciones
<qué faltó o qué es dudoso>
"""


@lru_cache(maxsize=16)
def build_research_agent(model: str, allow_web: bool):
    tools = [web_search, mock_search] if allow_web else [mock_search]
    # checkpointer=False: el sub-agente no hereda el RedisSaver del grafo padre (su estado interno es efímero).
    return create_react_agent(
        get_llm(model), tools=tools, prompt=RESEARCH_PROMPT, name="researcher", checkpointer=False
    )


async def research_node(state: OrchestratorState, config: RunnableConfig) -> dict:
    instruction = state.get("current_instruction") or state["user_query"]
    allow_web = state.get("search_approval") != "rejected"
    parts = [
        f"Fecha de hoy: {today()}. Priorizá los datos MÁS RECIENTES publicados a esta fecha.",
        f"Tarea asignada por el supervisor:\n{instruction}",
    ]
    if not allow_web:
        parts.append("NOTA: la búsqueda web paga fue RECHAZADA por un revisor humano. Usá solo mock_search "
                     "y declaralo en Limitaciones.")
    if state.get("validation_feedback") and state.get("research_findings"):
        parts.append(f"Esto es un REFINAMIENTO. Feedback del supervisor:\n{state['validation_feedback']}")
        parts.append(f"Tu entrega anterior (mejorala, no la repitas):\n{state['research_findings']}")

    task = [HumanMessage(content="\n\n".join(parts))]
    job_id = config.get("configurable", {}).get("thread_id", "anon")
    sub_config = {"recursion_limit": 12, "configurable": {"job_id": job_id}}
    result = await acall_with_fallback(
        lambda model: build_research_agent(model, allow_web).ainvoke({"messages": task}, sub_config),
        label="researcher",
    )
    output, tools_used = summarize_agent_run(result["messages"])

    return {
        "research_findings": output,
        "contributions": [contribution("researcher", state.get("step_count", 0), instruction, output, tools_used)],
        "messages": [AIMessage(content=output, name="researcher")],
    }

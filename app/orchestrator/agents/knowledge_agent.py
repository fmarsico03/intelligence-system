"""Agente de Conocimiento: RAG híbrido (BM25 + Pinecone) sobre la documentación técnica indexada.

A diferencia de researcher/analyst, NO es un agente ReAct con tool-loop: llama directo a la cadena RAG
(retrieve + generación grounded de app.rag.rag), que ya valida la veracidad de la respuesta ("No lo sé"
si no está en el contexto) y la tipa con Pydantic (`RAGResponse`). Más simple, menos tokens, y la
validación de veracidad ya viene incorporada en la cadena.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.orchestrator.agents._common import contribution
from app.orchestrator.state import OrchestratorState
from app.rag.rag import get_rag_response


async def knowledge_node(state: OrchestratorState) -> dict:
    instruction = state.get("current_instruction") or state["user_query"]
    if state.get("validation_feedback") and state.get("knowledge_findings"):
        instruction = (
            f"{instruction}\n\nEsto es un REFINAMIENTO. Feedback del supervisor:\n"
            f"{state['validation_feedback']}\n\nTu entrega anterior (mejorala, no la repitas):\n"
            f"{state['knowledge_findings']}"
        )

    result = await get_rag_response(instruction)
    if result.respuesta.strip().lower() == "no lo sé":
        output = "No lo sé: la documentación técnica indexada no cubre esta consulta."
    else:
        refs = ", ".join(result.referencias) or "sin referencias"
        output = f"{result.respuesta}\n\nFuentes (doc. técnica indexada): {refs}"

    return {
        "knowledge_findings": output,
        "contributions": [
            contribution("knowledge", state.get("step_count", 0), instruction, output, ["hybrid_retrieval"])
        ],
        "messages": [AIMessage(content=output, name="knowledge")],
    }

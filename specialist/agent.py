"""Specialist Agent: answers support-triage tasks from the knowledge base.

`handle_task` is the single integration point between the A2A server and RAG.
It returns a SpecialistResult or raises TaskFailure(code, message).
"""

import os
import time

from a2a_protocol.models import Source, SpecialistResult, TaskFailure, TaskRequest
from rag import retriever
from rag.errors import RagError
from rag.pipeline import answer_question

SLOW_SIMULATION_SECONDS = 60

AGENT_CARD = {
    "name": "Support Specialist Agent",
    "description": "Classifies IT support requests and recommends a resolution "
    "using the company support knowledge base (RAG).",
    "version": "0.2.0",
    "skills": [
        {
            "id": "triage_ticket",
            "description": "Given a support request and the categories a ticket form accepts, "
            "return a ticket category, resolution notes, and the knowledge-base sources used.",
        }
    ],
    "endpoints": {"submit": "POST /tasks", "status": "GET /tasks/{task_id}"},
    "task_statuses": ["submitted", "working", "completed", "failed"],
}


def warm_up() -> None:
    """Load the index and models up front so the first task isn't slow."""
    retriever.warm_up()
    if not os.getenv("GROQ_API_KEY"):
        print("[Specialist] WARNING: GROQ_API_KEY is not set; tasks will fail with missing_api_key.")


def _match_category(kb_category: str, allowed: list[str]) -> str | None:
    """Map the KB's category onto the form's options. No fuzzy mapping: an Email
    issue is not filed as Software; it comes back as None for manual routing."""
    if not allowed:
        return kb_category
    return next((option for option in allowed if option.lower() == kb_category.lower()), None)


def handle_task(request: TaskRequest) -> SpecialistResult:
    # Fault injection for the demo video: lets us show timeout/failure on demand.
    simulate = request.metadata.get("simulate")
    if simulate == "fail":
        raise TaskFailure("simulated_failure", "Specialist was asked to simulate a failure.")
    if simulate == "slow":
        time.sleep(SLOW_SIMULATION_SECONDS)

    try:
        answer = answer_question(request.question)
    except RagError as exc:
        raise TaskFailure(exc.code, exc.message) from exc

    return SpecialistResult(
        category=_match_category(answer.kb_category, request.allowed_categories),
        kb_category=answer.kb_category,
        resolution=answer.resolution,
        escalate=answer.escalate,
        confidence=answer.confidence,
        sources=[
            Source(
                source=chunk.source,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                score=round(chunk.vector_score, 3),
                rerank_score=round(chunk.rerank_score, 3),
            )
            for chunk in answer.sources
        ],
    )

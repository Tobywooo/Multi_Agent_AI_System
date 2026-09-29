"""Requester Agent: a rule-based coordinator.

User request -> A2A task to the Specialist -> poll for the result ->
validate it -> fill and submit the support form with Playwright -> verify.
"""

import os
import time

from a2a_protocol.client import (
    A2AClient,
    A2AError,
    AgentUnavailableError,
    TaskFailedError,
    TaskTimeoutError,
)
from a2a_protocol.models import SpecialistResult, Task, TaskRequest
from automation.ticket_form import TicketResult, TicketSubmissionError, submit_ticket

DEFAULT_SPECIALIST_URL = os.getenv("SPECIALIST_URL", "http://127.0.0.1:8001")

# The options in the support form's Category dropdown. The Specialist must pick
# from these; ticket_form re-checks against the live page before submitting.
FORM_CATEGORIES = ["Account Access", "Hardware", "Software", "Network"]


class RequesterError(Exception):
    """The Specialist's answer can't be used to file a ticket."""


def log(message: str) -> None:
    print(f"[Requester] {message}", flush=True)


def build_task(user_request: str, simulate: str | None) -> TaskRequest:
    metadata = {"simulate": simulate} if simulate else {}
    return TaskRequest(
        question=user_request, allowed_categories=FORM_CATEGORIES, metadata=metadata
    )


def validate_result(result: SpecialistResult) -> str:
    """Return the category to submit, or raise if the answer isn't usable for the form."""
    if result.category is None:
        raise RequesterError(
            f"The knowledge base classifies this as '{result.kb_category}', which the support "
            f"form does not offer ({', '.join(FORM_CATEGORIES)}). Route this ticket manually."
        )
    if result.category not in FORM_CATEGORIES:
        raise RequesterError(f"Specialist returned an unknown category '{result.category}'.")
    if not result.resolution.strip():
        raise RequesterError("Specialist returned an empty resolution.")
    return result.category


def format_history(task: Task) -> str:
    # Fast tasks can finish between polls, so show the Specialist's own record of every status.
    start = task.history[0].timestamp
    return " -> ".join(
        f"{event.status.value} (+{(event.timestamp - start).total_seconds():.2f}s)"
        for event in task.history
    )


def format_resolution_notes(result: SpecialistResult) -> str:
    cited = ", ".join(f"{s.source} ({s.section})" for s in result.sources)
    notes = result.resolution.strip()
    if result.escalate:
        notes += "\n\nEscalation recommended."
    return f"{notes}\n\nSources: {cited}"


def run(
    user_request: str,
    *,
    specialist_url: str = DEFAULT_SPECIALIST_URL,
    timeout: float = 30.0,
    poll_interval: float = 1.0,
    simulate: str | None = None,
    headed: bool = False,
    slow_mo: int = 0,
    hold_seconds: float = 0,
) -> TicketResult | None:
    """Run the full workflow. Returns the verified ticket, or None if any step failed."""
    log(f"Received request: {user_request!r}")
    client = A2AClient(specialist_url)

    try:
        card = client.get_agent_card()
        log(f"Connected to {card['name']} at {specialist_url}")

        ack = client.submit(build_task(user_request, simulate))
        log(f"Task submitted -> task_id={ack.task_id}, status={ack.status.value}")

        started = time.monotonic()

        def on_status_change(task: Task) -> None:
            log(f"Task {task.task_id} status: {task.status.value} ({time.monotonic() - started:.1f}s)")

        task = client.wait_for_result(
            ack.task_id,
            timeout=timeout,
            poll_interval=poll_interval,
            on_status_change=on_status_change,
        )
    except AgentUnavailableError as exc:
        log(f"FAILED: Specialist Agent is unavailable. {exc}")
        return None
    except TaskTimeoutError as exc:
        log(f"TIMEOUT: {exc}. Giving up; no ticket was submitted.")
        return None
    except TaskFailedError as exc:
        log(f"Task lifecycle: {format_history(exc.task)}")
        log(f"FAILED: Specialist task failed {exc} No ticket was submitted.")
        return None
    except A2AError as exc:
        log(f"FAILED: A2A communication error: {exc}")
        return None

    result = task.result
    log(f"Task lifecycle: {format_history(task)}")
    log(f"Specialist answer: category={result.category!r} (kb: {result.kb_category!r}), "
        f"confidence={result.confidence:.2f}, escalate={result.escalate}")
    log(f"  resolution: {result.resolution}")
    for source in result.sources:
        log(f"  source: {source.chunk_id} (similarity {source.score:.3f}, rerank {source.rerank_score})")

    try:
        category = validate_result(result)
    except RequesterError as exc:
        log(f"CANNOT FILE TICKET: {exc}")
        return None

    log("Opening support app and filling the ticket form...")
    try:
        ticket = submit_ticket(
            issue=user_request,
            category=category,
            resolution=format_resolution_notes(result),
            headless=not headed,
            slow_mo=slow_mo,
            hold_seconds=hold_seconds,
        )
    except TicketSubmissionError as exc:
        log(f"FAILED: Ticket submission did not verify. {exc}")
        return None

    log(f"SUCCESS: Ticket #{ticket.ticket_id} submitted and verified ({ticket.category}).")
    return ticket

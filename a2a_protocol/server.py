"""Generic A2A task server: accepts tasks, runs them in the background, reports status.

The server knows nothing about RAG. An agent plugs in a `handler` that turns a
TaskRequest into a SpecialistResult (or raises TaskFailure).

Endpoints:
    GET  /.well-known/agent.json   agent card (who I am, what I can do)
    POST /tasks                    submit a task -> 202 + task_id, status "submitted"
    GET  /tasks/{task_id}          current status, history, and result or error
"""

import logging
import threading
import traceback
import uuid
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from a2a_protocol.models import (
    SpecialistResult,
    StatusEvent,
    Task,
    TaskAck,
    TaskError,
    TaskFailure,
    TaskRequest,
    TaskStatus,
    utc_now,
)

logger = logging.getLogger("a2a.server")

TaskHandler = Callable[[TaskRequest], SpecialistResult]


class TaskStore:
    """In-memory, thread-safe task table. Tasks are lost when the server restarts."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def create(self, request: TaskRequest) -> Task:
        now = utc_now()
        task = Task(
            task_id=uuid.uuid4().hex[:12],
            status=TaskStatus.SUBMITTED,
            request=request,
            created_at=now,
            updated_at=now,
            history=[StatusEvent(status=TaskStatus.SUBMITTED, timestamp=now)],
        )
        with self._lock:
            self._tasks[task.task_id] = task
        return task.model_copy(deep=True)

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.model_copy(deep=True) if task else None

    def update(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: SpecialistResult | None = None,
        error: TaskError | None = None,
    ) -> None:
        now = utc_now()
        with self._lock:
            task = self._tasks[task_id]
            task.status = status
            task.updated_at = now
            task.history.append(StatusEvent(status=status, timestamp=now))
            task.result = result
            task.error = error
        logger.info("task %s -> %s", task_id, status.value)


def create_app(agent_card: dict[str, Any], handler: TaskHandler) -> FastAPI:
    app = FastAPI(title=agent_card["name"], description=agent_card.get("description", ""))
    store = TaskStore()

    def run_task(task_id: str, request: TaskRequest) -> None:
        store.update(task_id, TaskStatus.WORKING)
        try:
            result = handler(request)
        except TaskFailure as exc:
            store.update(task_id, TaskStatus.FAILED, error=TaskError(code=exc.code, message=exc.message))
        except Exception as exc:
            traceback.print_exc()
            error = TaskError(code="internal_error", message=f"{type(exc).__name__}: {exc}")
            store.update(task_id, TaskStatus.FAILED, error=error)
        else:
            store.update(task_id, TaskStatus.COMPLETED, result=result)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        # Opening the agent's URL in a browser lands on the interactive API docs.
        return RedirectResponse(url="/docs")

    @app.get("/.well-known/agent.json")
    def get_agent_card() -> dict[str, Any]:
        return agent_card

    @app.post("/tasks", status_code=202, response_model=TaskAck)
    def submit_task(request: TaskRequest) -> TaskAck:
        task = store.create(request)
        # Daemon thread so a long-running task never blocks server shutdown.
        threading.Thread(target=run_task, args=(task.task_id, request), daemon=True).start()
        return TaskAck(task_id=task.task_id, status=task.status, created_at=task.created_at)

    @app.get("/tasks/{task_id}", response_model=Task)
    def get_task(task_id: str) -> Task:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'")
        return task

    return app

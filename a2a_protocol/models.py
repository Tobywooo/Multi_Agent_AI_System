"""Message schemas for our A2A-style protocol, shared by both agents.

Lifecycle of a task:

    Requester                         Specialist
    POST /tasks  {TaskRequest}  --->  store task, start worker thread
                 <---  202 {TaskAck}  status = submitted
    GET /tasks/{id}             --->
                 <---  {Task}         status = working
    GET /tasks/{id}             --->
                 <---  {Task}         status = completed + result
                                      (or failed + error)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)


# ---- Requester -> Specialist -------------------------------------------------


class TaskRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's support request, verbatim.")
    allowed_categories: list[str] = Field(
        default_factory=list,
        description="Categories the target form accepts; the Specialist must pick from these.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Out-of-band options, e.g. {'simulate': 'slow' | 'fail'} for demos.",
    )


# ---- Specialist -> Requester -------------------------------------------------


class Source(BaseModel):
    source: str  # file name, e.g. "hardware.md"
    section: str  # e.g. "Resolution"
    chunk_id: str
    score: float  # vector (cosine) similarity, 0..1
    rerank_score: float | None = None  # cross-encoder relevance; higher is better


class SpecialistResult(BaseModel):
    """What the RAG pipeline must produce. The Requester types these values into the form."""

    category: str | None = Field(
        description="One of allowed_categories, or None if none of them fits."
    )
    kb_category: str | None = Field(
        description="Category the knowledge base assigns (may be outside allowed_categories)."
    )
    resolution: str = Field(min_length=1)
    escalate: bool = False
    confidence: float = Field(description="Best vector similarity among cited chunks (0..1).")
    sources: list[Source]


class TaskError(BaseModel):
    code: str  # machine-readable, e.g. "insufficient_context", "llm_error"
    message: str


class TaskFailure(Exception):
    """Raise from a task handler to end the task as `failed` with a specific error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class TaskAck(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime


class StatusEvent(BaseModel):
    status: TaskStatus
    timestamp: datetime


class Task(BaseModel):
    task_id: str
    status: TaskStatus
    request: TaskRequest
    created_at: datetime
    updated_at: datetime
    history: list[StatusEvent]
    result: SpecialistResult | None = None
    error: TaskError | None = None

"""Client side of the A2A protocol: submit a task, then poll until it finishes or times out."""

import time
from typing import Any, Callable

import requests

from a2a_protocol.models import Task, TaskAck, TaskError, TaskRequest, TaskStatus


class A2AError(Exception):
    """Base class for everything that can go wrong talking to another agent."""


class AgentUnavailableError(A2AError):
    """The remote agent could not be reached."""


class TaskNotFoundError(A2AError):
    """The remote agent does not know the task ID."""


class TaskFailedError(A2AError):
    def __init__(self, task: Task):
        self.task = task
        self.error = task.error or TaskError(code="unknown", message="Task failed without details")
        super().__init__(f"[{self.error.code}] {self.error.message}")


class TaskTimeoutError(A2AError):
    def __init__(self, task_id: str, timeout: float, last_status: TaskStatus | None):
        self.task_id = task_id
        self.last_status = last_status
        status = last_status.value if last_status else "unknown"
        super().__init__(f"Task {task_id} did not finish within {timeout:g}s (last status: {status})")


class A2AClient:
    def __init__(self, base_url: str, request_timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            response = requests.request(
                method, f"{self.base_url}{path}", timeout=self.request_timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise AgentUnavailableError(
                f"Could not reach agent at {self.base_url} ({type(exc).__name__})"
            ) from exc
        if response.status_code == 404:
            raise TaskNotFoundError(response.json().get("detail", "Not found"))
        if not response.ok:
            raise A2AError(f"{method} {path} returned HTTP {response.status_code}: {response.text}")
        return response

    def get_agent_card(self) -> dict[str, Any]:
        return self._request("GET", "/.well-known/agent.json").json()

    def submit(self, request: TaskRequest) -> TaskAck:
        response = self._request("POST", "/tasks", json=request.model_dump(mode="json"))
        return TaskAck.model_validate(response.json())

    def get(self, task_id: str) -> Task:
        return Task.model_validate(self._request("GET", f"/tasks/{task_id}").json())

    def wait_for_result(
        self,
        task_id: str,
        *,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
        on_status_change: Callable[[Task], None] | None = None,
    ) -> Task:
        """Poll until the task completes. Raises TaskFailedError or TaskTimeoutError otherwise."""
        deadline = time.monotonic() + timeout
        last_status: TaskStatus | None = None

        while True:
            task = self.get(task_id)
            if task.status != last_status:
                last_status = task.status
                if on_status_change:
                    on_status_change(task)

            if task.status == TaskStatus.COMPLETED:
                return task
            if task.status == TaskStatus.FAILED:
                raise TaskFailedError(task)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskTimeoutError(task_id, timeout, last_status)
            time.sleep(min(poll_interval, remaining))

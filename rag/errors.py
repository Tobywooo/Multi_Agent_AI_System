"""Errors the RAG pipeline raises; the Specialist maps each `code` to a failed A2A task."""


class RagError(Exception):
    code = "rag_error"

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class InsufficientContextError(RagError):
    """Nothing in the knowledge base is relevant enough to answer from."""

    code = "insufficient_context"


class LLMError(RagError):
    """The LLM call failed. Codes: missing_api_key, llm_auth_error, llm_unavailable,
    llm_invalid_output."""

    code = "llm_error"

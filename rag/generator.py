"""Grounded answer generation with Groq.

The LLM sees only the retrieved passages and must reply with JSON matching a
schema built per request: kb_category is limited to categories that appear in
the retrieved passages, and cited_chunks to their chunk IDs. On models with
strict structured outputs, Groq enforces that schema during decoding.
"""

import json
import os

import groq
from pydantic import BaseModel, ValidationError

from rag.config import GROQ_MODEL, LLM_TIMEOUT_SECONDS, STRICT_JSON_MODELS
from rag.errors import LLMError
from rag.retriever import RetrievedChunk

SYSTEM_PROMPT = """\
You are an IT support specialist. Triage the user's support request using ONLY \
the knowledge-base passages provided. Do not use outside knowledge.

Reply with a JSON object:
- answerable: false if none of the passages address the user's issue.
- kb_category: the Ticket Category of the procedure that best fits the issue. \
If the request involves a possible security incident (compromise, suspicious \
activity, phishing, unverifiable identity), prefer the Security procedure.
- resolution: 1-3 sentences of resolution notes for the support ticket, written \
for the support technician in the third person ("the user"). Every action must \
come from the passages; do not add steps, contacts, mailboxes, or tools the \
passages do not mention. Include any required verification step.
- escalate: true only if the request itself already meets an escalation \
criterion in the passages (for example suspected compromise or multiple users \
affected). Do not escalate issues that basic troubleshooting may still fix.
- cited_chunks: the IDs of the passages you relied on."""


# LLMs like typographic punctuation (non-breaking hyphens, curly quotes). Plain
# ASCII is safer for ticket text and for Windows consoles (cp1252).
_PUNCTUATION = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", " ": " ", "…": "...",
})


def clean_text(text: str) -> str:
    return text.translate(_PUNCTUATION).strip()


class LLMAnswer(BaseModel):
    answerable: bool
    kb_category: str
    resolution: str
    escalate: bool
    cited_chunks: list[str]


def _json_schema(chunks: list[RetrievedChunk]) -> dict:
    categories = sorted({c.ticket_category for c in chunks if c.ticket_category})
    return {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "kb_category": {"type": "string", "enum": categories},
            "resolution": {"type": "string"},
            "escalate": {"type": "boolean"},
            "cited_chunks": {
                "type": "array",
                "items": {"type": "string", "enum": [c.chunk_id for c in chunks]},
            },
        },
        "required": ["answerable", "kb_category", "resolution", "escalate", "cited_chunks"],
        "additionalProperties": False,
    }


def _user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    passages = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)
    return f"Support request: {question}\n\nKnowledge-base passages:\n\n{passages}"


def _call_groq(client: groq.Groq, question: str, chunks: list[RetrievedChunk]) -> str:
    schema = _json_schema(chunks)
    if GROQ_MODEL in STRICT_JSON_MODELS:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "triage_answer", "strict": True, "schema": schema},
        }
    else:
        # Older models: plain JSON mode, schema described in the prompt, validated below.
        response_format = {"type": "json_object"}

    extra = {"reasoning_effort": "low"} if "gpt-oss" in GROQ_MODEL else {}
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(question, chunks)},
            ],
            response_format=response_format,
            temperature=0,
            **extra,
        )
    except groq.AuthenticationError as exc:
        raise LLMError("Groq rejected the API key. Check GROQ_API_KEY in .env.", "llm_auth_error") from exc
    except (groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError) as exc:
        raise LLMError(f"Groq is unavailable: {type(exc).__name__}", "llm_unavailable") from exc
    except groq.APIStatusError as exc:
        raise LLMError(f"Groq returned HTTP {exc.status_code}: {exc.message}", "llm_error") from exc
    return completion.choices[0].message.content or ""


def _parse(raw: str, chunks: list[RetrievedChunk]) -> LLMAnswer:
    answer = LLMAnswer.model_validate(json.loads(raw))
    categories = {c.ticket_category for c in chunks}
    if answer.kb_category not in categories:
        raise ValueError(f"kb_category '{answer.kb_category}' is not in the retrieved passages")
    known_ids = {c.chunk_id for c in chunks}
    answer.cited_chunks = [cid for cid in answer.cited_chunks if cid in known_ids]
    answer.resolution = clean_text(answer.resolution)
    if not answer.resolution:
        raise ValueError("resolution is empty")
    return answer


def generate(question: str, chunks: list[RetrievedChunk], attempts: int = 2) -> LLMAnswer:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set. Add it to the .env file.", "missing_api_key")

    client = groq.Groq(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS, max_retries=1)
    last_error: Exception | None = None
    for _ in range(attempts):
        raw = _call_groq(client, question, chunks)
        try:
            return _parse(raw, chunks)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
    raise LLMError(f"LLM reply was not valid triage JSON: {last_error}", "llm_invalid_output")

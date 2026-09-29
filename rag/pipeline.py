"""RAG entry point: question -> retrieve + rerank -> grounded LLM answer.

Usage (test RAG on its own, no agents):
    python -m rag.pipeline "My keyboard has stopped working."
    python -m rag.pipeline "..." --retrieval-only    # no LLM / API key needed
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass

from rag.errors import InsufficientContextError, RagError
from rag.generator import generate
from rag.retriever import RetrievedChunk, retrieve, warm_up


@dataclass
class RagAnswer:
    kb_category: str
    resolution: str
    escalate: bool
    confidence: float  # best vector similarity among the cited chunks, 0..1
    sources: list[RetrievedChunk]  # the chunks the LLM cited


def answer_question(question: str) -> RagAnswer:
    chunks = retrieve(question)
    answer = generate(question, chunks)
    if not answer.answerable:
        # Second relevance check: the passages cleared the similarity gate, but the
        # LLM judged they don't actually address the request.
        raise InsufficientContextError(
            "The retrieved knowledge-base passages do not address this request."
        )

    cited = [c for c in chunks if c.chunk_id in answer.cited_chunks] or chunks
    return RagAnswer(
        kb_category=answer.kb_category,
        resolution=answer.resolution,
        escalate=answer.escalate,
        confidence=round(max(c.vector_score for c in cited), 3),
        sources=cited,
    )


def main() -> None:
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every character
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM step")
    args = parser.parse_args()

    warm_up()
    try:
        if args.retrieval_only:
            for c in retrieve(args.question):
                print(f"rerank {c.rerank_score:7.3f}  vec {c.vector_score:.3f}  {c.chunk_id}")
            return
        answer = answer_question(args.question)
    except RagError as exc:
        print(f"[{exc.code}] {exc.message}")
        raise SystemExit(1)

    result = asdict(answer)
    for source in result["sources"]:
        source.pop("text")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

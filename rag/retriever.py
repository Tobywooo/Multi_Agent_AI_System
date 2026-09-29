"""Two-stage retrieval: FAISS vector search, relevance gate, cross-encoder rerank.

Advanced technique: Intelligent Reranking (rag_techniques #17). The bi-encoder
embeds query and chunk separately, so similar-sounding chunks from the wrong
doc can win (e.g. "someone hacked my account" -> account_access rather than
security). A cross-encoder reads the query and chunk together and scores
their relevance directly, which fixes that ordering.
"""

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag.config import CANDIDATE_K, RELEVANCE_THRESHOLD, RERANK_TOP_N, RERANKER_MODEL
from rag.errors import InsufficientContextError
from rag.vector_store import load_vector_store


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    section: str
    ticket_category: str | None
    text: str  # contextual header + body, exactly as embedded
    vector_score: float  # cosine similarity, 0..1
    rerank_score: float  # cross-encoder logit; higher is better, not bounded


@lru_cache(maxsize=1)
def _store():
    return load_vector_store()


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def warm_up() -> None:
    """Load the index and both models now so the first request isn't slow."""
    _store()
    _reranker()


def retrieve(question: str) -> list[RetrievedChunk]:
    hits = _store().similarity_search_with_score(question, k=CANDIDATE_K)
    relevant = [(doc, score) for doc, score in hits if score >= RELEVANCE_THRESHOLD]
    if not relevant:
        best = hits[0][1] if hits else 0.0
        raise InsufficientContextError(
            f"No knowledge-base passage is relevant to this request "
            f"(best similarity {best:.2f} < threshold {RELEVANCE_THRESHOLD})."
        )

    rerank_scores = _reranker().predict([(question, doc.page_content) for doc, _ in relevant])
    chunks = [
        RetrievedChunk(
            chunk_id=doc.metadata["chunk_id"],
            source=doc.metadata["source"],
            section=doc.metadata["section"],
            ticket_category=doc.metadata["ticket_category"],
            text=doc.page_content,
            vector_score=float(vector_score),
            rerank_score=float(rerank_score),
        )
        for (doc, vector_score), rerank_score in zip(relevant, rerank_scores)
    ]
    chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    return chunks[:RERANK_TOP_N]

"""Embedding model and FAISS vector store helpers shared by ingest and retrieval."""

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL, VECTOR_STORE_DIR


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # Normalized vectors + inner product = cosine similarity, so search scores
    # land in a readable 0..1 range we can later threshold on.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: list[Document]) -> FAISS:
    return FAISS.from_documents(
        chunks,
        get_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def save_vector_store(store: FAISS) -> None:
    store.save_local(str(VECTOR_STORE_DIR))


def load_vector_store() -> FAISS:
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"No vector store found at {VECTOR_STORE_DIR}. Run `python -m rag.ingest` first."
        )
    # The docstore is pickled; safe here because we only load an index we built ourselves.
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )

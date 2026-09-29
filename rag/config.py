"""Shared settings for the RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
TEST_CASES_PATH = PROJECT_ROOT / "test_cases.json"

load_dotenv(PROJECT_ROOT / ".env")

# ---- Indexing ----------------------------------------------------------------

# Small, fast, runs locally on CPU -- no API key needed for embeddings.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Sections are split on markdown headers first; these limits only apply as a
# fallback when a single section is unusually long.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ---- Retrieval ---------------------------------------------------------------

# Cast a wide net with the vector index, then let the cross-encoder pick the best.
CANDIDATE_K = 10
RERANK_TOP_N = 4
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cosine similarity below this means "not about anything in the KB". Measured:
# off-topic questions score ~0.0, real support issues ~0.28-0.65. (Rerank logits
# don't separate these as cleanly, so the gate uses the vector score.)
RELEVANCE_THRESHOLD = 0.2

# ---- Generation --------------------------------------------------------------

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# Models where Groq guarantees the reply matches our JSON schema (constrained decoding).
STRICT_JSON_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
LLM_TIMEOUT_SECONDS = 20

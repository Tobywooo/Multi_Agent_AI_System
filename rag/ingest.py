"""Build the FAISS vector store from the knowledge base.

Pipeline: load markdown docs -> split by section -> prepend contextual chunk
headers -> embed -> save to disk.

Advanced technique: Contextual Chunk Headers (rag_techniques #10). Every KB
doc shares the same section layout, so a bare "Resolution" or "Escalation"
chunk is nearly identical across docs and says nothing about which issue it
belongs to. Prepending the document title, section name, and ticket category
gives each chunk's embedding that missing context.

Usage:
    python -m rag.ingest            # build and save the index
    python -m rag.ingest --check    # also run test_cases.json against it
    python -m rag.ingest --compare  # check with vs. without contextual headers
"""

import argparse
import json
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    KNOWLEDGE_BASE_DIR,
    TEST_CASES_PATH,
    VECTOR_STORE_DIR,
)
from rag.vector_store import create_vector_store, save_vector_store

TICKET_CATEGORY_SECTION = "Ticket Category"


def load_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    paths = sorted(kb_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown files found in {kb_dir}")
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in paths
    ]


def extract_ticket_category(markdown: str) -> str | None:
    match = re.search(rf"^##\s+{TICKET_CATEGORY_SECTION}\s*\n+\s*(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else None


def build_chunk_header(doc_title: str, section: str, ticket_category: str | None) -> str:
    lines = [f"Document: {doc_title}", f"Section: {section}"]
    if ticket_category:
        lines.append(f"Ticket Category: {ticket_category}")
    return "\n".join(lines)


def split_documents(documents: list[Document], use_headers: bool = True) -> list[Document]:
    section_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc_title"), ("##", "section")],
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for doc in documents:
        source = doc.metadata["source"]
        ticket_category = extract_ticket_category(doc.page_content)

        for section_doc in section_splitter.split_text(doc.page_content):
            doc_title = section_doc.metadata.get("doc_title", source)
            section = section_doc.metadata.get("section", "Overview")
            # The category already appears in every chunk header and in metadata,
            # so a standalone one-word "Hardware" chunk would only add noise.
            if section == TICKET_CATEGORY_SECTION:
                continue

            header = build_chunk_header(doc_title, section, ticket_category)
            for i, body in enumerate(size_splitter.split_text(section_doc.page_content)):
                chunks.append(
                    Document(
                        page_content=f"{header}\n\n{body}" if use_headers else body,
                        metadata={
                            "source": source,
                            "doc_title": doc_title,
                            "section": section,
                            "ticket_category": ticket_category,
                            "chunk_id": f"{Path(source).stem}:{section.lower().replace(' ', '_')}:{i}",
                        },
                    )
                )
    return chunks


def run_test_cases(store: FAISS, label: str, k: int = 3) -> int:
    """Print top-k retrievals per test case; return how many had the expected category at rank 1."""
    test_cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    hits = 0
    print(f"\n=== Retrieval check: {label} ===")
    for case in test_cases:
        results = store.similarity_search_with_score(case["request"], k=k)
        top_category = results[0][0].metadata["ticket_category"] if results else None
        hit = top_category == case["expected_category"]
        hits += hit
        print(f"\n[{'PASS' if hit else 'MISS'}] {case['request']}  (expected: {case['expected_category']})")
        for doc, score in results:
            print(f"    {score:.3f}  {doc.metadata['chunk_id']}  -> {doc.metadata['ticket_category']}")
    print(f"\n{label}: {hits}/{len(test_cases)} test cases retrieved the expected category at rank 1")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge-base vector store.")
    parser.add_argument("--check", action="store_true", help="run test_cases.json against the new index")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also build a header-less index in memory and compare retrieval",
    )
    args = parser.parse_args()

    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks")

    store = create_vector_store(chunks)
    save_vector_store(store)
    print(f"Saved vector store to {VECTOR_STORE_DIR}")

    if args.check or args.compare:
        with_headers = run_test_cases(store, "with contextual headers")
    if args.compare:
        baseline = create_vector_store(split_documents(documents, use_headers=False))
        without_headers = run_test_cases(baseline, "without contextual headers")
        print(f"\nSummary: with headers {with_headers} vs. without {without_headers}")


if __name__ == "__main__":
    main()

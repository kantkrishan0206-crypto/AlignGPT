"""Retrieval contracts for RAG experiments and backend scaffolds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    text: str
    source: str
    score: float


class InMemoryRetriever:
    """Simple keyword retriever with provenance for contract tests and demos."""

    def __init__(self, documents: tuple[RetrievedDocument, ...] = ()) -> None:
        self._documents = documents

    def search(self, query: str, limit: int = 3) -> tuple[RetrievedDocument, ...]:
        query_terms = set(query.casefold().split())
        ranked = []
        for document in self._documents:
            doc_terms = set(document.text.casefold().split())
            overlap = len(query_terms & doc_terms)
            if overlap:
                ranked.append((overlap + document.score, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return tuple(document for _, document in ranked[:limit])

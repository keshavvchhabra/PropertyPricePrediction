from __future__ import annotations

from typing import List, Optional

from rag.knowledge_base import load_market_knowledge
from rag.vector_store import MarketInsightVectorStore
from utils.config import Settings


class InsightRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.records, self.documents = load_market_knowledge(self.settings.knowledge_base_path)
        self.vector_store = MarketInsightVectorStore(
            vector_db_path=self.settings.vector_db_path,
            embedding_model=self.settings.embedding_model,
        )
        self.vector_store.ensure_store(self.records, self.documents)

    def retrieve_insights(self, query: str, top_k: Optional[int] = None) -> List[str]:
        results = self.vector_store.search(query, top_k or self.settings.retrieval_top_k)
        return [item["insight"] for item in results]


_default_retriever: InsightRetriever | None = None


def retrieve_insights(query: str, top_k: Optional[int] = None) -> List[str]:
    """Return the top-K relevant insights for a free-text query."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = InsightRetriever()
    return _default_retriever.retrieve_insights(query, top_k=top_k)

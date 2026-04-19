from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from advisor.schemas import PropertyInput, RetrievedContext

try:
    import faiss
except ImportError:  # pragma: no cover - bootstrap fallback
    faiss = None


class MarketRetriever:
    def __init__(self, documents: List[dict]) -> None:
        self.documents = documents
        corpus = [self._document_text(document) for document in documents]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = self.vectorizer.fit_transform(corpus).astype(np.float32)
        self.document_matrix = matrix.toarray()
        self.index = None
        if faiss is not None:
            self.index = faiss.IndexFlatL2(self.document_matrix.shape[1])
            self.index.add(self.document_matrix)

    def search(self, user_input: PropertyInput, top_k: int) -> List[RetrievedContext]:
        query = " ".join(
            [
                user_input.city,
                user_input.area,
                user_input.property_type,
                user_input.user_intent,
                " ".join(user_input.amenities),
            ]
        )
        query_vector = self.vectorizer.transform([query]).astype(np.float32).toarray()
        if self.index is not None:
            distances, indices = self.index.search(query_vector, min(top_k, len(self.documents)))
        else:
            distances = ((self.document_matrix - query_vector) ** 2).sum(axis=1)
            indices = np.argsort(distances)[: min(top_k, len(self.documents))]
            distances = np.array([distances[indices]], dtype=np.float32)
            indices = np.array([indices], dtype=np.int64)

        results: List[RetrievedContext] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            document = self.documents[int(idx)]
            results.append(
                RetrievedContext(
                    **document,
                    score=float(distance),
                )
            )
        return results

    @staticmethod
    def _document_text(document: dict) -> str:
        return " ".join(
            [
                document["city"],
                document["area"],
                document["property_type"],
                document["title"],
                document["summary"],
                document["risk_level"],
            ]
        )

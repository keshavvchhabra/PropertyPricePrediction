from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss
except ImportError:  # pragma: no cover - optional bootstrap fallback
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional bootstrap fallback
    SentenceTransformer = None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class MarketInsightVectorStore:
    """Persistent vector store using FAISS when available."""

    def __init__(self, vector_db_path: Path, embedding_model: str) -> None:
        self.vector_db_path = vector_db_path
        self.embedding_model_name = embedding_model
        self.index_path = self.vector_db_path / "market_knowledge.index"
        self.embeddings_path = self.vector_db_path / "market_knowledge_embeddings.npy"
        self.metadata_path = self.vector_db_path / "market_knowledge_metadata.json"
        self.vectorizer_path = self.vector_db_path / "market_knowledge_vectorizer.joblib"
        self.documents: List[str] = []
        self.records: List[dict] = []
        self.embeddings: np.ndarray | None = None
        self.index: Any = None
        self.backend = "sentence-transformers"
        self.embedder: Any = None
        self.vectorizer: TfidfVectorizer | None = None

    def ensure_store(self, records: List[dict], documents: List[str]) -> None:
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.records = records
        self.documents = documents
        desired_backend = self._resolve_backend()
        document_hash = self._document_hash(documents)

        if self._can_reuse_existing_store(document_hash, desired_backend):
            self._load_store_metadata()
            self._load_index_assets()
            return

        self.backend = desired_backend
        if self.backend == "sentence-transformers":
            embeddings = self._encode_with_sentence_transformers(documents)
        else:
            embeddings = self._encode_with_tfidf(documents, fit=True)

        self.embeddings = embeddings
        self._build_index(embeddings)
        self._write_store_metadata(document_hash)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.records:
            return []

        if self.backend == "sentence-transformers":
            query_vector = self._encode_with_sentence_transformers([query])
        else:
            query_vector = self._encode_with_tfidf([query], fit=False)

        query_vector = _normalize_rows(query_vector.astype(np.float32))
        top_k = min(top_k, len(self.records))

        if self.index is not None:
            scores, indices = self.index.search(query_vector, top_k)
            raw_scores = scores[0]
            raw_indices = indices[0]
        else:
            if self.embeddings is None:
                self.embeddings = np.load(self.embeddings_path).astype(np.float32)
            similarity = np.dot(self.embeddings, query_vector[0])
            raw_indices = np.argsort(-similarity)[:top_k]
            raw_scores = similarity[raw_indices]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(raw_scores, raw_indices):
            if int(idx) < 0:
                continue
            record = dict(self.records[int(idx)])
            record["score"] = float(score)
            results.append(record)
        return results

    def _resolve_backend(self) -> str:
        if SentenceTransformer is None:
            return "tfidf-fallback"

        try:
            self.embedder = SentenceTransformer(self.embedding_model_name)
            return "sentence-transformers"
        except Exception:
            self.embedder = None
            return "tfidf-fallback"

    def _encode_with_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.embedding_model_name)
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def _encode_with_tfidf(self, texts: List[str], fit: bool) -> np.ndarray:
        if fit:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            matrix = self.vectorizer.fit_transform(texts)
            joblib.dump(self.vectorizer, self.vectorizer_path)
        else:
            if self.vectorizer is None:
                self.vectorizer = joblib.load(self.vectorizer_path)
            matrix = self.vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)
        return _normalize_rows(dense)

    def _build_index(self, embeddings: np.ndarray) -> None:
        self.index = None
        if faiss is None:
            return

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.embeddings_path, embeddings)

    def _load_index_assets(self) -> None:
        self.embeddings = np.load(self.embeddings_path).astype(np.float32)
        if self.backend == "tfidf-fallback" and self.vectorizer_path.exists():
            self.vectorizer = joblib.load(self.vectorizer_path)
        if faiss is not None and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None

    def _can_reuse_existing_store(self, document_hash: str, desired_backend: str) -> bool:
        if not self.metadata_path.exists() or not self.embeddings_path.exists():
            return False

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if metadata.get("document_hash") != document_hash:
            return False
        if metadata.get("backend") != desired_backend:
            return False
        if metadata.get("embedding_model") != self.embedding_model_name and desired_backend == "sentence-transformers":
            return False
        if desired_backend == "tfidf-fallback" and not self.vectorizer_path.exists():
            return False
        return True

    def _load_store_metadata(self) -> None:
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.backend = metadata["backend"]
        self.documents = metadata["documents"]
        self.records = metadata["records"]

    def _write_store_metadata(self, document_hash: str) -> None:
        np.save(self.embeddings_path, self.embeddings)
        metadata = {
            "backend": self.backend,
            "embedding_model": self.embedding_model_name,
            "document_hash": document_hash,
            "documents": self.documents,
            "records": self.records,
        }
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @staticmethod
    def _document_hash(documents: List[str]) -> str:
        payload = json.dumps(documents, ensure_ascii=True, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

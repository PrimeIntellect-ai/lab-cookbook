from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class _Doc:
    id: str
    text: str


class TFIDFSearchIndex:
    def __init__(self) -> None:
        self._docs: list[_Doc] = []
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None

    def add_documents(self, docs: list[dict[str, str]]) -> None:
        for d in docs:
            self._docs.append(_Doc(id=d["id"], text=d["text"]))
        corpus = [d.text for d in self._docs]
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, k: int = 5) -> list[dict[str, float | str]]:
        if not self._docs or self._matrix is None:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        top_idxs = sims.argsort()[::-1][:k]
        return [
            {"id": self._docs[i].id, "score": float(sims[i])}
            for i in top_idxs
            if sims[i] > 0
        ]

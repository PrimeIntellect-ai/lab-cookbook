"""Semantic search indexes for document retrieval tools.

Two implementations provided:
    TFIDFSearchIndex  — TF-IDF based (requires only scikit-learn)
    SimpleSearchIndex — Alias for TFIDFSearchIndex

Both share the same interface:
    index.build(texts, doc_ids=None)
    results = index.search(query, k=5)
    # → [{"id": "doc_001", "text": "...", "score": 0.85}, ...]

For production with large corpora (>100K docs), consider ChromaDB:
    import chromadb
    client = chromadb.Client()
    collection = client.create_collection("docs")
    collection.add(documents=texts, ids=doc_ids)
    results = collection.query(query_texts=[query], n_results=k)

For dense embedding search, consider sentence-transformers:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(model.encode(query), embeddings)[0]
"""
from typing import Optional


class TFIDFSearchIndex:
    """TF-IDF search index. No GPU needed, fast for <100K docs.

    Uses bigram TF-IDF (1-2 word n-grams) with cosine similarity.
    Supports save/load via pickle for offline reuse.

    Example:
        index = TFIDFSearchIndex()
        index.build(
            texts=["The quick brown fox...", "Another document..."],
            doc_ids=["doc_001", "doc_002"],
        )
        results = index.search("quick fox", k=3)
        # [{"id": "doc_001", "text": "The quick brown fox...", "score": 0.72}]

    Tool integration example:
        async def search_docs(query: str, state: dict) -> str:
            results = index.search(query, k=5)
            return "\\n".join(f"[{r['id']}] {r['text'][:200]}" for r in results)
    """

    def __init__(self) -> None:
        self._vectorizer = None
        self._matrix = None
        self._doc_ids: list = []
        self._texts: list = []

    def build(
        self,
        texts: list,
        doc_ids: Optional[list] = None,
    ) -> None:
        """Build TF-IDF index from a list of texts.

        Args:
            texts:   List of document strings to index.
            doc_ids: Optional list of string IDs, one per text.
                     If None, uses "0", "1", "2", ... as IDs.

        Raises:
            ImportError: If scikit-learn is not installed.
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list of strings")

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._texts = list(texts)
        self._doc_ids = (
            list(doc_ids)
            if doc_ids is not None
            else [str(i) for i in range(len(texts))]
        )

        if len(self._doc_ids) != len(self._texts):
            raise ValueError(
                f"doc_ids length ({len(self._doc_ids)}) must match texts length ({len(self._texts)})"
            )

        self._vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,  # Apply log normalization to term frequencies
        )
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def search(self, query: str, k: int = 5) -> list:
        """Search for top-k documents matching the query.

        Returns list of dicts with keys: id, text, score.
        Results are sorted by score descending.
        Documents with score 0 are excluded.

        Args:
            query: Search query string.
            k:     Maximum number of results to return.

        Returns:
            List of dicts: [{"id": str, "text": str, "score": float}, ...]

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Index not built. Call build() first.")

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]

        # argsort ascending, then reverse to get descending order
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            {
                "id": self._doc_ids[i],
                "text": self._texts[i],
                "score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0.0
        ]

    def save(self, path: str) -> None:
        """Save the index to a pickle file for later reuse.

        Saves the vectorizer, TF-IDF matrix, doc IDs, and texts.
        Load with TFIDFSearchIndex.load(path).

        Args:
            path: File path to save to (e.g., "index.pkl").
        """
        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "matrix": self._matrix,
                    "doc_ids": self._doc_ids,
                    "texts": self._texts,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "TFIDFSearchIndex":
        """Load a previously saved index from a pickle file.

        Args:
            path: File path to load from (e.g., "index.pkl").

        Returns:
            A TFIDFSearchIndex instance ready for search().
        """
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls()
        index._vectorizer = data["vectorizer"]
        index._matrix = data["matrix"]
        index._doc_ids = data["doc_ids"]
        index._texts = data["texts"]
        return index

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self._texts)

    def __repr__(self) -> str:
        status = f"{len(self._texts)} docs" if self._texts else "not built"
        return f"TFIDFSearchIndex({status})"


# Convenience alias — identical to TFIDFSearchIndex
SimpleSearchIndex = TFIDFSearchIndex

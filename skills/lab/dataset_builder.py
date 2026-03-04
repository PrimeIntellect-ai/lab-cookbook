from __future__ import annotations

from datasets import Dataset


class DatasetBuilder:
    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, question: str, answer: str, info: dict | None = None) -> None:
        row = {"question": question, "answer": answer}
        if info is not None:
            row["info"] = info
        self._rows.append(row)

    def build(self) -> Dataset:
        return Dataset.from_list(self._rows)

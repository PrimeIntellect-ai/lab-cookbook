"""Dataset construction utilities for building RL training data.

Provides:
    DatasetBuilder — accumulate Q&A pairs, build HuggingFace Dataset
    load_jsonl     — load JSONL file as list of dicts
    save_jsonl     — save list of dicts as JSONL file

Dataset column schema (required by verifiers):
    question: str   — the question (for single-turn environments)
    answer:   str   — ground truth answer
    info:     dict  — task-specific metadata (type, source IDs, valid_answers, etc.)

For multi-turn environments, add a `prompt` column (list of message dicts)
instead of or in addition to `question`.
"""
import json
import math
from pathlib import Path
from typing import Optional


class DatasetBuilder:
    """Build a HuggingFace Dataset from question/answer/info triples.

    Accumulates rows via add(), then builds a HuggingFace Dataset with build()
    or splits into train/test with split().

    Example:
        builder = DatasetBuilder(system_prompt="You are a helpful assistant.")
        builder.add(question="What is 2+2?", answer="4")
        builder.add(question="Capital of France?", answer="Paris")
        dataset = builder.build()
        # HuggingFace Dataset with columns: question, answer, info

        train_ds, test_ds = builder.split(train_frac=0.8)
        print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    With metadata:
        builder.add(
            question="Find a patent about wireless charging",
            answer="patent_001",
            info={"valid_answers": ["patent_001", "patent_003"], "source": "USPTO"},
        )
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        """Create a new DatasetBuilder.

        Args:
            system_prompt: Optional system prompt to embed in `prompt` column.
                           If provided, each row also gets a `prompt` column
                           with [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": question}].
        """
        self.system_prompt = system_prompt
        self._rows: list = []

    def add(
        self,
        question: str,
        answer: str,
        info: Optional[dict] = None,
    ) -> None:
        """Add a Q&A pair to the dataset.

        Args:
            question: The question string.
            answer:   Ground truth answer string.
            info:     Optional metadata dict. Stored as-is in the `info` column.
        """
        row: dict = {
            "question": question,
            "answer": answer,
            "info": info if info is not None else {},
        }

        if self.system_prompt is not None:
            row["prompt"] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ]

        self._rows.append(row)

    def build(self) -> object:
        """Build a datasets.Dataset from accumulated rows.

        Returns:
            HuggingFace datasets.Dataset with all accumulated rows.

        Raises:
            ImportError: If the datasets library is not installed.
            ValueError: If no rows have been added.
        """
        if not self._rows:
            raise ValueError("No rows added. Call add() before build().")

        from datasets import Dataset
        return Dataset.from_list(self._rows)

    def split(self, train_frac: float = 0.8) -> tuple:
        """Split accumulated rows into train and test datasets.

        Rows are split in order (no shuffling) — shuffle before calling
        split() if you need random splits.

        Args:
            train_frac: Fraction of rows for training (0 < train_frac < 1).

        Returns:
            Tuple of (train_dataset, test_dataset) as HuggingFace Datasets.

        Example:
            train_ds, test_ds = builder.split(0.9)
        """
        if not self._rows:
            raise ValueError("No rows added. Call add() before split().")
        if not (0.0 < train_frac < 1.0):
            raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

        n_train = math.floor(len(self._rows) * train_frac)
        n_train = max(1, min(n_train, len(self._rows) - 1))  # At least 1 row in each split

        train_builder = DatasetBuilder(self.system_prompt)
        test_builder = DatasetBuilder(self.system_prompt)
        train_builder._rows = self._rows[:n_train]
        test_builder._rows = self._rows[n_train:]

        return train_builder.build(), test_builder.build()

    def save_jsonl(self, path: str) -> None:
        """Save accumulated rows to a JSONL file for inspection.

        Args:
            path: File path to write (e.g., "data/train.jsonl").
        """
        save_jsonl(self._rows, path)

    def shuffle(self, seed: int = 42) -> "DatasetBuilder":
        """Shuffle accumulated rows in place. Returns self for chaining.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            self (for method chaining)
        """
        import random
        rng = random.Random(seed)
        rng.shuffle(self._rows)
        return self

    def __len__(self) -> int:
        """Return the number of accumulated rows."""
        return len(self._rows)

    def __repr__(self) -> str:
        return f"DatasetBuilder({len(self._rows)} rows)"


def load_jsonl(path: str) -> list:
    """Load a JSONL file as a list of dicts.

    Skips blank lines. Each non-blank line must be valid JSON.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dicts, one per non-blank line.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.

    Example:
        rows = load_jsonl("data/train.jsonl")
        print(len(rows), rows[0].keys())
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_no} of {path}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
    return rows


def save_jsonl(data: list, path: str) -> None:
    """Save a list of dicts as a JSONL file.

    Creates parent directories if they don't exist.
    Writes one JSON object per line with a trailing newline.

    Args:
        data: List of JSON-serializable dicts.
        path: File path to write (created or overwritten).

    Example:
        save_jsonl([{"question": "...", "answer": "..."}], "data/train.jsonl")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

"""lab — Dataset construction and search utilities for prime-cookbook.

Provides:
    TFIDFSearchIndex / SimpleSearchIndex
        TF-IDF keyword search index. No GPU required. Fast for < 100K docs.
        Use for document retrieval tools in ToolEnv environments.

    DatasetBuilder
        Build HuggingFace datasets from question/answer/info triples.
        Supports split(), save_jsonl(), and len().

    load_jsonl / save_jsonl
        Lightweight JSONL I/O helpers.

    generate_ground_truth / GroundTruth
        Generate structured ground truth (answer + key_points + source_quotes)
        using GPT-4.1 for Level 3 open-ended environments.
"""

from .semantic_search import TFIDFSearchIndex, SimpleSearchIndex
from .dataset_builder import DatasetBuilder, load_jsonl, save_jsonl
from .ground_truth import generate_ground_truth, GroundTruth

__all__ = [
    "TFIDFSearchIndex",
    "SimpleSearchIndex",
    "DatasetBuilder",
    "load_jsonl",
    "save_jsonl",
    "generate_ground_truth",
    "GroundTruth",
]

"""prime-cookbook — RL environment recipes for Prime Intellect Lab.

Provides reusable reward functions (verifier skills) and dataset/search
utilities (lab skills) for building verifiers-based training environments.

Usage:
    from prime_cookbook.skills.verifiers import exact_match_reward, judge_reward
    from prime_cookbook.skills.lab import DatasetBuilder, SimpleSearchIndex
"""

from prime_cookbook.skills.verifiers import (
    exact_match_reward,
    contains_reward,
    set_match_reward,
    judge_reward,
    universal_rubric_reward,
    math_reward,
    extract_boxed_answer,
    code_reward,
    xml_parser_reward,
    last_line_reward,
    make_xml_parser,
)
from prime_cookbook.skills.lab import (
    SimpleSearchIndex,
    TFIDFSearchIndex,
    DatasetBuilder,
    load_jsonl,
    save_jsonl,
    generate_ground_truth,
    GroundTruth,
)

__version__ = "0.1.0"

__all__ = [
    # Verifier skills — reward functions
    "exact_match_reward",
    "contains_reward",
    "set_match_reward",
    "judge_reward",
    "universal_rubric_reward",
    "math_reward",
    "extract_boxed_answer",
    "code_reward",
    "xml_parser_reward",
    "last_line_reward",
    "make_xml_parser",
    # Lab skills — dataset and search utilities
    "SimpleSearchIndex",
    "TFIDFSearchIndex",
    "DatasetBuilder",
    "load_jsonl",
    "save_jsonl",
    "generate_ground_truth",
    "GroundTruth",
]

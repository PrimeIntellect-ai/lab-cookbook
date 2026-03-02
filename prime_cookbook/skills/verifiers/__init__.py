"""verifiers — Reward functions and parsers for vf.Rubric.

All reward functions are async, return float in [0.0, 1.0], and accept
extra **kwargs so they can be mixed freely in any rubric.

Quick reference:
    exact_match_reward   — binary: completion.lower() == answer.lower()
    contains_reward      — binary: answer in completion
    set_match_reward     — binary: completion in info["valid_answers"]
    judge_reward         — LLM judge comparison against answer
    universal_rubric_reward — full rubric with key_points + hallucination penalty
    math_reward          — extracts \\boxed{} and checks numeric/symbolic match
    code_reward          — executes code and checks test case outputs
    xml_parser_reward    — exact match on parsed <answer> XML tag
    last_line_reward     — exact match on last non-empty line
"""

from .exact_match import exact_match_reward, contains_reward, set_match_reward
from .judge_rubric import judge_reward, universal_rubric_reward
from .math_verify import math_reward, extract_boxed_answer
from .code_verify import code_reward
from .parsers import xml_parser_reward, last_line_reward, make_xml_parser

__all__ = [
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
]

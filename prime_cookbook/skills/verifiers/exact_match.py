"""Exact match verifiers for deterministic rewards.

All functions are async and accept `completion` and `answer` kwargs
for use with vf.Rubric.

Reward levels:
    exact_match_reward  → L1: single correct answer
    contains_reward     → L1: answer is substring of completion
    set_match_reward    → L1: multiple valid answers (search tasks)
"""


async def exact_match_reward(
    completion: str,
    answer: str,
    **kwargs,
) -> float:
    """Binary reward: 1.0 if completion (stripped, lowercased) equals answer.

    Best for: single-word or short phrase answers (yes/no/maybe, labels,
    entity names, short factual answers).

    Example:
        rubric = vf.Rubric(funcs=[exact_match_reward])
        # Dataset: {"question": "Capital of France?", "answer": "Paris"}
    """
    return 1.0 if completion.strip().lower() == answer.strip().lower() else 0.0


async def contains_reward(
    completion: str,
    answer: str,
    **kwargs,
) -> float:
    """Binary reward: 1.0 if answer appears anywhere in completion.

    Best for: when answer is a substring (e.g., a specific term or
    entity name that should appear in a longer response).

    NOT suitable for: numeric answers where partial match would give
    wrong signal (e.g., answer "12" matching "12345").

    Example:
        rubric = vf.Rubric(funcs=[contains_reward])
        # Dataset: {"question": "...", "answer": "aspirin"}
        # Rewards completions that mention aspirin anywhere.
    """
    return 1.0 if answer.strip().lower() in completion.lower() else 0.0


async def set_match_reward(
    completion: str,
    info: dict,
    **kwargs,
) -> float:
    """Binary reward: 1.0 if completion matches any answer in a precomputed set.

    Best for: search/retrieval tasks where multiple valid answers exist
    (e.g., return any one of several relevant documents).

    The dataset `info` dict must contain `valid_answers`: a list of
    acceptable answer strings. Comparison is case-insensitive and stripped.

    Example:
        # In dataset generation:
        info = {"valid_answers": ["patent_001", "patent_003", "patent_007"]}

        rubric = vf.Rubric(funcs=[set_match_reward])
        # Rewards if model outputs any of the valid patent IDs.
    """
    valid_answers = info.get("valid_answers", [])
    if not valid_answers:
        return 0.0
    answer_normalized = completion.strip().lower()
    return 1.0 if any(
        answer_normalized == v.strip().lower() for v in valid_answers
    ) else 0.0

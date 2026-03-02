"""Math verification helpers.

Extracts \\boxed{} answers and verifies mathematical equivalence.
For use with vf.MathRubric or standalone in custom rubrics.

NOTE: For production use, prefer vf.MathRubric which uses the
math-verify library for proper symbolic equivalence checking.
This module provides a lightweight fallback that handles common cases.
"""
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract the last \boxed{} expression from a model response.

    Handles nested braces correctly (e.g., \boxed{\frac{1}{2}}).
    Searches for the LAST occurrence so that scratchpad intermediate
    answers are ignored in favour of the final answer.

    Example:
        >>> extract_boxed_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_boxed_answer(r"So \boxed{x=1} then \boxed{\frac{3}{4}}")
        '\\frac{3}{4}'
        >>> extract_boxed_answer("No boxed answer here")
        None
    """
    # Search for the last \boxed{ occurrence (handles both \\ and \ variants)
    # Try double-backslash form first (common in raw strings / JSON)
    idx = text.rfind("\\\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed{")
    if idx == -1:
        return None

    # Find the opening brace
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None

    # Walk forward tracking brace depth to find the matching close
    depth = 1
    i = brace_start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        # Unmatched brace — malformed response
        return None

    return text[brace_start + 1 : i - 1].strip()


async def math_reward(
    completion: str,
    answer: str,
    **kwargs,
) -> float:
    r"""Math reward using \boxed{} extraction and numeric/string comparison.

    Returns 1.0 if extracted answer matches reference, 0.0 otherwise.

    Extraction strategy (in order):
    1. Extract last \boxed{} expression
    2. Fall back to last non-empty line of completion

    Comparison strategy (in order):
    1. String equality (stripped, no case folding — math is case-sensitive)
    2. Float equality (handles "3.0" == "3", "1/2" is NOT handled — use vf.MathRubric)

    Limitations:
        - Does not handle symbolic equivalence (e.g., "x+1" == "1+x")
        - Does not simplify fractions or expand expressions
        - For symbolic math, use vf.MathRubric (requires math-verify package)

    Example:
        rubric = vf.Rubric(funcs=[math_reward])
        # Dataset: {"question": "...", "answer": "42"}
    """
    extracted = extract_boxed_answer(completion)

    if extracted is None:
        # Fall back to last non-empty line
        lines = [line.strip() for line in completion.strip().split("\n") if line.strip()]
        extracted = lines[-1] if lines else ""

    ref = answer.strip()
    pred = extracted.strip()

    if not pred:
        return 0.0

    # 1. String equality
    if pred == ref:
        return 1.0

    # 2. Float equality (handles "3.0" == "3", "0.5" == ".5", etc.)
    try:
        if float(pred) == float(ref):
            return 1.0
    except ValueError:
        pass

    return 0.0

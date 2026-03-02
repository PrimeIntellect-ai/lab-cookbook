"""LLM-judge reward functions for open-ended responses.

Use when answers cannot be verified deterministically (synthesis,
analysis, explanation tasks). Requires OPENAI_API_KEY.

Design principles:
- Universal rubric + content-specific ground truth beats per-question rubrics
- Penalize hallucination heavily (x0.2) — better to undertrain than train on bad signal
- NO REGEX for parsing judge output — use json.loads() + str.find() fallback

NOTE: OPENAI_API_KEY must be set in the environment when using judge reward functions.
      Call vf.ensure_keys(["OPENAI_API_KEY"]) in your environment's load_environment()
      rather than at import time, so the module can be imported in test environments.
"""
import json

# Universal rubric — same for all open-ended questions.
# Ground truth (answer, key_points, source_quotes) provides content specificity.
UNIVERSAL_JUDGE_PROMPT = """You are evaluating a response to a question.

Question: {question}
Reference answer: {reference_answer}
Key points that must be covered: {key_points}
Source quotes from the corpus: {source_quotes}

Agent response:
{response}

Score the response on these criteria (return JSON only, no other text):
- factually_accurate: bool (are all claims correct relative to the source material?)
- no_hallucination: bool (does the response avoid inventing facts not in the corpus?)
- covers_key_points: list of strings (which key points from the list above are addressed?)
- answers_question: bool (does it directly answer what was asked?)
- cites_evidence: bool (does it reference specific content as evidence?)
- final_score: float 0.0-1.0 (overall quality)

Return only valid JSON, starting with {{"""


async def universal_rubric_reward(
    completion: str,
    prompt: list,
    info: dict,
    judge: object,
    **kwargs,
) -> float:
    """Universal rubric reward for open-ended questions.

    Evaluates completion against structured ground truth using an LLM judge.
    Applies hallucination penalty (x0.2) and factual error penalty (x0.5).

    Requires info dict with:
        reference_answer: str   — model answer to compare against
        key_points: list[str]   — factual claims that must be present
        source_quotes: list[str] — direct quotes from source material

    Requires a judge callable injected via rubric.add_class_object("judge", ...).

    Example:
        judge_rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")
        rubric = vf.Rubric(funcs=[universal_rubric_reward])
        rubric.add_class_object("judge", judge_rubric.judge)
        env = vf.ToolEnv(dataset=dataset, rubric=rubric)

    See: prime_cookbook/skills/lab/ground_truth.py for generating info dicts.
    """
    # Extract question from prompt (last user message)
    question = ""
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            question = content if isinstance(content, str) else str(content)
            break

    reference_answer = info.get("reference_answer", "")
    key_points = info.get("key_points", [])
    source_quotes = info.get("source_quotes", [])

    judge_prompt = UNIVERSAL_JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        key_points="\n".join(f"- {p}" for p in key_points),
        source_quotes="\n".join(f'- "{q}"' for q in source_quotes),
        response=completion,
    )

    judge_response = await judge(judge_prompt)

    # Parse judge output — no regex, use json.loads() with str.find() fallback
    try:
        result = json.loads(judge_response)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback: locate final_score by string search
        idx = judge_response.find("final_score")
        if idx == -1:
            return 0.0
        chunk = judge_response[idx:idx + 40]
        score = _extract_first_float(chunk)
        if score is None:
            return 0.0
        return max(0.0, min(1.0, score))

    score = float(result.get("final_score", 0.0))
    score = max(0.0, min(1.0, score))

    # Penalize hallucination heavily — better to undertrain than train on wrong signal
    if not result.get("no_hallucination", True):
        score *= 0.2
    elif not result.get("factually_accurate", True):
        score *= 0.5

    return score


async def judge_reward(
    completion: str,
    answer: str,
    judge: object,
    **kwargs,
) -> float:
    """Simple judge reward comparing completion to a reference answer.

    Returns a float 0.0-1.0 based on how well the completion matches
    the reference answer according to the judge.

    Best for: quick L3 environments where you don't need the full
    universal_rubric_reward with key_points and source_quotes.

    Requires a judge callable injected via rubric.add_class_object("judge", ...).

    Example:
        judge_rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")
        rubric = vf.Rubric(funcs=[judge_reward])
        rubric.add_class_object("judge", judge_rubric.judge)
    """
    prompt = (
        "Rate how well this response answers the question compared to the reference.\n\n"
        f"Reference answer: {answer}\n\n"
        f"Response to evaluate: {completion}\n\n"
        'Return JSON only: {"score": 0.0-1.0, "reason": "one sentence"}'
    )

    response = await judge(prompt)

    try:
        result = json.loads(response)
        return max(0.0, min(1.0, float(result.get("score", 0.0))))
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: find score by string search
        idx = response.find('"score"')
        if idx == -1:
            idx = response.find("score")
        if idx == -1:
            return 0.0
        chunk = response[idx:idx + 30]
        score = _extract_first_float(chunk)
        return max(0.0, min(1.0, score)) if score is not None else 0.0


def _extract_first_float(text: str) -> float | None:
    """Extract the first float-like token from a short text string.

    Used as a fallback when json.loads() fails. Tries each whitespace-
    delimited token after stripping common JSON punctuation.

    Returns None if no float found.
    """
    for part in text.split():
        cleaned = part.strip(',:}"\'[]{}()')
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None

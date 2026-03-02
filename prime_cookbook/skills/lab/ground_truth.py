"""Ground truth generation for Level 3 open-ended environments.

Use GPT-4.1 to generate structured ground truth for questions
that cannot be verified deterministically.

Output schema (GroundTruth dataclass):
    answer:        str        — model prose answer to compare against
    key_points:    list[str]  — specific facts the answer must contain
    source_quotes: list[str]  — direct quotes from source material

The key_points and source_quotes fields are used by universal_rubric_reward
(in prime_cookbook.skills.verifiers.judge_rubric) to provide content-specific
grounding to the universal judge prompt.

See: docs/reward-design.md#ground-truth-generation
"""
import json
from dataclasses import dataclass, field
from typing import Optional

GROUND_TRUTH_SYSTEM = """You are generating ground truth data for RL training.

For each question and source material, produce a structured answer with:
1. A clear, factual prose answer (2-4 sentences)
2. 3-5 specific key points — individual facts or claims that a correct answer must contain
3. 2-3 direct quotes from the source material that support the answer

Rules:
- All claims must be grounded in the provided source material
- Key points should be specific and verifiable, not vague
- Source quotes must be exact verbatim text from the source material
- Do NOT invent information not present in the source

Return valid JSON only, with exactly these keys: answer, key_points, source_quotes"""

GROUND_TRUTH_USER_TEMPLATE = """Question: {question}

Source material:
{context}

Generate ground truth JSON:
{{
    "answer": "...",
    "key_points": ["specific fact 1", "specific fact 2", "specific fact 3"],
    "source_quotes": ["exact quote from source 1", "exact quote from source 2"]
}}"""


@dataclass
class GroundTruth:
    """Structured ground truth for a single Q&A pair.

    Used as the `info` dict payload for Level 3 open-ended environments.

    Attributes:
        answer:        Prose answer to the question.
        key_points:    List of specific facts the answer must contain.
        source_quotes: List of direct quotes from the source material.

    Example:
        gt = GroundTruth(
            answer="The filing date was March 15, 2019.",
            key_points=["Filing date: March 15, 2019", "Inventor: John Smith"],
            source_quotes=["filed on March 15, 2019", "invented by John Smith"],
        )
        # Use as info dict:
        info = {
            "reference_answer": gt.answer,
            "key_points": gt.key_points,
            "source_quotes": gt.source_quotes,
        }
    """

    answer: str
    key_points: list = field(default_factory=list)
    source_quotes: list = field(default_factory=list)

    def to_info_dict(self) -> dict:
        """Convert to the info dict format expected by universal_rubric_reward.

        Returns:
            dict with keys: reference_answer, key_points, source_quotes
        """
        return {
            "reference_answer": self.answer,
            "key_points": self.key_points,
            "source_quotes": self.source_quotes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        """Create a GroundTruth from a dict (e.g., loaded from JSONL).

        Accepts both flat format (answer/key_points/source_quotes) and
        the info dict format (reference_answer/key_points/source_quotes).
        """
        answer = data.get("answer") or data.get("reference_answer", "")
        key_points = data.get("key_points", [])
        source_quotes = data.get("source_quotes", [])
        return cls(answer=answer, key_points=key_points, source_quotes=source_quotes)


async def generate_ground_truth(
    question: str,
    context: str,
    client: Optional[object] = None,
    model: str = "gpt-4.1",
) -> GroundTruth:
    """Generate structured ground truth for an open-ended question.

    Calls GPT-4.1 (or the specified model) to produce a GroundTruth with
    answer, key_points, and source_quotes grounded in the provided context.

    Args:
        question: The question to answer.
        context:  Source material to ground the answer in (e.g., document text).
        client:   AsyncOpenAI client instance. Creates a new one if None.
        model:    Model to use for generation (default: "gpt-4.1").

    Returns:
        GroundTruth dataclass with answer, key_points, and source_quotes.

    Raises:
        ImportError: If openai is not installed.
        json.JSONDecodeError: If the model returns malformed JSON (unlikely with json_object mode).
        KeyError: If the model omits required fields (handled gracefully with defaults).

    Example:
        gt = await generate_ground_truth(
            question="When was patent US10123456 filed?",
            context="US10123456 was filed on March 15, 2019 by inventor John Smith...",
        )
        info = gt.to_info_dict()
        # → {"reference_answer": "...", "key_points": [...], "source_quotes": [...]}
    """
    from openai import AsyncOpenAI

    if client is None:
        client = AsyncOpenAI()

    user_prompt = GROUND_TRUTH_USER_TEMPLATE.format(
        question=question,
        context=context,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GROUND_TRUTH_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    return GroundTruth(
        answer=result.get("answer", ""),
        key_points=result.get("key_points", []),
        source_quotes=result.get("source_quotes", []),
    )


async def generate_ground_truth_batch(
    questions: list,
    contexts: list,
    client: Optional[object] = None,
    model: str = "gpt-4.1",
    max_concurrent: int = 10,
) -> list:
    """Generate ground truth for multiple questions concurrently.

    Args:
        questions:      List of question strings.
        contexts:       List of context strings (one per question).
        client:         AsyncOpenAI client. Creates a new one if None.
        model:          Model to use for generation.
        max_concurrent: Maximum concurrent API calls (default: 10).

    Returns:
        List of GroundTruth objects, one per question.

    Example:
        ground_truths = await generate_ground_truth_batch(
            questions=["Q1?", "Q2?"],
            contexts=["Context 1...", "Context 2..."],
        )
        # ground_truths[0].answer, ground_truths[0].key_points, ...
    """
    import asyncio
    from openai import AsyncOpenAI

    if len(questions) != len(contexts):
        raise ValueError(
            f"questions ({len(questions)}) and contexts ({len(contexts)}) must have equal length"
        )

    if client is None:
        client = AsyncOpenAI()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _generate_one(q: str, c: str) -> GroundTruth:
        async with semaphore:
            return await generate_ground_truth(q, c, client=client, model=model)

    tasks = [_generate_one(q, c) for q, c in zip(questions, contexts)]
    return await asyncio.gather(*tasks)

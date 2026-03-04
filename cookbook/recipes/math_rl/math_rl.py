"""Math reasoning environment — GSM8K (OpenAI) with MathRubric.

Primary dataset: openai/gsm8k (main split).
Fallback dataset: small synthetic arithmetic set when HF download is unavailable.
"""
from __future__ import annotations

import random

import verifiers as vf
from datasets import Dataset, load_dataset

SYSTEM_PROMPT = """You are a math expert. Solve each problem step by step.
Always put your final answer in \\boxed{} at the end."""


def _extract_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip().replace(",", "")


def _load_gsm8k(num_examples: int, split: str = "train") -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if num_examples != -1:
        ds = ds.select(range(min(num_examples, len(ds))))
    ds = ds.map(
        lambda row: {
            "question": row["question"],
            "answer": _extract_gsm8k_answer(row["answer"]),
        },
        remove_columns=ds.column_names,
    )
    return ds


def _synthetic_fallback(n: int = 200, seed: int = 42) -> Dataset:
    rng = random.Random(seed)
    rows = []
    ops = [
        lambda a, b: (f"What is {a} + {b}?", str(a + b)),
        lambda a, b: (f"What is {a} - {b}?", str(a - b)),
        lambda a, b: (f"What is {a} * {b}?", str(a * b)),
    ]
    for _ in range(n):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        q, ans = rng.choice(ops)(a, b)
        rows.append({"question": q, "answer": ans})
    return Dataset.from_list(rows)


def load_environment(
    num_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    try:
        dataset = _load_gsm8k(num_examples=num_examples)
    except Exception:
        n = 500 if num_examples == -1 else num_examples
        dataset = _synthetic_fallback(n=n, seed=seed)

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=vf.MathRubric(),
    )


if __name__ == "__main__":
    env = load_environment(num_examples=10)
    print(f"Dataset size: {len(env.dataset)}")
    print(f"Sample question: {env.dataset[0]['question']}")
    print(f"Sample answer:   {env.dataset[0]['answer']}")

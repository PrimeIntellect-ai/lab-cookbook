"""Document Search — Level 2 (multi-step reasoning over metadata).

Same tools as L1 but questions require reasoning ACROSS multiple documents:
  - Comparisons  ("Which was founded more recently: X or Y?")
  - Aggregations ("How many Software companies are in the corpus?")
  - Arithmetic   ("How many years ago was Acme Corp founded? (current year = 2025)")
  - Rankings     ("Which company in the corpus has the most employees?")

Reward: deterministic programmatic check — not LLM-as-judge.
Starting reward with 1B model: ~0.10-0.20 (harder than L1).
Typically saturates to ~0.80-0.90 within 30-50 steps.

Uses the same corpus as L1 (shared module-level state).
"""
from __future__ import annotations

import random
import verifiers as vf
from datasets import Dataset

from prime_cookbook.skills.lab import DatasetBuilder
from prime_cookbook.recipes.document_search.document_search_l1 import (
    _CORPUS,
    _ensure_corpus_loaded,
    search_docs,
    get_metadata,
)

SYSTEM_PROMPT = """You are a research analyst with access to a company database.
Use search_docs and get_metadata to look up information, then reason step by step.

Rules:
- For year-ago questions assume the current year is 2025.
- For comparison questions output only the company name that matches.
- For count/number questions output only the integer.
- Your final answer must be on its own line with no extra text.
"""

CURRENT_YEAR = 2025

# ---------------------------------------------------------------------------
# Q&A generation — multi-step questions
# ---------------------------------------------------------------------------

def _build_qa_pairs(corpus: dict, n: int = 400, seed: int = 42) -> Dataset:
    rng = random.Random(seed)
    builder = DatasetBuilder()
    docs = list(corpus.values())

    def _founded_recently(d1, d2):
        if d1["founded"] == d2["founded"]:
            return d1["name"]  # tie — first
        return d1["name"] if d1["founded"] > d2["founded"] else d2["name"]

    def _more_employees(d1, d2):
        return d1["name"] if d1["employees"] >= d2["employees"] else d2["name"]

    def _years_ago(d):
        return str(CURRENT_YEAR - d["founded"])

    questions_generated = 0
    attempts = 0

    while questions_generated < n and attempts < n * 10:
        attempts += 1
        choice = rng.randint(0, 4)

        if choice == 0:
            # Comparison: which was founded more recently?
            d1, d2 = rng.sample(docs, 2)
            q = f"Which company was founded more recently: {d1['name']} or {d2['name']}?"
            a = _founded_recently(d1, d2)
            builder.add(q, a)

        elif choice == 1:
            # Comparison: which has more employees?
            d1, d2 = rng.sample(docs, 2)
            q = f"Which company has more employees: {d1['name']} or {d2['name']}?"
            a = _more_employees(d1, d2)
            builder.add(q, a)

        elif choice == 2:
            # Arithmetic: how many years ago was X founded?
            d = rng.choice(docs)
            q = f"How many years ago was {d['name']} founded? (current year is {CURRENT_YEAR})"
            a = _years_ago(d)
            builder.add(q, a)

        elif choice == 3:
            # Aggregation: how many companies are in industry X?
            industry = rng.choice([
                "Software", "Hardware", "Biotech", "Finance", "Retail",
                "Healthcare", "Energy", "Manufacturing", "Media", "Logistics",
            ])
            count = sum(1 for d in docs if d["industry"] == industry)
            if count == 0:
                continue
            q = f"How many companies in the database are in the {industry} industry?"
            a = str(count)
            builder.add(q, a)

        else:
            # Rank: which company has the highest revenue?
            sample = rng.sample(docs, 4)
            best = max(sample, key=lambda d: d["revenue_millions"])
            names = ", ".join(d["name"] for d in sample)
            q = f"Of these companies, which has the highest revenue: {names}?"
            a = best["name"]
            builder.add(q, a)

        questions_generated += 1

    return builder.build()


# ---------------------------------------------------------------------------
# Reward — deterministic exact match
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


async def multi_step_reward(completion: str, answer: str, **kwargs) -> float:
    """Return 1.0 if the last non-empty line of completion matches the answer."""
    lines = [ln.strip() for ln in completion.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    prediction = _normalize(lines[-1])
    expected = _normalize(answer)
    if prediction == expected:
        return 1.0
    # Partial credit: answer is contained in completion (lenient for long names)
    if expected in prediction or prediction in expected:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------

def load_environment(
    num_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    """Load Document Search L2 environment.

    Args:
        num_examples: Number of examples to use (-1 = all ~400).
        seed: Random seed.

    Returns:
        ToolEnv requiring multi-step reasoning over the document corpus.
    """
    _ensure_corpus_loaded(seed=seed)
    dataset = _build_qa_pairs(_CORPUS, seed=seed)
    if num_examples != -1:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    rubric = vf.Rubric(funcs=[multi_step_reward], weights=[1.0])

    return vf.ToolEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_docs, get_metadata],
        rubric=rubric,
    )


if __name__ == "__main__":
    _ensure_corpus_loaded()
    env = load_environment(num_examples=10)
    print(f"Dataset size: {len(env.dataset)}")
    for row in env.dataset.select(range(3)):
        print(f"  Q: {row['question']}")
        print(f"  A: {row['answer']}")

"""Math reasoning environment — Level 1 recipe.

Simplest environment pattern: SingleTurnEnv + MathRubric.
Model receives an arithmetic/algebra problem, answers in \\boxed{}.

Dataset: 500 procedurally generated arithmetic problems.
Reward: vf.MathRubric (\\boxed{} extraction + symbolic equivalence).

Expected reward with gpt-4.1-mini: 1.000 (saturates immediately).
Useful as a sanity check that your setup works end-to-end.

For training signal, use math_rl_l2.py (harder) or switch to GSM8K.
"""
import random
import verifiers as vf
from datasets import Dataset

SYSTEM_PROMPT = """You are a math expert. Solve each problem step by step.
Always put your final answer in \\boxed{} at the end."""


def _generate_problems(n: int = 500, seed: int = 42) -> Dataset:
    """Generate arithmetic problems with known answers."""
    rng = random.Random(seed)
    rows = []

    ops = [
        # (template_fn) → (question, answer)
        lambda a, b: (f"What is {a} + {b}?", str(a + b)),
        lambda a, b: (f"What is {a} - {b}?", str(a - b)),
        lambda a, b: (f"What is {a} * {b}?", str(a * b)),
        lambda a, b: (f"What is {a * b} / {b}?", str(a)),  # exact integer division
    ]

    for _ in range(n):
        op_fn = rng.choice(ops)
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        question, answer = op_fn(a, b)
        rows.append({"question": question, "answer": answer})

    return Dataset.from_list(rows)


def load_environment(
    num_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    """Load math reasoning environment.

    Args:
        num_examples: Limit examples (-1 = all). Use 50-100 for quick testing.
        seed: Random seed for problem generation.

    Returns:
        SingleTurnEnv with MathRubric.
    """
    n = 500 if num_examples == -1 else num_examples
    dataset = _generate_problems(n, seed=seed)

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=vf.MathRubric(),
    )


if __name__ == "__main__":
    # Quick smoke test
    env = load_environment(num_examples=10)
    print(f"Dataset size: {len(env.dataset)}")
    print(f"Sample question: {env.dataset[0]['question']}")
    print(f"Sample answer:   {env.dataset[0]['answer']}")

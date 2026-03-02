"""Tool use environment — stateless tools recipe.

Four deterministic tools expose a small, learnable API surface.
The model must decide WHICH tool to call and pass correct arguments.

Tools:
  calculate(expression)           — evaluate arithmetic expressions
  count_words(text)               — count space-separated words
  reverse_string(text)            — reverse a string character by character
  convert_units(value, from, to)  — simple unit conversions

Dataset: 100 questions with fully deterministic answers.
Reward: exact match after normalization (no regex).

Expected starting reward with gpt-4.1-mini: ~0.70-0.85.
Good difficulty sweet spot for 1B models: ~0.25-0.40.

Design notes:
  - Tools are kept stateless (no shared state between calls)
  - NO regex anywhere — str.find(), str.split(), json.loads() only
  - Reward is async and returns float 0.0-1.0
"""
import random
import verifiers as vf
from datasets import Dataset

SYSTEM_PROMPT = """You are a helpful assistant with access to tools. \
Use them to answer questions precisely.

After calling a tool, analyse its output and give your final answer as a single \
plain value with no extra text.

Examples of final answers: "42", "hello", "5 words", "3.28 feet"
"""

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression and return the numeric result.

    Supports +, -, *, / and parentheses. Input must be a safe numeric expression.

    Args:
        expression: Arithmetic expression string, e.g. "(3 + 4) * 2"

    Returns:
        Result as a string, e.g. "14" or "3.5"
    """
    # Allow only safe characters before eval
    allowed = set("0123456789 +-*/().")
    for ch in expression:
        if ch not in allowed:
            return f"Error: unsafe character '{ch}' in expression"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 6))
    except Exception as exc:
        return f"Error: {exc}"


def count_words(text: str) -> int:
    """Count the number of words in the given text.

    Words are defined as space-separated tokens.

    Args:
        text: Input text string.

    Returns:
        Integer word count.
    """
    return len(text.split())


def reverse_string(text: str) -> str:
    """Return the input string reversed character by character.

    Args:
        text: Input string.

    Returns:
        Reversed string.
    """
    return text[::-1]


# Supported conversions: (from_unit, to_unit) -> (factor, unit_label)
_CONVERSIONS: dict[tuple[str, str], tuple[float, str]] = {
    ("km", "miles"): (0.621371, "miles"),
    ("miles", "km"): (1.60934, "km"),
    ("kg", "lbs"): (2.20462, "lbs"),
    ("lbs", "kg"): (0.453592, "kg"),
    ("celsius", "fahrenheit"): (None, "°F"),   # special formula
    ("fahrenheit", "celsius"): (None, "°C"),   # special formula
    ("meters", "feet"): (3.28084, "feet"),
    ("feet", "meters"): (0.3048, "meters"),
    ("liters", "gallons"): (0.264172, "gallons"),
    ("gallons", "liters"): (3.78541, "liters"),
}


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between common units.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit (e.g. "km", "kg", "celsius").
        to_unit: Target unit (e.g. "miles", "lbs", "fahrenheit").

    Returns:
        Converted value with unit label, e.g. "6.21 miles".
    """
    key = (from_unit.lower(), to_unit.lower())
    if key not in _CONVERSIONS:
        return f"Error: conversion from '{from_unit}' to '{to_unit}' not supported"

    factor, label = _CONVERSIONS[key]

    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = value * 9 / 5 + 32
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5 / 9
    else:
        result = value * factor

    return f"{round(result, 4)} {label}"


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _build_dataset(seed: int = 42) -> Dataset:
    """Build 100 questions that each require exactly one tool call."""
    rng = random.Random(seed)
    rows = []

    # ---- calculate (25 questions) ----
    for _ in range(25):
        a = rng.randint(2, 50)
        b = rng.randint(2, 50)
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            q = f"What is {a} * {b}?"
            ans = str(a * b)
        elif op == "+":
            q = f"What is {a} + {b}?"
            ans = str(a + b)
        else:
            q = f"What is {a} - {b}?"
            ans = str(a - b)
        rows.append({"question": q, "answer": ans})

    # ---- count_words (25 questions) ----
    word_pool = ["apple", "banana", "cherry", "dog", "elephant", "frog",
                 "grape", "honey", "iris", "jungle", "koala", "lemon"]
    for _ in range(25):
        n_words = rng.randint(2, 8)
        words = rng.choices(word_pool, k=n_words)
        phrase = " ".join(words)
        q = f'How many words are in the phrase: "{phrase}"?'
        rows.append({"question": q, "answer": str(n_words)})

    # ---- reverse_string (25 questions) ----
    strings = ["hello", "world", "python", "agent", "learning", "reward",
               "science", "data", "model", "train", "token", "verify"]
    for _ in range(25):
        s = rng.choice(strings)
        q = f'What is the reverse of "{s}"?'
        rows.append({"question": q, "answer": s[::-1]})

    # ---- convert_units (25 questions) ----
    conversion_qs = [
        (5.0, "km", "miles", convert_units(5.0, "km", "miles")),
        (10.0, "km", "miles", convert_units(10.0, "km", "miles")),
        (1.0, "miles", "km", convert_units(1.0, "miles", "km")),
        (70.0, "kg", "lbs", convert_units(70.0, "kg", "lbs")),
        (150.0, "lbs", "kg", convert_units(150.0, "lbs", "kg")),
        (100.0, "celsius", "fahrenheit", convert_units(100.0, "celsius", "fahrenheit")),
        (32.0, "fahrenheit", "celsius", convert_units(32.0, "fahrenheit", "celsius")),
        (2.0, "meters", "feet", convert_units(2.0, "meters", "feet")),
        (6.0, "feet", "meters", convert_units(6.0, "feet", "meters")),
        (3.0, "liters", "gallons", convert_units(3.0, "liters", "gallons")),
    ]
    for _ in range(25):
        val, fu, tu, ans = rng.choice(conversion_qs)
        q = f"Convert {val} {fu} to {tu}."
        rows.append({"question": q, "answer": ans})

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.strip().lower()
    parts = text.split()
    return " ".join(parts)


async def exact_match_reward(completion: str, answer: str, **kwargs) -> float:
    """Return 1.0 if the final line of the completion exactly matches the answer."""
    # Use the last non-empty line as the model's final answer
    lines = [ln.strip() for ln in completion.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    prediction = _normalize(lines[-1])
    expected = _normalize(answer)
    return 1.0 if prediction == expected else 0.0


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------

def load_environment(
    num_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    """Load the tool-use environment.

    Args:
        num_examples: Limit dataset size (-1 = all 100).
        seed: Random seed for dataset generation.

    Returns:
        ToolEnv with four stateless tools and exact-match reward.
    """
    dataset = _build_dataset(seed=seed)
    if num_examples != -1:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    rubric = vf.Rubric(
        funcs=[exact_match_reward],
        weights=[1.0],
    )

    return vf.ToolEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        tools=[calculate, count_words, reverse_string, convert_units],
        rubric=rubric,
    )


if __name__ == "__main__":
    env = load_environment(num_examples=10)
    print(f"Dataset size: {len(env.dataset)}")
    for row in env.dataset.select(range(3)):
        print(f"  Q: {row['question']}")
        print(f"  A: {row['answer']}")
        print()

"""Sandbox Code — PythonEnv recipe.

The model writes Python code to solve programming challenges.
Code is executed inside a Prime Sandbox (isolated container).
Reward: fraction of test cases that pass.

Dataset: 50 simple coding challenges with test cases.

Requires:
  - PRIME_API_KEY for sandbox access
  - verifiers>=0.1.10 with sandbox dependencies

This recipe demonstrates:
  - vf.PythonEnv (code execution in isolated containers)
  - Test-case-based partial rewards
  - Structured code extraction from model output
  - Sandbox lifecycle management (handled automatically by verifiers)

Expected starting reward with gpt-4.1-mini: ~0.80-0.90.
Expected starting reward with 1B model: ~0.20-0.40.
"""
from __future__ import annotations

import random
import verifiers as vf
from datasets import Dataset

SYSTEM_PROMPT = """You are an expert Python programmer. Solve the given coding challenge.

Rules:
- Write ONLY a Python function with the exact signature specified.
- Do NOT include example usage, imports beyond what is needed, or test code.
- Your function must handle all edge cases mentioned in the problem.
- Output ONLY the function definition, nothing else.

Example output format:
def solve(nums: list[int]) -> int:
    return sum(nums)
"""

# ---------------------------------------------------------------------------
# Dataset: 50 coding challenges
# ---------------------------------------------------------------------------

_CHALLENGES = [
    # (description, function_signature, test_cases)
    (
        "Write a function that returns the sum of all numbers in a list.",
        "def solve(nums: list) -> int:",
        [([1, 2, 3], 6), ([0], 0), ([-1, 1], 0), ([10, 20, 30], 60)],
    ),
    (
        "Write a function that returns the maximum value in a list.",
        "def solve(nums: list) -> int:",
        [([1, 2, 3], 3), ([5, 3, 1], 5), ([-1, -5, -2], -1), ([42], 42)],
    ),
    (
        "Write a function that reverses a string.",
        "def solve(s: str) -> str:",
        [("hello", "olleh"), ("", ""), ("a", "a"), ("abcd", "dcba")],
    ),
    (
        "Write a function that checks if a number is even.",
        "def solve(n: int) -> bool:",
        [(2, True), (3, False), (0, True), (-4, True), (7, False)],
    ),
    (
        "Write a function that returns the factorial of n (n >= 0).",
        "def solve(n: int) -> int:",
        [(0, 1), (1, 1), (5, 120), (3, 6), (4, 24)],
    ),
    (
        "Write a function that returns True if a string is a palindrome.",
        "def solve(s: str) -> bool:",
        [("racecar", True), ("hello", False), ("", True), ("a", True), ("abba", True)],
    ),
    (
        "Write a function that counts the vowels in a string (a, e, i, o, u).",
        "def solve(s: str) -> int:",
        [("hello", 2), ("", 0), ("aeiou", 5), ("rhythm", 0), ("Python", 1)],
    ),
    (
        "Write a function that returns the Fibonacci number at position n (0-indexed, F(0)=0, F(1)=1).",
        "def solve(n: int) -> int:",
        [(0, 0), (1, 1), (5, 5), (10, 55), (7, 13)],
    ),
    (
        "Write a function that returns a list of even numbers from the input list.",
        "def solve(nums: list) -> list:",
        [([1, 2, 3, 4], [2, 4]), ([], []), ([1, 3, 5], []), ([2, 4], [2, 4])],
    ),
    (
        "Write a function that flattens a list of lists by one level.",
        "def solve(nested: list) -> list:",
        [([[1, 2], [3, 4]], [1, 2, 3, 4]), ([[1]], [1]), ([], []), ([[1, 2, 3]], [1, 2, 3])],
    ),
    (
        "Write a function that returns the number of words in a sentence.",
        "def solve(sentence: str) -> int:",
        [("hello world", 2), ("", 0), ("one", 1), ("a b c d", 4)],
    ),
    (
        "Write a function that returns the unique elements of a list (preserving order).",
        "def solve(nums: list) -> list:",
        [([1, 2, 1, 3], [1, 2, 3]), ([], []), ([1, 1, 1], [1]), ([1, 2, 3], [1, 2, 3])],
    ),
    (
        "Write a function that returns True if a list is sorted in ascending order.",
        "def solve(nums: list) -> bool:",
        [([1, 2, 3], True), ([3, 2, 1], False), ([], True), ([1], True), ([1, 1, 2], True)],
    ),
    (
        "Write a function that returns the second largest number in a list.",
        "def solve(nums: list) -> int:",
        [([1, 2, 3], 2), ([5, 5, 3], 5), ([10, 1], 1), ([1, 2, 3, 4, 5], 4)],
    ),
    (
        "Write a function that capitalizes the first letter of each word in a sentence.",
        "def solve(s: str) -> str:",
        [("hello world", "Hello World"), ("python", "Python"), ("a b c", "A B C")],
    ),
    (
        "Write a function that returns the product of all numbers in a list.",
        "def solve(nums: list) -> int:",
        [([1, 2, 3], 6), ([5, 10], 50), ([1], 1), ([2, 3, 4], 24)],
    ),
    (
        "Write a function that removes duplicates from a list while preserving order.",
        "def solve(items: list) -> list:",
        [([1, 2, 1, 3, 2], [1, 2, 3]), ([], []), (["a", "b", "a"], ["a", "b"])],
    ),
    (
        "Write a function that returns True if a number is prime.",
        "def solve(n: int) -> bool:",
        [(2, True), (3, True), (4, False), (17, True), (1, False), (0, False)],
    ),
    (
        "Write a function that returns the GCD of two numbers.",
        "def solve(a: int, b: int) -> int:",
        [(12, 8, 4), (100, 75, 25), (7, 3, 1), (0, 5, 5)],
    ),
    (
        "Write a function that converts a list of (key, value) tuples to a dictionary.",
        "def solve(pairs: list) -> dict:",
        [([("a", 1), ("b", 2)], {"a": 1, "b": 2}), ([], {}), ([("x", 10)], {"x": 10})],
    ),
    (
        "Write a function that returns the cumulative sum list of an input list.",
        "def solve(nums: list) -> list:",
        [([1, 2, 3], [1, 3, 6]), ([5], [5]), ([], []), ([1, 1, 1, 1], [1, 2, 3, 4])],
    ),
    (
        "Write a function that counts the frequency of each character in a string as a dict.",
        "def solve(s: str) -> dict:",
        [("aab", {"a": 2, "b": 1}), ("", {}), ("abc", {"a": 1, "b": 1, "c": 1})],
    ),
    (
        "Write a function that returns a list of all indices where a value appears in a list.",
        "def solve(lst: list, val) -> list:",
        [([1, 2, 1, 3, 1], 1, [0, 2, 4]), ([], 5, []), ([1, 2, 3], 9, [])],
    ),
    (
        "Write a function that zips two lists into a list of tuples.",
        "def solve(a: list, b: list) -> list:",
        [([1, 2], ["a", "b"], [(1, "a"), (2, "b")]), ([], [], [])],
    ),
    (
        "Write a function that takes a dict and returns a new dict with keys and values swapped.",
        "def solve(d: dict) -> dict:",
        [({"a": 1, "b": 2}, {1: "a", 2: "b"}), ({}, {})],
    ),
    (
        "Write a function that rotates a list to the right by k positions.",
        "def solve(lst: list, k: int) -> list:",
        [([1, 2, 3, 4, 5], 2, [4, 5, 1, 2, 3]), ([1, 2, 3], 0, [1, 2, 3]), ([], 3, [])],
    ),
    (
        "Write a function that returns the longest word in a sentence.",
        "def solve(sentence: str) -> str:",
        [("the quick brown fox", "quick"), ("hello", "hello"), ("a bb ccc", "ccc")],
    ),
    (
        "Write a function that chunks a list into sublists of size n.",
        "def solve(lst: list, n: int) -> list:",
        [([1, 2, 3, 4], 2, [[1, 2], [3, 4]]), ([1, 2, 3], 2, [[1, 2], [3]]), ([], 3, [])],
    ),
    (
        "Write a function that returns the median of a list of numbers (float).",
        "def solve(nums: list) -> float:",
        [([1, 2, 3], 2.0), ([1, 2, 3, 4], 2.5), ([5], 5.0), ([3, 1, 2], 2.0)],
    ),
    (
        "Write a function that returns True if all elements in a list are unique.",
        "def solve(lst: list) -> bool:",
        [([1, 2, 3], True), ([1, 2, 1], False), ([], True), ([1], True)],
    ),
    (
        "Write a function that returns the transpose of a 2D matrix (list of lists).",
        "def solve(matrix: list) -> list:",
        [([[1, 2], [3, 4]], [[1, 3], [2, 4]]), ([[1, 2, 3]], [[1], [2], [3]])],
    ),
    (
        "Write a function that returns all permutations of a list (as a list of lists).",
        "def solve(lst: list) -> list:",
        [([1, 2], [[1, 2], [2, 1]]), ([1], [[1]]), ([], [[]])],
    ),
    (
        "Write a function that performs binary search and returns the index of target in a sorted list, or -1 if not found.",
        "def solve(nums: list, target: int) -> int:",
        [([1, 3, 5, 7, 9], 5, 2), ([1, 2, 3], 4, -1), ([1], 1, 0)],
    ),
    (
        "Write a function that flattens a deeply nested list.",
        "def solve(nested) -> list:",
        [([1, [2, [3, 4], 5]], [1, 2, 3, 4, 5]), ([1, 2], [1, 2]), ([[[[1]]]], [1])],
    ),
    (
        "Write a function that groups a list of dicts by a given key.",
        "def solve(items: list, key: str) -> dict:",
        [
            (
                [{"type": "a", "v": 1}, {"type": "b", "v": 2}, {"type": "a", "v": 3}],
                "type",
                {"a": [{"type": "a", "v": 1}, {"type": "a", "v": 3}], "b": [{"type": "b", "v": 2}]},
            ),
        ],
    ),
    (
        "Write a function that computes the dot product of two vectors (lists of numbers).",
        "def solve(a: list, b: list) -> float:",
        [([1, 2, 3], [4, 5, 6], 32), ([1, 0], [0, 1], 0), ([2, 3], [3, 2], 12)],
    ),
    (
        "Write a function that returns the number of islands in a binary grid (1=land, 0=water).",
        "def solve(grid: list) -> int:",
        [
            ([[1, 1, 0], [0, 0, 0], [1, 0, 1]], 3),
            ([[1, 1, 1], [1, 1, 1]], 1),
            ([[0]], 0),
        ],
    ),
    (
        "Write a function that finds the longest common prefix of a list of strings.",
        "def solve(strs: list) -> str:",
        [
            (["flower", "flow", "flight"], "fl"),
            (["dog", "racecar", "car"], ""),
            (["same", "same"], "same"),
        ],
    ),
    (
        "Write a function that checks whether a given string has balanced parentheses.",
        "def solve(s: str) -> bool:",
        [
            ("(())", True), ("()()", True), ("(()", False), ("", True), (")(", False),
        ],
    ),
    (
        "Write a function that merges two sorted lists into one sorted list.",
        "def solve(a: list, b: list) -> list:",
        [
            ([1, 3, 5], [2, 4, 6], [1, 2, 3, 4, 5, 6]),
            ([], [1, 2], [1, 2]),
            ([1], [1], [1, 1]),
        ],
    ),
    (
        "Write a function that returns the running maximum of a list.",
        "def solve(nums: list) -> list:",
        [
            ([3, 1, 4, 1, 5, 2], [3, 3, 4, 4, 5, 5]),
            ([1], [1]),
            ([5, 4, 3], [5, 5, 5]),
        ],
    ),
    (
        "Write a function that returns the nth row of Pascal's triangle as a list.",
        "def solve(n: int) -> list:",
        [(0, [1]), (1, [1, 1]), (4, [1, 4, 6, 4, 1]), (3, [1, 3, 3, 1])],
    ),
    (
        "Write a function that converts Roman numerals to an integer.",
        "def solve(s: str) -> int:",
        [("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58), ("MCMXCIV", 1994)],
    ),
    (
        "Write a function that returns all subsets of a list.",
        "def solve(lst: list) -> list:",
        [
            ([1, 2], [[], [1], [2], [1, 2]]),
            ([], [[]]),
        ],
    ),
    (
        "Write a function that finds two numbers in a list that sum to a target and returns their indices.",
        "def solve(nums: list, target: int) -> list:",
        [([2, 7, 11, 15], 9, [0, 1]), ([3, 2, 4], 6, [1, 2])],
    ),
    (
        "Write a function that returns the number of ways to make change for amount using given coin denominations.",
        "def solve(amount: int, coins: list) -> int:",
        [(5, [1, 2, 5], 4), (3, [2], 0), (0, [1], 1)],
    ),
    (
        "Write a function that implements a simple Caesar cipher (shift right by k).",
        "def solve(text: str, k: int) -> str:",
        [("abc", 3, "def"), ("xyz", 3, "abc"), ("ABC", 1, "BCD"), ("hello", 0, "hello")],
    ),
    (
        "Write a function that computes the edit distance between two strings.",
        "def solve(s: str, t: str) -> int:",
        [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "abc", 0), ("a", "b", 1)],
    ),
    (
        "Write a function that returns the majority element (appears more than n/2 times) from a list.",
        "def solve(nums: list) -> int:",
        [([3, 2, 3], 3), ([2, 2, 1, 1, 1, 2, 2], 2), ([1], 1)],
    ),
    (
        "Write a function that returns all prime numbers up to n (inclusive) using the Sieve of Eratosthenes.",
        "def solve(n: int) -> list:",
        [(10, [2, 3, 5, 7]), (1, []), (20, [2, 3, 5, 7, 11, 13, 17, 19])],
    ),
]


def _build_dataset(seed: int = 42) -> Dataset:
    """Build the 50-challenge dataset."""
    rng = random.Random(seed)
    rows = []

    for desc, sig, test_cases in _CHALLENGES:
        # Format test cases as a string for the question
        tests_str = "\n".join(
            f"  {sig.split('(')[0].replace('def ', '')}({', '.join(repr(a) for a in (args if isinstance(args, tuple) else (args,)))}) == {repr(expected)}"
            if not isinstance(args, tuple)
            else f"  ({', '.join(repr(a) for a in args)}) == {repr(expected)}"
            for *args, expected in [
                (tc[:-1], tc[-1]) if len(tc) > 2 else ((tc[0],), tc[1])
                for tc in test_cases
            ]
        )
        question = (
            f"{desc}\n\nFunction signature: `{sig}`\n\n"
            f"Your function will be tested with these inputs (among others)."
        )
        rows.append({
            "question": question,
            "answer": sig,  # ground truth is the signature (actual eval is sandbox)
            "info": {
                "test_cases": str(test_cases),
                "signature": sig,
                "description": desc,
            },
        })

    rng.shuffle(rows)
    return Dataset.from_list(rows[:50])


# ---------------------------------------------------------------------------
# Reward — test-case pass rate via sandbox
# ---------------------------------------------------------------------------

async def code_reward(completion: str, answer: str, info: dict, **kwargs) -> float:
    """Execute model's code in sandbox and return fraction of tests passed.

    Requires PRIME_API_KEY.  Falls back to 0.0 if sandbox is unavailable.
    """
    # Extract code block from completion
    code = _extract_code(completion)
    if not code:
        return 0.0

    test_cases_str = info.get("test_cases", "[]") if info else "[]"
    try:
        test_cases = eval(test_cases_str, {"__builtins__": {}})  # noqa: S307
    except Exception:
        return 0.0

    # Build test harness
    test_harness = _build_test_harness(code, test_cases)

    try:
        sandbox = kwargs.get("sandbox")
        if sandbox is None:
            return 0.0
        result = await sandbox.run_python(test_harness, timeout=10)
        return _parse_pass_rate(result)
    except Exception:
        return 0.0


def _extract_code(completion: str) -> str:
    """Extract Python code from model output.

    Looks for ```python ... ``` blocks, then falls back to raw code.
    Uses str.find() only — no regex.
    """
    # Try fenced code block
    fence_start = completion.find("```python")
    if fence_start != -1:
        code_start = completion.find("\n", fence_start) + 1
        code_end = completion.find("```", code_start)
        if code_end != -1:
            return completion[code_start:code_end].strip()

    # Try plain ``` block
    fence_start = completion.find("```")
    if fence_start != -1:
        code_start = completion.find("\n", fence_start) + 1
        code_end = completion.find("```", code_start)
        if code_end != -1:
            return completion[code_start:code_end].strip()

    # Use lines starting with "def " as the start of code
    lines = completion.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("def "):
            return "\n".join(lines[i:]).strip()

    return completion.strip()


def _build_test_harness(code: str, test_cases: list) -> str:
    """Build a test runner script."""
    lines = [code, "", "passed = 0", "total = 0"]
    for tc in test_cases:
        args = tc[:-1]
        expected = tc[-1]
        args_str = ", ".join(repr(a) for a in args)
        lines.append(f"total += 1")
        lines.append(f"try:")
        lines.append(f"    result = solve({args_str})")
        lines.append(f"    if result == {repr(expected)}: passed += 1")
        lines.append(f"except Exception: pass")
    lines.append("print(f'PASS_RATE:{passed}/{total}')")
    return "\n".join(lines)


def _parse_pass_rate(output: str) -> float:
    """Parse 'PASS_RATE:n/m' from sandbox output."""
    marker = "PASS_RATE:"
    pos = output.find(marker)
    if pos == -1:
        return 0.0
    fraction = output[pos + len(marker):].split()[0]
    slash = fraction.find("/")
    if slash == -1:
        return 0.0
    try:
        passed = int(fraction[:slash])
        total = int(fraction[slash + 1:])
        return passed / total if total > 0 else 0.0
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------

def load_environment(
    num_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    """Load the Sandbox Code environment.

    Requires PRIME_API_KEY for sandbox access.

    Args:
        num_examples: Number of challenges (-1 = all 50).
        seed: Random seed.

    Returns:
        PythonEnv with test-case-fraction reward.
    """
    dataset = _build_dataset(seed=seed)
    if num_examples != -1:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    rubric = vf.Rubric(funcs=[code_reward], weights=[1.0])

    return vf.PythonEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
    )


if __name__ == "__main__":
    env = load_environment(num_examples=5)
    print(f"Dataset size: {len(env.dataset)}")
    for row in env.dataset.select(range(3)):
        print(f"  {row['question'][:80]}...")

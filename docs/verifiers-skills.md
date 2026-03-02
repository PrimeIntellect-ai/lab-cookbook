# Verifier Skills

All reward functions live in `prime_cookbook/skills/verifiers/`. Import and compose into `vf.Rubric`.

```python
from prime_cookbook.skills.verifiers import (
    exact_match_reward,
    contains_reward,
    set_match_reward,
    universal_rubric_reward,
    judge_reward,
    math_reward,
    code_reward,
    xml_parser_reward,
    last_line_reward,
)
import verifiers as vf

rubric = vf.Rubric(funcs=[exact_match_reward])
```

---

## exact_match_reward

**When to use:** Short, deterministic answers. Entity extraction, classification, factoid QA.

**Pitfalls:** Case-sensitive by default. Use `normalize=True` for production.

```python
from prime_cookbook.skills.verifiers import exact_match_reward

def reward(completion: str, state: dict, **kwargs) -> float:
    answer = state["info"]["answer"]
    return exact_match_reward(completion, answer, normalize=True)

rubric = vf.Rubric(funcs=[reward])
```

**Returns:** `1.0` if match, `0.0` otherwise.

**Signature:**
```python
exact_match_reward(
    completion: str,
    answer: str,
    normalize: bool = False,    # lowercase + strip whitespace
    field: str = "answer",      # XML/tag to extract from completion
) -> float
```

---

## contains_reward

**When to use:** When exact match is too strict but you need a deterministic signal. Answer may appear in a longer response.

**Pitfalls:** Reward hacking — model can include every possible answer to guarantee reward. Add length penalty if needed.

```python
from prime_cookbook.skills.verifiers import contains_reward

def reward(completion: str, state: dict, **kwargs) -> float:
    answer = state["info"]["answer"]
    base = contains_reward(completion, answer, normalize=True)
    # Optional: penalize verbose responses
    length_penalty = max(0, 1 - len(completion) / 2000)
    return base * (0.8 + 0.2 * length_penalty)
```

**Returns:** `1.0` if answer appears in completion, `0.0` otherwise.

---

## set_match_reward

**When to use:** Tasks with multiple valid answers (synonym sets, search results). Partial credit for partial overlap.

```python
from prime_cookbook.skills.verifiers import set_match_reward

def reward(completion: str, state: dict, **kwargs) -> float:
    valid_answers = state["info"]["valid_answers"]  # list of strings
    return set_match_reward(
        completion,
        valid_answers,
        threshold=0.5,    # fraction of valid answers that must be present
        normalize=True,
    )
```

**Returns:** Fraction of valid answers found in completion (0.0–1.0), `0.0` if below threshold.

**Note:** Order-insensitive. Uses `contains_reward` internally for each answer.

---

## universal_rubric_reward

**When to use:** Open-ended Q&A where you have a reference answer and want more nuance than exact match. Scores on semantic similarity using a lightweight model.

```python
from prime_cookbook.skills.verifiers import universal_rubric_reward

def reward(completion: str, state: dict, **kwargs) -> float:
    reference = state["info"]["answer"]
    return universal_rubric_reward(
        completion,
        reference,
        min_score=0.3,    # below this → 0.0 (hard cutoff)
    )
```

**Returns:** Semantic similarity score (0.0–1.0).

**Note:** Uses a local embedding model (no API calls). ~10ms per call.

---

## judge_reward

**When to use:** Simple pass/fail based on an LLM judge. Cheaper alternative to `JudgeRubric` when you only need binary feedback.

```python
from prime_cookbook.skills.verifiers import judge_reward

async def reward(completion: str, state: dict, **kwargs) -> float:
    question = state["info"]["question"]
    reference = state["info"]["answer"]
    return await judge_reward(
        completion,
        question=question,
        reference=reference,
        judge_model="gpt-4.1-mini",
        prompt="Does the response correctly answer the question? Score 0 or 1.",
    )
```

**Returns:** `1.0` or `0.0` (judge output is parsed as binary).

**Note:** For multi-criterion judging, use `vf.JudgeRubric` instead.

---

## math_reward

**When to use:** Math problems where the answer is in `\boxed{}` notation (GSM8K, MATH, competition problems).

```python
from prime_cookbook.skills.verifiers import math_reward

rubric = vf.Rubric(funcs=[math_reward])
```

**Extraction logic:**
1. Finds the last `\boxed{...}` in the completion
2. Normalizes: strips whitespace, removes trailing zeros, normalizes fractions
3. Compares against ground truth with the same normalization

**Returns:** `1.0` if extracted answer matches ground truth, `0.0` otherwise.

**Example:**
```python
# These all score 1.0 against ground truth "42"
completion_a = "...therefore x = \\boxed{42}"
completion_b = "...the answer is \\boxed{42.0}"
completion_c = "\\boxed{6 \\times 7} = \\boxed{42}"  # uses last boxed
```

---

## code_reward

**When to use:** Code generation tasks where correctness is verified by execution.

⚠️ **Sandbox warning:** `code_reward` executes untrusted model output. Always run inside a `SandboxEnv` or `PythonEnv`. Never call from a `SingleTurnEnv` on the host machine.

```python
from prime_cookbook.skills.verifiers import code_reward

def reward(completion: str, state: dict, **kwargs) -> float:
    test_cases = state["info"]["tests"]
    return code_reward(
        completion,
        test_cases=test_cases,
        language="python",
        timeout=10,           # seconds per test case
        sandbox=state.get("sandbox"),
    )
```

**Returns:** Fraction of test cases passed (0.0–1.0).

**Test case format:**
```python
test_cases = [
    {"input": "add(1, 2)", "expected": "3"},
    {"input": "add(-1, 1)", "expected": "0"},
]
```

---

## xml_parser_reward

**When to use:** Model outputs structured XML that must be valid and contain specific fields.

```python
from prime_cookbook.skills.verifiers import xml_parser_reward

def reward(completion: str, state: dict, **kwargs) -> float:
    required_fields = ["answer", "reasoning", "confidence"]
    return xml_parser_reward(
        completion,
        required_fields=required_fields,
        field_validators={
            "confidence": lambda x: x in ["high", "medium", "low"],
        },
    )
```

**Returns:** 
- `1.0` — valid XML, all required fields present and valid
- `0.5` — valid XML, but missing or invalid fields
- `0.0` — invalid XML

**Tip:** Use this when your system prompt instructs the model to format output as XML. More robust than regex for structured extraction.

---

## last_line_reward

**When to use:** Model is instructed to put the final answer on the last line. Simpler than XML parsing.

```python
from prime_cookbook.skills.verifiers import last_line_reward

def reward(completion: str, state: dict, **kwargs) -> float:
    answer = state["info"]["answer"]
    return last_line_reward(
        completion,
        answer=answer,
        normalize=True,
        strip_prefix="Answer:",   # optional: strip label before comparing
    )
```

**Returns:** `1.0` if last non-empty line matches answer, `0.0` otherwise.

**Example system prompt:**
```
Think step-by-step. Write your final answer on the last line, starting with "Answer:".
```

---

## Composing Multiple Rewards

Combine rewards in a single rubric. Scores are multiplied by default; set `combine="sum"` for additive.

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import exact_match_reward, xml_parser_reward

def structure_reward(completion, state, **kwargs):
    return xml_parser_reward(completion, required_fields=["answer", "reasoning"])

def answer_reward(completion, state, **kwargs):
    return exact_match_reward(completion, state["info"]["answer"],
                               field="answer")  # extract from <answer> tag

rubric = vf.Rubric(
    funcs=[structure_reward, answer_reward],
    weights=[0.3, 0.7],   # weighted sum
    combine="sum",
)
```

---

## Related Docs

- [Reward Design](reward-design.md)
- [Lab Skills](lab-skills.md)
- [Environment Types](environment-types.md)

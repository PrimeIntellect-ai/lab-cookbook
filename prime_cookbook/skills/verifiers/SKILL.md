# Verifier Skills

Reusable reward functions for `vf.Rubric`. Drop any of these into your rubric — they're all async functions that return a float `0.0–1.0`.

## Available Skills

| File | Functions | Use When |
|------|-----------|----------|
| `exact_match.py` | `exact_match_reward` | Single-word / label answers (yes/no/maybe, category names) |
| `exact_match.py` | `contains_reward` | Answer is a substring somewhere in the completion |
| `exact_match.py` | `set_match_reward` | Search tasks where multiple valid answers exist (pass `valid_answers` list in `info`) |
| `judge_rubric.py` | `universal_rubric_reward` | Open-ended Q&A with `reference_answer`, `key_points`, `source_quotes` in `info` |
| `judge_rubric.py` | `judge_reward` | Simple judge comparison against a reference string |
| `math_verify.py` | `math_reward` | Math answers wrapped in `\boxed{}` |
| `math_verify.py` | `extract_boxed_answer` | Helper — extract `\boxed{}` content from any string |
| `code_verify.py` | `code_reward` | Code generation with test cases (subprocess, no sandbox) |
| `parsers.py` | `xml_parser_reward` | Structured `<answer>` XML tags |
| `parsers.py` | `last_line_reward` | Model puts final answer on the last line |

## Quick Start

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import exact_match_reward, universal_rubric_reward

# L1 — deterministic, fast
rubric = vf.Rubric(funcs=[exact_match_reward])
env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)

# L3 — LLM judge with universal rubric
judge_rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")
rubric = vf.Rubric(funcs=[universal_rubric_reward])
rubric.add_class_object("judge", judge_rubric.judge)
env = vf.ToolEnv(dataset=dataset, rubric=rubric, tools=[...])
```

## Reward Function Signature

All functions follow the verifiers rubric convention — async, receive kwargs by name:

```python
async def my_reward(
    completion: str,   # model's final response text
    answer: str,       # ground truth from dataset
    info: dict,        # extra metadata from dataset row
    prompt: list,      # conversation history (list of messages)
    judge: callable,   # judge callable (if JudgeRubric attached)
    **kwargs,          # absorbs any extra rubric args
) -> float:
    ...
```

## Difficulty Tiers

| Tier | Verifier | Starting Reward Target |
|------|----------|----------------------|
| L1 | `exact_match_reward` | 0.15–0.35 |
| L2 | `set_match_reward` or custom deterministic | 0.10–0.25 |
| L3 | `universal_rubric_reward` | 0.10–0.20 |

If starting reward is outside the target range: **too easy → add constraints; too hard → break into sub-tasks or use L1 first**.

## Key Rules

- **NO REGEX** in reward functions — use `str.find()`, `str.split()`, `json.loads()`
- Reward functions must be `async def` 
- Always return a `float` in `[0.0, 1.0]`
- For `universal_rubric_reward`: penalize hallucination ×0.2, factual errors ×0.5
- Use `vf.ensure_keys(["OPENAI_API_KEY"])` at module level for judge skills

## Adding a New Verifier Skill

1. Add your function to the appropriate file (or create a new `.py`)
2. Export it from `__init__.py`
3. Document it in this SKILL.md table
4. Add a usage example to `docs/verifiers-skills.md`

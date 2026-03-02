# Reward Design

Getting rewards right is the most important part of any RL environment. Bad rewards → reward hacking, slow learning, or no learning at all.

---

## The 3-Level Curriculum

Structure environments in difficulty levels. Each level is a separate environment entry in your config.

| Level | Reward Signal | Example | Model Solve Rate Target |
|-------|--------------|---------|------------------------|
| L1 | Exact match / deterministic | Precise entity extraction | 0.3–0.5 |
| L2 | Structured but lenient | Partial credit for near-miss | 0.2–0.4 |
| L3 | LLM judge | Open-ended question answering | 0.15–0.35 |

Start at L1. Only introduce L2/L3 after the model plateaus on L1.

```python
# L1 — deterministic, fast, cheap
rubric_l1 = vf.Rubric(funcs=[exact_match_reward])

# L2 — partial credit, still deterministic
rubric_l2 = vf.Rubric(funcs=[fuzzy_match_reward])

# L3 — LLM judge, expensive but flexible
rubric_l3 = vf.JudgeRubric(
    judge_model="gpt-4.1-mini",
    criteria=["accuracy", "completeness", "no_hallucination"],
    weights=[3, 2, 3],
)
```

---

## Difficulty Calibration

**Sweet spot: 15–35% base solve rate.**

If the model already solves >60% of tasks before any training, the reward signal is too dense — the model won't learn much. If it solves <10%, gradients vanish.

```python
# Quick calibration script
import verifiers as vf

env = MyEnv(dataset=dataset, rubric=rubric)
scores = vf.quick_eval(env, model="gpt-4.1-nano", n=100)
print(f"Base solve rate: {(scores > 0.5).mean():.1%}")
# Target: 15–35%
```

**Adjustments:**
- Too easy (>50%): add constraints, require citations, increase precision threshold
- Too hard (<10%): break into sub-tasks, add hint in system prompt, use L1 instead

---

## The Universal Rubric Pattern

One rubric class, multiple datasets. Parameterize on ground truth rather than hardcoding.

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import universal_rubric_reward

def make_rubric(ground_truth_col: str = "answer"):
    def reward(completion, state, **kwargs):
        answer = state["info"].get(ground_truth_col, "")
        return universal_rubric_reward(completion, answer)
    return vf.Rubric(funcs=[reward])

# Reuse same rubric for different datasets
math_rubric = make_rubric("solution")
qa_rubric = make_rubric("answer")
search_rubric = make_rubric("expected_result")
```

---

## Hallucination and Factual Error Penalties

Apply multiplicative penalties on top of base reward:

```python
def hallucination_penalty(completion, state, **kwargs):
    """Penalize responses containing known false claims."""
    base_score = state.get("base_reward", 1.0)
    if contains_hallucination(completion, state["info"]):
        return base_score * 0.2  # 80% penalty
    return base_score

def factual_error_penalty(completion, state, **kwargs):
    """Partial penalty for factual errors (not full hallucination)."""
    base_score = state.get("base_reward", 1.0)
    error_count = count_factual_errors(completion, state["info"])
    return base_score * (0.5 ** error_count)  # 50% per error
```

**Rule of thumb:**
- Confirmed hallucination: `× 0.2`
- Factual error / wrong citation: `× 0.5`
- Off-topic / refuses to answer: `× 0.0`

---

## Ground Truth Generation with GPT-4.1

For L3 environments, generate reference answers offline to avoid slow judge calls in the training loop:

```python
from prime_cookbook.skills.lab import generate_ground_truth
from openai import OpenAI

client = OpenAI()

ground_truths = generate_ground_truth(
    questions=dataset["question"],
    model="gpt-4.1",                  # use the best model for GT
    system_prompt="Answer precisely and cite your sources.",
    batch_size=50,
)

# Save and reuse — don't regenerate on every run
import json
with open("data/ground_truth.jsonl", "w") as f:
    for gt in ground_truths:
        f.write(json.dumps(gt.dict()) + "\n")
```

---

## NO REGEX Rule

**Never use regex to parse model outputs.** Use `str.find()` and `json.loads()` instead.

```python
# ❌ BAD — regex breaks on slight formatting variations
import re
match = re.search(r"<answer>(.*?)</answer>", completion)
answer = match.group(1) if match else ""

# ✅ GOOD — explicit, predictable
start = completion.find("<answer>")
end = completion.find("</answer>")
if start != -1 and end != -1:
    answer = completion[start + len("<answer>"):end].strip()
else:
    answer = ""

# ✅ GOOD — for JSON outputs
try:
    data = json.loads(completion)
    answer = data.get("answer", "")
except json.JSONDecodeError:
    answer = ""
```

Regex silently fails on minor variations (extra spaces, different capitalization). Explicit parsing fails loudly, which is better for debugging.

---

## JudgeRubric Setup

```python
import verifiers as vf

rubric = vf.JudgeRubric(
    judge_model="gpt-4.1-mini",   # cheap, fast, consistent
    criteria=[
        "accuracy",          # is the answer factually correct?
        "no_hallucination",  # does it avoid fabricating info?
        "covers_key_points", # does it address all parts of the question?
        "answers_question",  # is it actually responsive?
        "cites_evidence",    # does it support claims with evidence?
    ],
    weights=[3, 3, 2, 1, 1],   # sum to 10, normalize internally
    system_prompt=(
        "You are an expert evaluator. Score each criterion 0–1. "
        "Be strict: partial credit only for partial correctness."
    ),
)
```

**Tips:**
- Use `gpt-4.1-mini` for judge, not `gpt-4.1` — 5× cheaper, nearly identical calibration
- Keep criteria count ≤ 5 to avoid judge inconsistency
- Weight correctness and no-hallucination highest
- Cache judge calls when evaluating the same completion multiple times

---

## Common Pitfalls

### Reward Hacking
The model exploits the reward function rather than the task.

**Symptoms:** Reward increases but qualitative output degrades. High reward, wrong answers.

**Fixes:**
- Use multiple reward signals (accuracy + brevity + no-hallucination)
- Randomize phrasing of eval questions
- Periodically update judge prompts
- L3 judge with diverse criteria is harder to hack than L1 exact match

### Sparse Rewards
Model gets 0 for almost every rollout early in training.

**Symptoms:** No reward signal, training stalls immediately.

**Fixes:**
- Check difficulty calibration (see above)
- Add intermediate rewards (partial credit for correct approach)
- Use `rollouts_per_example=16` to increase chances of at least one positive rollout per batch

### Judge Inconsistency
LLM judge gives wildly different scores for semantically identical answers.

**Symptoms:** High variance in reward for similar completions.

**Fixes:**
- Add few-shot examples to judge prompt
- Use temperature=0 for judge inference
- Add explicit scoring rubric ("score 1.0 if..., 0.5 if..., 0.0 if...")
- Cross-validate with a second judge model on 10% of examples

---

## Related Docs

- [Verifier Skills](verifiers-skills.md) — available reward functions
- [Lab Skills](lab-skills.md) — ground truth generation tools
- [Document Search Recipe](recipes/document-search.md) — full 3-level curriculum example

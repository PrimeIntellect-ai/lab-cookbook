# Math Reasoning

Single-turn math environment using GSM8K. Model must produce a boxed final answer. Primary use: sanity-check that your training pipeline works end-to-end.

**Environment type:** `SingleTurnEnv`  
**Reward:** `math_reward` (boxed answer extraction + normalization)  
**Dataset:** `openai/gsm8k` (train split, 7473 examples)

---

## Quick Start

```bash
prime env install prime_cookbook/recipes/math_rl

prime eval run recipe-math-rl --model gpt-4.1-mini
# Expected: reward mean ~0.94

prime rl run prime_cookbook/recipes/math_rl/config.toml
```

---

## Expected Metrics

| Model | Reward Mean | Notes |
|-------|-------------|-------|
| gpt-4.1-mini | ~0.94 | Near-ceiling; not ideal for training |
| Qwen2.5-7B-Instruct | ~0.88 | Good eval baseline |
| Qwen2.5-1.5B-Instruct | ~0.22 | Good training difficulty for 1.5B |
| Qwen2.5-0.5B-Instruct | ~0.08 | Too sparse; switch to simpler math |

---

## Training Notes

GSM8K saturates quickly for models ≥7B. Use it as a pipeline sanity check, not as a serious RL benchmark.

```toml
# config.toml
[model]
name = "Qwen/Qwen2.5-1.5B-Instruct"

[training]
max_steps = 200        # saturates by ~100 steps for 1.5B on GSM8K
batch_size = 64
rollouts_per_example = 8

[sampling]
max_tokens = 512
temperature = 1.0

[[env]]
id = "recipe-math-rl"
weight = 1.0
```

**What to expect:**
- Steps 0–20: Reward rises from ~0.22 → ~0.50
- Steps 20–80: Plateau around 0.75–0.85
- Steps 80+: Diminishing returns

---

## Extension Ideas

**Harder math (recommended for 7B+ models):**
```python
# Switch dataset to MATH-500 (competition problems)
dataset = load_dataset("hendrycks/competition_math", split="test")
```

**Chain-of-thought supervision:**
```python
system_prompt = (
    "Solve step-by-step. Show your reasoning. "
    "Put your final answer in \\boxed{}."
)
```

**Multi-step reward (partial credit for correct steps):**
```python
from prime_cookbook.skills.verifiers import math_reward, last_line_reward

def step_reward(completion, state, **kwargs):
    # Reward for correct intermediate steps (heuristic)
    n_steps = completion.count("=")
    return min(1.0, n_steps / 5) * 0.2  # up to 20% partial credit

rubric = vf.Rubric(
    funcs=[math_reward, step_reward],
    weights=[0.8, 0.2],
    combine="sum",
)
```

---

## Related

- [Environment Types](../environment-types.md)
- [Verifier Skills](../verifiers-skills.md) — `math_reward` details
- [Training Config](../training-config.md) — per-recipe config example

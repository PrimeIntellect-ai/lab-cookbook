# Multi-Env

Multi-task training using `EnvGroup`. Combines math reasoning, tool use, and document search into a single training run. Prevents catastrophic forgetting and trains a more general model.

**Environment type:** `EnvGroup`  
**Reward:** Per-environment reward functions (composited by sampler)  
**Dataset:** Union of per-env datasets, sampled by weight

---

## Quick Start

```bash
prime env install prime_cookbook/recipes/multi_env

prime eval run recipe-multi-env --model gpt-4.1-mini
# Shows per-env breakdown

prime rl run prime_cookbook/recipes/multi_env/config.toml
```

---

## Environment Overview

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import math_reward, exact_match_reward
from prime_cookbook.skills.lab import TFIDFSearchIndex

# Build individual environments
math_env = vf.SingleTurnEnv(
    dataset=gsm8k_train,
    rubric=vf.Rubric(funcs=[math_reward]),
    system_prompt="Solve the math problem. Put your answer in \\boxed{}.",
)

tool_env = vf.ToolEnv(
    dataset=tool_tasks,
    rubric=vf.Rubric(funcs=[answer_reward]),
    tools=[calculator, lookup],
    max_turns=8,
)

search_env = DocSearchL1(
    dataset=search_questions,
    index=TFIDFSearchIndex.load("data/wiki_index.pkl"),
    max_turns=10,
)

# Combine into EnvGroup
env = vf.EnvGroup(
    envs=[math_env, tool_env, search_env],
    weights=[1.0, 1.5, 2.0],   # search is harder, sample more
)
```

---

## Training Config

```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[training]
max_steps = 5000
batch_size = 128
rollouts_per_example = 8
learning_rate = 5e-6

[sampling]
max_tokens = 2048
temperature = 1.0

# Directly specify envs with weights (alternative to EnvGroup in code)
[[env]]
id = "recipe-math-rl"
weight = 1.0

[[env]]
id = "recipe-tool-use"
weight = 1.5

[[env]]
id = "recipe-document-search-l1"
weight = 2.0

[[env]]
id = "recipe-document-search-l2"
weight = 1.0
```

---

## Per-Env Metrics

`prime eval run` on a multi-env shows breakdown per environment:

```
Evaluating recipe-multi-env with Qwen2.5-7B-Instruct (step 2000)
──────────────────────────────────────────────────────────────────
  Overall reward mean: 0.58 ± 0.31

  Per-environment:
  recipe-math-rl              0.84 ± 0.18   (+0.06 vs base)
  recipe-tool-use             0.63 ± 0.27   (+0.12 vs base)
  recipe-document-search-l1   0.66 ± 0.29   (+0.15 vs base)
  recipe-document-search-l2   0.41 ± 0.31   (+0.09 vs base)
──────────────────────────────────────────────────────────────────
```

---

## Weight Tuning

Weights control **sampling frequency**, not loss scale. Higher weight = more rollouts from that env.

**Rules of thumb:**
- Weight proportional to task difficulty (harder tasks need more training signal)
- If one env's reward is plateauing while others are still improving, reduce its weight
- If one env's reward is collapsing (reward hacking), reduce its weight

```python
# Dynamic weights — adjust mid-training based on eval metrics
# Currently requires manual config edit + restart
# Planned: online weight adjustment via prime rl update

[[env]]
id = "recipe-math-rl"
weight = 0.5           # reduce once reward > 0.8

[[env]]
id = "recipe-document-search-l3"
weight = 3.0           # increase as model gets better at L1/L2
```

---

## Why Multi-Env Training Works

Single-task RL fine-tuning often causes **catastrophic forgetting** — the model improves on the target task but degrades on everything else. Multi-env training:

1. **Regularizes** — gradient updates from different tasks interfere constructively
2. **Prevents shortcut learning** — reward hacks for one task often hurt others
3. **Builds generalizable skills** — "search then answer" generalizes across tool-use and document-search
4. **Keeps language quality high** — math env keeps response formatting clean

**Evidence:** In internal evals, a 7B model trained multi-env outperforms task-specific fine-tuned models on held-out tasks (zero-shot generalization: +8–12% average).

---

## Advanced: Adaptive Curriculum with EnvGroup

```python
import verifiers as vf

class AdaptiveCurriculumGroup(vf.EnvGroup):
    """Dynamically reweights envs based on rolling reward averages."""

    def __init__(self, envs, initial_weights, window=100):
        super().__init__(envs=envs, weights=initial_weights)
        self.window = window
        self.reward_history = {i: [] for i in range(len(envs))}

    def update_weights(self, env_idx: int, reward: float):
        """Call after each rollout to update sampling weights."""
        hist = self.reward_history[env_idx]
        hist.append(reward)
        if len(hist) > self.window:
            hist.pop(0)

        # Increase weight for envs where model is in learning range (0.2–0.6)
        new_weights = []
        for i, h in self.reward_history.items():
            if not h:
                new_weights.append(1.0)
                continue
            mean = sum(h) / len(h)
            # Peak weight at mean=0.35 (sweet spot), taper off above/below
            learning_signal = 1.0 - abs(mean - 0.35) / 0.35
            new_weights.append(max(0.1, learning_signal))

        self.weights = new_weights
```

---

## Expected Outcomes

| Metric | Single-Task (Math) | Multi-Env (3 tasks) |
|--------|-------------------|---------------------|
| Math eval (base task) | +0.15 ↑ | +0.06 ↑ |
| Tool-use eval (unseen) | -0.08 ↓ | +0.12 ↑ |
| Search eval (unseen) | -0.05 ↓ | +0.15 ↑ |
| Instruction following | -0.12 ↓ | -0.02 → |

Multi-env sacrifices peak performance on the primary task but dramatically improves generalization.

---

## Related

- [Environment Types](../environment-types.md) — EnvGroup
- [Math Reasoning Recipe](math-reasoning.md)
- [Tool Use Recipe](tool-use.md)
- [Document Search Recipe](document-search.md)
- [Training Config](../training-config.md)

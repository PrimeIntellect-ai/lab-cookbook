# Prime Cookbook

Collection of RL environment recipes for [Prime Intellect](https://primeintellect.ai) Lab — batteries-included environments for training language models with the [verifiers](https://github.com/willccbb/verifiers) library.

## What This Is

A structured library of **skills** (reusable building blocks) and **recipes** (complete, runnable environments) for RL fine-tuning with verifiable rewards. Think of it as a cookbook: skills are ingredients, recipes are dishes.

```
prime_cookbook/
├── skills/
│   ├── verifiers/   # reward functions, parsers, rubrics
│   └── lab/         # dataset builders, search indices, ground truth generation
└── recipes/
    ├── math_rl/         # math reasoning (GSM8K-style)
    ├── tool_use/        # stateless tool calling
    ├── document_search/ # 3-level curriculum search
    ├── word_game/       # multi-turn word games
    ├── sandbox_code/    # code execution in sandboxes
    └── multi_env/       # multi-task EnvGroup training
```

## Prerequisites

- Python 3.10+
- `verifiers` library (`pip install verifiers`)
- `prime` CLI (`pip install prime-rl`)
- `OPENAI_API_KEY` (for LLM judge rewards and ground truth generation)
- `PRIME_API_KEY` (for training on Prime Inference)

## Architecture

```
prime rl run config.toml
        │
        ▼
  verifiers rollout loop
  ┌─────────────────────────────────────────┐
  │  dataset → prompt → model → response    │
  │         ↑                    ↓          │
  │    env_response         rubric.score()  │
  │    (tool results,       (reward signal) │
  │     next turn, done)                    │
  └─────────────────────────────────────────┘
        │
        ▼
  prime CLI → optimizer → weight update
```

The verifiers library owns the rollout loop. The `prime` CLI handles optimization (GRPO/AIPO). Your recipe defines:
1. A dataset (prompts + optional ground truth)
2. An environment (inherits from one of 6 env types)
3. A rubric (reward function)

## Two Content Types

**Skills** — import and compose:
```python
from prime_cookbook.skills.verifiers import math_reward, exact_match_reward
from prime_cookbook.skills.lab import TFIDFSearchIndex, DatasetBuilder
```

**Recipes** — install and run:
```bash
prime env install prime_cookbook/recipes/math_rl
prime eval run recipe-math-rl --model gpt-4.1-mini
prime rl run prime_cookbook/recipes/math_rl/config.toml
```

## Related Docs

- [Getting Started](getting-started.md) — install, eval, train in 5 steps
- [Environment Types](environment-types.md) — which env class to use
- [Reward Design](reward-design.md) — curriculum, rubrics, pitfalls
- [Training Config](training-config.md) — TOML reference
- [Recipes →](recipes/) — individual recipe docs

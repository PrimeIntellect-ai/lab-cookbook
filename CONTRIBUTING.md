# Contributing a New Recipe

This guide walks through adding a new environment recipe to `prime-cookbook`. Follow these steps in order — each has a verification checkpoint.

---

## Step 1: Create the environment file

Create `environments/<name>/<name>.py` with a `load_environment()` function:

```python
# environments/my-recipe/my_recipe.py
"""My Recipe — one-line description of what this environment trains.

Reward level: L1 / L2 / L3
Starting reward (untrained): ~0.20 on Llama-3.2-3B-Instruct
"""
import verifiers as vf
from prime_cookbook.skills.verifiers.exact_match import exact_match_reward


SYSTEM_PROMPT = """You are a helpful assistant. Answer concisely.
Put your final answer on the last line."""


def build_dataset() -> vf.Dataset:
    """Build and return the training dataset."""
    from datasets import Dataset
    # Load or generate your data here
    rows = [
        {"question": "What is 2+2?", "answer": "4", "info": {}},
        # ... more examples
    ]
    return Dataset.from_list(rows)


def build_rubric() -> vf.Rubric:
    """Build the reward rubric."""
    return vf.Rubric(funcs=[exact_match_reward])


def load_environment() -> vf.Environment:
    """Entry point — returns the configured environment.
    
    Called by `prime eval run` and `prime rl run`.
    """
    dataset = build_dataset()
    rubric = build_rubric()
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )
```

**Checkpoint**: Run `python -c "from my_recipe import load_environment; env = load_environment(); print(env)"` — should not raise.

---

## Step 2: Add pyproject.toml

Create `environments/<name>/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "prime-cookbook-my-recipe"
version = "0.1.0"
description = "One-line description of this recipe"
requires-python = ">=3.10"
dependencies = [
    "prime-cookbook>=0.1.0",
    "verifiers>=0.1.10",
    # Any recipe-specific deps
]

[tool.hatch.build.targets.wheel]
packages = ["my_recipe"]  # or the folder name

[tool.verifiers.environment]
entrypoint = "my_recipe:load_environment"

[tool.verifiers.eval]
num_examples = 100
rollouts_per_example = 4
```

**Checkpoint**: `pip install -e environments/<name>` should succeed.

---

## Step 3: Create the recipe documentation

Create `docs/recipes/<name>.md`:

```markdown
# My Recipe

## What This Trains

Brief description of the skill being trained. What does the agent learn to do?

## Dataset

- Source: (where the data comes from)
- Size: N examples (split: X train / Y eval)
- Format: question + answer + info

## Reward Design

Level: L1 / L2 / L3

Describe the reward function:
- What counts as correct?
- Any partial credit?
- Hallucination penalties?

## Tools

List any tools the agent can call:
| Tool | Description | Returns |
|------|-------------|---------|

## Expected Results

| Model | Eval Reward (untrained) | Eval Reward (after 500 steps) |
|-------|------------------------|------------------------------|
| Llama-3.2-3B-Instruct | 0.XX | 0.XX |
| Qwen2.5-7B-Instruct | 0.XX | 0.XX |

## Training Curve

(Add a screenshot or W&B link after training)

## Notes & Gotchas

Any quirks, known failure modes, or tips.
```

---

## Step 4: Add config.toml for training

Create `environments/<name>/config.toml`:

```toml
[train]
model = "meta-llama/Llama-3.2-3B-Instruct"
environment = "my_recipe:load_environment"
rollouts_per_example = 8
learning_rate = 1e-5
max_steps = 1000
warmup_steps = 50
clip_ratio = 1.2

[eval]
judge_model = "gpt-4.1-mini"
num_examples = 100
rollouts_per_example = 4
eval_every = 100

[logging]
wandb_project = "prime-cookbook"
wandb_run_name = "my-recipe-3b"
```

---

## Step 5: Test with prime CLI

```bash
# Install the recipe
prime env install environments/<name>

# Run eval (should complete without errors)
prime eval run environments/<name>

# Check starting reward is in [0.15, 0.35]
# If too high: add harder examples
# If too low: simplify or add easier warm-up examples
```

**Checkpoint**: Eval runs cleanly, starting reward is in the sweet spot.

---

## Step 6: Update root README.md

Add your recipe to the table in `README.md`:

```markdown
| [my-recipe](environments/my-recipe/) | L1 | Exact match | One-line description |
```

---

## Step 7: Show expected eval score and training curve

In `environments/<name>/README.md`, add:

1. **Starting reward** (eval on base model before training)
2. **Final reward** (eval after training for N steps)
3. A training curve plot or W&B link (if available)

This lets users calibrate expectations and verify their setup is working.

---

## Checklist

```
[ ] environments/<name>/<name>.py        — load_environment() implemented
[ ] environments/<name>/pyproject.toml   — entrypoint + eval defaults
[ ] environments/<name>/config.toml      — training config
[ ] environments/<name>/README.md        — expected scores + curve
[ ] docs/recipes/<name>.md              — extended documentation
[ ] README.md updated                    — recipe added to table
[ ] prime eval run passes                — no errors, reward in [0.15, 0.35]
[ ] AGENTS.md compliance checked         — no regex, tools stateless, async rewards
```

---

## Common Mistakes

**Reward always 0.0**: Check that your reward function is `async def` and that it's actually being called. Add a `print()` statement temporarily to verify.

**Reward always 1.0**: Your task is too easy for the base model. Increase difficulty.

**JSON parsing errors**: The judge is returning malformed JSON. Check the judge prompt and add a fallback parser (see `judge_rubric.py` for the pattern).

**Import errors**: Make sure your recipe's `pyproject.toml` lists all dependencies and you've run `pip install -e environments/<name>`.

**`load_environment()` is slow**: Move any expensive initialization (model loading, index building) into `load_environment()` not into tool functions — tools are called per rollout but the environment is initialized once.

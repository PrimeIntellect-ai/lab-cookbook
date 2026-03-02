# Eval Guide

Evaluate environments before and during training.

---

## Running Evals

```bash
# Evaluate on Prime Inference (recommended)
prime eval run recipe-math-rl --model gpt-4.1-mini

# Specify number of examples
prime eval run recipe-math-rl --model gpt-4.1-mini --n 200

# Evaluate a local checkpoint
prime eval run recipe-math-rl \
  --model ./checkpoints/step-500 \
  --n 100

# Multiple environments
prime eval run recipe-document-search-l1 recipe-document-search-l2 \
  --model gpt-4.1-mini
```

---

## Reading Results

```
Evaluating recipe-document-search-l1 with gpt-4.1-mini
──────────────────────────────────────────────────────
  examples evaluated:   200
  reward mean:          0.71
  reward std:           0.28
  reward > 0:           89.5%    (non-zero reward)
  reward > 0.5:         78.0%    (partial credit+)
  reward = 1.0:         52.0%    (full credit)

  tool call stats:
    total_tool_calls:   438      (avg 2.19 per rollout)
    search_calls:       438
    avg_turns:          3.2

  timing:
    avg_time_per_rollout:  4.2s
    total_eval_time:       840s
──────────────────────────────────────────────────────
```

**What to look for:**
- `reward mean` — primary quality metric
- `reward > 0` — are rollouts non-degenerate? Should be >80%
- `reward std` — high std is fine (diverse tasks); near-zero std suggests reward collapse
- `avg_turns` — is the model using the tool budget sensibly?

---

## Eval vs Training Split

Always use separate datasets for eval and training.

```python
from datasets import load_dataset

dataset = load_dataset("my_org/my_dataset")

# Standard split
train_ds = dataset["train"]   # used during RL training
eval_ds  = dataset["test"]    # never seen during training

# If no official split, do it manually
from datasets import Dataset
full = dataset["train"].shuffle(seed=42)
n = len(full)
train_ds = full.select(range(int(n * 0.9)))
eval_ds  = full.select(range(int(n * 0.9), n))
```

In your environment:
```python
env = MyEnv(dataset=train_ds, rubric=rubric)

# Separate eval env
eval_env = MyEnv(dataset=eval_ds, rubric=rubric)
```

---

## Smoke Testing Locally

Before pushing to Prime Inference, validate the environment locally with a small sample:

```bash
# Install env in local mode
prime env install prime_cookbook/recipes/my_recipe --local

# Quick smoke test (5 examples, cheap model)
prime eval run recipe-my-recipe \
  --model gpt-4.1-nano \
  --n 5 \
  --local
```

Checklist before training:
- [ ] `reward mean` is in the 0.15–0.35 range for the target model size
- [ ] No crash on first rollout
- [ ] Tool calls appear in stats (for tool envs)
- [ ] Eval set has no overlap with train set
- [ ] `reward > 0` > 30% (model can get any reward at all)

---

## Expected Scores by Recipe

Evaluated with `gpt-4.1-mini` unless noted.

| Recipe | Eval Model | Reward Mean | Notes |
|--------|-----------|-------------|-------|
| math-rl | gpt-4.1-mini | ~0.94 | Saturates quickly; use GSM8K-hard or MATH-500 for harder tasks |
| tool-use | gpt-4.1-mini | ~0.72 | Variable by task type |
| document-search L1 | gpt-4.1-mini | ~0.71 | Good baseline difficulty |
| document-search L2 | gpt-4.1-mini | ~0.48 | Harder; partial credit helps |
| document-search L3 | gpt-4.1-mini | ~0.61 | LLM judge more lenient |
| word-game | gpt-4.1-mini | ~0.38 | Hard; sparse reward |
| sandbox-code | gpt-4.1-mini | ~0.55 | Code correctness via execution |
| multi-env | gpt-4.1-mini | varies | Per-env metrics shown separately |

**Small model expectations** (Qwen2.5-1.5B on math-rl):
```
reward mean:  0.22    ← appropriate difficulty for training
reward > 0.5: 28.0%
```

---

## Tracking Eval Over Training

`prime eval run` can be run as a background job during training to track reward over time:

```bash
# Run eval every 100 training steps (in a separate terminal)
watch -n 300 "prime eval run recipe-math-rl \
  --model ./checkpoints/latest \
  --n 100 \
  --tag step=$(cat checkpoints/latest/step.txt)"
```

Or configure periodic eval in `config.toml`:
```toml
[eval]
every_n_steps = 100
n_examples = 100
model = "gpt-4.1-mini"
```

---

## Related Docs

- [Getting Started](getting-started.md)
- [Training Config](training-config.md)
- [Reward Design](reward-design.md)

# Math Reasoning (SingleTurnEnv + MathRubric)

The simplest possible verifiers environment: a `SingleTurnEnv` paired with `MathRubric`.  
Use this recipe to verify that your training stack is wired up correctly end-to-end before moving to harder tasks.

## What it does

Generates 500 arithmetic problems (addition, subtraction, multiplication, exact-integer division).  
The model must answer in `\boxed{}` format.  `vf.MathRubric` extracts the boxed value and checks symbolic equivalence — no string matching.

## Setup

```bash
pip install verifiers>=0.1.10
```

## Quick eval (no training)

```python
from prime_cookbook.recipes.math_rl.math_rl import load_environment
import verifiers as vf

env = load_environment(num_examples=50)
vf.evaluate(env, model="gpt-4.1-mini", rollouts_per_example=4)
```

## Training run

```bash
prime rl run config.toml
```

Expected metrics after training:
| Step | Mean Reward | Notes |
|------|-------------|-------|
| 0    | ~0.90       | Strong base for small math |
| 5    | ~0.97       | Near saturation |
| 10+  | 1.00        | Use a harder dataset |

## When to use this pattern

- **Sanity checking** a new model or training setup
- **Template** for any single-turn Q&A task
- **Baseline** before introducing tools or multi-turn reasoning

## Tips for harder math

- Replace the generated dataset with **GSM8K**: `from datasets import load_dataset; ds = load_dataset("gsm8k", "main", split="train")`
- For MATH competition problems: `load_dataset("lighteval/MATH", split="train")`
- Increase `max_tokens` to 1024+ for chain-of-thought reasoning
- Target difficulty: model solves 30–70% of problems before training starts; adjust problem difficulty accordingly

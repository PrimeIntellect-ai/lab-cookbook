# Multi-Environment (EnvGroup — math + tools + word game)

Combines three distinct environments into a single training run using `vf.EnvGroup`.  
This is the recipe to use when you want a model to generalise across multiple capability dimensions.

## Sub-environments

| Environment | Type | Task | Starting Reward |
|-------------|------|------|----------------|
| `math_rl` | SingleTurnEnv | Arithmetic in `\boxed{}` | ~0.70 |
| `tool_use` | ToolEnv | 4 stateless tools | ~0.30 |
| `word_game` | MultiTurnEnv | Wordle (6 guesses) | ~0.15 |

Combined starting reward: **~0.38** (weighted average)

## Setup

```bash
pip install verifiers>=0.1.10
```

## Quick eval

```python
from prime_cookbook.recipes.multi_env.multi_env import load_environment
import verifiers as vf

env = load_environment(math_examples=50, tools_examples=50, word_examples=50)
vf.evaluate(env, model="gpt-4.1-mini", rollouts_per_example=4)
```

## Training run

```bash
prime rl run config.toml
```

## How vf.EnvGroup works

```python
vf.EnvGroup(
    envs=[math_env, tools_env, word_env],
    env_names=["math", "tools", "word"],
)
```

- Each batch interleaves examples from all environments
- Each environment uses its own reward function
- The combined reward signal is normalized per-environment before advantage computation
- Sub-environment metrics are logged separately (e.g., `math/mean_reward`, `tools/mean_reward`)

## Why multi-environment training?

**Generalization**: A model trained on only math tends to lose general tool-use ability.  
EnvGroup prevents reward collapse to any single task.

**Capability balance**: Harder environments (word game) benefit from the momentum of  
easier ones (math) during early training.

**Efficiency**: One training run produces a model capable of multiple behaviors,  
rather than requiring separate fine-tuning runs.

## Scaling up

To add a fourth environment:

```python
from prime_cookbook.recipes.document_search import document_search_l1

doc_env = document_search_l1.load_environment(num_examples=200)

vf.EnvGroup(
    envs=[math_env, tools_env, word_env, doc_env],
    env_names=["math", "tools", "word", "docs"],
)
```

## Tips

- **Balance example counts**: Equal counts per environment (~200 each) work well for similar-sized environments
- **Match difficulty**: Avoid one environment dominating with very high or very low reward
- **Monitor per-env metrics**: If one environment's reward drops, investigate that sub-environment
- **Start reward target**: Aim for 0.25-0.45 across the combined group for good training signal

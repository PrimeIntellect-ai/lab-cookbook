# Tool Use (ToolEnv — 4 stateless tools)

A `vf.ToolEnv` environment that teaches a model to call stateless utility tools.  
Four tools cover calculation, string manipulation, and unit conversion — all with fully deterministic answers.

## Tools

| Tool | Description | Example |
|------|-------------|---------|
| `calculate(expression)` | Evaluate arithmetic (+, -, *, /) | `"(3 + 4) * 2"` → `"14"` |
| `count_words(text)` | Count space-separated words | `"hello world"` → `2` |
| `reverse_string(text)` | Reverse characters | `"hello"` → `"olleh"` |
| `convert_units(value, from_unit, to_unit)` | Unit conversion | `5.0, "km", "miles"` → `"3.1069 miles"` |

## Setup

```bash
pip install verifiers>=0.1.10
```

## Quick eval

```python
from prime_cookbook.recipes.tool_use.tool_use import load_environment
import verifiers as vf

env = load_environment(num_examples=50)
vf.evaluate(env, model="gpt-4.1-mini", rollouts_per_example=4)
```

## Training run

```bash
prime rl run config.toml
```

## Expected metrics

| Model | Starting reward | After 50 steps |
|-------|----------------|----------------|
| GPT-4.1-mini | ~0.85 | ~0.97 |
| Llama-3.2-1B | ~0.25 | ~0.65 |
| Llama-3.2-3B | ~0.45 | ~0.80 |

## Key patterns demonstrated

- **`vf.ToolEnv`** — automatic JSON schema extraction from Python function signatures
- **Stateless tools** — no shared state; safe to call in any order, any number of times
- **`vf.Rubric`** with a custom async reward function
- Final-answer extraction: reward function reads the **last non-empty line** as the prediction

## Adapting to new domains

To add a custom tool, just add a Python function with a docstring and type annotations.
`vf.ToolEnv` extracts the JSON schema automatically:

```python
def lookup_population(city: str) -> str:
    """Return the approximate population of a city.
    Args:
        city: City name.
    Returns:
        Population as a string, e.g. "8.3 million".
    """
    ...  # implement lookup
```

Then add it to the `tools=[...]` list in `load_environment()`.

# Sandbox Code (PythonEnv — isolated code execution)

The model generates Python solutions to programming challenges.  
Code is executed inside a **Prime Sandbox** (isolated container).  
Reward is the fraction of test cases that pass.

## Requirements

```bash
pip install verifiers>=0.1.10
export PRIME_API_KEY=your-key-here
```

## What it does

- **50 coding challenges** from basic (sum a list) to intermediate (edit distance, Sieve of Eratosthenes)
- Model outputs a Python `solve()` function
- Sandbox runs the test harness and reports `PASS_RATE:n/m`
- Reward = `n / m` (fraction of tests passed)

## Setup & Quick eval

```python
from prime_cookbook.recipes.sandbox_code.sandbox_code import load_environment
import verifiers as vf

env = load_environment(num_examples=10)
vf.evaluate(env, model="gpt-4.1-mini", rollouts_per_example=2)
```

## Training run

```bash
prime rl run config.toml
```

## Expected metrics

| Model | Starting reward | After 100 steps |
|-------|----------------|-----------------|
| GPT-4.1-mini | ~0.85 | ~0.97 |
| Llama-3.2-3B | ~0.35 | ~0.65 |
| Llama-3.2-1B | ~0.15 | ~0.40 |

## Key patterns

### PythonEnv lifecycle
```python
env = vf.PythonEnv(
    dataset=dataset,
    system_prompt=SYSTEM_PROMPT,
    rubric=rubric,
)
```
`PythonEnv` manages the sandbox lifecycle:
- Creates an isolated Python REPL container per rollout
- Executes code in the container
- Cleans up after each rollout

### Code extraction (no regex)
The `_extract_code()` helper looks for:
1. ` ```python ... ``` ` fenced blocks
2. Plain ` ``` ` blocks
3. First line starting with `def `

### Test harness pattern
```python
def _build_test_harness(code, test_cases):
    # Embeds user code + test runner into a single script
    # Outputs: "PASS_RATE:3/5"
```

## Challenge categories

| Category | Count | Examples |
|----------|-------|---------|
| Basic list ops | 10 | sum, max, reverse, count |
| String manipulation | 8 | palindrome, vowels, capitalize |
| Math & number theory | 8 | factorial, Fibonacci, prime, GCD |
| Data structures | 8 | dict ops, group-by, chunk |
| Algorithms | 8 | binary search, merge sort, BFS islands |
| Dynamic programming | 8 | coin change, edit distance, Pascal's triangle |

## Adapting to new domains

Replace `_CHALLENGES` with domain-specific coding tasks.  
Test cases are the key — each problem needs deterministic, verifiable outputs.

For **harder** problems: increase `max_tokens` to 2048+ for chain-of-thought code generation.

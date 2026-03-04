# Sandbox Code

Code generation + execution environment. The model writes Python code inside a sandboxed container, executes it, observes output, and iterates until the task is solved. All execution happens in isolated Prime Sandboxes — no host system access.

**Environment type:** `PythonEnv`  
**Reward:** Test case pass rate (0.0–1.0 based on fraction of tests passing)  
**Dataset:** 1,500 programming problems (LeetCode-style + data analysis tasks)

---

## Quick Start

```bash
prime env install prime_cookbook/recipes/sandbox_code

prime eval run recipe-sandbox-code --model gpt-4.1-mini
# Expected: reward mean ~0.55

prime rl run prime_cookbook/recipes/sandbox_code/config.toml
```

---

## Environment Overview

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import code_reward

class SandboxCodeEnv(vf.PythonEnv):
    """
    Model writes Python code, executes it in sandbox,
    sees output, iterates. Rewarded on test case pass rate.
    """

    docker_image = "python:3.11-slim"
    cpu_cores = 1
    memory_gb = 2
    timeout_minutes = 5

    system_prompt = """You are an expert Python programmer.

Solve the given programming task by writing and executing Python code.
Use the `python` tool to run code blocks. You can run code multiple times
and observe output before submitting your final solution.

When you have a working solution, state "DONE" and include the final code."""

    def get_rubric(self):
        def reward(completion: str, state: dict, **kwargs) -> float:
            test_cases = state["info"]["tests"]
            sandbox = state.get("sandbox")

            # Run final code against test cases in sandbox
            return code_reward(
                completion,
                test_cases=test_cases,
                sandbox=sandbox,
                timeout=10,
            )

        return vf.Rubric(funcs=[reward])
```

---

## Dataset Format

Each example needs a problem description and test cases:

```python
{
    "question": "Write a function `add(a, b)` that returns the sum of two numbers.",
    "tests": [
        {"input": "add(1, 2)", "expected": "3"},
        {"input": "add(-1, 1)", "expected": "0"},
        {"input": "add(0, 0)", "expected": "0"},
    ],
    "difficulty": "easy",  # easy | medium | hard
    "tags": ["math", "functions"],
}
```

For more complex problems, tests can include full setup code:

```python
{
    "question": "Implement a function `two_sum(nums, target)` that returns indices of the two numbers that add up to target.",
    "tests": [
        {
            "setup": "nums = [2, 7, 11, 15]; target = 9",
            "input": "two_sum(nums, target)",
            "expected": "[0, 1]",
        },
        {
            "setup": "nums = [3, 2, 4]; target = 6",
            "input": "two_sum(nums, target)",
            "expected": "[1, 2]",
        },
    ],
}
```

---

## Reward Design

```python
from prime_cookbook.skills.verifiers import code_reward

def test_pass_reward(completion: str, state: dict, **kwargs) -> float:
    """Fraction of test cases passing."""
    tests = state["info"]["tests"]
    sandbox = state.get("sandbox")
    return code_reward(completion, test_cases=tests, sandbox=sandbox, timeout=10)

def solution_present_reward(completion: str, state: dict, **kwargs) -> float:
    """Small reward for submitting any runnable code (prevents giving up early)."""
    has_code = "def " in completion or "class " in completion
    return 0.1 if has_code else 0.0

rubric = vf.Rubric(
    funcs=[test_pass_reward, solution_present_reward],
    weights=[0.9, 0.1],
    combine="sum",
)
```

---

## Expected Metrics

| Model | Reward Mean | All Tests Pass | 0 Tests Pass |
|-------|-------------|---------------|-------------|
| gpt-4.1-mini | ~0.55 | ~48% | ~12% |
| Qwen2.5-Coder-7B | ~0.62 | ~54% | ~8% |
| Qwen2.5-7B-Instruct | ~0.41 | ~32% | ~18% |
| Qwen2.5-Coder-1.5B | ~0.28 | ~20% | ~25% |

**Recommendation:** Use `Qwen2.5-Coder-*` variants — they significantly outperform general instruction-tuned models on code tasks.

---

## Training Config

```toml
[model]
name = "Qwen/Qwen2.5-Coder-7B-Instruct"

[training]
max_steps = 2000
batch_size = 64
rollouts_per_example = 4   # sandboxes are slow (~3-5s startup), keep low

[sampling]
max_tokens = 4096           # code can be long + multiple iterations
temperature = 0.8

[[env]]
id = "recipe-sandbox-code"
weight = 1.0
```

**Performance note:** Sandbox startup adds ~3–5s latency per rollout. Keep `rollouts_per_example` at 4 to avoid long batch times.

---

## Difficulty Levels

The recipe ships three env IDs for curriculum training:

```bash
# Level 1: single-function tasks, 2-3 test cases
prime eval run recipe-sandbox-code-easy

# Level 2: multi-function, edge cases, 5-10 test cases
prime eval run recipe-sandbox-code-medium

# Level 3: algorithms, data structures, 10+ test cases
prime eval run recipe-sandbox-code-hard
```

Curriculum config:
```toml
[[env]]
id = "recipe-sandbox-code-easy"
weight = 3.0

[[env]]
id = "recipe-sandbox-code-medium"
weight = 2.0

[[env]]
id = "recipe-sandbox-code-hard"
weight = 1.0
```

---

## Security Notes

- **Always use `PythonEnv` or `SandboxEnv`** — never execute model-generated code on the host
- Sandboxes are isolated containers with no internet access by default
- Each rollout gets a fresh container (no state leakage between rollouts)
- Set `memory_gb` and `timeout_minutes` to prevent resource exhaustion
- The `code_reward` function only runs code inside the sandbox's Python REPL

---

## Extension Ideas

**Data analysis tasks:**
```python
# Provide a CSV and ask the model to analyze it
problem = {
    "question": "Load sales.csv and compute monthly revenue totals.",
    "setup_files": {"sales.csv": csv_content},
    "tests": [{"input": "result['2024-01']", "expected": "45230.50"}],
}
```

**Library installation in sandbox:**
```python
class DataScienceEnv(vf.PythonEnv):
    docker_image = "python:3.11-slim"

    async def setup_sandbox(self, sandbox):
        await sandbox.exec("pip install pandas numpy matplotlib -q")
        await super().setup_sandbox(sandbox)
```

---

## Related

- [Environment Types](../environment-types.md) — PythonEnv, SandboxEnv
- [Verifier Skills](../verifiers-skills.md) — `code_reward`
- [Training Config](../training-config.md)

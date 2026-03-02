# Environment Types

Six base classes, each a superset of the previous. Pick the simplest one that covers your task.

```
MultiTurnEnv (base)
├── SingleTurnEnv          — one response, no tools
└── ToolEnv                — stateless tool calling
    ├── MCPEnv             — MCP server integration
    └── StatefulToolEnv    — per-rollout state
        ├── SandboxEnv     — containerized bash
        │   └── PythonEnv  — persistent Python REPL
        └── CliAgentEnv    — custom agent code in sandbox
```

`EnvGroup` is orthogonal — wraps any number of envs for multi-task training.

---

## SingleTurnEnv

One prompt → one response → score. No tool calls, no follow-up turns.

**When to use:** Classification, math reasoning, closed-book Q&A, instruction following.

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import math_reward

dataset = vf.load_dataset("openai/gsm8k", split="train")

rubric = vf.Rubric(funcs=[math_reward])

env = vf.SingleTurnEnv(
    dataset=dataset,
    rubric=rubric,
    system_prompt="Solve the math problem. Put your final answer in \\boxed{}.",
)
```

**Key parameters:**
- `dataset` — HuggingFace dataset with `prompt`/`question` column + optional `answer`
- `rubric` — `vf.Rubric` or `vf.JudgeRubric`
- `system_prompt` — optional string or callable

---

## ToolEnv

Model can call any number of tools per turn. Tools are Python functions; schema is auto-extracted.

**When to use:** Web search, calculator, API calls, any stateless tool use.

```python
import verifiers as vf
from prime_cookbook.skills.verifiers import exact_match_reward

def search(query: str) -> str:
    """Search the web and return a summary."""
    return web_search(query)  # your implementation

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

env = vf.ToolEnv(
    dataset=dataset,
    rubric=vf.Rubric(funcs=[exact_match_reward]),
    tools=[search, calculator],
    max_turns=10,
)
```

**Key parameters:**
- `tools` — list of Python functions (sync or async)
- `max_turns` — hard stop on conversation length
- `stop_errors` — list of exception types that terminate rollout vs return as tool response

**Tracked metrics:** `total_tool_calls`, `<tool_name>_calls` per tool.

---

## StatefulToolEnv

Like `ToolEnv` but with per-rollout state. Hidden args (not in model's schema) can be injected at call time.

**When to use:** Search with a shared index, database queries, anything that needs per-episode context.

```python
import verifiers as vf

class DocSearchEnv(vf.StatefulToolEnv):
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def setup_state(self, state):
        state["query_count"] = 0
        return state

    def search(self, query: str, _state: dict) -> str:
        """Search the document index."""
        _state["query_count"] += 1
        return self.index.search(query)

    def get_tools(self):
        return [self.search]
```

**Key parameters:**
- `args_to_skip` — list of arg names hidden from model's tool schema (e.g., `["_state"]`)
- `update_tool_args()` — override to inject state into every tool call at runtime

---

## SandboxEnv

Containerized bash shell via [Prime Sandboxes](https://docs.primeintellect.ai/sandboxes). Each rollout gets a fresh container.

**When to use:** Bash tasks, file manipulation, multi-step shell workflows.

```python
import verifiers as vf

env = vf.SandboxEnv(
    dataset=dataset,
    rubric=rubric,
    docker_image="ubuntu:22.04",
    cpu_cores=1,
    memory_gb=2,
    timeout_minutes=5,
    labels={"recipe": "sandbox-code"},
)
```

**Key parameters:**
- `docker_image` — container image
- `cpu_cores`, `memory_gb`, `disk_size_gb` — resource limits
- `timeout_minutes` — per-rollout wall-clock timeout
- `environment_vars` — dict of env vars injected into container

**Tracked metrics:** `sandbox_ready_wait_time`, `sandbox_command_execution_time`.

---

## PythonEnv

Persistent Python REPL inside a sandbox. State (variables, imports) persists across turns.

**When to use:** Code generation + execution, data analysis, multi-step computation.

```python
import verifiers as vf

env = vf.PythonEnv(
    dataset=dataset,
    rubric=rubric,
    system_prompt=(
        "You are a Python programmer. Write and execute code to solve the task. "
        "Use the `python` tool to run code blocks."
    ),
    docker_image="python:3.11-slim",
    memory_gb=4,
    timeout_minutes=10,
)
```

**Tracked metrics:** Inherits `SandboxEnv` metrics + `python_ready_wait_time`.

**Note:** The Python REPL state persists within a rollout but resets between rollouts. Do not rely on global state for correctness checking.

---

## CliAgentEnv

Runs arbitrary agent code inside a sandbox. Prime intercepts API requests the agent code makes and replays them through the RL loop.

**When to use:** Computer use, browser agents, complex multi-step agents that need their own loop.

```python
import verifiers as vf

class BrowserAgentEnv(vf.CliAgentEnv):
    agent_script = "prime_cookbook/recipes/browser_agent/agent.py"

    sandbox_config = {
        "docker_image": "my-browser-env:latest",
        "cpu_cores": 2,
        "memory_gb": 4,
        "timeout_minutes": 15,
    }

    def evaluate(self, state) -> float:
        # Check if task was completed
        return check_task_completed(state)
```

**RolloutGatewayMixin** — opt-in for server-side rollout routing (recommended for production):
```python
class MyEnv(vf.RolloutGatewayMixin, vf.CliAgentEnv):
    use_gateway = True
```

---

## EnvGroup

Combines multiple environments for multi-task training. Rollouts are sampled proportionally by weight.

**When to use:** Training on heterogeneous tasks simultaneously, curriculum across difficulty levels.

```python
import verifiers as vf

math_env = vf.SingleTurnEnv(...)
search_env = DocSearchEnv(...)
code_env = vf.PythonEnv(...)

env = vf.EnvGroup(
    envs=[math_env, search_env, code_env],
    weights=[1.0, 2.0, 1.5],  # proportional sampling
)
```

**Key parameters:**
- `envs` — list of environment instances
- `weights` — sampling weights (normalized internally)

See [multi-env recipe](recipes/multi-env.md) for a complete example.

---

## Choosing an Environment

| Task | Use |
|------|-----|
| Math / QA / classification | `SingleTurnEnv` |
| Tool-augmented reasoning | `ToolEnv` |
| Search with shared index | `StatefulToolEnv` |
| Shell / file tasks | `SandboxEnv` |
| Code gen + execution | `PythonEnv` |
| Full agent in container | `CliAgentEnv` |
| Multiple tasks | `EnvGroup` |

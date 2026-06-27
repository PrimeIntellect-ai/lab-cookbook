# Coding Agents and Sandboxes

Code-oriented v1 environments use the same taskset/harness split. Put verification and scoring on the taskset; choose a harness and runtime for how the agent acts.

## Tool-backed Python

`math-python` exposes a persistent Python execution tool. The server owns the live interpreter process for one rollout, while `PythonState` records serializable counters and history on `trace.state`:

```python
class PythonToolset(vf.Toolset[PythonToolConfig, PythonState]):
    TOOL_PREFIX = "python"

    @vf.tool
    async def execute(self, code: str) -> str:
        ...
```

The eval config allows multiple turns so the model can call the tool and then answer:

```toml
num_tasks = 5
num_rollouts = 2
max_turns = 4

[taskset]
id = "math-python"
num_tasks = 50
```

## Harbor Tasks

Harbor tasks are data plus tests. The taskset stages tests and scores inside the rollout runtime; the harness only controls the agent loop.

```toml
[taskset]
id = "opencode-harbor"
tasks = ["regex-log"]
ignore_dockerfile = true

[harness]
id = "opencode-harbor"
runtime = { type = "docker" }
```

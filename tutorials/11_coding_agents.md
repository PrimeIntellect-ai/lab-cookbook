# Coding Agent Environments

Code-executing environments raise the stakes: the model's output *runs*. In this tutorial you will see the two main shapes — a math taskset with a persistent Python interpreter tool, and a **Harbor** task suite driven by a real coding-agent harness in Docker — and learn how runtimes decide where all of this executes.

The taskset/harness split carries the load here: verification and scoring stay on the taskset; the harness and its runtime decide how and where the agent acts.

**You need:** [Build Your First Environment](5_build_first_environment.md) and [Tool Use and Search](9_tools.md); Docker running locally for the Harbor half.

## Where code runs: runtimes

A quick map before the examples. Rollouts execute in a **runtime**, selected by config:

- `subprocess` — local Python subprocesses. Debugging only: rollouts can leak side effects into each other (one process editing a harness config file affects the rest).
- `docker` — containers on your machine. Real isolation, locally.
- Sandbox runtimes (`prime`, `modal`) — remote, for production: training and high-concurrency eval.

The same environment moves between them by config alone — develop against `subprocess` or `docker`, scale on `prime`.

## Tool-backed execution: `math-python`

`math-python` (from `environments/math_python/math_python.py`) poses competition math problems and exposes one tool: a **persistent** Python interpreter, so variables survive across calls within a rollout.

```python
class PythonState(vf.State):
    executions: int = 0
    restarts: int = 0
    last_error: str = ""
    history: list[str] = Field(default_factory=list)


class PythonToolset(vf.Toolset[PythonToolConfig, PythonState]):
    TOOL_PREFIX = "python"

    async def setup_task(self, task: MathPythonTask) -> None:
        await self._start_worker()          # one live interpreter per rollout

    @vf.tool
    async def execute(self, code: str) -> str:
        """Execute Python code in the rollout's persistent interpreter."""
        ...
```

Two details are the lesson:

- **The state split from [Tool Use and Search](9_tools.md) in action.** The live interpreter process is unserializable, so it lives on `self` and is created in `setup_task` — per rollout. The *serializable* facts about it (`executions`, `restarts`, `history`, `last_error`) live in `self.state` and surface on `trace.state`, where rewards and your debugging can read them.
- **Tools must fail politely.** On timeout or a dead worker, `execute` restarts the interpreter and returns an error *string* to the model — the rollout continues and the model can retry. Raising would instead abort the rollout as an infrastructure error. Return errors the model can act on; raise only when the environment itself is broken ([Designing Rewards](7_rewards.md) draws the same line for scoring code).

Scoring is deterministic — `math_verify` compares the model's `\boxed{}` answer to ground truth. Tools change the *how*, never the *what counts*. The config allows the tool-then-answer loop some room (`configs/10/math-python-eval.toml`):

```toml
num_tasks = 5
num_rollouts = 2
max_turns = 4                   # call the tool, read output, answer

[taskset]
id = "math-python"
num_tasks = 50

[taskset.tools]
timeout_seconds = 60
```

```bash
prime eval run @ configs/10/math-python-eval.toml
```

In the traces, `tool_messages` shows the computations; `trace.state.executions` tells you at a glance whether the model actually used the interpreter or answered from memory.

## Harbor tasks: `opencode-harbor`

[Harbor](https://www.harborframework.com) tasks are **data plus tests**: a task directory with instructions, an optional container image, and a verifier script. verifiers ships built-in support, so a Harbor-based taskset is a few lines:

```python
import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset

class OpenCodeHarborConfig(HarborConfig):
    dataset: str = "hello-world"
    ignore_dockerfile: bool = True      # skip per-task image builds

class OpenCodeHarborTaskset(HarborTaskset, vf.Taskset[HarborTask, OpenCodeHarborConfig]):
    pass
```

No `load_tasks`, no `@vf.reward`: `HarborTaskset` loads the task directories and inherits scoring from each task's own verifier — the taskset stages the tests and scores *inside the rollout runtime*, where the agent's edits actually happened. Your subclass only pins the dataset and adjusts knobs. Two useful ones for tasks with tight limits: `timeout_multiplier` (scales agent and verifier timeouts) and `resource_multiplier` (scales CPU/memory/disk).

The interesting half is the harness. The `opencode_harbor` package bundles a custom `vf.Harness` that installs the OpenCode coding agent into the task container, points it at the interception endpoint (never directly at a provider — that would bypass trace capture), and runs it against the task prompt. Custom harness authoring is its own topic ([Guide 12](../guides/12-custom-harnesses/README.md)); here you only select it:

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 1
num_rollouts = 1

[sampling]
max_tokens = 4096

[taskset]
id = "opencode-harbor"
tasks = ["regex-log"]           # one task from environments/opencode_harbor/tasks/
ignore_dockerfile = true

[harness]
id = "opencode-harbor"
runtime = { type = "docker" }   # needs Docker running locally
```

```bash
prime eval run @ configs/10/opencode-harbor.toml
```

Watch the division of labor in the trace: the harness produced an agent transcript (shell commands, file edits), and the taskset's inherited verifier produced the reward. Neither reached into the other.

When a Harbor task lacks a usable image, build and publish one with `prime images push` ([docs](https://docs.primeintellect.ai/sandboxes/images)) and set it as the task's `image` — the build happens in the cloud, no local Docker needed.

## Try it

- Run `math-python` with `--taskset.tools.timeout_seconds 5` and find a trace where the interpreter restarted (`trace.state.restarts > 0`) — then read how the model handled the error message.
- Browse `environments/opencode_harbor/tasks/regex-log/` to see what a Harbor task directory contains, then point `tasks = [...]` at a different bundled task.
- Re-run the Harbor eval with `--harness.disabled_tools '["bash"]'`-style restrictions (harness-dependent) and observe how the agent adapts — or fails.

## Next

You've completed the Ramping up series. The [recipes](README.md#recipes) put these pieces to work on real use cases — [Compare Harnesses](12_compare_harnesses.md) and [Search Agent](17_search_agent.md) are natural continuations of this tutorial.

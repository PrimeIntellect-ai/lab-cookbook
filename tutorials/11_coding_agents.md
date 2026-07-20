# Coding Agent Environments

Code-executing environments raise the stakes: the model's output *runs*. This tutorial uses two v1 cookbook packages: a math taskset with a persistent Python interpreter tool, and a **Harbor** suite driven by a coding-agent harness in Docker.

The taskset/harness split carries the load here: verification and scoring stay on the taskset; the harness and its runtime decide how and where the agent acts.

**You need:** [Build Your First Environment](5_build_first_environment.md) and [Tool Use and Search](10_tools.md); Docker running locally for the Harbor half.

## Where code runs: runtimes

A quick map before the examples. Rollouts execute in a **runtime**, selected by config:

- `subprocess` — local Python subprocesses. Debugging only: rollouts can leak side effects into each other (one process editing a harness config file affects the rest).
- `docker` — containers on your machine. Real isolation, locally.
- Sandbox runtimes (`prime`, `modal`) — remote, for production: training and high-concurrency eval.

The same environment moves between them by config alone — develop against `subprocess` or `docker`, scale on `prime`.

## Tool-backed execution: `math-python`

`math-python` poses competition math problems and exposes one **persistent** Python interpreter, so variables survive across calls within a rollout. A clean v1 port separates row data, behavior, and the tool server:

```python
class MathPythonData(vf.TaskData):
    answer: str


class PythonState(vf.State):
    executions: int = 0
    restarts: int = 0
    last_error: str = ""
    history: list[str] = Field(default_factory=list)


class PythonToolConfig(vf.ToolsetConfig):
    timeout_seconds: float = 60
    runtime: vf.RuntimeConfig = vf.DockerConfig(
        image="python:3.11-slim",
        workdir="/tmp",
    )


class PythonToolset(vf.Toolset[PythonToolConfig, PythonState]):
    TOOL_PREFIX = "python"

    async def setup_task(self, task: MathPythonData) -> None:
        await self._start_worker()          # one live interpreter per rollout

    @vf.tool
    async def execute(self, code: str) -> str:
        """Execute Python code in the rollout's persistent interpreter."""
        ...


class MathPythonTaskConfig(vf.TaskConfig):
    tools: PythonToolConfig = PythonToolConfig()


class MathPythonTask(vf.Task[MathPythonData, PythonState, MathPythonTaskConfig]):
    tools = (PythonToolset,)

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(trace.last_reply, self.data.answer)
```

Two details are the lesson:

- **The state split from [Tool Use and Search](10_tools.md) in action.** `setup_task` receives `MathPythonData`. The live interpreter process is unserializable, so it lives on the tool server's `self`; serializable facts live in `self.state` and surface on `trace.state`.
- **Tools must fail politely.** On timeout or a dead worker, `execute` restarts the interpreter and returns an error *string* to the model — the rollout continues and the model can retry. Raising would instead abort the rollout as an infrastructure error. Return errors the model can act on; raise only when the environment itself is broken ([Designing Rewards](7_rewards.md) draws the same line for scoring code).

`Taskset.load()` should build `MathPythonData` and wrap each row in `MathPythonTask(..., self.config.task)`. Scoring then reads ground truth through `self.data` and the completion through `trace.last_reply`. The expected v1 config shape is:

```toml
num_tasks = 5
num_rollouts = 2
max_turns = 4                   # call the tool, read output, answer

[taskset]
id = "math-python"
num_tasks = 50

[taskset.task.tools]
timeout_seconds = 60
runtime = { type = "docker", image = "python:3.11-slim", workdir = "/tmp" }
```

Run the shipped config:

```bash
uv run eval @ configs/11/math-python-eval.toml
```

Typed `vf.ToolMessage` records show computations; `trace.state.executions` tells you whether the model used the interpreter.

## Harbor tasks: `opencode-harbor`

[Harbor](https://www.harborframework.com) tasks are **data plus tests**: a task directory with instructions, a container image, and a verifier script. `verifiers` ships the parser and reward implementation. The cookbook's local taskset loads its bundled task directories:

```python
import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.taskset import HarborConfig, HarborTask, parse_task


class OpenCodeHarborConfig(HarborConfig):
    dataset: str = "bundled"
    require_image: bool = True


class OpenCodeHarborTaskset(vf.Taskset[HarborTask, OpenCodeHarborConfig]):
    def load(self):
        task_dirs = sorted(bundled_tasks_dir().glob("*/task.toml"))
        for idx, task_toml in enumerate(task_dirs):
            yield HarborTask(
                parse_task(task_toml.parent, idx, self.config),
                self.config.task,
            )
```

No custom row model or reward is needed: `parse_task` produces Harbor data, and `HarborTask.solved` stages and runs each task's verifier inside the rollout runtime.

The matching `OpenCodeHarness` installs the coding agent into the task runtime, points it at the interception endpoint, and runs it against `trace.task.data.prompt`. Run one prebuilt task with:

```bash
uv run eval @ configs/11/opencode-harbor.toml
```

The trace keeps the boundary visible: the harness owns the agent transcript and `HarborTask.solved` owns the reward. Neither reaches into the other.

When a Harbor task lacks a usable image, build and publish one before evaluation. [Harbor Tasksets](20_harbor.md) covers `task.toml`, `prime images push`, bulk prebuilds, image overrides, and multimodal prompt handling.

## Try it

- Set `--taskset.task.tools.timeout-seconds 5` on `math-python` and inspect how the model handles a restart.
- Browse `environments/opencode_harbor/tasks/regex-log/` to see what a Harbor task directory contains, then point `tasks = [...]` at a different bundled task.
- Re-run the Harbor eval with `--harness.disabled_tools '["bash"]'`-style restrictions (harness-dependent) and observe how the agent adapts — or fails.

## Next

You've completed the Ramping up series. The [recipes](README.md#recipes) put these pieces to work on real use cases — [Compare Harnesses](12_compare_harnesses.md) and [Search Agent](17_search_agent.md) are natural continuations of this tutorial.

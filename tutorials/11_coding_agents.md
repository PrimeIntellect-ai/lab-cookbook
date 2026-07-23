# Coding Agent Environments

Code-executing environments raise the stakes: the model's output *runs*. This tutorial uses two v1 cookbook packages: a math taskset with a persistent Python interpreter tool, and a **[Harbor](https://www.harborframework.com/docs)** suite driven by the built-in `pi` coding-agent harness in Docker — ending with everything you need to package and run Harbor tasksets of your own.

The taskset/harness split carries the load here: verification and scoring stay on the taskset; the harness and its runtime decide how and where the agent acts.

**You need:** [Build Your First Environment](5_build_first_environment.md) and [Tool Use and Search](10_tools.md); Docker running locally and Python 3.12 or newer for the Harbor half. The cookbook's `opencode-harbor` dependency already requests `verifiers[harbor]`, so sync it with:

```bash
uv sync --python 3.12
```

In a standalone project, add the `verifiers[harbor]` extra explicitly.

## Where code runs: runtimes

A quick map before the examples. Rollouts execute in a **runtime**, selected by config:

- `subprocess` — local Python subprocesses. Debugging only: rollouts can leak side effects into each other (one process editing a harness config file affects the rest).
- `docker` — containers on your machine. Real isolation, locally.
- Sandbox runtimes (`prime`, `modal`) — remote, for production: training and high-concurrency eval.

The same environment moves between them by config alone — develop against `subprocess` or `docker`, scale on `prime`.

## Tool-backed execution: `math-python`

`math-python` poses competition math problems and exposes one **persistent** Python interpreter, so variables survive across calls within a rollout. A clean v1 port separates row data, behavior, and the tool server:

```python
import verifiers.v1 as vf
from pydantic import Field


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
    async def correct_answer(self, trace: vf.Trace) -> float:
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

The agent side needs no custom code: the built-in `pi` harness (`[harness] id = "pi"`) installs the [Pi](https://github.com/earendil-works/pi) coding-agent CLI into the task runtime (x64 and arm64 builds), reaches the interception endpoint through a custom OpenAI-compatible provider, and runs against the task prompt. Run one prebuilt task with:

```bash
uv run eval @ configs/11/harbor-smoke.toml
```

The trace keeps the boundary visible: the harness owns the agent transcript and `HarborTask.solved` owns the reward. Neither reaches into the other.

The rest of this tutorial covers the Harbor format itself: what a task contains, the minimal Python wrapper, prebuilt images, and prompt handling.

## What a Harbor task contains

One task is one directory that looks like this:

```text
regex-log/
├── instruction.md
├── task.toml
├── environment/
│   └── Dockerfile
├── tests/
│   ├── test.sh
│   └── test_outputs.py
└── solution/
    └── solve.sh
```

`instruction.md` becomes the agent prompt. `solution/` is useful for author validation but is not given to the agent. `task.toml` provides the general configuration for a task, including timeouts, resources, metadata, and most importantly, a pullable image:

```toml
version = "1.0"

[verifier]
timeout_sec = 900.0

[agent]
timeout_sec = 900.0

[environment]
docker_image = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox/alexgshaw/regex-log:20251031"
cpus = 1
memory = "2G"
storage = "10G"
```

After the harness edits the live container, `HarborTaskset` copies `tests/` into it and runs `tests/test.sh`. That script must write a numeric score to:

```text
/logs/verifier/reward.txt
```

A binary verifier ends with `echo 1 > /logs/verifier/reward.txt` on success and `echo 0 > /logs/verifier/reward.txt` on failure. A missing, empty, or non-numeric file scores `0`.

## Run the local smoke test

The smallest useful config pins one bundled task (`configs/11/harbor-smoke.toml`):

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 1
num_rollouts = 1

[sampling]
max_tokens = 4096

[taskset]
id = "opencode-harbor"
tasks = ["hello-world"]

[harness]
id = "pi"
runtime = { type = "docker" }
```

Check resolution first, then run one rollout:

```bash
uv run eval @ configs/11/harbor-smoke.toml --dry-run
uv run eval @ configs/11/harbor-smoke.toml
```

Here, “local” describes the importable Python package and bundled task data in `environments/opencode_harbor`; Docker still pulls the public `python:3.11-slim` image declared by `hello-world`.

## The minimal wrapper

For a Harbor package that needs no custom behavior, the Python side is only a typed config and taskset:

```python
import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class MyHarborConfig(HarborConfig):
    dataset: str = "harbor/hello-world"


class MyHarborTaskset(
    HarborTaskset,
    vf.Taskset[HarborTask, MyHarborConfig],
):
    pass
```

`dataset` selects the Harbor data package. It is separate from `[taskset].id`, which selects the Python wrapper:

- **Harbor Hub:** use an `org/name` dataset, optionally pinned as `org/name@ref`.
- **Local or legacy registry:** use a bare dataset name and set `registry_path`; `repo` and `registry_url` cover the other Harbor registry selectors.
- **Environment Hub:** an id such as `org/my-harbor-env@version` can select a published Python taskset instead of a local package.

Filter a downloaded package with `tasks = ["task-name"]`. To relax authored limits without editing task data:

```toml
[taskset]
id = "my-harbor"
timeout_multiplier = 2.0
resource_multiplier = 1.5
```

`timeout_multiplier` scales both agent and verifier timeouts. `resource_multiplier` scales CPU, memory, and disk; it does not scale GPUs.

## Use prebuilt images

`HarborTaskset` does **not** build `environment/Dockerfile` during a rollout. It uses `[environment].docker_image`; a Dockerfile-only task is rejected because silently running it on the harness image would test the wrong environment. `ignore_dockerfile = true` explicitly permits that fallback, but is only correct when the harness image already contains everything the task needs.

Build one task image in the Prime registry:

```bash
prime images push opencode-harbor.x86.regex-log:latest \
  --context environments/opencode_harbor/tasks/regex-log/environment \
  --platform linux/amd64
```

Or discover and build every Dockerfile-only Harbor task in a directory of your own tasks:

```bash
prime images push-bulk \
  --harbor path/to/your/tasks \
  --name-template "my-tasks.x86.{dir}" \
  --tag latest \
  --platform linux/amd64 \
  --dry-run      # lists what would build; drop the flag to build for real
```

`push-bulk --harbor` skips tasks that already declare `docker_image` — every bundled task under `environments/opencode_harbor/tasks` does, so pointing it there reports nothing to build. Both image commands build remotely; the rollout only pulls the resulting image.

If you cannot change upstream `task.toml` files, override the frozen `TaskData.image` while loading:

```python
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset

IMAGE_TEMPLATE = "<registry-ref>/opencode-harbor.x86.{task}:latest"


class MyHarborConfig(HarborConfig):
    dataset: str = "org/dataset"
    ignore_dockerfile: bool = True


class MyHarborTaskset(
    HarborTaskset,
    vf.Taskset[HarborTask, MyHarborConfig],
):
    def load(self) -> list[HarborTask]:
        return [
            HarborTask(
                task.data.model_copy(
                    update={
                        "image": IMAGE_TEMPLATE.format(
                            task=Path(task.data.task_dir).name
                        )
                    }
                ),
                task.config,
            )
            for task in super().load()
        ]
```

Replace `<registry-ref>` with the prefix printed by `prime images push`. The copied `TaskData` points each task at its prebuilt image, so no runtime build is needed.

## Prompt compatibility

Harbor's loader reads `instruction.md` as a plain Python `str`, which every CLI-agent harness accepts. Images cannot be piped through a command-line prompt as plain text. Attach them to the task as typed `vf.Messages` while loading:

```python
class ImageHarborTaskset(HarborTaskset, vf.Taskset[HarborTask, MyHarborConfig]):
    def load(self) -> list[HarborTask]:
        return [
            HarborTask(
                task.data.model_copy(
                    update={
                        "prompt": [
                            vf.UserMessage(
                                content=[
                                    vf.TextContentPart(text=task.data.prompt_text),
                                    vf.ImageUrlContentPart(
                                        image_url=vf.ImageUrlSource(url=image_url(task))
                                    ),
                                ]
                            )
                        ]
                    }
                ),
                task.config,
            )
            for task in super().load()
        ]
```

The harness must then declare `SUPPORTS_MESSAGE_PROMPT = True`; the built-in `pi`, `default`, and `codex` harnesses do. A harness that can't render rich prompts should instead require a string prompt, so message-list prompts fail fast rather than being flattened silently.

## Current parity gaps

`HarborTaskset` does not yet implement every Harbor feature. Known gaps include sandbox `no-network` policy, shared versus separate verifier environments, and multi-step tasks. Check the [Harbor documentation](https://www.harborframework.com/docs) before adopting those features, and validate both a known-good solution and a no-op task before trusting a new reward.

## Try it

- Set `--taskset.task.tools.timeout-seconds 5` on `math-python` and inspect how the model handles a restart.
- Browse `environments/opencode_harbor/tasks/regex-log/` to see what a Harbor task directory contains, then point `tasks = [...]` at a different bundled task.
- Re-run the Harbor eval with `--harness.disabled-tools bash`-style restrictions (space-separated tool names; which names exist is harness-dependent) and observe how the agent adapts — or fails.

## Next

→ [Best Practices](12_best_practices.md) closes the Ramping up series with the authoring checklist. Then the [recipes](../recipes/README.md) put these pieces to work on real use cases — [Build Your Own Coding-Agent Harness](../recipes/coding_agent_harness.md) and [Search Agent](../recipes/search_agent.md) are natural continuations of this tutorial.

# Golden Path

This is the v1 authoring contract used by the cookbook.

## Package Contract

A v1 environment package exports one `vf.Taskset` subclass. It may also export one bundled `vf.Harness` subclass when the package owns a custom agent loop.

```python
import verifiers.v1 as vf


class MyTask(vf.Task):
    answer: str


class MyConfig(vf.TasksetConfig):
    num_tasks: int = 100


class MyTaskset(vf.Taskset[MyTask, MyConfig]):
    def load_tasks(self) -> list[MyTask]:
        ...

    @vf.reward(weight=1.0)
    async def correct(self, task: MyTask, trace: vf.Trace) -> float:
        ...


__all__ = ["MyTaskset"]
```

The taskset id selects the package. Loader resolution imports the package/module named by that id and finds the exported taskset class. When the same module exports a harness, that taskset id is also the default harness id.

## Tasks

Subclass `vf.Task` for row-specific data. Required base fields include `idx` and `prompt`; set `prompt=None` only when a user simulator opens the conversation. Use `system_prompt`, `image`, `workdir`, `timeout`, and `resources` when the task needs them.

`load_tasks(self) -> list[TaskT]` runs once at environment load. Use it for dataset loading, filtering, and constructing typed tasks.

## Config

Subclass `vf.TasksetConfig` for taskset knobs. Nested configs are normal Pydantic config fields:

```python
class SearchConfig(vf.TasksetConfig):
    max_examples: int | None = None
    tools: SearchToolConfig = SearchToolConfig()
```

TOML mirrors that shape:

```toml
[taskset]
id = "wiki-search"
max_examples = 250

[taskset.tools]
shared = true
```

## Scoring

Use `@vf.reward`, `@vf.metric`, `@vf.group_reward`, and `@vf.stop`. The framework injects arguments by name. Individual rewards and metrics may request `task`, `trace`, and `runtime`; group rewards request `task` and `traces`.

Read finished rollouts from `vf.Trace`:

- `trace.task`: typed task.
- `trace.assistant_messages`: model messages in order.
- `trace.tool_messages`: tool results.
- `trace.state`: typed mutable rollout state.
- `trace.info`: persisted metadata/artifacts.
- `trace.num_turns`, `trace.stop_condition`, `trace.error`: lifecycle status.

## Tools

Tools are `vf.Toolset` servers. Use `setup` for expensive task-agnostic resources, `setup_task` for per-task inputs, and `self.state` for serializable mutable rollout state:

```python
class SearchState(vf.State):
    queries: int = 0


class SearchToolset(vf.Toolset[SearchToolConfig, SearchState]):
    TOOL_PREFIX = "search"

    async def setup_task(self, task: MyTask) -> None:
        self.answer = task.answer

    @vf.tool
    async def query(self, text: str) -> list[str]:
        self.state.queries += 1
        ...

if __name__ == "__main__":
    SearchToolset.run()
```

A taskset exposes tools with `tools(self, task) -> list[vf.Toolset]`. Use `shared`, `colocated`, `fork`, `runtime`, or `url` on the tool config to control placement.

Do not store mutable per-rollout state on `self` for shared servers. Put serializable rollout state in `self.state`; use a per-rollout server or `shared + fork` only for process-local objects that cannot be serialized.

## User Simulators

User simulators are `vf.User` servers with `respond(self, message: str) -> vf.Messages`. They are wired through `Taskset.user(self, task)`. Use a typed `vf.State` subclass when the simulator and rewards share rollout state.

## Harnesses

A harness is a `vf.Harness` subclass that implements `launch(ctx, trace, runtime, endpoint, secret, mcp_urls) -> vf.ProgramResult`. Use a custom harness only for custom agent loops. Most tasksets should run with built-in harnesses.

## Eval Config

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 10
num_rollouts = 2
max_turns = 6

[sampling]
max_tokens = 1024

[taskset]
id = "my-taskset"

[harness]
id = "default"
```

Run with `uv run eval @ config.toml` or `uv run eval my-taskset -n 10 -r 2`.

## Training Config

```toml
[[orchestrator.train.env]]
name = "my-taskset"
max_turns = 6
taskset = { id = "my-taskset" }
harness = { id = "default", runtime = { type = "subprocess" } }
```

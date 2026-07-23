# Best Practices

You've built tasksets, judges, rewards, toolsets, user simulators, and harnesses. This closing tutorial is the checklist that keeps them clean — where each kind of logic belongs, and the failure modes that account for most broken environments. Skim it now; return to it every time you're about to publish an environment.

**You need:** the Ramping-up series ([5](5_build_first_environment.md)–[11](11_coding_agents.md)); this tutorial consolidates its design rules in one place.

## Public Surface

Use `import verifiers.v1 as vf` in v1 code. Expose exactly one taskset class through `__all__` unless the package also intentionally exports a bundled harness.

Config is the public API. Put runner-tunable fields on `vf.TasksetConfig`, `vf.ToolsetConfig`, `vf.UserConfig`, or `vf.HarnessConfig`. Keep constants, templates, and fixed rubric text in code or package data.

## Tasks

Return typed `vf.Task` subclasses from `load(self) -> list[TaskT]`. Do dataset loading, filtering, and slicing there. Use config fields like `dataset_split`, `num_tasks`, or `difficulty` when the runner should control the task list — and pin external data sources (dataset revisions) in config so runs stay reproducible.

## Scoring

Rewards and metrics are async decorated methods. Name only the injected arguments you need: `task`, `trace`, and optionally `runtime`.

Read model output from `trace.assistant_messages`, tool output from `trace.tool_messages`, persisted data from `trace.info`, and mutable rollout state from `trace.state`.

A reward of 0.0 must mean "the model failed," never "my scoring code broke" — raise for broken environments, reply with text for recoverable model mistakes ([Designing Rewards](7_rewards.md)).

## Tools and Users

Tools and user simulators are servers. Put expensive task-agnostic resources in `setup`, per-task inputs in `setup_task`, and serializable mutable rollout state in `self.state`. Put expensive read-only setup behind `shared = true`; use colocated servers when the server must share the harness runtime workspace.

## Harnesses

Most tasksets should run with built-in harnesses. Write a custom `vf.Harness` only when the agent loop itself is different ([Build Your Own Coding-Agent Harness](../recipes/coding_agent_harness.md)). The harness drives the rollout; the taskset still owns task data and scoring.

## Runtime Placement

Choose runtime at the harness or tool/user config boundary. Use task `image`, `resources`, and `timeout` when requirements vary per row. Use `Taskset.NEEDS_CONTAINER = True` when subprocess execution is nonsensical for the entire taskset.

## Failure Mode Checklist

- `__all__` does not expose exactly one taskset class.
- A reward reads a dict-shaped completion instead of `Trace` fields.
- Tool-using tasksets stop after one turn before the model can answer after tool results.
- Config sets fields that are not on the typed config class.
- A shared tool server stores mutable rollout data on `self` instead of `self.state`.
- A tool config carries per-task data that should come from `setup_task`.
- A coding task scores in the harness instead of taskset `finalize` or reward.

## Next

That's the full toolbox. The [recipes](../recipes/README.md) put it to work on complete use cases — and when you have an environment of your own to publish, run this page's checklist against it first.

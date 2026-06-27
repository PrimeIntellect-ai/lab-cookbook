# Review v1 Environments

Use this skill to review Verifiers v1 environment packages.

## Contract

A v1 package imports `verifiers.v1 as vf` and exports a `vf.Taskset` subclass through `__all__`. There is no environment loader in the v1 authoring path. The taskset id selects the importable package/module; the loader finds the exported taskset class.

Expected pieces:

- `vf.Task`: typed row data.
- `vf.TasksetConfig`: runner-tunable taskset knobs.
- `vf.Taskset[TaskT, ConfigT, StateT]`: tasks, tools, user simulator, rewards, metrics, and task lifecycle hooks.
- `vf.Toolset`: MCP tool server with `@vf.tool` methods and a runnable module guard.
- `vf.User`: framework-driven user simulator with `respond`.
- `vf.Harness`: custom rollout program, only when built-ins are not enough.
- `vf.Trace`: the source of truth for scoring.

## Review Steps

1. Confirm every v1 module imports `verifiers.v1 as vf`.
2. Confirm `__all__` exports exactly one taskset class unless a bundled harness is intentionally exported too.
3. Confirm `load_tasks(self) -> list[TaskT]` returns typed task objects and uses `self.config` for runner-tunable choices.
4. Trace one rollout: prompt, tool/user interactions, stop condition, finalize, rewards, metrics.
5. Confirm rewards read `trace.assistant_messages`, `trace.tool_messages`, `trace.state`, or `trace.info`, not framework-internal dict shapes.
6. Confirm tool/user servers have `if __name__ == "__main__": Server.run()` and that placement config matches state-sharing needs.
7. Confirm tool-using tasksets do not stop before the model can answer after tool results.
8. Confirm config TOMLs set only real typed config fields.
9. Run import/export tests and at least a small model-free smoke where possible.

## Findings to Prioritize

- Loader or import contract violations.
- Rewards that cannot see the data they score.
- Tool/user state isolation bugs, especially with `shared = true`.
- Harness/taskset responsibility confusion.
- Config fields that no longer exist.
- Behavior narrowed during migration without a note in `reference/v1-authoring-gaps.md`.

# mini-loop

A deliberately minimal CLI-agent harness, built from scratch in the [Build Your Own Coding-Agent Harness](../../recipes/coding_agent_harness.md) recipe.

The agent program (`mini_loop/program.py`) is a ~60-line uv script: one chat completion per step, the model answers with a single ```bash block, the program runs it in the task container and feeds exit code + output back. The harness (`mini_loop/harness.py`) prepares the script in the rollout runtime and launches it against the interception endpoint.

Select it on any string-prompt taskset with `[harness] id = "mini-loop"`.

## Config knobs

- `max_steps` (default 20) — model turns before the agent gives up.
- `command_timeout_seconds` (default 120) — per-command wall-clock budget.

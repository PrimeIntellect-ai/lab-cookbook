# Coding Agents and Sandboxes

Train and evaluate coding agents in isolated runtimes.

Sandboxed execution environments let the model inspect and write code, run commands safely in a container, collect logs, and score edits programmatically and scalably.

This guide considers two examples of coding agent environments:
- [prime/math-python](https://app.primeintellect.ai/dashboard/environments/prime/math-python) -- a lightweight Python REPL environment for solving math problems with code
- [prime/opencode-harbor](https://app.primeintellect.ai/dashboard/environments/prime/opencode-harbor) -- a full CLI agent environment that runs OpenCode on Harbor tasks

## Sandboxes as Tools

`math-python` asks math questions that are easier to solve with code than by mental arithmetic. The model gets a Python tool backed by a sandbox, uses it for calculations, and returns a final boxed answer.

Run a small eval:

```bash
prime eval run prime/math-python \
  -m openai/gpt-5.4-nano \
  -n 5 \
  -r 2 \
  -t 1024
```

Or run with a config file:

```toml
# [configs/09/math-python-eval.toml](../../configs/09/math-python-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/math-python"
num_examples = 5
rollouts_per_example = 2

[eval.sampling]
max_tokens = 1024
```

```bash
prime eval run configs/09/math-python-eval.toml
```

This is the smallest useful sandbox pattern: one task, one Python toolset, one isolated runtime, and a deterministic reward. The Python tool starts an IPython kernel inside a plain `python:3.11-slim` sandbox and installs the packages it needs during sandbox setup, so the example shows exactly what is available at execution time.

In the implementation, `MathPythonTaskset` owns the dataset, prompt, reward, and Python tool methods. `MathPythonTaskset.load_toolsets(config)` exposes those tools through a `vf.Toolset` with the sandbox image and package install command attached to that toolset.

When a toolset needs a sandbox, keep the sandbox contract on the `vf.Toolset` that exposes the tools. Use an owned `vf.SandboxConfig(...)` for standalone tasksets like `math-python`. If the toolset should share a CLI/program sandbox when one exists, use `vf.SandboxConfig(prefer="program", ...)`; use `sandbox="program"` only when the toolset cannot run without the harness-owned program sandbox.

## Move to a CLI Agent

`opencode-harbor` runs a real coding agent inside a sandbox. Each task comes from Harbor: an instruction, files or setup scripts, a Docker image, and tests that determine reward.

Run a small eval:

```bash
prime eval run prime/opencode-harbor -m openai/gpt-5.4-mini
```

Or run with a config file:

```toml
# [configs/09/opencode-harbor.toml](../../configs/09/opencode-harbor.toml)
model = "openai/gpt-5.4-mini"
save_results = true

[[eval]]
env_id = "prime/opencode-harbor"

[eval.taskset]
task_names = ["regex-log"]

[eval.harness]
max_turns = 4

[eval.harness.program]
disabled_tools = ["webfetch", "question"]
```

```bash
prime eval run configs/09/opencode-harbor.toml
```

A broader baseline run without `task_names` (5 examples × 3 rollouts, the
env's defaults from `pyproject.toml`) cost roughly **$3.04** end-to-end against
`gpt-5.4-mini`. Expect a longer eval than the text-only examples.

The reward comes from the task tests, not from judging the final message. That makes coding-agent environments useful for training, but it also means broken tests, missing dependencies, or unrealistic timeouts can dominate results.

Use the same override split in eval TOML — see
[configs/09/opencode-harbor.toml](../../configs/09/opencode-harbor.toml):

```toml
[[eval]]
env_id = "prime/opencode-harbor"

[eval.taskset]
task_names = ["regex-log"]

[eval.harness]
max_turns = 4

[eval.harness.program]
disabled_tools = ["webfetch", "question"]
```

The taskset fields choose Harbor tasks and sandbox defaults. The harness fields
change how OpenCode runs the task. Program fields are nested under
`harness.program` because they configure the command program inside the harness,
not the task distribution.

## How the Pieces Fit

The Hub IDs are `prime/math-python` and `prime/opencode-harbor`. After install, the local import/package IDs are `math-python` and `opencode-harbor`.

In `math_python`:

- the Taskset samples math questions
- the Taskset exposes a Python tool backed by a sandbox
- the reward checks the boxed answer by symbolic equivalence

In `opencode_harbor`:

- the Harbor Taskset loads task instructions, files, sandbox settings, and tests
- the OpenCode Harness runs the CLI agent inside the sandbox
- the reward is computed from the Harbor verifier output

Start with the smallest sandbox that proves the scoring loop, then add richer task state, files, commands, and full agent harnesses when the task requires them.

When the baseline eval runs cleanly, train against the same environment ID as in [Training with RL](../03-training-with-rl/README.md).

## Next

In [Synthetic Agent Environments](../11-synthetic-agent-environments/README.md), you will simulate a small world in memory and have an agent interact with it through tools.

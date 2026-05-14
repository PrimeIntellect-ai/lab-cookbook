# Coding Agents and Sandboxes

Evaluate code-producing agents in isolated runtimes.

Coding environments need more than text comparison. The environment should let the model inspect or write code, execute commands safely, collect logs, and score the result from actual behavior.

This guide starts with [`primeintellect/math-python`](https://app.primeintellect.ai/dashboard/environments/primeintellect/math-python), a lightweight Python-tool environment, then moves to [`primeintellect/opencode-harbor`](https://app.primeintellect.ai/dashboard/environments/primeintellect/opencode-harbor), a full CLI-agent environment that runs OpenCode on Harbor tasks.

## Warm Up with Math Python

`math-python` asks math questions that are easier to solve with code than by mental arithmetic. The model gets a Python tool backed by a sandbox, uses it for calculations, and returns a final boxed answer.

Run a small eval:

```bash
prime eval run primeintellect/math-python \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 2 \
  -t 1024
```

```text
TODO: expected output
```

Open the eval results:

```bash
prime lab view --evals
```

```text
TODO: expected output
```

Inspect:

- the Python code the model ran
- any command errors or tracebacks
- whether the final answer is boxed
- whether symbolic verification matches your judgment
- how many tool calls the model needed

This is the smallest useful sandbox pattern: one task, one Python tool, one isolated runtime, and a deterministic reward.

## Move to a CLI Agent

`opencode-harbor` runs a real coding agent inside a sandbox. Each task comes from Harbor: an instruction, files or setup scripts, a Docker image, and tests that determine reward.

Run a small eval using `configs/06/opencode-harbor.toml` (model `openai/gpt-5.4-mini`, environment defaults for everything else):

```bash
prime eval run configs/06/opencode-harbor.toml
```

```text
TODO: expected output
```

A baseline run of this config (5 examples × 3 rollouts, the env's defaults from `pyproject.toml`) cost roughly **$3.04** end-to-end against `gpt-5.4-mini`.

Open the eval results:

```bash
prime lab view --evals
```

```text
TODO: expected output
```

Inspect:

- the task instruction
- the files and working directory available to the agent
- the OpenCode log
- commands run inside the sandbox
- test output and reward
- whether the agent timed out, failed setup, or failed the task itself

The reward comes from the task tests, not from judging the final message. That makes coding-agent environments useful for training, but it also means broken tests, missing dependencies, or unrealistic timeouts can dominate results.

## How the Pieces Fit

The Hub IDs are `math-python` and `opencode-harbor`. The source packages are `math_python` and `opencode_harbor`.

In `math_python`:

- the Taskset samples math questions
- the Harness exposes a Python tool in a sandbox
- the reward checks the boxed answer by symbolic equivalence

In `opencode_harbor`:

- the Harbor Taskset loads task instructions, files, sandbox settings, and tests
- the OpenCode Harness runs the CLI agent inside the sandbox
- the reward is computed from the Harbor verifier output

You do not need to build all of this at once. Start with the smallest sandbox that proves the scoring loop, then add richer task state, files, commands, and full agent harnesses when the task requires them.

## Before Training

Before launching RL on a coding-agent environment, check:

- baseline reward is not all zero from setup failures
- failed rollouts contain useful agent behavior to improve
- timeouts are long enough for plausible solutions
- tests fail for the right reasons
- sandbox logs are saved and readable

When those conditions hold, train against the same environment ID just as you did in [Training with RL](../03-training-with-rl/README.md).

## Next

In [Tool Use and Search](../07-tool-use-and-search/README.md), you will work with environments where the model searches a document corpus before answering.

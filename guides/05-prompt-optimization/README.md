# Prompt Optimization

Use GEPA<a href="../../reference/glossary.md#gepa">¹</a> to improve an environment prompt before changing model weights.

RL changes the model. GEPA changes the prompt. It is useful when the environment already has a meaningful scoring signal, but the model behavior depends heavily on the system prompt, tool instructions, output format, or task strategy.

This guide uses [`primeintellect/wordle`](https://app.primeintellect.ai/dashboard/environments/primeintellect/wordle), because its behavior is easy to inspect: the model sees game state, chooses guesses, and is scored on whether it solves the puzzle.

## Check the Baseline

Run a small eval first:

```bash
prime eval run primeintellect/wordle \
  -m openai/gpt-5.5 \
  -n 20 \
  -r 1 \
  -t 1024
```

Expect the standard eval summary with a run id, rollout progress, reward metrics, token usage, and saved results path.

Open the eval results:

```bash
prime lab view --evals
```

This opens the eval results view in Lab.

Read a few failed rollouts. GEPA is most useful when the model is trying the task but needs better guidance. If the scoring is broken, the task is impossible, or the model cannot follow the environment loop at all, fix that before optimizing the prompt.

## Run GEPA

Run GEPA against the same environment and model:

```bash
prime gepa run primeintellect/wordle -m openai/gpt-5.5
```

The command prints optimization progress and a results directory containing the best prompt artifacts.

For a reusable run, create `configs/gepa/wordle.toml`:

```toml
model = "openai/gpt-5.5"
save_to_environment = true

[[env]]
id = "primeintellect/wordle"
```

Then run:

```bash
prime gepa run configs/gepa/wordle.toml
```

Use the printed results directory to inspect GEPA artifacts after the run.

GEPA evaluates prompt candidates<a href="../../reference/glossary.md#prompt-candidate">²</a> against environment feedback and writes artifacts to a results directory:

```text
Saving results to /path/to/results/
```

The most important artifact is:

```text
/path/to/results/system_prompt.txt
```

That file contains the optimized system prompt. With `save_to_environment = true`<a href="../../reference/glossary.md#save-to-environment">³</a>, GEPA also saves the prompt into the environment's `prompts/` folder when the environment is available locally.

## Evaluate the Optimized Prompt

Create `configs/eval/wordle-gepa.toml`:

```toml
model = "openai/gpt-5.5"
save_results = true

[[eval]]
env_id = "primeintellect/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024, temperature = 0.7 }
env_args = { path_to_system_prompt = "environments/wordle/prompts/system_prompt.txt" }
```

The `env_args`<a href="../../reference/glossary.md#env-args">⁴</a> map passes environment-specific settings into the run.

Run the eval:

```bash
prime eval run configs/eval/wordle-gepa.toml
```

Expect another eval summary with the same model and sample count as the baseline, so the comparison isolates the prompt change.

Then compare the baseline and GEPA runs:

```bash
prime lab view --evals
```

Keep the model, sample count, rollout count, and sampling settings fixed while comparing prompts. The only thing that should differ between runs is the system prompt loaded through `path_to_system_prompt`.

## Decide Whether to Keep It

Adopt the optimized prompt if it improves the metric and the rollouts look better for the right reasons.

Look for:

- more consistent task strategy
- fewer formatting or tool-use mistakes
- higher score on examples the baseline could plausibly solve
- no new shortcuts that exploit the metric without solving the task, which is a common form of reward hacking<a href="../../reference/glossary.md#reward-hacking">⁵</a>

If the prompt helps, use it in future eval or training configs by passing the same environment argument. If it does not help, inspect the GEPA artifacts and baseline failures before spending a larger optimization budget.

## Next

In [Coding Agents and Sandboxes](../06-coding-agents-and-sandboxes/README.md), you will evaluate environments where the model writes or runs code inside a sandbox.

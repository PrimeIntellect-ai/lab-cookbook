# Prompt Optimization

Use GEPA to improve an environment prompt before changing model weights.

RL changes the model. GEPA changes the prompt. It is useful when the environment already has a meaningful scoring signal, but the model behavior depends heavily on the system prompt, tool instructions, output format, or task strategy.

This guide uses [prime/wordle](https://app.primeintellect.ai/dashboard/environments/prime/wordle), because its behavior is easy to inspect: the model sees game state, chooses guesses, and is scored on whether it solves the puzzle.

## Check the Baseline

Run a small eval first:

```bash
prime eval run prime/wordle \
  -m openai/gpt-5.5 \
  -n 20 \
  -r 1 \
  -t 1024
```

Or run with a config file:

```toml
# [configs/05/wordle-eval.toml](../../configs/05/wordle-eval.toml)
model = "openai/gpt-5.5"
save_results = true

[[eval]]
env_id = "prime/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024 }
```

```bash
prime eval run configs/05/wordle-eval.toml
```

GEPA is most useful when the model is trying the task but needs better guidance. If the scoring is broken, the task is impossible, or the model cannot follow the environment loop at all, fix that before optimizing the prompt.

## Run GEPA

Run GEPA with a config file:

```toml
# [configs/05/wordle-gepa.toml](../../configs/05/wordle-gepa.toml)
model = "openai/gpt-5.5"
reflection_model = "openai/gpt-5.5"
save_to_environment = true

[[env]]
env_id = "prime/wordle"

[gepa]
max_calls = 500
num_train = 100
num_val = 50
minibatch_size = 3
max_concurrent = 32

[sampling]
max_tokens = 1024
```

```bash
prime gepa run configs/05/wordle-gepa.toml
```

The command prints optimization progress and a results directory containing the best prompt artifacts.

GEPA evaluates prompt candidates against environment feedback and writes artifacts to a results directory:

```text
Saving results to /path/to/results/
```

The most important artifact is:

```text
/path/to/results/system_prompt.txt
```

That file contains the optimized system prompt. With save_to_environment = true, GEPA also saves the prompt into the environment's `prompts/` folder when the environment is available locally.

## Evaluate the Optimized Prompt

Run the same eval shape with the optimized prompt:

```bash
prime eval run prime/wordle \
  -m openai/gpt-5.5 \
  -n 20 \
  -r 1 \
  -t 1024 \
  -a '{"path_to_system_prompt": "environments/wordle/prompts/system_prompt.txt"}'
```

Or run with a config file:

```toml
# [configs/05/wordle-gepa-eval.toml](../../configs/05/wordle-gepa-eval.toml)
model = "openai/gpt-5.5"
save_results = true

[[eval]]
env_id = "prime/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024, temperature = 0.7 }
env_args = { path_to_system_prompt = "environments/wordle/prompts/system_prompt.txt" }
```

```bash
prime eval run configs/05/wordle-gepa-eval.toml
```

Keep the model, sample count, rollout count, and sampling settings fixed while comparing prompts. The only thing that should differ between runs is the system prompt loaded through `path_to_system_prompt`. Compare against the baseline with `prime eval view`.

## Decide Whether to Keep It

Adopt the optimized prompt if it improves the metric and the rollouts look better for the right reasons.

Look for:

- more consistent task strategy
- fewer formatting or tool-use mistakes
- higher score on examples the baseline could plausibly solve
- no new shortcuts that exploit the metric without solving the task, which is a common form of reward hacking

If the prompt helps, use it in future eval or training configs by passing the same environment argument. If it does not help, inspect the GEPA artifacts and baseline failures before spending a larger optimization budget.

## Next

In [Coding Agents and Sandboxes](../06-coding-agents-and-sandboxes/README.md), you will evaluate environments where the model writes or runs code inside a sandbox.

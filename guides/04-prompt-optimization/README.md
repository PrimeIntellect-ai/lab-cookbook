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
# [configs/04/wordle-eval.toml](../../configs/04/wordle-eval.toml)
model = "openai/gpt-5.5"
save_results = true

[[eval]]
env_id = "prime/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024 }
```

```bash
prime eval run configs/04/wordle-eval.toml
```

GEPA is most useful when the model is trying the task but needs better guidance. If the scoring is broken, the task is impossible, or the model cannot follow the environment loop at all, fix that before optimizing the prompt.

## Run GEPA

Run GEPA with a config file:

```toml
# [configs/04/wordle-gepa.toml](../../configs/04/wordle-gepa.toml)
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
prime gepa run configs/04/wordle-gepa.toml
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
  -a '{"taskset": {"prompt_path": "environments/wordle/prompts/system_prompt.txt"}}'
```

Or run with a config file:

```toml
# [configs/04/wordle-gepa-eval.toml](../../configs/04/wordle-gepa-eval.toml)
model = "openai/gpt-5.5"
save_results = true

[[eval]]
env_id = "prime/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024, temperature = 0.7 }
taskset = { prompt_path = "environments/wordle/prompts/system_prompt.txt" }
```

```bash
prime eval run configs/04/wordle-gepa-eval.toml
```

Keep the model, sample count, rollout count, and sampling settings fixed while comparing prompts. The only thing that should differ between runs is the system prompt loaded through `prompt_path` on the wordle taskset config.

## Next

In [Warm Starts with SFT](../05-warm-starts-with-sft/README.md), you will use SFT to give a model a stronger starting policy before further RL.

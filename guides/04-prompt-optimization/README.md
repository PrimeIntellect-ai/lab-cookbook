# Prompt Optimization

Use GEPA to improve an environment prompt before changing model weights.

RL changes the model. GEPA changes the prompt. It is useful when the environment already has a meaningful scoring signal, but the model behavior depends heavily on the system prompt, tool instructions, output format, or task strategy.

This guide uses the local `wordle` environment, also published as
[prime/wordle](https://app.primeintellect.ai/dashboard/environments/prime/wordle),
because its behavior is easy to inspect: the model sees game state, chooses
guesses, and is scored on whether it solves the puzzle.

## Build the Wordle Environment

Install the local package before optimizing it:

```bash
prime env install wordle
```

The environment is intentionally small but shows the reusable taskset pattern:
`WordleTaskset` subclasses `tasksets.TextArenaTaskset`, owns the prompt and
rewards, and runs through the default harness.

```python
class WordleTasksetConfig(TextArenaTasksetConfig):
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    user: WordleUserConfig | None = WordleUserConfig()
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = None


class WordleTaskset(TextArenaTaskset[WordleTasksetConfig]):
    def load_system_prompt(
        self, config: WordleTasksetConfig
    ) -> vf.PromptInput | vf.SystemPromptConfig | None:
        if config.system_prompt is not None:
            return config.system_prompt
        return vf.SystemPromptConfig(path="prompts/system_prompt.txt")
```

`load_system_prompt` is what makes Wordle a good GEPA example: GEPA can write
an optimized prompt into `prompts/system_prompt.txt`, and the next eval picks it
up without changing the reward code or the harness.

## Check the Baseline

Run a small eval first:

```bash
prime eval run wordle \
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
env_id = "wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024 }
taskset = { num_train_examples = 100, num_eval_examples = 20 }
harness = { max_turns = 6 }
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
env_id = "wordle"
taskset = { num_train_examples = 100, num_eval_examples = 50 }
harness = { max_turns = 6 }

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

That file contains the optimized system prompt. With `save_to_environment = true`,
GEPA also saves the prompt into the environment's `prompts/` folder when the
environment is available locally. The Wordle taskset loads
`prompts/system_prompt.txt` through `load_system_prompt`, so a saved prompt
becomes the environment default.

## Evaluate the Optimized Prompt

Run the same eval shape with the optimized prompt:

```bash
prime eval run wordle \
  -m openai/gpt-5.5 \
  -n 20 \
  -r 1 \
  -t 1024
```

Or run with a config file:

```toml
# [configs/04/wordle-gepa-eval.toml](../../configs/04/wordle-gepa-eval.toml)
model = "openai/gpt-5.5"
save_results = true

[[eval]]
env_id = "wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024, temperature = 0.7 }
taskset = { num_train_examples = 100, num_eval_examples = 20 }
harness = { max_turns = 6 }
```

```bash
prime eval run configs/04/wordle-gepa-eval.toml
```

Keep the model, sample count, rollout count, and sampling settings fixed while
comparing prompts. The only thing that should differ between runs is the prompt
file loaded by `WordleTaskset.load_system_prompt`.

## Next

In [Warm Starts with SFT](../05-warm-starts-with-sft/README.md), you will use SFT to give a model a stronger starting policy before further RL.

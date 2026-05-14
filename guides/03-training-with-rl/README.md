# Training with RL

Launch a Hosted Training run with an environment.

RL is useful once an environment has a reward signal you trust. The model samples multiple rollouts for each task, the environment scores those rollouts, and the trainer updates the model toward higher-reward behavior.

For the first run, use the Hub version of the reverse-text environment. If you built a local version in the previous guide, using the Hub copy keeps training stable while you keep iterating on your local copy.

## Check the Baseline

Run a small eval before training:

```bash
prime eval run primeintellect/reverse-text \
  -m openai/gpt-5-nano \
  -n 10 \
  -r 2 \
  -t 512
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

Look for two things before launching training:

- the reward is not already perfect
- the rollouts show fixable mistakes, not broken prompts or broken scoring

If the model gets every example right, there is little to learn. If every score is zero and the rollouts look unrelated to the task, fix the environment or prompt before training.

## Choose a Training Model

See the current Hosted Training models with:

```bash
prime train models
```

```text
TODO: expected output
```

Hosted Training currently supports these model families:

- **gpt-oss**: reasoning MoE models with effort control.
- **Qwen 3.5 / 3.6**: dense and MoE models; natively multimodal.
- **Nemotron 3**: hybrid reasoning MoE models.
- **Llama 3.2 Instruct**: dense instruct models.

Reasoning controls go directly under `[sampling]`:

```toml
[sampling]
max_tokens = 512
reasoning_effort = "medium" # gpt-oss: low, medium, high
```

For `gpt-oss`, `reasoning_effort` can be `"low"`, `"medium"`, or `"high"`. The default is `"medium"`.

For Nemotron 3 and Qwen 3.5 / 3.6, use `enable_thinking`:

```toml
[sampling]
max_tokens = 512
enable_thinking = true # Nemotron 3 and Qwen 3.5 / 3.6
```

The default is `true`. Set it to `false` when you want shorter, more direct responses or want to compare thinking and non-thinking rollouts.

## Hyperparameters and Intuitions

Status: TODO

For a first run, the config below is intentionally small. Once it works, these are the knobs that change run behavior most, in roughly the order you'll want to think about them.

TODO: turn each bullet into a paragraph with concrete intuitions and failure modes.

- **`rollouts_per_example`.** How many attempts the trainer samples per task. RL learns from advantage across rollouts in the same group, so this can't be 1. Typical range 4–16. Higher → cleaner gradient signal, more compute per step. Drop it when rollouts are expensive (long contexts, sandboxes); raise it when the reward is noisy or the model is exploring.
- **`batch_size`.** Total rollouts consumed per step. Bigger batches stabilize learning and make reward curves smoother; smaller batches step faster but noisier. Keep `batch_size` a multiple of `rollouts_per_example`.
- **Learning rate.** Default is usually right. If reward collapses or oscillates wildly in the first 20 steps, the LR is too high. If it plateaus from the start and never moves, it may be too low.
- **KL / clip controls.** These keep the policy from drifting too far from the reference model. Loosen them if learning stalls because the model can't change enough; tighten them if the model degrades on out-of-distribution prompts during training.
- **`max_tokens` and reasoning effort.** Reasoning models need enough budget to think *and* produce the answer. If completions are getting truncated mid-answer, raise `max_tokens` before changing anything else. Reasoning effort trades cost for capability — start at `"medium"` and only move up if the smaller setting hits a ceiling.
- **Eval cadence.** How often a held-out eval runs during training. Frequent enough to catch regressions, sparse enough to not dominate cost. Configure under `[[eval]]` once the basic loop is working.
- **Difficulty filtering and oversampling.** When most tasks are too easy or too hard, the gradient is wasted. The `train-with-environments` skill covers this in depth.

The general intuition: change one knob at a time, watch the rollouts, and compare against the previous run's curves. The [Choosing a Model](../01-environments-and-evals/README.md#choosing-a-model) section in 01 covers model-level tradeoffs that show up here too.

## Write a Training Config

Create `configs/rl/reverse-text.toml`:

```toml
model = "openai/gpt-oss-20b"
max_steps = 100

batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 512
reasoning_effort = "medium"

[[env]]
id = "primeintellect/reverse-text"
```

The main fields are:

- `model`: the base model to train.
- `max_steps`: how long the run should train before stopping.
- `batch_size`: the number of rollout samples consumed per training step.
- `rollouts_per_example`: how many attempts to sample for the same task.
- `[sampling]`: generation settings used during rollout collection, including reasoning controls.
- `[[env]]`: the environment or environments used for training.

For a first run, keep the config small. Bigger batches, more rollouts, validation, evals, W&B, and checkpoint policies can come later once the basic learning loop is working.

## Launch Training

Start the run:

```bash
prime train configs/rl/reverse-text.toml
```

The command prints a run ID along with the command for streaming logs from the new Hosted Training run. Follow logs with:

```bash
prime train logs <run_id> -f
```

```text
TODO: expected output
```

Open the Lab viewer to follow training:

```bash
prime lab view --training
```

```text
TODO: expected output
```

## Decide Whether It Is Learning

Early in the run, watch for:

- reward moving upward over time
- rollout samples becoming more consistent
- completions following the expected answer format
- no repeated environment errors in the logs
- no obvious reward bug where bad completions receive high scores

For reverse-text, the first "aha" is usually visible in rollouts before it is obvious from aggregate metrics: the model starts reversing more characters in the right order, then exact matches become more common.

## Advanced Configs

Once the basic run is learning, the same config can be extended with a few common knobs. See [Advanced Configurations](https://docs.primeintellect.ai/hosted-training/advanced-configs) for the full reference.

### Sampling Temperature

Set `temperature` under `[sampling]` to control rollout diversity. The default is `1.0`; raise it slightly to encourage exploration when rollouts within a group look too similar.

```toml
[sampling]
max_tokens = 512
temperature = 1.1
```

### Multiple Environments

Add more `[[env]]` sections to train across environments at once, and use `[buffer].env_ratios` to set the mix:

```toml
[[env]]
id = "primeintellect/reverse-text"

[[env]]
id = "primeintellect/wordle"

[buffer]
env_ratios = [0.5, 0.5]
```

The ratios match the order of the `[[env]]` sections.

### Online Evaluation

Add an `[eval]` block to evaluate periodically during training without stopping the run:

```toml
[eval]
interval = 25
num_examples = 64
rollouts_per_example = 1
eval_base_model = true

[[eval.env]]
id = "primeintellect/reverse-text"
```

Results show up alongside training metrics in `prime lab view --training`. Use `eval_base_model = true` on the first run with a new environment to anchor the curve against the untrained model.

## Next

In [Warm Starts with SFT](../04-warm-starts-with-sft/README.md), you will use SFT to give a model a stronger starting policy before RL.

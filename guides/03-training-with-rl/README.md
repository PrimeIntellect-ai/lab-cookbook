# Training with RL

Launch a Hosted Training run with an environment.

RL is useful once an environment has a reward signal you trust. The model samples multiple rollouts for each task, the environment scores those rollouts, and the trainer updates the model toward higher-reward behavior.

For the first run, use the Hub version of the reverse-text environment. If you built a local version in the previous guide, this keeps the training path stable while you are still editing local code.

## Check the Baseline

Run a small eval before training:

```bash
prime eval run primeintellect/reverse-text \
  -m openai/gpt-5-nano \
  -n 10 \
  -r 2 \
  -t 512
```

Open the eval results:

```bash
prime lab view --evals
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

Hosted Training currently supports these model families:

- **gpt-oss**: reasoning MoE models with effort control.
- **Qwen 3.5 / 3.6**: dense and MoE models; Qwen 3.5 and 3.6 models are natively multimodal.
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
temperature = 0.7

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

The command prints a run ID and the log command for the new Hosted Training run. Follow logs with:

```bash
prime train logs <run_id> -f
```

Open the Lab viewer to follow training:

```bash
prime lab view --training
```

## Decide Whether It Is Learning

Early in the run, watch for:

- reward moving upward over time
- rollout samples becoming more consistent
- completions following the expected answer format
- no repeated environment errors in the logs
- no obvious reward bug where bad completions receive high scores

For reverse-text, the first "aha" is usually visible in rollouts before it is obvious from aggregate metrics: the model starts reversing more characters in the right order, then exact matches become more common.

## Next

In [Warm Starts with SFT](../04-warm-starts-with-sft/README.md), you will use SFT to give a model a stronger starting policy before RL.

# Training with RL

Launch RL runs in your environment with Hosted Training.

Reinforcement learning is useful for improving a model's behavior once an environment has a reward signal you trust. The model samples multiple rollouts for each task, the environment scores those rollouts, and the trainer updates the model's policy toward higher-reward behavior.

Run the reverse-text eval from [Building Your First Environment](../02-building-your-first-environment/README.md#evaluate-it) before launching training. Ensure your environment shows rewards with some variance in scores across rollouts before starting training.

## Choose a Training Model

See the current Hosted Training models with:

```bash
prime train models
```

Hosted Training currently supports these model families:

- **gpt-oss**: MoE models with `reasoning_effort` control.
- **Qwen 3.5 / 3.6**: dense and MoE models with `enable_thinking` control; natively multimodal.
- **Nemotron 3**: MoE models with `enable_thinking` control.
- **Llama 3.2 Instruct**: dense instruct models.

Reasoning controls go directly under `[sampling]`:

```toml
[sampling]
max_tokens = 512
enable_thinking = false
```

For `gpt-oss` models, `reasoning_effort` can be `"low"`, `"medium"`, or `"high"`. The default is `"medium"`.

For Nemotron 3 and Qwen 3.5 / 3.6, use enable_thinking:

```toml
[sampling]
max_tokens = 512
enable_thinking = true # Nemotron 3 and Qwen 3.5 / 3.6
```

The default is `true`. Set it to `false` when you want shorter, more direct responses or want to compare thinking and non-thinking rollouts.

## Hyperparameters and Intuitions

For a first run, the config below is intentionally small. Once it works, these are the knobs that change run behavior most, in roughly the order you'll want to think about them. Below is more information on the components of the config.

**rollouts_per_example.** This controls how many attempts the trainer samples per task. RL learns from advantage across rollouts in the same group, so this can't be 1. Typical values fall between 4 and 16. Higher values produce a cleaner gradient signal but cost more compute per step. Drop it when rollouts are expensive (long contexts, sandboxes); raise it when the reward is noisy or the model is exploring.

**batch_size.** This is the total number of rollouts consumed per training step. Bigger batches stabilize learning and make reward curves smoother; smaller batches step faster but noisier. Keep `batch_size` a multiple of `rollouts_per_example` so sample throughput stays predictable.

**learning_rate.** This sets the learning rate for the LoRA adapter and defaults to `1e-4`. The default is usually a decent starting point for tasks with rewards between 0 and 1. If reward collapses or oscillates wildly in the first 20 steps, the learning rate is too high. If it plateaus from the start and never moves, it may be too low.

`**max_tokens` and thinking mode.** Under `[sampling]`, `max_tokens` sets the maximum number of tokens the model can generate per response turn. Models that support thinking need enough budget to reason through the problem and produce the answer. If completions are getting truncated mid-answer, raise max_tokens before changing anything else. Thinking is toggled for hybrid reasoning models (`Qwen`, `Nemotron`) via `[sampling].enable_thinking` — it produces extended chain-of-thought reasoning before the final answer, which helps on multi-step tasks at the cost of longer outputs. Note that `gpt-oss` models instead support `reasoning_effort` (`low`, `medium`, `high`) under `[sampling]`.

**Eval cadence.** The `[eval].interval` field controls how often a held-out eval runs during training. Set it frequent enough to catch regressions but sparse enough to not dominate cost. Configure under `[eval]` once the basic training loop is working.

## Write a Training Config

Use [configs/03/reverse-text-rl.toml](../../configs/03/reverse-text-rl.toml) as a small reverse-text starter:

```toml
# [configs/03/reverse-text-rl.toml](../../configs/03/reverse-text-rl.toml)
model = "meta-llama/Llama-3.2-1B-Instruct"

max_steps = 100
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 1024

[[env]]
id = "prime/reverse-text"
```

The main fields are:

- `model`: the base model to train.
- `max_steps`: how long the run should train before stopping.
- `batch_size`: the number of rollout samples consumed per training step.
- `rollouts_per_example`: how many attempts to sample for the same task.
- `[sampling]`: generation settings used during rollout collection, including reasoning controls.
- `[[env]]`: the environment or environments used for training.

For a first run, keep the config small. Bigger batches, more rollouts, validation, evals, W&B, and checkpoint policies can come later once the basic learning loop is working.

## Example Run Costs

The table below shows observed costs from small Hosted Training runs. Treat these as rough reference points rather than quotes: pricing can change, and the final cost depends on the model, rollout length, token usage, and any extra evals or logging you enable. Each run used the same small training shape: `max_steps = 100`, `batch_size = 128`, `rollouts_per_example = 8`, and `max_tokens = 1024`.


| Environment             | Model                              | What it shows                             | Observed cost |
| ----------------------- | ---------------------------------- | ----------------------------------------- | ------------- |
| `prime/reverse-text`    | `meta-llama/Llama-3.2-1B-Instruct` | Cheapest baseline text-only RL run        | $0.14         |
| `prime/reverse-text`    | `Qwen/Qwen3.5-0.8B`                | Same task on a similarly small Qwen model | $0.14         |
| `prime/reverse-text`    | `meta-llama/Llama-3.2-3B-Instruct` | Same task on the next Llama size up       | $0.35         |
| `prime/reverse-text`    | `Qwen/Qwen3.5-2B`                  | Same task on a small dense Qwen model     | $0.54         |
| `prime/wordle` | `meta-llama/Llama-3.2-1B-Instruct` | Same small model on a multi-turn game     | $0.73         |
| `prime/reverse-text`    | `Qwen/Qwen3.5-4B`                  | Same task on a larger dense Qwen model    | $3.24         |


The main pattern is that text-only runs on the smallest models can stay well under a dollar, while multi-turn environments and larger models move the cost up quickly. For the current per-token model prices, run `prime train models` before launching a larger experiment.

## Launch Training

Start the run:

```bash
prime train configs/03/reverse-text-rl.toml
```

The command prints a run ID along with the command for streaming logs from the new Hosted Training run. Follow logs with:

```bash
prime train logs <run_id> -f
```

Early logs should show the run starting, the environment loading, rollout collection beginning, and step-level training metrics. Repeated environment errors here usually mean the environment or secrets need fixing before you wait on the run.

Open the Lab viewer to follow training:

```bash
prime lab view --training
```

This opens the Hosted Training view in Lab, where you can follow run status, logs, reward curves, loss curves, checkpoints, and online evals if configured.

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

Set temperature under [sampling] to control rollout diversity. The default is `1.0`; raise it slightly to encourage exploration when rollouts within a group look too similar.

```toml
[sampling]
max_tokens = 512
temperature = 1.1
```

### Multiple Environments

Add more `[[env]]` sections to train across environments at once, and use [buffer].env_ratios to set the mix:

```toml
[[env]]
id = "prime/reverse-text"

[[env]]
id = "prime/wordle"

[buffer]
env_ratios = [0.5, 0.5]
```

The ratios match the order of the `[[env]]` sections.

### Environment Overrides

Training uses the same `taskset`/`harness` config split as evaluation — the standard `verifiers.v1` config surface. 

```toml
[[env]]
id = "prime/wordle"

[env.taskset]
num_train_examples = 512
num_eval_examples = 128

[env.harness]
max_turns = 6
```

### Online Evaluation

Add an `[eval]` block to evaluate periodically during training without stopping the run:

```toml
[eval]
interval = 25
num_examples = 64
rollouts_per_example = 1
eval_base_model = true

[[eval.env]]
id = "prime/reverse-text"
```

Results show up alongside training metrics in `prime lab view --training`. Use `eval_base_model = true` on the first run with a new environment to anchor the curve against the untrained model.

## Next

In [Prompt Optimization](../04-prompt-optimization/README.md), you will improve an environment prompt with GEPA before changing model weights.

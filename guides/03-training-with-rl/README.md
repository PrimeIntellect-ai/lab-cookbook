# 03 — Training with RL

The environment you built in Guide 02 is not just an eval, but also is a valid **training signal**.

In this guide you will take the same `reverse-text` taskset and embed it in a [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) training config, in order to get a tiny model to get better at reversing text. Nothing about the environment changes: the taskset, harness, and runtime split is identical, and the reward you wrote becomes the reinforcement learning signal that the model will be trained based on.

## Before you train: validate the environment

Training is expensive and slow to debug. Never start a full run to discover whether your scoring works; instead, you should establish that with cheap eval passes first:

```bash
# resolve the config without any model calls
prime eval run reverse-text --dry-run

# run the model-free gold validation, if the taskset implements one
prime eval validate reverse-text -n 20 --runtime.type subprocess

# small live run: mixed rewards on the target base model is what you want
prime eval run reverse-text -n 20 -r 4 --shuffle --model meta-llama/Llama-3.2-1B-Instruct
```

Before moving on, check the traces (Guide 01) for:

- correct prompts and expected assistant behavior;
- the intended reward on known successes *and* known failures;
- **mixed** rewards on the base model you plan to train — if every rollout scores 0.0 or every rollout scores 1.0, there is no gradient signal to learn from;
- bounded turns, tokens, and time.

This is where the reward design from Guide 02 pays off: `reverse-text` scores a similarity ratio rather than exact match, so a partially-correct model still gets partial credit, and this provides a much smoother signal to climb.

## The training config

Training consumes the same v1 environment definition as eval. Instead of a `[taskset]` section, embed the environment under `[[orchestrator.train.env]]` (`configs/03/reverse-text-rl.toml`):

```toml
model = "meta-llama/Llama-3.2-1B-Instruct"
max_steps = 100                     # trainer knobs live at the top level
batch_size = 128
rollouts_per_example = 8            # the "group" for group-relative advantage

[sampling]
max_tokens = 1024

[[orchestrator.train.env]]          # the environment, embedded
name = "reverse-text"
taskset = { id = "reverse-text" }
harness = { id = "default" }
```

The split is deliberate:

- **Trainer knobs** (`max_steps`, `batch_size`, `rollouts_per_example`, learning rate, ...) sit at the training config level.
- **Environment knobs** (taskset id and its typed config fields, harness, runtime, `max_turns`) sit on the embedded env entry — exactly the fields you already know from eval configs, in inline-table form.

`rollouts_per_example` deserves a moment. Most RL training with LLMs is based on compute-relative advantage (e.g. [GRPO]([https://huggingface.co/learn/llm-course/en/chapter12/3b](https://huggingface.co/learn/llm-course/en/chapter12/3b))), which compares rollouts of the *same* task against each other: 8 rollouts per task means each task contributes a group whose internal reward spread drives the update. Groups where every rollout scores the same teach nothing — which is why the mixed-reward check above is a gate, not a nice-to-have.

## Launch

Training runs from a `prime-rl` checkout (it needs NVIDIA GPUs and its own inference entrypoint — follow that repository's install docs):

```bash
uv run rl @ configs/03/reverse-text-rl.toml --dry-run   # resolve and validate first
uv run rl @ configs/03/reverse-text-rl.toml
```

The training stack serves the environment over the v1 env-server path and collects `Trace` objects just like eval does, so when something looks wrong mid-run, you debug it the same way by reading the traces.

## Multi-environment training

Add more `[[orchestrator.train.env]]` entries when you deliberately train across tasks. Every entry needs a unique `name`:

```toml
[[orchestrator.train.env]]
name = "reverse-text"
taskset = { id = "reverse-text" }
harness = { id = "default" }

[[orchestrator.train.env]]
name = "wordle"
max_turns = 6
taskset = { id = "wordle", num_tasks = 512 }
harness = { id = "default" }
```

Two cautions when mixing environments: use `ratio` on each entry if you want weighted sampling between them, and keep reward scales comparable — an environment whose rewards run 0–10 will dominate one scoring 0–1 even when the sampling looks balanced.

## Diagnosing failures

When a training run misbehaves, classify the failure before touching hyperparameters. Valid-but-low rewards are a model/task problem; `ProviderError`, `HarnessError`, `ToolsetError`, and `SandboxError` are infrastructure problems visible in the traces; only after valid traces arrive and loss/KL still misbehave is it a trainer problem. Learning-rate changes do not fix invalid traces.

## Try it

- Run the eval gate above and count how many of the 80 rollouts land strictly between 0 and 1 — that is your gradient signal.
- Take the environment you built from Guide 02's exercises and write its `[[orchestrator.train.env]]` entry. Would its exact-match reward pass the mixed-reward gate on a 1B model? If not, sketch a partial-credit version.



## Next

→ [04 — Prompt Optimization](../04-prompt-optimization/README.md): a cheaper knob than RL — measure prompt changes with the same eval loop.
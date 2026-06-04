# Environments and Evals

In Lab, evals are environments.

If you've run or read about a benchmark like GSM8K, MMLU, or SWE-bench, you already have the mental model: an eval is a collection of tasks plus a way to score a model's attempts on them. An *environment* is that same unit — tasks and scoring — packaged behind a single entry point so anything in Lab can load it and run rollouts against it. The name is borrowed from reinforcement learning, where tasks and a reward signal are what a model *trains* against; the choice is deliberate, because in Lab the package you use to grade a model is the same package you'd use to train one. No need to rewrite your evals.

An environment packages the work you want a model or agent to do. It samples tasks, produces rollouts, and computes metrics from the results. The same environment can be used for benchmarking models and prompts, generating synthetic data, optimizing harnesses, and training with RL or other algorithms.

Environments live in your workspace as well as on the [Environments Hub](https://app.primeintellect.ai/dashboard/environments). This guide uses the local `gsm8k` and `wordle` environments provided in the `environments/` directory.

In GSM8K, tasks are math questions with expected integer answers, and the model must return the answer in a `\boxed{}` format in a single turn. In Wordle, the model is given up to 6 turns to guess a 5-letter word, and the environment provides feedback after each guess.

## Evaluate GSM8K

Run a small eval:

```bash
prime eval run prime/gsm8k \
  -m openai/gpt-5.4-nano \
  -n 10 \
  -r 2
```

This evaluates 10 examples with 2 rollouts per example. Results are saved automatically.

This can also be done with a config file:

```toml
# [configs/01/first-eval.toml](../../configs/01/first-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/gsm8k"
num_examples = 10
rollouts_per_example = 2
```

```bash
prime eval run configs/01/first-eval.toml
```

The terminal summary includes metrics like average reward, token usage, and error rate, as well as an example rollout.

```bash
prime eval view
```

## Read the Rollouts

Open a few individual rollouts before focusing on the aggregate score. Each rollout shows one model attempt, including the prompt, completion, score, and any task data captured by the environment. The eval viewer also shows distribution-level metrics, timing, and other details.

As you read, check whether:

- the model understood the task
- repeated rollouts for the same task behave differently
- failures have an obvious cause
- the score matches your judgment
- any task needs clearer data, constraints, or scoring

This is the basic eval loop: evaluate a model, read the rollouts, and decide whether the task, prompt, harness, model, or metric needs to change.

## Choosing a Model

There are several factors to consider when selecting a model:

**Open vs. closed.** Lab supports both open-weights and closed-weights models for running evaluations, prompt optimization, and other non-training workflows. Evaluations support the standard API protocols for OpenAI and Anthropic compatible model endpoints:
- `/v1/responses` (OpenAI)
- `/v1/chat/completions` (OpenAI)
- `/v1/completions` (OpenAI)
- `/v1/messages` (Anthropic)

The same environments you use for evaluating closed frontier models can be used for training your own models on top of an open base model. See [Training with RL](../03-training-with-rl/README.md#choose-a-training-model) for how to connect a training-compatible model to the same environments.

**Cost, speed, and capability.** Start with a cheap, fast model — `openai/gpt-5.4-nano`, `anthropic/claude-haiku-4.5`, or a small open model like `Qwen/Qwen3.5-0.8B` — to confirm the environment and scoring work, then step up when you're iterating on prompts or checking the ceiling. Many evals use OpenAI or Anthropic models: pass a Prime Inference id to `-m` as above, or an alias from [configs/endpoints.toml](../../configs/endpoints.toml) with your own API key. Run `prime inference models` if you want to browse options or compare pricing. If a bigger model doesn't move scores, the bottleneck is probably the environment, not the model.

**Reasoning controls.** Many model families, including `Qwen3.5` / `Qwen3.6`, `Nemotron`, and `gpt-oss`, support thinking mode — extended chain-of-thought before the final answer, toggled via `[sampling].enable_thinking` (or `reasoning_effort` for `gpt-oss`). This helps on multi-step tasks (math, code, logic) but inflates output length and cost. When comparing models, try a few reasoning settings so you see the cost-performance tradeoffs, not just the best-case score.

**Multimodal support.** Many open-source models are text-only. If your tasks involve images, screenshots, or diagrams, you will need to choose a model family that supports multimodal input. As of May 2026, we recommend the `Qwen3.5`/`Qwen3.6` family for evaluation and training with multimodal support. Closed frontier models from OpenAI, Anthropic, and Google all support multimodal input for evaluation, as well as some flagship open models. See the [Multimodal Environments](../09-multimodal-environments/README.md) guide for how to build environments that pass non-text observations.


## Run a Small Suite

Once you want to run more than one environment in a single pass, move the eval settings into a config file. This keeps the model, sampling settings, and environment arguments together.

Use [configs/01/first-eval-suite.toml](../../configs/01/first-eval-suite.toml):

```toml
# [configs/01/first-eval-suite.toml](../../configs/01/first-eval-suite.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/gsm8k"
num_examples = 10
rollouts_per_example = 2
sampling_args = { max_tokens = 1024 }

[[eval]]
env_id = "prime/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024 }

[eval.taskset]
num_eval_examples = 20

[eval.harness]
max_turns = 6
```
Run the suite:

```bash
prime eval run configs/01/first-eval-suite.toml
```

Use this pattern when you want to compare model behavior across environments,
compare a base model to a trained adapter, or re-run the same checks after
changing a prompt or config. Environment-specific overrides stay next to each
`[[eval]]`: `taskset` changes the task source or difficulty, while `harness`
changes rollout execution.

## Next

In [Building Your First Environment](../02-building-your-first-environment/README.md), you will build a small environment yourself and use evals to check whether it is ready for training.

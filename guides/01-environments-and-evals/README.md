# Environments and Evals

In Lab, evals<a href="../../reference/glossary.md#eval">¹</a> are environments<a href="../../reference/glossary.md#environment">²</a>.

If you've run or read about a benchmark like GSM8K, MMLU, or SWE-bench, you already have the mental model: an eval is a collection of tasks<a href="../../reference/glossary.md#task">³</a> plus a way to score a model's attempts on them. An *environment* is that same unit — tasks and scoring — packaged behind a single entry point so anything in Lab can load it and run rollouts<a href="../../reference/glossary.md#rollout">⁴</a> against it. The name is borrowed from reinforcement learning, where tasks and a reward signal<a href="../../reference/glossary.md#reward-signal">⁵</a> are what a model *trains* against; the choice is deliberate, because in Lab the package you use to grade a model is the same package you'd use to train one. No need to rewrite your evals.

An environment packages the work you want a model or agent to do. It samples tasks, produces rollouts, and computes metrics<a href="../../reference/glossary.md#metric">⁶</a> from the results. The same environment can be used for benchmarking models and prompts, generating synthetic data, optimizing harnesses<a href="../../reference/glossary.md#harness">⁷</a>, and training with RL or other algorithms.

Environments can live locally in your workspace or on the Environments Hub. This guide uses [`primeintellect/gsm8k`](https://app.primeintellect.ai/dashboard/environments/primeintellect/gsm8k), a Hub environment.

Later guides also use [`primeintellect/wordle`](https://app.primeintellect.ai/dashboard/environments/primeintellect/wordle), a game environment with clear task state and simple success criteria.

We'll focus on the two pieces you need first: tasks and metrics. In GSM8K, the tasks are math questions with expected final answers. The metric checks whether each rollout reaches the right answer, and that same score can serve as a reward signal during later optimization.

Tools, sandboxes, browser sessions, user simulators, and custom harnesses make environments more powerful, but they are not part of this first eval.

## Evaluate GSM8K

GSM8K is a familiar math eval. It is also a Lab environment, which means you can evaluate any compatible model against it from the CLI.

Run a small eval:

```bash
prime eval run primeintellect/gsm8k \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 2
```

```text
TODO: expected output
```

This evaluates 5 examples with 2 rollouts per example. Results are saved automatically.

Open the Lab viewer to inspect eval results:

```bash
prime lab view --evals
```

```text
TODO: expected output
```

## Read the Rollouts

Open a few individual rollouts before focusing on the aggregate score. Each rollout shows one model attempt, including the prompt, completion<a href="../../reference/glossary.md#completion">⁸</a>, score, and any task data captured by the environment.

As you read, check whether:

- the model understood the task
- repeated rollouts for the same task behave differently
- failures have an obvious cause
- the score matches your judgment
- any task needs clearer data, constraints, or scoring

This is the basic eval loop: evaluate a model, read the rollouts, and decide whether the task, prompt, model, or metric needs to change.

## Choosing a Model

Status: TODO

Every eval starts with `-m provider/model`, where the provider<a href="../../reference/glossary.md#provider">⁹</a> is the service serving the model. Which model you pick shapes what the eval tells you, what it costs, and how long it takes.

TODO: walk through the tradeoffs the reader is implicitly making when they type a model name.

- **Size vs. cost vs. latency.** When a small, fast model is enough; when you need a bigger one to tell whether the environment is the bottleneck or the model is.
- **Reasoning controls.** Models with `reasoning_effort` (e.g. gpt-oss, Nemotron) vs. plain instruct models; when the extra thinking time pays for itself.
- **Tool-use and JSON reliability.** Which families follow tool schemas cleanly enough to evaluate; what to watch for in rollouts when they don't.
- **Multimodal support.** Which families accept images; cross-link to [Multimodal Environments](../10-multimodal-environments/README.md).
- **Open vs. closed.** When you want a model you can also train (covered in [Training with RL](../03-training-with-rl/README.md#choose-a-training-model)) vs. a frontier model you only evaluate against.

Practical defaults: a small, cheap model for the smoke-eval pass, then a stronger model once the environment is stable. The same `-m` flag swaps between them with no other changes.

## Run a Small Suite

Once you want to run more than one environment in a single pass, move the eval settings into a config file. This keeps the model, sampling settings, and environment arguments together.

Create `configs/eval/first-suite.toml`:

```toml
model = "openai/gpt-5-nano"
save_results = true

[[eval]]
env_id = "primeintellect/gsm8k"
num_examples = 20
rollouts_per_example = 2
sampling_args = { max_tokens = 1024 }

[[eval]]
env_id = "primeintellect/wordle"
num_examples = 20
rollouts_per_example = 1
sampling_args = { max_tokens = 1024, temperature = 0.7 }
```
# link to configs/00/first-eval-suite.toml

The `save_results`<a href="../../reference/glossary.md#save-results">¹⁰</a> field keeps the run visible after it finishes.


Run the suite:

```bash
prime eval run configs/00/first-suite.toml
```

```text
TODO: expected output
```

Use this pattern when you want to compare model behavior across environments, compare a base model to a trained adapter, or re-run the same checks after changing a prompt or config.

## Next

In [Building Your First Environment](../02-building-your-first-environment/README.md), you will build a small environment yourself and use evals to check whether it is ready for training.

---

### Footnotes

# Environments and Evals

In Lab, evals are environments.

An environment packages the work you want a model or agent to do. It samples tasks, produces rollouts, and computes metrics from the results. The same environment can be used for benchmarking models and prompts, generating synthetic data, optimizing harnesses, and training with RL or other algorithms.

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

This evaluates 5 examples with 2 rollouts per example, using the default value of 1024 for `max_tokens`. Results are saved automatically. This is equivalent to running:

```toml
model = "openai/gpt-5-nano"
save_results = true

[[eval]]
env_id = "primeintellect/gsm8k"
num_examples = 5 
rollouts_per_example = 2
sampling_args = { max_tokens = 1024 }
```
# link to configs/00/first-eval.toml

Open the Lab viewer to inspect eval results:

```bash
prime lab view --evals
```

## Read the Rollouts

Open a few individual rollouts before focusing on the aggregate score. Each rollout shows one model attempt, including the prompt, completion, score, and any task data captured by the environment.

As you read, check whether:

- the model understood the task
- repeated rollouts for the same task behave differently
- failures have an obvious cause
- the score matches your judgment
- any task needs clearer data, constraints, or scoring

This is the basic eval loop: evaluate a model, read the rollouts, and decide whether the task, prompt, model, or metric needs to change.

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


Run the suite:

```bash
prime eval run configs/00/first-suite.toml
```

Use this pattern when you want to compare model behavior across environments, compare a base model to a trained adapter, or re-run the same checks after changing a prompt or config.

## Next

In [Building Your First Environment](../02-building-your-first-environment/README.md), you will build a small environment yourself and use evals to check whether it is ready for training.

---

### Footnotes
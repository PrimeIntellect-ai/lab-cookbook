# Environments and Evals

In Lab, evals are environments.

An environment packages the work you want a model or agent to do. It samples tasks, produces rollouts, and computes metrics from the result. The same environment can be used for benchmarking models and prompts, generating synthetic data, optimizing harnesses, and training with RL or other algorithms.

Environments can live locally in your workspace or on the Environments Hub. This guide uses [`primeintellect/gsm8k`](https://app.primeintellect.ai/dashboard/environments/primeintellect/gsm8k), a Hub environment.

We'll focus on the two pieces you need first: tasks and metrics. In GSM8K, the tasks are math questions with expected final answers. The metric checks whether each rollout reaches the right answer, and that score can be weighted as a reward for optimization later.

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

This evaluates 5 examples with 2 rollouts per example. Results are saved automatically.

Open the Lab viewer to inspect eval results:

```bash
prime lab view --evals
```

## What to Inspect

Start with individual rollouts. The aggregate score tells you whether the model did well; the rollouts tell you why.

Look for:

- the prompt sent to the model
- the model completion
- the reward or metric value
- whether repeated rollouts for the same task behave differently
- examples where the model failed for a clear reason
- examples where the scoring rule seems too strict, too loose, or ambiguous

This is the basic eval loop: evaluate a model, inspect rollouts, and decide whether the task, prompt, model, or metric needs to change.

## Next

In [Building Your First Environment](../02-building-your-first-environment/README.md), you will build a small environment yourself and use evals to check whether it is ready for training.

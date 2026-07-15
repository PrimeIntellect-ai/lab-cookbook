# 0 — Hello, Prime Intellect Lab

Welcome! 

This series of tutorials is meant to get you up and running on Prime Intellect Lab, from an empty terminal to evaluating frontier models, training a models with reinforcement learning, and automatically optimizing prompts. In this set of tutorials and recipes, no prior experience with evals or RL is assumed.

## What is Prime Intellect Lab?

Prime Intellect Lab is a stack for working with **environments**: packaged, scored tasks for language models.

That sentence is dense, so let's unpack it with an example. Suppose you want to know how good a model is at grade-school math. You need three things:

1. **Tasks**: a set of math problems, each with a known correct answer.
2. **A way to run the model on them**: send each problem to the model, collect its answers.
3. **Scoring**: compare each answer to the known one and produce a number.

Bundle those three together and you have an *environment*. Once a task is packaged as an environment, one artifact serves every purpose:

- **Evaluate** any model on it — locally from your terminal, or hosted on the platform.
- **Train** a model on it with reinforcement learning: the score becomes the reward signal that trains the model to do better on the task.
- **Optimize prompts** against it: measure a prompt change with the exact same scoring, or let an algorithm evolve the prompt for you.

The score means the same thing everywhere. A model's eval number, its training reward, and a prompt experiment are all computed by the same code, so you can use the same abstractions both to evaluate and train.

## The pieces

You will meet these components across the tutorials. Each one of the pieces is is introduced properly when you first use it, but here is the global overview:

- **The `prime` CLI** — your front door to everything: `prime eval run` evaluates, `prime rl run` launches training, `prime gepa run` optimizes prompts, `prime env init` scaffolds new environments. One tool, installed in Tutorial 1.
- **[verifiers](https://github.com/PrimeIntellect-ai/verifiers)** — the open-source Python framework environments are written in. It defines the building blocks (tasksets, harnesses, traces) and runs the actual rollouts. You don't need to write any verifiers code for these tutorials — the CLI drives it for you.
- **The Environment Hub** — a catalog of ready-made environments published by Prime Intellect and the community (math, games, coding tasks, agent benchmarks...). Anything on the Hub is runnable by id, like `primeintellect/reverse-text`. You can publish your own with `prime env push`.
- **Prime Inference** — a unified model gateway. One API key gives you access to models from many providers (OpenAI, Anthropic, open-weights models, ...) behind one OpenAI-compatible endpoint. All the tutorials' model calls go through it by default, so you never juggle per-provider keys.
- **[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)** — the open-source RL trainer. You can run it yourself on your own GPUs, or —
- **Hosted Training** — the platform runs prime-rl for you. You submit a config with `prime rl run`, the platform provisions the GPUs, trains, streams you metrics, and hands back model checkpoints. This is how Tutorial 3 trains a model without you owning a single GPU.
- **Hosted Evals** — the same idea for evaluations: add `--hosted` to an eval and it runs on platform infrastructure instead of your laptop, with results viewable in the web dashboard.
- **Sandboxes** — isolated cloud containers where agentic environments (e.g. coding tasks) execute safely. They stay behind the scenes in these tutorials, but they are what makes it safe to let a model run shell commands.

A mental picture that will serve you well:

```
              ┌─────────────────────────────────────────┐
              │            an ENVIRONMENT                │
              │   tasks  +  how to run  +  how to score  │
              └────────────────────┬────────────────────┘
                                   │  one definition, three uses
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
    evaluation (T2)         RL training (T3)      prompt optimization (T4)
   "how good is it?"      "make it better by      "make it better by
                          updating the weights"    updating the prompt"
```

## The road ahead

| Tutorial | What you'll do | What you'll need |
| --- | --- | --- |
| [1 — Setup](1_setup.md) | Install the CLI, authenticate, run a smoke test. | 10 minutes, a terminal |
| [2 — Your First Eval](2_first_eval.md) | Measure GPT-5.4 mini on grade-school math, then on multi-turn Wordle. | A few cents of inference credit |
| [3 — Your First RL Run](3_first_rl.md) | Train a small model to reverse text, on hosted GPUs, and watch it improve live. | Training credits |
| [4 — Your First GEPA Run](4_first_gepa.md) | Evolve a better prompt automatically — improvement without touching weights. | Inference credits |

Each tutorial builds on the previous one but explains what it uses, so you can also dip in wherever you like. After the Basics, the [Ramping up series](README.md#ramping-up) introduces the building blocks one at a time — including how to build environments of your own — and the [recipes](README.md#recipes) apply them to real use cases.

Ready? → [1 — Setup](1_setup.md)

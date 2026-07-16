# 0: Hello, Prime Intellect Lab

Welcome! 

This series of tutorials is meant to get you up and running on Prime Intellect Lab, from an empty terminal to evaluating frontier models, training a models with reinforcement learning, and automatically optimizing prompts. We'll be starting gently, so no prior experience with evals or RL is assumed.

## What is Prime Intellect Lab?

Prime Intellect Lab is a stack for working with **environments**: packaged, scored tasks for language models.

For example, suppose you want to know how good a model is at grade-school math. You need three things:

1. **Tasks**: a set of math problems, including the correct solution for each problem.
2. **A way to run the model on them**: send each problem to the model, and then collect the model's answers.
3. **Scoring**: compare each answer to the correct one and produce a score.

An environment is simply a bundle of those three things. Packaging your use-case as an environment allows you to use that environment for more than one purpose. You can:

- **Evaluate** any model on it, either on your local machine from your terminal, or hosted on the Prime Intellect platform.
- **Train** a model on it with reinforcement learning (also available both locally and on the platform), where the score becomes the reward signal that trains the model to do better on the task.
- **Optimize prompts** against it: measure performance given the current system prompt, then iterate on changing the system prompt to maximize performance, or let an algorithm evolve the prompt for you (more on that later).

The score means the same thing everywhere. A model's eval number and its training/optimization reward are all computed by the same code, so you can use the same abstractions both to evaluate and train.

## The pieces 

You will meet these components across the tutorials. Each one of the pieces is is introduced properly when you first use it, but let's go through a quick overview.

Your front door to everything is the ****`prime` **CLI**. You can use it to run all the functionality from your terminal. For example, `prime eval run` evaluates, `prime rl run` launches training, `prime gepa run` optimizes prompts, `prime env init` scaffolds new environments. We will be installing it in the next tutorial.

The platform operates on environments, and the environments are written in **[verifiers](https://github.com/PrimeIntellect-ai/verifiers)** — our open-source Python framework. It defines the building blocks (tasksets, harnesses, traces) and runs the actual rollouts (trajectories of the model solving the tasks). For the initial few tutorials, you won't need to write any verifiers code, but in the later ones, we will also be getting into how to build your own environments for the use-cases you care about.

Once the environments are built, they need to be catalogued somewhere. That place is our Environment Hub -- it includes ready-made environments published by both Prime Intellect and the community, involving tasks ranging from knowledge and math to coding, games, agent benchmarks and more. Any environment on the hub is downloadable and runnable. You can publish your own with the `prime` CLI by running `prime env push`.

So we have `verifiers` to build the environments and the Hub to store them, but how do we actually evaluate how good a model is at a given task? We run evaluations, and with the prime-cli it's a matter of setting up a config and then running `prime eval run`. Crucially, we also support hosted evaluations, which run on our platform infrastructure instead of your own machine, and the results are viewable on the platform dashboard. This way, you can run evals on models without needing to run inference yourself or provide your own API keys. Instead, it all runs via Prime Inference, which is our unified model gateway -- via one API key, you gain access to models from many providers.

Of course, we also want to train our models on a given environment. For that purpose, we've built **[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)**, our open-source framework for large-scale asynchronous reinforcement learning. It's powerful and proven for usage on anywhere from a single GPU to 1000+ GPUs. As with evals, you could install it and run things on your own cluster, but you could also have our platform run `prime-rl` for you. All you do is submit a config with the CLI (`prime rl run`) and the platform provisions GPUs, trains, streams your metrics for monitoring on the dashboard, and offers you the possibility to use the trained model checkpoint. 

Some environments require the agent being trained to do complex actions, such as writing and executing code. To do this safely, we use our **Sandboxes** -- these are isolated, disposable Docker containers that make it possible to run agents in a clean, reproducible runtime without worry. In these tutorials, they'll stay behind the scenes, but they are what makes it safe for e.g. 32 instances of a model to run arbitrary shell commands during an evaluation.

A mental picture that will serve you well:


## The road ahead


| Tutorial                                   | What you'll do                                                                  | What you'll need                |
| ------------------------------------------ | ------------------------------------------------------------------------------- | ------------------------------- |
| [1 — Setup](1_setup.md)                    | Install the CLI, authenticate, run a smoke test.                                | 10 minutes, a terminal          |
| [2 — Your First Eval](2_first_eval.md)     | Measure GPT-5.4 mini on grade-school math, then on multi-turn Wordle.           | A few cents of inference credit |
| [3 — Your First RL Run](3_first_rl.md)     | Train a small model to reverse text, on hosted GPUs, and watch it improve live. | Training credits                |
| [4 — Your First GEPA Run](4_first_gepa.md) | Evolve a better prompt automatically — improvement without touching weights.    | Inference credits               |


Each tutorial builds on the previous one but explains what it uses, so you can also dip in wherever you like. After the Basics, the [Ramping up series](README.md#ramping-up) introduces the building blocks one at a time — including how to build environments of your own — and the [recipes](README.md#recipes) apply them to real use cases.

Ready? → [1 — Setup](1_setup.md)
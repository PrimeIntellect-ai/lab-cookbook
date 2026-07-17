# Your First Eval

In this tutorial, we'll be evaluating models, starting from a simple grade school math environment, and then running our model in a multi-turn environment where it can try an answer, then retry over and over. We will run the evaluations, understand the different parameters that are important, and go over viewing the outputs.

You need the setup from [Tutorial 1](1_setup.md): an authenticated CLI, and a terminal sitting inside the `lab-cookbook` directory.

## Command breakdown

Let's run our benchmark on one of the most popular simple math benchmarks (GSM8K), which we have as a local copy in the `environments/` directory. To keep the run cheap, let's run it on a relatively cheap model. We'll also need to specify a few more arguments (explained in the table below), and run it like as follows: 

```bash
uv run eval gsm8k_v1 \
  --num-tasks 10 \
  --num-rollouts 2 \
  --max-concurrent 8 \
  --model openai/gpt-5.4-mini \
  --sampling.max-tokens 1024
```

While it's running, have a look at what the commands denote:


| Part                          | Meaning                                                                                                                                                                                                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gsm8k_v1`                    | The local environment to run — GSM8K is a classic dataset of grade-school word problems ("Natalia sold clips to 48 of her friends..."), each with a known numeric answer.                                  |
| `--num-tasks 10`              | Evaluate 10 tasks (problems). Without it, you'd run the whole dataset.                                                                                                                                     |
| `--num-rollouts 2`            | Give the model **2 attempts per task**. Language models are sampled, so the same model can succeed once and fail once on the same problem — multiple attempts show you that variance instead of hiding it. |
| `--max-concurrent 8`          | Keep at most 8 rollouts in flight. This bounds local load without changing what is evaluated.                                                                                                              |
| `--model openai/gpt-5.4-mini` | Which model to evaluate, routed through Prime Inference.                                                                                                                                                   |
| `--sampling.max-tokens 1024`  | Cap each model response at 1024 tokens, so a rambling answer can't run forever (or run up your bill).                                                                                                      |


You're running 20 attempts (10 tasks × 2 rollouts per task) stream by, each finishing with a score, and then a summary with the mean **reward** — for GSM8K, `1.0` means the extracted final answer matched the known one, `0.0` means it didn't. 

When you ran the command, the following happened:

1. The CLI loaded the **taskset** containing the problems, plus the scoring code that knows each correct answer.
2. For each attempt, a **harness** sent the problem to the model and collected its reply. Today's harness (`default`) is a simple chat loop, but later tutorials use harnesses that wrap entire coding agents.
3. Each attempt produced a **rollout**, and its complete record (including every message, the score, why it stopped) was saved as a **trace**.
4. The scoring code read each trace and computed the **reward**.

Each evaluation run saves the outputs:

```
outputs/<taskset>--<model>--<harness>/<run-id>/
├── config.toml      # the fully-resolved settings of this run
├── traces.jsonl     # one trace per line — the raw evidence
└── eval.log         # logs
```

In `traces.jsonl`, a line contains the task data, typed message graph, rewards, metrics, stop condition, and any errors for one rollout. A mean score tells you how often the model succeeded or failed, but you can look at traces to more closely inspect what the model did, where it succeeded and where it failed.

## Reproducibility

Command-line flags are great for exploring, for important results, you want to have reproducible configs. Any eval can be a TOML file, so let's port the above CLI command into a TOML file (saved under `configs/02/gsm8k-eval.toml`):

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 10
num_rollouts = 2
max_concurrent = 8

[sampling]
max_tokens = 1024

[taskset]
id = "gsm8k_v1"
```

To confirm that the config resolves successfully, we can run our command with a `--dry-run` flag:

```bash
uv run eval @ configs/02/gsm8k-eval.toml --dry-run
```

If all is okay, run the eval with the config:

```bash
uv run eval @ configs/02/gsm8k-eval.toml
```

Flags and files compose. When both are used for a certain parameter, flags have a precedence, so `uv run eval @ configs/02/gsm8k-eval.toml --num-tasks 2` is a quick two-task test of a saved config. Every run saves its resolved `config.toml` next to its traces, so the result is reproducible.

## Multi-turn evals

The GSM8K tasks ask one question and expect one answer. Of course, tasks are not always like this; often they are *conversations*, where the model acts, the environment responds, and the model responds. The classic word game Wordle is a perfect miniature, where the model has to guess a five-letter word, learn which letters it guessed correctly, and then guess again:

```bash
uv run eval wordle_v1 \
  --num-tasks 5 \
  --num-rollouts 1 \
  --max-concurrent 5 \
  --model openai/gpt-5.4-mini \
  --max-turns 6 \
  --sampling.max-tokens 1024
```

This is a multi-turn environment, so we also need to specify how many times we'd like our agent to retry. This is done via the `--max-turns 6` flag, which places a hard ceiling on conversation length, enforced by the framework, so no rollout can loop forever. Six turns matches the real game's six guess maximum.

In `traces.jsonl`, you'll see the model's guess, the environment's feedback on each letter, and the next guess narrowing down, followed by the reward.

## Things to try

- Compare different models on a given benchmark; for example, see how different models in the same model family compare.
- See the effect of different sampling parameters; for example, how does increasing the temperature parameter affect math performance?

## Recap

You measured two models on two environments, learned the vocabulary (taskset, harness, rollout, trace, reward), read failing traces instead of trusting a mean, and made a run reproducible with a config file.

So far the model was fixed and we just *measured* it. Time to change the model itself.

Next → [3 — Your First RL Run](3_first_rl.md): train a small model to get better at a task, live on hosted GPUs.
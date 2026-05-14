# Building Your First Environment

Build a local environment and evaluate it with Lab.

The task is simple to state and surprisingly hard to solve: given a piece of text, return the characters in reverse order. Even capable models drop characters or reverse word-by-word, and accuracy falls sharply without chain-of-thought. The payoff is a clean continuous reward — longest-common-subsequence between the model's answer and the true reversal — that is robust to reward hacking and trains quickly under RL.

You will build this as `reverse-text`. You can also inspect the finished Hub environment at [`primeintellect/reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text).

## Create the Package

From your Lab workspace, scaffold a local environment package:

```bash
prime env init reverse-text
```

```text
TODO: expected output
```

This creates `environments/reverse_text/` with a starter `reverse_text.py` and `pyproject.toml`. Open `reverse_text.py` — you will replace its contents as you go.

## Define Your Tasks

The first thing an environment needs is some tasks for the model to attempt. Here, we'll use [`PrimeIntellect/Reverse-Text-RL`](https://huggingface.co/datasets/PrimeIntellect/Reverse-Text-RL). Each row gives you a piece of text:

```python
{"prompt": "The quick brown fox jumps over the lazy dog."}
```

Build a chat-style `prompt` from the text and pair it with the reversed `answer` the model should produce:

```python
from datasets import load_dataset


DATASET_NAME = "PrimeIntellect/Reverse-Text-RL"


def source():
    rows = []
    for row in load_dataset(DATASET_NAME, split="train"):
        text = row["prompt"]
        rows.append({
            "prompt": [{"role": "user", "content": text}],
            "answer": text[::-1],
        })
    return rows
```

The taskset will call `source` once and cache the rows.

## Add a Reward

Tell the model where to put its answer with a system prompt:

```python
SYSTEM_PROMPT = (
    "Reverse the text character-by-character. Put your answer in "
    "<reversed_text> tags."
)
```

A reward is an `async` function decorated with `@vf.reward`. It receives the immutable `task` and the `state` produced by the rollout, and returns a float. Pull the tagged answer out of the model's reply and score it against the true reversal with a longest-common-subsequence ratio, so partial answers get partial credit:

```python
from difflib import SequenceMatcher

import verifiers.v1 as vf


@vf.reward(weight=1.0)
async def lcs_reward(task, state) -> float:
    text = state["completion"][-1]["content"]
    response = text.split("<reversed_text>", 1)[-1].split("</reversed_text>", 1)[0].strip()
    return SequenceMatcher(None, response, task["answer"]).ratio()
```

If either tag is missing, the splits fall through to the raw completion.

## Wire It Together

Two `load_*` functions tie the pieces together. `load_taskset` packages the source, system prompt, and reward. `load_environment` wraps the taskset in `vf.Env`, which is what evals and trainers load:

```python
def load_taskset(config: vf.TasksetConfig) -> vf.Taskset:
    return vf.Taskset(
        source=source,
        system_prompt=SYSTEM_PROMPT,
        rewards=[lcs_reward],
        config=config,
    )


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=load_taskset(config=config.taskset))
```

By default, `vf.Env` sends each prompt to the model and hands the response back to the taskset for scoring.

## Check the Package

Make sure `environments/reverse_text/pyproject.toml` declares the entrypoint and dependencies:

```toml
[project]
name = "reverse-text"
version = "0.1.0"
description = "Reverse text character by character."
requires-python = ">=3.11"
dependencies = [
    "verifiers",
    "datasets",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["reverse_text.py", "pyproject.toml"]

[project.entry-points."verifiers.environments"]
reverse-text = "reverse_text:load_environment"

[tool.verifiers.eval]
num_examples = 20
rollouts_per_example = 2
```

Install the environment into your workspace:

```bash
prime env install reverse-text
```

```text
TODO: expected output
```

## Evaluate It

Run a small eval:

```bash
prime eval run reverse-text \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 2 \
  -t 512
```

```text
TODO: expected output
```

Open the Lab viewer:

```bash
prime lab view --evals
```

```text
TODO: expected output
```

Read a few rollouts. For reverse-text, check whether the model copied the string forward, reversed only words, dropped punctuation, or produced the right characters in the wrong order. `lcs_reward` tells you how close it got.

## Why This Environment Works

Reverse-text has a clear task, a deterministic answer, and a graded reward. That makes it a good first training target:

- the task is easy to generate at scale
- failures are easy to inspect
- partial credit gives the model a learning signal even before it fully solves the task

The next guide uses this same environment to launch an RL run and watch reward improve.

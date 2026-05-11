# Building Your First Environment

Build a local reverse-text environment and evaluate it with Lab.

Reverse-text asks a model to reverse a string character by character. It is small enough to understand in one file, but still useful for training because partial progress is measurable: a model can get some reward for producing a mostly-correct reversal before it learns the task perfectly.

## Create the Environment

From your Lab workspace, create a local environment package:

```bash
prime env init reverse-text
```

This creates `environments/reverse_text/`. Open `environments/reverse_text/reverse_text.py` and replace the starter implementation with a Taskset-based environment:

```python
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from datasets import load_dataset

import verifiers.v1 as vf


DATASET_NAME = "PrimeIntellect/Reverse-Text-RL"
SYSTEM_PROMPT = (
    "Reverse the text character-by-character. Put your answer in "
    "<reversed_text> tags."
)


def completion_text(completion: object) -> str:
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    return str(completion or "")


def extract_reversed_text(text: str) -> str:
    start_tag = "<reversed_text>"
    end_tag = "</reversed_text>"
    start = text.find(start_tag)
    if start == -1:
        return text.strip()
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def build_source(
    dataset_name: str = DATASET_NAME,
    dataset_split: str = "train",
    max_examples: int | None = None,
):
    def source():
        dataset = load_dataset(dataset_name, split=dataset_split)
        for index, row in enumerate(dataset):
            if max_examples is not None and index >= max_examples:
                break

            text = str(row["prompt"])
            yield {
                "task_id": f"reverse-text-{index}",
                "prompt": [{"role": "user", "content": text}],
                "answer": text[::-1],
                "info": {"text": text},
                "max_turns": 1,
            }

    return source


@vf.update
async def parse_answer(task: dict[str, Any], state: dict[str, Any]) -> None:
    text = completion_text(state.get("completion"))
    state["parsed_answer"] = extract_reversed_text(text)


@vf.metric
async def exact_match(task: dict[str, Any], state: dict[str, Any]) -> float:
    return float(state.get("parsed_answer") == task["answer"])


@vf.reward(weight=1.0)
async def lcs_reward(task: dict[str, Any], state: dict[str, Any]) -> float:
    prediction = str(state.get("parsed_answer") or "")
    answer = str(task["answer"])
    return SequenceMatcher(None, prediction, answer).ratio()


def load_taskset(
    dataset_name: str = DATASET_NAME,
    dataset_split: str = "train",
    max_examples: int | None = None,
    config: vf.TasksetConfig | None = None,
) -> vf.Taskset:
    return vf.Taskset(
        source=build_source(dataset_name, dataset_split, max_examples),
        system_prompt=SYSTEM_PROMPT,
        updates=[parse_answer],
        metrics=[exact_match],
        rewards=[lcs_reward],
        config=config,
    )


def load_environment(
    dataset_name: str = DATASET_NAME,
    dataset_split: str = "train",
    max_examples: int | None = None,
    config: vf.EnvConfig | None = None,
) -> vf.Env:
    config = config or vf.EnvConfig()
    return vf.Env(
        taskset=load_taskset(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_examples=max_examples,
            config=config.taskset,
        )
    )
```

The important shape is:

- `build_source()` loads task rows.
- `parse_answer()` extracts the model's answer from each rollout.
- `exact_match()` records a strict metric.
- `lcs_reward()` gives partial credit and is weighted as the training reward.
- `load_environment()` wraps the taskset in `vf.Env`, which uses the default model harness.

You are not building a custom harness yet. The default harness sends the prompt to the model, records the response, and lets the taskset score the rollout.

## Check the Package

Make sure `environments/reverse_text/pyproject.toml` includes the environment entrypoint and dependencies:

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

## Evaluate It

Run a small eval:

```bash
prime eval run reverse-text \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 2 \
  -t 512
```

Open the Lab viewer:

```bash
prime lab view --evals
```

Read a few rollouts. For reverse-text, check whether the model copied the string forward, reversed only words, dropped punctuation, or produced the right characters in the wrong order. The `exact_match` metric tells you whether the output was perfect; the `lcs_reward` score tells you how close it was.

## Why This Environment Works

Reverse-text has a clear task, a deterministic answer, and a graded reward. That makes it a good first training target:

- the task is easy to generate at scale
- failures are easy to inspect
- exact match is strict enough for evaluation
- partial credit gives the model a learning signal before it solves the task

The next guide uses this same environment to launch an RL run and inspect whether reward improves.

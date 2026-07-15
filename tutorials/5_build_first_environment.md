# Build Your First Environment

In the Basics you ran existing environments. Now you will build one. The task we've chosen is deliberately trivial — reversing a string — so that all the attention goes to the *shape* of an environment: how tasks are typed, how a rollout knows when to stop, and how a reward reads the trace.

The finished version ships with the cookbook at `environments/reverse_text/reverse_text.py`, so you can run everything in this tutorial immediately and read along in the real file.

## Scaffolding

For your own environments, the CLI generates the package skeleton:

```bash
prime env init my-env
```

This creates a package under `./environments` with a taskset stub, a `pyproject.toml`, and the `__all__` export wired up. (Flags like `-T`, `-U`, and `-H` add tool servers, user simulators, and custom harnesses — later tutorials cover those. For most environments, a taskset is all you need.)

Here we'll skip the scaffold and dissect the existing `reverse_text` package instead.

## The anatomy of a taskset

A minimal v1 environment is four pieces, in order: a typed `Task`, a typed `TasksetConfig`, a `Taskset` subclass, and an `__all__` export. Here is the whole environment:

```python
import re
from difflib import SequenceMatcher

import verifiers.v1 as vf

SYSTEM = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
TAG = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)


class ReverseTextTask(vf.Task):
    answer: str


class ReverseTextConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"


class ReverseTextTaskset(vf.Taskset[ReverseTextTask, ReverseTextConfig]):
    def load_tasks(self) -> list[ReverseTextTask]:
        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        return [
            ReverseTextTask(
                idx=i,
                prompt=row["prompt"],
                system_prompt=SYSTEM,
                answer=row["prompt"][::-1],
            )
            for i, row in enumerate(rows)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def lcs(self, task: ReverseTextTask, trace: vf.Trace) -> float:
        completion = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        match = TAG.search(completion or "")
        response = match.group(1).strip() if match else ""
        return SequenceMatcher(None, response, task.answer).ratio()


__all__ = ["ReverseTextTaskset"]
```

Let's look at each individual piece and explain what it does.

### 1. The Task: a single row

```python
class ReverseTextTask(vf.Task):
    answer: str
```

A `Task` is one problem. The base class already carries `idx`, `prompt`, and `system_prompt`; you add whatever ground truth scoring will need — here, the expected reversed string. Task fields are **immutable per-row facts**. If a reward needs it to score a rollout, it belongs on the task.

### 2. The Config: knobs tunable by the runner

```python
class ReverseTextConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"
```

Config fields are the opposite of task fields: **things a user may want to change at run time** without touching the environment code. They surface automatically in the `[taskset]` section of a TOML config and as `--taskset.*` CLI flags:

```bash
prime eval run reverse-text -n 3 --taskset.dataset-split train
```

Typical config knobs: dataset splits, difficulty settings, judge model names. If your taskset has nothing to configure, use the empty base `vf.TasksetConfig` directly.

### 3. The Taskset: loading, stopping, scoring

`load_tasks` turns a data source into typed tasks. It runs once at startup; here it pulls a Hugging Face dataset and computes the answer by reversing the prompt.

```python
@vf.stop
async def single_turn(self, trace: vf.Trace) -> bool:
    return trace.num_turns >= 1
```

A `@vf.stop` condition tells the framework when a rollout is finished. It receives the live trace after each turn; returning `True` ends the rollout. Reversing text takes one reply, so we stop after one turn. (Wordle, from [tutorial 2](2_first_eval.md), instead relied on the env-level `max_turns` cap plus game logic.)

```python
@vf.reward(weight=1.0)
async def lcs(self, task: ReverseTextTask, trace: vf.Trace) -> float:
    ...
```

A `@vf.reward` function is the score. Note its signature — it receives the **typed task** (the ground truth) and the **trace** (what the model did), and returns a float. This one extracts the `<reversed_text>` tag from the last assistant message and computes a similarity ratio, so partial credit is possible — a smoother signal than exact match, which matters once you train on the environment. Weights, metrics, group rewards, and the rest of the scoring toolbox get their own tutorial: [Designing Rewards](7_rewards.md).

### 4. The export

```python
__all__ = ["ReverseTextTaskset"]
```

`__all__` exposes exactly one taskset class; this is how the loader resolves `taskset.id = "reverse-text"` to your class. Everything else in the module is private wiring.

## Run it

```bash
prime eval run @ configs/02/reverse-text-eval.toml
```

Then open the run's `results.jsonl` (see [tutorial 2](2_first_eval.md)) and trace the reward by hand: find `assistant_messages[-1]`, apply the tag regex mentally, and check the score matches. Being able to reproduce a reward from a trace is the core debugging skill for environments.

## The boundaries, restated

- **Task fields** are the immutable per-row truth used by scoring.
- **Config fields** are runner-tunable knobs exposed through `[taskset]` / `--taskset.*`.
- **Rewards read the** `Trace`; they never parse framework internals.
- `__all__` **exposes exactly one taskset class** for loader resolution.

## Try it

- Add a second reward, `format(self, task, trace)`, that returns 1.0 when the `<reversed_text>` tag is present at all, with `weight=0.2`. Re-run and watch both rewards appear in the traces.
- Add a `num_tasks: int | None = None` config field and slice the dataset in `load_tasks` — then drive it with `--taskset.num-tasks 5`.
- Scaffold a fresh environment with `prime env init word-count` and write a taskset that asks the model to count words in a sentence, scored by exact match.

## Next

→ [Judges](6_judges.md): when correctness is semantic and exact match won't do — score with an LLM judge, controlled through config.

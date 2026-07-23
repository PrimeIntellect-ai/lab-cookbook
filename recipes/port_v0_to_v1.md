# Port a v0 Environment to v1

Environments written for verifiers v0 — the `load_environment(...) -> vf.Environment` shape with datasets, `Rubric`s, and parsers — are deprecated. New authoring targets v1: typed tasksets, harnesses, and traces. This recipe is for the environments you already have: it shows how to keep running a v0 package unchanged through the legacy bridge, then walks a complete port using an environment that exists in both generations — reverse-text.

**You need:** [Build Your First Environment](../tutorials/5_build_first_environment.md), so the v1 vocabulary (TaskData / Task / TasksetConfig / Taskset) is familiar. The v1 result ships in this repo at `environments/reverse_text_v1/`.

## Option zero: don't port yet — run it through the bridge

The v1 eval CLI can run a v0 environment as-is by setting a legacy env id *instead of* a taskset id:

```bash
uv run eval --id primeintellect/reverse-text -n 2
```

The bridge installs the v0 package from the Hub on demand, loads it via `load_environment`, and runs it through the v1 rollout machinery — construction kwargs pass through `--args`. Use this to preserve existing released environments while you migrate. Do not copy the v0 shape for new authoring.

## The starting point: what a v0 env looks like

A v0 package exposes one factory function; everything lives inside it:

```python
import verifiers as vf


def load_environment() -> vf.Environment:
    dataset = ...                        # rows with "prompt" / "answer" columns

    async def lcs_reward_func(completion, answer) -> float:
        match = TAG.search(completion[-1]["content"] or "")
        response = match.group(1).strip() if match else ""
        return SequenceMatcher(None, response, answer).ratio()

    rubric = vf.Rubric(funcs=[lcs_reward_func])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, system_prompt=SYSTEM)
```

Note the shapes that will change: rows are untyped dict columns, rewards are free functions receiving a dict-shaped `completion`, configuration is whatever kwargs `load_environment` happens to accept, and the environment *class* (`SingleTurnEnv`, `ToolEnv`, ...) encodes the interaction pattern.

## The target: the same environment in v1

The full port (`environments/reverse_text_v1/reverse_text_v1/taskset.py`):

```python
import verifiers.v1 as vf


class ReverseTextData(vf.TaskData):
    answer: str


class ReverseTextTask(vf.Task[ReverseTextData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def lcs(self, trace: vf.Trace) -> float:
        match = _TAG.search(trace.last_reply or "")
        response = match.group(1).strip() if match else ""
        return SequenceMatcher(None, response, self.data.answer).ratio()


class ReverseTextConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"


class ReverseTextTaskset(vf.Taskset[ReverseTextTask, ReverseTextConfig]):
    def load(self) -> list[ReverseTextTask]:
        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        return [
            ReverseTextTask(
                ReverseTextData(idx=i, prompt=row["prompt"], system_prompt=SYSTEM, answer=row["prompt"][::-1]),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
```



## The migration, step by step

Each v0 concept has exactly one v1 home:


| v0                                                                    | v1                                                                                                                          |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Dataset row (dict columns)                                            | A `vf.TaskData` subclass — typed, immutable, per-row truth.                                                                 |
| `load_environment(**kwargs)` kwargs                                   | Fields on a `vf.TasksetConfig` subclass — surfaced as `[taskset]` TOML and `--taskset.*` flags.                             |
| Dataset loading inside the factory                                    | `Taskset.load(self)` returning typed tasks.                                                                                 |
| `Rubric` reward functions                                             | `@vf.reward` methods on the `Task`, reading `vf.Trace` — never a dict-shaped completion.                                    |
| `Parser` classes                                                      | Plain parsing inside the reward (a regex, a helper) — no framework object needed.                                           |
| Environment class choice (`SingleTurnEnv`, `ToolEnv`, `MultiTurnEnv`) | Not a class: single-turn is a `@vf.stop` (or simply no tools/user); tools are a `vf.Toolset`; the user side is a `vf.User`. |
| `system_prompt=` env kwarg                                            | `TaskData.system_prompt`, set per row in `load()`.                                                                          |
| The implicit chat loop                                                | A harness — `default` unless the program itself is custom.                                                                  |


In order:

1. **Create a** `vf.TaskData` **subclass** for the per-row fields scoring needs (here: `answer`).
2. **Move constructor kwargs onto a** `vf.TasksetConfig` **subclass** (here: dataset name and split). What was buried in code becomes runner-visible config.
3. **Move dataset loading into** `Taskset.load()`, building typed rows.
4. **Move each rubric function onto the Task as a** `@vf.reward` **method.** The signature changes from `(completion, answer)` to `(self, trace)`: the completion is `trace.last_reply`, ground truth is `self.data.answer`. Weights move from `Rubric(weights=...)` to the decorator.
5. **Replace the environment class with behavior on the task**: `SingleTurnEnv` becomes the `single_turn` stop condition; a v0 `ToolEnv`'s tool functions become `@vf.tool` methods on a `vf.Toolset`; a v0 user simulator becomes a `vf.User` with `respond`.
6. **Export exactly one taskset class through** `__all__` — this is how `taskset.id` resolves to your code.
7. **Delete** `load_environment`**.** If downstream users still need it, keep the v0 package published and point them at the bridge; don't ship both shapes in one package.



## Verify the port

The two generations score the same task the same way, so let them referee each other:

```bash
uv run eval --id primeintellect/reverse-text -n 10 --model openai/gpt-5.4-nano     # v0 via bridge
uv run eval reverse_text_v1 -n 10 --model openai/gpt-5.4-nano                      # your port
```

Same model, same sampling, same task count. The means won't match exactly (different sampled completions), but they must be *statistically indistinguishable* — and for any single completion text, the v0 rubric and the v1 reward must produce the identical score. If they don't, diff the parsing first; ports lose points in the regex, not the algebra.

Then run the cookbook's authoring checks — `uv run pytest tests/` includes an export-shape test (`__all__` exposes exactly one taskset) that catches the most common porting mistake.

## Things to try

- Port a v0 `ToolEnv`: the tool functions move nearly verbatim onto a `vf.Toolset` with `@vf.tool`, and the interesting decision becomes placement (per-task vs shared) — see [Tool Use and Search](../tutorials/10_tools.md).
- Keep one intentionally-wrong reward during the port (e.g. skip the tag regex) and watch the verification step catch it — cheap practice for a port where you *didn't* plant the bug.
- After porting, add the thing v0 couldn't express: a `@vf.metric`, a group reward, or a second judge — the reason the port is worth it is everything in [Designing Rewards](../tutorials/7_rewards.md).



## Recap

Run legacy packages through the bridge (`--id`) until you're ready; port by giving every v0 concept its one v1 home — rows to `TaskData`, kwargs to `TasksetConfig`, rubric functions to `@vf.reward` methods on the task, environment classes to task behavior; and verify by scoring the same completions with both generations. The port's payoff is the surface v0 never had: typed config as public API, traces instead of dicts, and harness independence.
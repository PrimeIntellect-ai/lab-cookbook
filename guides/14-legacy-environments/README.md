# Legacy Environments

Reference for the original Verifiers environment API — `MultiTurnEnv` subclasses scored by a `Rubric`. The main guides teach the current model (`verifiers.v1`, or "v1") grounded around the `Taskset` and `Harness` objects; this page is for reading and maintaining Hub packages built on the older API, and for migrating their pieces onto the current model.

**We recommend using the current v1 API for new Lab environments.** Existing v0 environments are still supported, but new environments should follow the path outlined in [Building Your First Environment](../02-building-your-first-environment/README.md) onward. We recommend migrating  environments to the current model when possible. Several new and upcoming features will only be supported for `verifiers.v1` environments.

## Original (v0) vs Current (v1) at a Glance

The original API builds an `Environment` object directly — a dataset plus a `Rubric` of reward functions, with environment *type* (single-turn, tool, multi-turn) chosen by which class you instantiate. The current model splits the same responsibilities into a `Taskset` (data, prompts, tools, rewards) and a `Harness` (execution), bound by `vf.Env`.

| Concern | Original | Current |
| --- | --- | --- |
| Top-level object | `vf.SingleTurnEnv(...)`, `vf.ToolEnv(...)`, a `vf.MultiTurnEnv` subclass | `vf.Env(taskset=..., harness=...)` |
| Task data | `dataset=` / `eval_dataset=` arguments | `load_tasks(split)` on a `Taskset` subclass |
| System prompt | `system_prompt=` argument | `TasksetConfig.system_prompt` / `load_system_prompt(config)` |
| Scoring | a `vf.Rubric` of reward functions | `@vf.reward` methods on the `Taskset` |
| Metrics | `rubric.add_metric(...)` | `@vf.metric` methods on the `Taskset` |
| Tools | `vf.ToolEnv` / `vf.StatefulToolEnv` / `vf.MCPEnv` | `load_toolsets(config)` returning `vf.Toolset`s |
| Judging | `vf.JudgeRubric` | judge call inside a `@vf.reward` method |
| Configuration | constructor keyword arguments | typed `TasksetConfig` / `HarnessConfig` |

Golden references for the current model: [reverse_text](../../environments/reverse_text/reverse_text.py), [wordle](../../environments/wordle/wordle.py), [wiki_search](../../environments/wiki_search/wiki_search.py).

## Single-Turn Environments

The simplest legacy environment is a dataset plus one reward function, wrapped in `SingleTurnEnv`:

```python
import verifiers as vf
from datasets import Dataset


def load_environment():
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": "What is 2+2?"}], "answer": "4"},
    ])

    async def correct_answer(completion, answer) -> float:
        return 1.0 if answer in completion[-1]["content"] else 0.0

    return vf.SingleTurnEnv(dataset=dataset, rubric=vf.Rubric(funcs=[correct_answer]))
```

`load_environment` returns a fully-built `Environment` object. Compare to the current model, where `load_environment` only binds a taskset and harness and the work lives on classes.

## Datasets

These environments take a Hugging Face `Dataset` directly through the `dataset=` argument, with an optional `eval_dataset=` for evaluation. Rows use the same columns as current tasks — `prompt` (a list of messages) or `question` (a string the framework wraps), plus optional `answer` and `info`. A `system_prompt=` argument prepends a system message.

```python
return vf.SingleTurnEnv(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    system_prompt="You are a helpful math tutor.",
    rubric=rubric,
)
```

## Rubrics

Scoring is a `Rubric` object that holds reward functions and their weights. Reward functions are plain async functions that request rollout data by parameter name:

```python
async def correct_answer(completion, answer) -> float:
    return 1.0 if answer in completion[-1]["content"] else 0.0

async def length_reward(completion) -> float:
    return 1.0 if len(completion[-1]["content"]) < 500 else 0.5

rubric = vf.Rubric(funcs=[correct_answer, length_reward], weights=[1.0, 0.1])
```

The final reward is the weighted sum. Functions can be added after construction with `rubric.add_reward_func(fn, weight=...)`, and monitor-only signals with `rubric.add_metric(fn)`.

Related rubric types:

- **`vf.JudgeRubric`** — holds a judge model client and exposes a `judge` callable to reward functions for LLM-based scoring.
- **`vf.MathRubric`** — symbolic math verification of `\boxed{}` answers via `math-verify`.
- **`vf.RubricGroup`** — combines several rubrics into one scoring surface, summing their rewards and collecting all metrics.

In the current model, all of this becomes `@vf.reward` and `@vf.metric` methods on the `Taskset`; a judge is a model call inside a reward method rather than a `JudgeRubric`.

## Tool Environments

The original API picks tool behavior by class:

- **`vf.ToolEnv`** — stateless tools passed at construction (`tools=[...]`), with a `max_turns` limit.
- **`vf.StatefulToolEnv`** — tools that need per-rollout state, with hidden arguments injected via `update_tool_args`.
- **`vf.MCPEnv`** — tools served by MCP servers, connected automatically.
- **`vf.SandboxEnv` / `vf.PythonEnv`** — containerized bash / Python execution.

```python
vf_env = vf.ToolEnv(dataset=dataset, tools=[calculate, lookup], rubric=rubric, max_turns=10)
```

In the current model, tools are methods returned from `load_toolsets(config)` inside a `vf.Toolset`, and a sandbox is attached to the toolset rather than selected by environment class — see [Tool Use and Search](../08-tool-use-and-search/README.md) and [Coding Agents and Sandboxes](../10-coding-agents-and-sandboxes/README.md).

## Custom Multi-Turn Environments

For interaction patterns beyond tool calling, you subclass `vf.MultiTurnEnv` and override `env_response` to produce the environment's reply each turn:

```python
class MyGameEnv(vf.MultiTurnEnv):
    async def env_response(self, messages, state):
        feedback = process(messages[-1]["content"])
        return [{"role": "user", "content": feedback}]
```

In the current model, the environment's between-turn reply lives in a `vf.User` subclass's `get_response`, and the rollout loop is the harness's — see [Prompt Optimization](../04-prompt-optimization/README.md#how-a-multi-turn-rollout-runs).

## Migrating

1. Subclass `vf.Taskset[MyTasksetConfig]` and move the dataset logic into `load_tasks(split)`.
2. Move reward functions onto the class as `@vf.reward` methods and metrics as `@vf.metric` methods; drop the standalone `Rubric`.
3. Move tools into `load_toolsets(config)` returning a `vf.Toolset`; drop `ToolEnv`/`StatefulToolEnv` subclassing.
4. Put judge and other settings on `MyTasksetConfig` instead of constructor arguments.
5. Export `load_taskset(config: MyTasksetConfig) -> MyTaskset`, and return `vf.Env(taskset=vf.load_taskset(config=config.taskset), harness=vf.load_harness(config=config.harness))` from `load_environment`.
6. Smoke-eval with `prime eval run` before pushing.

For environments owned by an external runtime, keep a separate `load_harness` — see [opencode_harbor](../../environments/opencode_harbor/opencode_harbor.py) and [Custom Harnesses](../12-custom-harnesses/README.md).

## Next

Return to the main curriculum at [Building Your First Environment](../02-building-your-first-environment/README.md), or use the public Prime docs for platform plumbing.

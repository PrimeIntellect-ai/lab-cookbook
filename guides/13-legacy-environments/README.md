# Legacy Environments

Reference for older Verifiers environment patterns. **Do not use these in new Lab environments** — the main guides teach the current Taskset/Harness model instead.

Read this page when maintaining Hub environments that have not migrated, or when porting code from older tutorials.

## Current vs Legacy

| Current (main guides) | Legacy |
| --- | --- |
| `class MyTaskset(vf.Taskset[MyTasksetConfig])` | `vf.Taskset(source=..., rewards=[fn, ...])` |
| `load_tasks(split)` / `load_system_prompt(config)` on the class | Module-level `source()` generator |
| `@vf.reward` methods on the Taskset | Module-level `@vf.reward` functions passed to `Rubric` |
| `load_taskset` + `vf.load_taskset` in `load_environment` | Inline `vf.Taskset(...)` construction |
| `load_toolsets(config)` returning named toolsets | `ToolEnv` / `StatefulToolEnv` / `MCPEnv` subclasses |
| Judge config on `TasksetConfig` + direct API client | `vf.JudgeRubric` + `rubric.add_*` |
| `@vf.reward(weight=0.0)` for metrics | `rubric.add_metric()` |

Golden references for the current pattern: [reverse_text](../../environments/reverse_text/reverse_text.py), [wordle](../../environments/wordle/wordle.py), [wiki_search](../../environments/wiki_search/wiki_search.py).

## Rubric-Based Scoring

Legacy environments often built a standalone rubric:

```python
rubric = vf.Rubric(
    funcs=[correct_answer, format_bonus],
    weights=[1.0, 0.2],
)
```

Related types:

- **`vf.JudgeRubric`** — stores judge model config and exposes a `judge` callable to reward functions
- **`vf.MathRubric`** — symbolic math verification via `\boxed{}` parsing
- **`vf.RubricGroup`** — combines multiple rubrics into one scoring surface

Metrics were registered separately:

```python
rubric.add_metric(response_length)  # weight=0
```

In current environments, declare multiple `@vf.reward` methods on the Taskset instead. Use `weight=0.0` for metrics-only signals.

## source() Tasksets

Legacy tasksets passed a generator or zero-arg builder:

```python
def source():
    for row in dataset:
        yield {"prompt": [...], "answer": ...}


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.Taskset(
            source=source,
            system_prompt=SYSTEM_PROMPT,
            rewards=[lcs_reward],
            config=config.taskset,
        )
    )
```

Current environments implement `load_tasks(split)` on a Taskset subclass and export `load_taskset` plus a direct `load_environment` factory.

## v0 Environment Classes

These remain in Verifiers for older packages but are not the Lab default:

- **`ToolEnv`** — stateless tools wired at environment construction
- **`StatefulToolEnv`** — per-rollout mutable tool state
- **`MCPEnv`** — globally available read-only MCP servers

New tool environments should use `load_toolsets(config)` on a Taskset subclass, as in [wiki_search](../../environments/wiki_search/wiki_search.py).

## Other Legacy Patterns

- **`load_environment(split="train")`** — split selection belongs on `TasksetConfig`, not as a loader positional arg
- **`path_to_system_prompt`** on eval config — use `taskset.system_prompt`, with `{ path = "..." }` for prompt files
- **`@vf.update` + judge side effects** — prefer a single `@vf.reward` method that owns judge I/O (see [wiki_search](../../environments/wiki_search/wiki_search.py)); `@vf.update` for scoring is legacy
- **`state.get_endpoint_config(api="chat")` inside reward functions** — breaks after rollout teardown; valid inside custom **harness** programs that proxy third-party agent calls, not for judges
- **`import verifiers.v1 as vf`** — use `import verifiers as vf`

## Migrating

1. Subclass `Taskset` and move `source()` logic into `load_tasks(split)`.
2. Move `@vf.reward` functions onto the class as methods; drop standalone `Rubric` construction.
3. Export `load_taskset(config: MyTasksetConfig) -> MyTaskset`.
4. In `load_environment`, return `vf.Env(taskset=vf.load_taskset(config=config.taskset), harness=vf.Harness(config=config.harness))`.
5. Smoke-eval with `prime eval run` before pushing.

For harness-backed envs (OpenCode, sandboxes), keep separate `load_harness` — see [opencode_harbor](../../environments/opencode_harbor/opencode_harbor.py).

## Next

Return to the main curriculum at [Building Your First Environment](../02-building-your-first-environment/README.md), or see [Lab Configuration](../../reference/lab-configuration.md) for platform plumbing.

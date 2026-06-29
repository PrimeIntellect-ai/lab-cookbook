# Legacy Environments

This page is only for v0 packages that expose `load_environment`. New environments should use the v1 taskset/harness path described in the rest of the cookbook.

A v0 package usually looks like this:

```python
import verifiers as vf


def load_environment() -> vf.Environment:
    dataset = ...
    rubric = vf.Rubric(...)
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

The v1 eval CLI can still run v0 environments through the legacy bridge by setting a legacy env id instead of a taskset id:

```bash
uv run eval --id reverse-text -n 2
```

Use this bridge to preserve existing released environments. Do not copy this shape for new v1 authoring.

## Moving a v0 Env to v1

1. Create a `vf.Task` subclass for each dataset row.
2. Move constructor kwargs into a `vf.TasksetConfig` subclass.
3. Move dataset loading into `Taskset.load_tasks(self) -> list[TaskT]`.
4. Move reward functions onto the taskset as `@vf.reward` methods that read `vf.Trace`.
5. Move callable tools into a `vf.Toolset` class with `@vf.tool` methods.
6. Move synthetic users into a `vf.User` class with `respond`.
7. Export the taskset class through `__all__`.

# Setup

This cookbook is a workspace of local v1 environment packages plus runnable configs.

## Sync

```bash
uv sync
```

The root workspace installs the local packages under `environments/` and the dev tools used by tests.

## Useful commands

```bash
uv run eval @ configs/01/first-eval.toml
uv run eval reverse-text -n 3 -r 2
uv run validate reverse-text -n 3 --runtime.type subprocess
uv run pytest -q
```

The v1 eval command takes either a taskset id directly or a TOML file prefixed with `@`. A TOML file selects exactly one taskset and, optionally, one harness.

## Local package shape

Each environment package exports one `vf.Taskset` subclass from its importable module:

```python
from my_env.taskset import MyTaskset

__all__ = ["MyTaskset"]
```

Single-file cookbook examples put the class and `__all__` in `my_env.py`; package examples can re-export from `__init__.py`.

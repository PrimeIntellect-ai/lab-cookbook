# 00 — Setup

Welcome! This cookbook teaches you how to build, evaluate, and train on **Verifiers v1 environments** with Prime Intellect Lab. Each guide is a self-contained walkthrough, and the code it references lives right here in the repository: runnable environment packages under `environments/` and matching run configs under `configs/`.

In this guide you will:

1. Install the tools (`uv` and the `prime` CLI).
2. Sync the cookbook workspace so the local environments are importable.
3. Run one tiny eval to confirm everything works.

## 1. Install the tools

You need two things: `[uv](https://docs.astral.sh/uv/)` to manage the Python workspace, and the `prime` CLI to run environments.

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the prime CLI
uv tool install prime
```

The `prime` CLI is the front door for everything in this cookbook: it scaffolds environments (`prime env init`), runs evaluations (`prime eval run`), and installs agent skills (`prime lab setup`).

### Recommended: install the Lab skills

If you work with a coding agent (Claude Code, Codex, ...), install the repository skills so your agent knows the authoring contract:

```bash
prime lab setup
```

The interactive installer walks you through it. The skills are more comprehensive than these guides, which are written for human consumption.

## 2. Sync the workspace

This repository is a `uv` workspace: every folder under `environments/` is a small installable package, and the root `pyproject.toml` wires them together.

```bash
uv sync
```

This installs `verifiers`, all local environment packages, and the dev tools used by the tests. The `prime` CLI automatically detects the workspace and runs commands inside its virtual environment, so you only need to do this once (and again whenever you add a new environment package).

## 3. Run something

> **Current status:** the released `prime` CLI does not support verifiers v1 environments yet. Until it does, every `prime eval run ...` command in these guides can be run from the workspace as `uv run eval ...` (and `prime eval validate` as `uv run validate`). The guides use the `prime` spelling throughout because that is the target interface.

Confirm your setup with a three-task eval of the `reverse-text` environment:

```bash
prime eval run reverse-text -n 3 -r 2
```

This runs 3 tasks (`-n 3`) with 2 rollouts each (`-r 2`). If you see per-task rewards stream by and a summary at the end, you are ready.

Most guides use TOML configs instead of CLI flags. The `@` prefix tells the CLI to read a config file:

```bash
prime eval run @ configs/01/first-eval.toml
```

A config file selects exactly one taskset and, optionally, one harness. You can validate a config without spending tokens by adding `--dry-run`:

```bash
prime eval run @ configs/01/first-eval.toml --dry-run
```

Finally, the test suite is plain pytest:

```bash
uv run pytest -q
```



## What's in an environment package?

A quick preview before Guide 01 explains the concepts. Each environment package exports one `vf.Taskset` subclass from its importable module:

```python
from my_env.taskset import MyTaskset

__all__ = ["MyTaskset"]
```

Single-file cookbook examples put the class and `__all__` directly in `my_env.py`; larger packages re-export from `__init__.py`. Either way, `__all__` names exactly one taskset class — that is how the loader finds it.

## Troubleshooting

- `prime: command not found` — re-run `uv tool install prime` and make sure `~/.local/bin` is on your `PATH` (`uv tool update-shell` fixes this).
- **Taskset not found** — run commands from inside the cookbook so the CLI can find the workspace, and re-run `uv sync` if you just added a package.
- **Model/auth errors** — evals call a model API; make sure your provider credentials are configured (e.g. via `prime config` or the usual API-key environment variables).



## Next

→ [01 — Environments and Evals](../01-environments-and-evals/README.md): what tasksets, harnesses, and traces actually are, and how to read an eval's output.
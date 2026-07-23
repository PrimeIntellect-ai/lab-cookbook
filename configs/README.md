# configs

Starter TOMLs for v1 eval and GEPA, plus hosted training examples.

## Layout

| Path | Used by | Shape |
| --- | --- | --- |
| Numbered directories | Guide-specific commands | v1 eval or training configs used in each guide. |
| `eval/` | `uv run eval @ <file>` | One v1 taskset per file, tuned for a model family. |
| `rl/` | Hosted training / prime-rl configs | Training configs embedding v1 env definitions under `[[orchestrator.train.env]]`. |
| `gepa/` | Legacy `vf-gepa` examples | Older v0 GEPA configs retained for reference. New configs should use the v1 `gepa` command. |
| `endpoints.toml` | Prime CLI helpers | Endpoint aliases for commands that still read the shared registry. |

## v1 Eval Configs

A v1 eval config selects one taskset and, optionally, one harness:

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 20
num_rollouts = 1
max_turns = 6

[sampling]
max_tokens = 1024

[taskset]
id = "wordle_v1"

[harness]
id = "default"
```

Run it with:

```bash
uv run eval @ configs/04/wordle-eval.toml
```

Taskset-owned fields go under `[taskset]`. Harness-owned fields go under `[harness]`. Tool and user configs nest under the taskset field that owns them, for example `[taskset.tools]`.

## Training Configs

Training configs embed the same v1 env definition:

```toml
[[orchestrator.train.env]]
name = "wiki-search"
max_turns = 8
taskset = { id = "wiki_search_v1" }
harness = { id = "default" }
```

Add another `[[orchestrator.train.env]]` block only when you deliberately train on multiple environments.

## GEPA Configs

The current `uv run gepa` CLI uses the same v1 taskset and harness loader as eval. `configs/04/wordle-gepa.toml` is the canonical local v1 example; files under `configs/gepa/` are older v0 examples retained for reference.

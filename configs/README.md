# configs

Starter TOMLs for the Lab CLIs (`prime eval run`, `prime gepa run`, `prime train`), plus a shared `endpoints.toml` for evaluating against hosted inference endpoints. Each TOML is a working example: pick the one whose model family you want and edit in place.

## Layout

| Path | Used by | Purpose |
|---|---|---|
| `eval/` | `prime eval run <file>` | Evaluate a model on one or more environments. |
| `gepa/` | `prime gepa run <file>` | Optimize environment prompts with GEPA. |
| `rl/` | `prime train <file>` | RL training on one or more environments. |
| `endpoints.toml` | `prime eval run -m <endpoint_id>` | Named hosted endpoints using an endpoint alias (Prime Inference, Anthropic, OpenAI, etc.) for evals against non-trainable models. |

Each subdirectory ships one TOML per supported model family. They differ only along three axes: the **`model`** identifier (and reflection_model for GEPA), the **`max_tokens`** sampling budget, and the **default environment** preselected in the `[[env]]`/`[[eval]]` block. Everything else (eval volume, GEPA loop sizes, RL batch shape) is held constant so the files diff cleanly.

## Environment overrides

Environment-specific knobs belong to either the Taskset or Harness config.
Eval and GEPA configs use sibling block tables:

```toml
[[eval]]
env_id = "prime/wordle"

[eval.taskset]
num_eval_examples = 20

[eval.harness]
max_turns = 6
```

Training configs use nested tables under the relevant `[[env]]` block:

```toml
[[env]]
id = "prime/opencode-harbor"

[env.taskset]
task_names = ["regex-log"]

[env.harness]
max_turns = 4

[env.harness.program]
disabled_tools = ["webfetch", "question"]
```

Eval and GEPA configs validate `taskset` and `harness` sections against the
environment's typed `TasksetConfig` and `HarnessConfig` subclasses before
`load_environment` runs.

For one-off CLI overrides, pass v1 child config through the root `config`
argument:

```bash
prime eval run prime/wordle \
  -a '{"config":{"taskset":{"num_eval_examples":20},"harness":{"max_turns":6}}}'
```

## The model TOMLs

| File | Family | Variants (uncomment one) | Default `max_tokens` | RL `batch_size` | Default env |
|---|---|---|---|---|---|
| `qwen-3-5.toml` | Qwen3.5 dense | `0.8B`, `2B`, `4B` *(active)*, `9B` | 1024 | 128 | `reverse-text` |
| `qwen-3-5-moe.toml` | Qwen3.5 MoE | `35B-A3B` *(active)*, `122B-A10B`, `397B-A17B` | 2048 | 256 | `wiki-search` |
| `llama-3.toml` | Llama 3.2 Instruct | `1B` *(active)*, `3B` | 1024 | 128 | `reverse-text` |
| `nemotron-3.toml` | NVIDIA Nemotron 3 | `Nano-30B-A3B` *(active)*, `Super-120B-A12B` | 2048 | 256 | `wiki-search` |
| `gpt-oss.toml` | OpenAI gpt-oss | `20b` *(active)*, `120b` | 1024 | 128 | `reverse-text` |

Notes that vary by family beyond the table:

- **`gpt-oss` GEPA and RL** set `reasoning_effort = "low"` under `[sampling]`. Raise it for harder tool/reasoning tasks at the cost of tokens.
- **`gepa/*`** files duplicate the model into `reflection_model` so the reflection LLM matches by default. Keep them in sync when switching sizes, or point `reflection_model` at a stronger model for better mutation quality.

## How to pick one

1. **Pick the family by scale and task shape.** Small dense (`qwen-3-5`, `llama-3`) and `gpt-oss-20b` are the right default for quick iteration, format-following tasks, and the smallest training runs, so `reverse-text` is preselected. MoE / larger reasoning models (`qwen-3-5-moe`, `nemotron-3`) are preselected on `wiki-search` because they handle multi-turn tool use and longer contexts better; 2048 `max_tokens` reflects that.
2. **Pick the size within the family by uncommenting one `model` line.** For `gepa/*`, update both `model` and `reflection_model` together.
3. **Pick the environment by uncommenting one `[[env]]` / `[[eval]]` block.** The preselected env matches what the family runs well; swap it if you have a specific target.
4. **Tune the loop knobs only after a smoke run works.** `max_tokens` controls per-rollout budget; RL `batch_size` and `rollouts_per_example` control sample throughput; GEPA max_calls / num_train / num_val / minibatch_size control optimizer cost. Use max_concurrent to cap parallel GEPA calls.

For RL-specific extensions — eval/val schedules, online difficulty filtering, oversampling, multi-env ratios — see the `train-with-environments` skill and the Hosted Training docs at <https://docs.primeintellect.ai/hosted-training>.

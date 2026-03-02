# Training Config Reference

All training is launched via:
```bash
prime rl run path/to/config.toml
```

---

## Minimal Config

```toml
[model]
name = "Qwen/Qwen2.5-1.5B-Instruct"

[training]
max_steps = 1000

[[env]]
id = "recipe-math-rl"
weight = 1.0
```

---

## Full Reference

```toml
# ── Model ──────────────────────────────────────────────────────────────────
[model]
name = "Qwen/Qwen2.5-7B-Instruct"   # HuggingFace model ID or local path
# revision = "main"                 # git ref (optional)
# dtype = "bfloat16"                # bfloat16 | float16 | float32

# ── Training ───────────────────────────────────────────────────────────────
[training]
max_steps = 2000
batch_size = 128                     # total tokens per optimizer step
rollouts_per_example = 8             # G in GRPO; higher = better advantage estimate
learning_rate = 1e-5
warmup_steps = 50

# ── Sampling ───────────────────────────────────────────────────────────────
[sampling]
max_tokens = 1024                    # max new tokens per turn
temperature = 1.0                    # higher = more diverse rollouts
top_p = 1.0

# ── Checkpointing ──────────────────────────────────────────────────────────
[checkpoint]
save_every = 100                     # steps between saves
output_dir = "checkpoints/"

# ── W&B ────────────────────────────────────────────────────────────────────
[wandb]
project = "prime-cookbook"
name = "math-rl-7b"
# offline = true                    # use if no W&B API key

# ── Environments ───────────────────────────────────────────────────────────
[[env]]
id = "recipe-math-rl"
weight = 1.0

[[env]]                              # add more for multi-task
id = "recipe-document-search-l1"
weight = 2.0
```

---

## Batch Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `batch_size` | Token budget per optimizer step | 128 | Increase for larger models |
| `rollouts_per_example` | Completions per prompt (GRPO G) | 8 | 4–16 range; higher = better advantage estimate, more compute |
| `max_tokens` | Max new tokens per turn | 1024 | Decrease for faster rollouts, increase for reasoning tasks |

**Rollouts per example trade-off:**
- `rollouts_per_example=4` — fast, noisy advantage estimates
- `rollouts_per_example=8` — good default
- `rollouts_per_example=16` — slow but strong, especially for sparse rewards

---

## Passing Secrets

Never put API keys in config files. Use `--env-var`:

```bash
prime rl run config.toml \
  --env-var OPENAI_API_KEY=$OPENAI_API_KEY \
  --env-var PRIME_API_KEY=$PRIME_API_KEY \
  --env-var MY_CUSTOM_KEY=value
```

Environment variables are forwarded to all environment processes.

---

## CLI Overrides

Any config value can be overridden from the CLI using dot-notation:

```bash
prime rl run config.toml \
  --model.name "Qwen/Qwen2.5-3B-Instruct" \
  --training.max_steps 500 \
  --training.learning_rate 5e-6 \
  --sampling.temperature 0.8
```

Useful for sweeps without editing files.

---

## Model Calibration

Match task difficulty to model size. A 1B model should not start on L3 tasks.

| Model Size | Recommended Tasks | `rollouts_per_example` | `max_tokens` |
|------------|------------------|------------------------|--------------|
| 0.5B–1.5B | L1 only (exact match, simple math) | 8–16 | 512 |
| 3B–7B | L1 + L2, simple tool use | 8 | 1024 |
| 7B–14B | L1/L2/L3, multi-turn tool use | 4–8 | 2048 |
| 30B–72B | Full curriculum, complex agents | 4 | 4096 |

---

## Per-Recipe Configs

### math_rl — Quick sanity check
```toml
[model]
name = "Qwen/Qwen2.5-1.5B-Instruct"

[training]
max_steps = 200
batch_size = 64
rollouts_per_example = 8

[sampling]
max_tokens = 512

[[env]]
id = "recipe-math-rl"
weight = 1.0
```

### document_search — 3-level curriculum
```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[training]
max_steps = 3000
batch_size = 128
rollouts_per_example = 8
learning_rate = 5e-6

[sampling]
max_tokens = 2048
temperature = 0.9

[[env]]
id = "recipe-document-search-l1"
weight = 3.0

[[env]]
id = "recipe-document-search-l2"
weight = 2.0

[[env]]
id = "recipe-document-search-l3"
weight = 1.0
```

### sandbox_code — Code execution
```toml
[model]
name = "Qwen/Qwen2.5-Coder-7B-Instruct"

[training]
max_steps = 2000
batch_size = 64
rollouts_per_example = 4   # sandboxes are slow, keep low

[sampling]
max_tokens = 4096
temperature = 0.8

[[env]]
id = "recipe-sandbox-code"
weight = 1.0
```

### multi_env — Multi-task
```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[training]
max_steps = 5000
batch_size = 128
rollouts_per_example = 8

[[env]]
id = "recipe-math-rl"
weight = 1.0

[[env]]
id = "recipe-tool-use"
weight = 1.5

[[env]]
id = "recipe-document-search-l1"
weight = 2.0
```

---

## Related Docs

- [Getting Started](getting-started.md)
- [Eval Guide](eval-guide.md)
- [Environment Types](environment-types.md)

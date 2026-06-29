# Training with RL

Training consumes the same v1 environment definition as eval. In a training config, embed the env under `[[orchestrator.train.env]]`:

```toml
model = "meta-llama/Llama-3.2-1B-Instruct"
max_steps = 100
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 1024

[[orchestrator.train.env]]
name = "reverse-text"
taskset = { id = "reverse-text" }
harness = { id = "default" }
```

Use the same taskset id and typed config fields you used for eval. The training stack serves the env over the v1 env-server path, so the taskset/harness/runtime split is unchanged.

## Multi-env Training

Add more `[[orchestrator.train.env]]` entries when you intentionally train across tasks:

```toml
[[orchestrator.train.env]]
name = "reverse-text"
taskset = { id = "reverse-text" }
harness = { id = "default" }

[[orchestrator.train.env]]
name = "wordle"
max_turns = 6
taskset = { id = "wordle", num_tasks = 512 }
harness = { id = "default" }
```

Keep environment knobs on the embedded env config. Keep optimizer and trainer knobs at the training config level.

# Warm Starts with SFT

Use SFT to give a model a better starting policy before further RL or as a standalone warm start.

SFT is useful when the model needs examples of the right behavior before reward optimization becomes efficient. Instead of bringing a separate dataset, Lab can generate demonstrations from an environment using a stronger teacher model, then train the target model on those demonstrations.

## When to Use It

Use SFT when:

- the environment is correct, but the base model rarely gets useful reward
- the model needs to learn an interaction pattern before RL can improve it
- you want to inspect teacher-generated demonstrations before training with rewards

Do not use SFT to paper over broken rewards. If the environment gives high scores to bad behavior, SFT will not fix the underlying training signal.

## Write an SFT Config

Set `loss = "sft"`, point at a teacher model, and reuse the wordle taskset/harness overrides from earlier guides:

```toml
# [configs/05/wordle-sft.toml](../../configs/05/wordle-sft.toml)
max_steps = 100
batch_size = 32
rollouts_per_example = 4
model = "openai/gpt-oss-20b"
loss = "sft"

[sampling]
max_tokens = 2048
reasoning_effort = "medium"

[teacher]
model = "openai/gpt-oss-120b"

[teacher.sampling]
max_tokens = 2048
reasoning_effort = "medium"

[[env]]
id = "primeintellect/wordle"

[env.taskset]
num_train_examples = 512
num_eval_examples = 128

[env.harness]
max_turns = 6

[eval]
interval = 5
num_examples = 32
rollouts_per_example = 1
eval_base_model = true

[[eval.env]]
id = "primeintellect/wordle"
```

The fields specific to SFT are:

- `loss = "sft"` switches the run from RL to SFT.
- `[teacher].model` selects the model that generates demonstrations.
- `[teacher.sampling]` controls teacher generation — including reasoning controls (`reasoning_effort` for gpt-oss, `enable_thinking` for Qwen/Nemotron). If the teacher emits empty completions, raise `max_tokens` or turn reasoning off so it finishes before hitting the budget.

Everything else mirrors an RL config: `[[env]]`, `[env.taskset]`, `[env.harness]`, and `[eval]` work the same way.

## Launch Training

Start the run:

```bash
prime train configs/05/wordle-sft.toml
```

The command prints a run ID and the log-streaming command. Follow logs with:

```bash
prime train logs <run_id> -f
```

Watch the first few rollouts for empty teacher completions — a sign that the teacher is hitting the token budget mid-reasoning and producing nothing usable. If that happens, raise `[teacher.sampling].max_tokens` or set the matching reasoning control off, then restart.

## After SFT

Once the run finishes, you have two next steps:

- Continue with on-policy distillation against the SFT adapter — see [On-Policy Distillation](../06-on-policy-distillation/README.md).
- Skip distillation and run RL directly on the adapter with [configs/05/wordle-rl.toml](../../configs/05/wordle-rl.toml):

```toml
# [configs/05/wordle-rl.toml](../../configs/05/wordle-rl.toml)
model = "openai/gpt-oss-20b:my-sft-lora-distilled-from-oss-120b-distill"

[sampling]
max_tokens = 512
reasoning_effort = "medium"

[[env]]
id = "primeintellect/wordle"

[env.taskset]
num_train_examples = 512
num_eval_examples = 128

[env.harness]
max_turns = 6
```

```bash
prime train configs/05/wordle-rl.toml
```

## Next

In [On-Policy Distillation](../06-on-policy-distillation/README.md), you will refine the SFT adapter with dense teacher feedback on the student's own rollouts.

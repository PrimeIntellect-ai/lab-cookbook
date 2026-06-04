# Warm Starts with SFT

**SFT in Lab is currently in beta and will roll out to all users shortly.**

Use SFT to give a model a better starting policy before further RL or as a standalone warm start.

SFT is useful when the model needs examples of the right behavior before reward optimization becomes efficient. Instead of bringing a separate dataset, Lab can generate demonstrations from an environment using a stronger teacher model, then train the target model on those demonstrations.

## When to Use It

Use SFT when:

- the environment is correct, but the base model rarely gets useful reward
- the model needs to learn an interaction pattern before RL can improve it
- you want to inspect teacher-generated demonstrations before training with rewards

Do not use SFT to paper over broken rewards. If the environment gives high scores to bad behavior, SFT will not fix the underlying training signal.

## Write an SFT Config

Configure the `loss` field to `"sft"` and choose a teacher model:

```toml
# [configs/05/wordle-sft.toml](../../configs/05/wordle-sft.toml)
model = "openai/gpt-oss-20b"
loss = "sft"

[teacher]
model = "openai/gpt-oss-120b"

[teacher.sampling]
max_tokens = 2048
reasoning_effort = "medium"

[[env]]
id = "prime/wordle"

[env.args.taskset]
num_train_examples = 512
num_eval_examples = 128

[env.args.harness]
max_turns = 6
```

This config trains `openai/gpt-oss-20b` with SFT on demonstrations generated from `prime/wordle`.

The key fields are:

- loss = "sft" switches the run from RL to SFT.
- `[teacher].model` selects the model that generates demonstrations.
- `[teacher.sampling]` controls teacher generation.
- `[[env]]` points to the environment that defines tasks and scoring.
- `[env.args.taskset]` and `[env.args.harness]` tune the same environment
  components you used during eval.

## Launch Training

Start the run:

```bash
prime train configs/05/wordle-sft.toml
```

The command prints a run ID along with the command for streaming logs from the new Hosted Training run. Follow logs with:

```bash
prime train logs <run_id> -f
```

After SFT, evaluate the adapter on the same environment. For on-policy distillation from the SFT adapter, see [On-Policy Distillation](../06-on-policy-distillation/README.md). To skip straight to RL, use [configs/05/wordle-rl.toml](../../configs/05/wordle-rl.toml):

```toml
# [configs/05/wordle-rl.toml](../../configs/05/wordle-rl.toml)
model = "openai/gpt-oss-20b:my-sft-lora-distilled-from-oss-120b-distill"

[sampling]
max_tokens = 512
reasoning_effort = "medium"

[[env]]
id = "prime/wordle"

[env.args.taskset]
num_train_examples = 512
num_eval_examples = 128

[env.args.harness]
max_turns = 6
```

```bash
prime train configs/05/wordle-rl.toml
```

## Next

In [On-Policy Distillation](../06-on-policy-distillation/README.md), you will refine the SFT adapter with dense teacher feedback on the student's own rollouts.

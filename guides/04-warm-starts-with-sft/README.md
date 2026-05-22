# Warm Starts with SFT

**SFT in Lab is currently in beta and will roll out to all users shortly.**

Use SFT to give a model a better starting policy before RL.

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
# [configs/04/sft-wordle.toml](../../configs/04/sft-wordle.toml)
model = "openai/gpt-oss-20b"
loss = "sft"

[teacher]
model = "openai/gpt-oss-120b"

[teacher.sampling]
max_tokens = 2048
reasoning_effort = "medium"

[[env]]
id = "prime/wordle"
```

This config trains `openai/gpt-oss-20b` with SFT on demonstrations generated from `prime/wordle`.

The key fields are:

- loss = "sft" switches the run from RL to SFT.
- `[teacher].model` selects the model that generates demonstrations.
- [teacher].replay controls whether the run reuses saved demonstrations instead of generating fresh ones.
- `[teacher.sampling]` controls teacher generation.
- `[[env]]` points to the environment that defines tasks and scoring.

## Launch Training

Start the run:

```bash
prime train configs/04/sft-wordle.toml
```

The command prints a run ID along with the command for streaming logs from the new Hosted Training run. Follow logs with:

```bash
prime train logs <run_id> -f
```

## Inspect the Warm Start

For an SFT warm start, inspect both the generated demonstrations and the trained model's rollouts. The run is doing useful work if the target model starts to copy the teacher's task strategy, answer format, and interaction pattern.

After SFT, evaluate the adapter against the same environment. If it improves baseline behavior without breaking the reward contract, use the checkpoint as the starting point for an RL run.

## Continue with RL

RL uses the same config shape. To train on top of an existing LoRA, set `model` to the adapter you want to start from and omit `loss`, since RL is the default.

Use [configs/04/wordle-from-sft.toml](../../configs/04/wordle-from-sft.toml):

```toml
# [configs/04/wordle-from-sft.toml](../../configs/04/wordle-from-sft.toml)
model = "openai/gpt-oss-20b:my-sft-lora-distilled-from-oss-120b-distill"

[sampling]
max_tokens = 512
reasoning_effort = "medium"

[[env]]
id = "prime/wordle"
```

```bash
prime train configs/04/wordle-from-sft.toml
```

The run should behave like a normal RL training job, but with the SFT adapter as the starting model.

This keeps the environment fixed while changing the starting policy: SFT teaches the model what good behavior looks like, and RL optimizes that behavior against the environment reward.

## Next

In [Prompt Optimization](../05-prompt-optimization/README.md), you will improve an environment prompt without changing model weights.

# Warm Starts with SFT

Use SFT<a href="../../reference/glossary.md#sft">¹</a> to give a model a better starting policy before RL.

SFT is useful when the model needs examples of the right behavior before reward optimization becomes efficient. Instead of bringing a separate dataset, Lab can generate demonstrations<a href="../../reference/glossary.md#demonstration">²</a> from an environment using a stronger teacher model<a href="../../reference/glossary.md#teacher-model">³</a>, then train the target model on those demonstrations.

SFT in Lab is currently in beta and will roll out to all users shortly. If `loss = "sft"` is not enabled for your account yet, the config shape below is still the intended public flow.

## When to Use It

Use SFT when:

- the environment is correct, but the base model rarely gets useful reward
- the model needs to learn an interaction pattern before RL can improve it
- you want to inspect teacher-generated demonstrations before training with rewards

Do not use SFT to paper over broken rewards. If the environment gives high scores to bad behavior, SFT will not fix the underlying training signal.

## Write an SFT Config

Create `configs/sft-oss-20b.toml`:

```toml
model = "openai/gpt-oss-20b"
loss = "sft" # default = "rl"

[teacher]
model = "openai/gpt-oss-120b"
save = true
replay = false

[teacher.sampling]
max_tokens = 2048
reasoning_effort = "medium"

[[env]]
id = "primeintellect/wordle"
```

This config trains `openai/gpt-oss-20b` with SFT on demonstrations generated from `primeintellect/wordle`.

The key fields are:

- `loss = "sft"`<a href="../../reference/glossary.md#loss">⁴</a> switches the run from RL to SFT.
- `[teacher].model` selects the model that generates demonstrations.
- `[teacher].save` keeps generated demonstrations available for inspection and reuse.
- `[teacher].replay`<a href="../../reference/glossary.md#replay">⁵</a> controls whether the run reuses saved demonstrations instead of generating fresh ones.
- `[teacher.sampling]` controls teacher generation.
- `[[env]]` points to the environment that defines tasks and scoring.

## Launch Training

Start the run:

```bash
prime train configs/sft-oss-20b.toml
```

When SFT is enabled for your account, the command starts a Hosted Training run and prints a run id plus a log-streaming command.

Follow logs:

```bash
prime train logs <run_id> -f
```

The logs should show teacher demonstration generation before the target model trains on those demonstrations.

Open the Lab viewer:

```bash
prime lab view --training
```

This opens the Hosted Training view in Lab.

## Inspect the Warm Start

For an SFT warm start<a href="../../reference/glossary.md#warm-start">⁶</a>, inspect both the generated demonstrations and the trained model's rollouts. The run is doing useful work if the target model starts to copy the teacher's task strategy, answer format, and interaction pattern.

After SFT, evaluate the adapter<a href="../../reference/glossary.md#adapter">⁷</a> against the same environment. If it improves baseline behavior without breaking the reward contract, use the checkpoint as the starting point for an RL run.

## Continue with RL

RL uses the same config shape. To train on top of an existing LoRA<a href="../../reference/glossary.md#lora">⁸</a>, set `model` to the adapter you want to start from and omit `loss`, since RL is the default.

Create `configs/rl/wordle-from-sft.toml`:

```toml
model = "openai/gpt-oss-20b:my-sft-lora-distilled-from-oss-120b-distill"

[sampling]
max_tokens = 512
reasoning_effort = "medium"

[[env]]
id = "primeintellect/wordle"
```

Then launch RL from that adapter:

```bash
prime train configs/rl/wordle-from-sft.toml
```

The run should behave like a normal RL training job, but with the SFT adapter as the starting model.

This keeps the environment fixed while changing the starting policy: SFT teaches the model what good behavior looks like, and RL optimizes that behavior against the environment reward.

## Next

In [Prompt Optimization](../05-prompt-optimization/README.md), you will improve an environment prompt without changing model weights.

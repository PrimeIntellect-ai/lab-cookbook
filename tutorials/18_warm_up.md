# Recipe — SFT Warm-up, then RL

[Tutorial 3](3_first_rl.md) had a rule: RL needs *mixed* rewards. If the base model fails a task every single time, every group of rollouts scores all-zero, there's no better-than-average behavior to reinforce, and the run burns GPU-hours learning nothing. This is the **cold-start problem**, and it's the standard failure mode of pointing RL at a task genuinely beyond the model.

The standard cure is a two-stage pipeline: first **SFT warm-up** — supervised fine-tuning on demonstrations from a stronger teacher model, which lifts the student from *never succeeds* to *sometimes succeeds* — then **RL from that checkpoint**, which takes over exactly where imitation stops helping. In this recipe you'll run both stages on Hosted Training, from two small TOML files.

**You need:** tutorials [1](1_setup.md)–[3](3_first_rl.md), and training credits for two runs.

## Why two stages — the intuition

The two methods teach differently, and the difference is exactly complementary:

- **SFT is imitation.** A teacher model rolls out in the environment; the student is trained to reproduce the teacher's behavior, token by token. It's sample-efficient and works from zero — but its ceiling is the teacher: imitation can't exceed what it imitates.
- **RL is trial and error.** The student's *own* attempts are scored, and above-average behavior is reinforced. No ceiling but the reward — yet it needs those attempts to sometimes succeed, or there's no gradient at all. No floor support.

SFT provides the floor, RL removes the ceiling. Neither alone gets a weak model to strong performance on a hard task.

## Step 0: confirm you actually have a cold start

Don't run the pipeline on reflex — it's two runs' worth of credits, and if the base model already gets, say, 15% with partial credit, plain RL is cheaper and fine. Check first:

```bash
prime eval run primeintellect/alphabet-sort-v1 -n 20 -r 4 --model Qwen/Qwen3.5-0.8B
```

Our task is `alphabet-sort` — sort a list of words alphabetically, a multi-turn task that's trivially verifiable but reliably brutal for a 0.8B model. Read the reward distribution, not just the mean: **lots of zeros with a few non-zero stragglers → plain RL will limp but work; essentially all zeros → cold start confirmed**, warm-up earns its cost. (If your rollouts error instead of scoring zero, fix that first — that's infrastructure, not difficulty.)

## Stage 1: SFT warm-up

Generate a config and switch it to SFT mode:

```bash
prime rl init primeintellect/alphabet-sort-v1
```

```toml
# warmup-sft.toml
model = "Qwen/Qwen3.5-0.8B"
loss = "sft"                        # <- stage 1 is supervised
max_steps = 60
batch_size = 128

[teacher]                           # who demonstrates
model = "openai/gpt-oss-120b"

[teacher.sampling]
max_tokens = 2048
reasoning_effort = "medium"

[sampling]
max_tokens = 2048

[[env]]
name = "alphabet-sort"
taskset = { id = "primeintellect/alphabet-sort-v1" }
harness = { id = "default" }

[eval]                              # measure every run, always
interval = 20

[[eval.env]]
name = "alphabet-sort-eval"
taskset = { id = "primeintellect/alphabet-sort-v1" }
harness = { id = "default" }
num_examples = 50
```

The moving parts:

- **`loss = "sft"`** flips the trainer from reinforcement to supervised distillation.
- **`[teacher]`** names the demonstrator — a much larger model, called through Prime Inference by default. The teacher rolls out **in the same environment**, which is the quiet advantage of doing SFT here rather than on a static demo dataset: demonstrations are on *your* task distribution, in *your* format, scored by *your* reward.
- **The `[eval]` block** gives you the base model's score at step 0 (your baseline, for free) and the student's progress every 20 steps.

Launch and watch, exactly as in tutorial 3:

```bash
prime rl run warmup-sft.toml
prime rl logs <run-id>
prime rl metrics <run-id>
```

Success for this stage is *not* mastery. You're watching the eval score move off the floor — from ~0 to even 15–30%. That's the mixed-reward regime RL needs; that's the whole job of stage 1. When the run finishes, grab its checkpoint:

```bash
prime rl checkpoints <run-id>
```

## Stage 2: RL from the warm checkpoint

Second config — same environment, two changes: the loss, and where the weights start:

```toml
# warmup-rl.toml
model = "Qwen/Qwen3.5-0.8B"
loss = "rl"                          # <- stage 2 reinforces
checkpoint_id = "<from stage 1>"     # <- start from the warmed-up student
max_steps = 150
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 2048
temperature = 1.0                    # RL wants exploration — don't sample greedily

[[env]]
name = "alphabet-sort"
taskset = { id = "primeintellect/alphabet-sort-v1" }
harness = { id = "default" }

[eval]
interval = 25

[[eval.env]]
name = "alphabet-sort-eval"
taskset = { id = "primeintellect/alphabet-sort-v1" }
harness = { id = "default" }
num_examples = 50
```

```bash
prime rl run warmup-rl.toml
```

Now the dynamics to watch (`prime rl metrics`, `prime rl distributions`):

- **Step-0 eval ≈ stage 1's final eval** — confirmation the checkpoint actually carried over.
- **Groups are mixed from the very first step** — `prime rl distributions` shows rewards spread within groups instead of the all-zero wall the base model would have produced. That spread *is* the training signal you bought with stage 1.
- **The score climbs past the SFT plateau** — RL reinforcing the student's own successes, including strategies the teacher never demonstrated. This is the "ceiling removed" half of the bargain.

## The receipts

The pipeline hands you a four-point story, all measured by the same environment on the same held-out tasks:

| Measurement | Where it came from |
| --- | --- |
| Base model | stage 1's step-0 eval |
| After SFT | stage 1's final eval |
| RL start (sanity check) | stage 2's step-0 eval |
| After RL | stage 2's final eval |

If you want the strongest version of the claim, add the control: a plain-RL run from the base model (stage 2's config minus `checkpoint_id`). On a true cold start its curve stays flat while the warmed run climbs — the cleanest possible picture of what the warm-up bought.

## Things to try

- **Vary the teacher.** Re-run stage 1 with a weaker teacher: the student's SFT plateau tracks teacher quality, but does the *RL endpoint* care as much? (Often less than you'd expect — RL only needed the floor.)
- **Shorten stage 1.** 60 steps → 20: how little imitation is enough to unlock RL? The cheapest sufficient warm-up is the one you want.
- **Watch for imitation ceilings.** If stage 2 barely improves on stage 1, your task may be *solved by imitation* — a sign the task is too easy to need RL at all, or the reward has no headroom above "format it like the teacher".

## Recap

Cold start = all-zero groups = no RL signal. Diagnose it with a cheap eval before spending; fix it with `loss = "sft"` plus a `[teacher]` that demonstrates in the same environment; then `loss = "rl"` with `checkpoint_id` picking up the warmed weights. SFT buys the floor, RL removes the ceiling, and the periodic evals in both runs give you the before/during/after numbers to prove each stage paid for itself.

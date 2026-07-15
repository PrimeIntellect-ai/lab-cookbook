# Recipe — Generalist Training

Every training run so far taught one model one task. But specialize a small model on text reversal and you've built a text-reverser, not a better model — and quite possibly made it *worse* at everything else. In this recipe you'll train on a **mix** of environments spanning three different capabilities — math, a multi-turn game, and tool use — and, crucially, measure the thing single-env training never has to answer for: **transfer**. Did the model get better at tasks it never trained on?

**You need:** tutorials [1](1_setup.md)–[3](3_first_rl.md); [Search Agent](17_search_agent.md) helps for the tool-use leg. This is the most expensive recipe in the set — it's a real multi-task training run — so smoke-test small.

## The mix

Three environments, three genuinely different behaviors:

| Leg | Environment | Capability it exercises |
| --- | --- | --- |
| Math | `primeintellect/gsm8k_v1` | Single-turn chain-of-thought reasoning to a verifiable answer. |
| Game | `primeintellect/wordle_v1` | Multi-turn state tracking — act, read feedback, adapt. |
| Tools | `primeintellect/wiki_search_v1` | Deciding when and how to call tools, grounding answers in results. |

The premise of generalist training is that these share underlying skills — careful reading, working memory, checking before committing — so practicing all three should beat practicing any one, *even on that one's own turf*, and should move tasks none of them cover. That's a hypothesis, not a law. This recipe is how you test it rather than assume it.

## The config

Multi-env training is just multiple `[[env]]` blocks. The interesting decisions are annotated:

```toml
# generalist.toml
model = "Qwen/Qwen3.5-0.8B"
loss = "rl"
max_steps = 200
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 2048
temperature = 1.0

[[env]]
name = "math"                # every env needs a unique name
ratio = 2                    # sampling weight — see below
taskset = { id = "primeintellect/gsm8k_v1", split = "train" }
harness = { id = "default" }

[[env]]
name = "game"
ratio = 1
max_turns = 6                # per-env knobs live on the env entry
taskset = { id = "primeintellect/wordle_v1" }
harness = { id = "default" }

[[env]]
name = "tools"
ratio = 1
max_turns = 8
taskset = { id = "primeintellect/wiki_search_v1", max_examples = 512, tools = { shared = true } }
harness = { id = "default" }
```

Three decisions worth dwelling on:

**Ratios are the diet.** `ratio` sets relative sampling: 2:1:1 here means half the training tasks are math. Ratios are how you feed weaker areas more practice — but set them *once you have evidence*, not by vibes; the per-env reward curves (below) are that evidence.

**Reward scales must be comparable.** The trainer optimizes one blended signal, so an environment whose rewards run 0–10 quietly dominates ones scoring 0–1 regardless of your carefully chosen ratios. All three envs here score in [0, 1]; when you build your own mix, check each env's reward range *first* — this is the classic silent failure of multi-env training.

**Difficulty should overlap the model.** A leg the model scores ~0% on contributes almost nothing (all-zero groups — the [warm-up recipe](18_warm_up.md)'s problem, now per-leg); a leg at ~100% contributes nothing either. Run the cheap per-leg baseline evals before training and swap or re-tier any leg outside roughly 10–90%.

## The transfer probe — decide it before you train

Here's the part that makes this an experiment instead of a vibe: pick your held-out environments **now**, before training, and wire them into the run's `[eval]` block. Two kinds:

```toml
[eval]
interval = 25

# in-distribution: held-out data from the training tasks
[[eval.env]]
name = "math-test"
taskset = { id = "primeintellect/gsm8k_v1", split = "test" }
harness = { id = "default" }
num_examples = 100

[[eval.env]]
name = "tools-test"
taskset = { id = "primeintellect/wiki_search_v1" }
harness = { id = "default" }
num_examples = 50

# TRANSFER: never trained on, chosen to probe each capability
[[eval.env]]
name = "transfer-math500"               # harder math than gsm8k
taskset = { id = "primeintellect/math500_v1" }
harness = { id = "default" }
num_examples = 100

[[eval.env]]
name = "transfer-alphabet-sort"         # multi-turn procedure, not wordle
taskset = { id = "primeintellect/alphabet-sort-v1" }
harness = { id = "default" }
num_examples = 50

[[eval.env]]
name = "transfer-unscramble"            # string manipulation none of the legs teach
taskset = { id = "primeintellect/unscramble_v1" }
harness = { id = "default" }
num_examples = 50
```

The transfer set is chosen with intent: `math500` asks whether gsm8k practice generalizes *up* in difficulty; `alphabet-sort` asks whether Wordle's multi-turn discipline carries to a different procedure; `unscramble` is the long shot — nothing in the mix teaches it directly, so movement there is the strongest generalization evidence. Every one gets a step-0 measurement automatically, which is your baseline.

Launch:

```bash
prime rl run generalist.toml
```

## Reading a multi-env run

The aggregate reward curve is nearly meaningless here — it blends three tasks and the diet you chose. Go per-environment from the start:

```bash
prime rl metrics <run-id>          # per-env training reward curves
prime rl distributions <run-id>    # per-env group spreads
prime rl rollouts <run-id>         # spot-check actual behavior per leg
```

The patterns that matter:

- **All legs climbing** — the healthy case; the capabilities are at least not fighting.
- **One leg flat at ~0** — it's too hard for this model; it contributes no signal and just dilutes the batch. Drop it or feed easier tiers.
- **One leg saturating early** — it's done teaching; its continued share of the batch is mostly wasted. This is when adjusting `ratio` for a follow-up run is justified — by evidence.
- **One leg climbing while another *declines*** — genuine task interference. Small models have limited capacity; this is the signal to rebalance ratios, or accept the trade-off consciously.

Then, at the end, the question you set this all up to answer — the transfer table:

| Eval env | Step 0 | Final | Trained on? |
| --- | --- | --- | --- |
| gsm8k (test) | — | — | yes |
| wiki-search | — | — | yes |
| math500 | — | — | **no** |
| alphabet-sort | — | — | **no** |
| unscramble | — | — | **no** |

Gains on the top rows say training worked. Gains on the bottom rows say something *general* improved. And if you want the comparison that settles arguments, run the control: a single-env run (math only, same total budget) and compare its transfer row against the mix's. That pair of runs is the difference between "we believe in task diversity" and "task diversity gained us 6 points on held-out environments, here's the table."

## Things to try

- **Reweigh from evidence:** take your first run's per-leg curves, adjust ratios (starve the saturated leg, feed the struggling one), re-run, compare transfer tables.
- **Add a fourth leg** from the Hub (`prime env list`) — instruction following (`ifeval_v1`) is a natural complement. Watch whether it lifts or drags the others.
- **Probe forgetting:** add an eval env the *base* model was already decent at. Generalist training that improves the transfer set while tanking prior abilities is a different trade than the table above admits.
- **Combine recipes:** infinite tasksets with per-leg difficulty tiers ([Infinite Tasksets](14_infinite_tasksets.md)) turn the static diet into a curriculum.

## Recap

A generalist run is multiple `[[env]]` blocks with unique names, ratios as the diet, comparable reward scales as the non-negotiable, and per-leg difficulty that overlaps the model's ability. Judge it per-environment, never by the blended curve — and define the real success metric before training: held-out environments, measured at step 0 and at the end, with a single-env control if you want the claim to stick.

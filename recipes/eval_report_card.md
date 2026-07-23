# A Model Report Card

A new model drops. The launch post quotes three benchmarks; your use case appears in none of them. In this recipe you'll build a **report card**: a small battery of evaluations across distinct capabilities, run with settings you chose deliberately, producing numbers you can defend when someone asks *"how did you get that?"*

**You need:** tutorials [1](../tutorials/1_setup.md)–[2](../tutorials/2_first_eval.md), and inference credits proportional to your ambition (the smoke-scale version costs little; full runs cost real money — estimates below on how to stage it).

## Pick the battery

A report card should test *different* capabilities, not the same one five times. A solid default, all from the Environment Hub:

| Capability | Environment | What it measures |
| --- | --- | --- |
| Math reasoning | `primeintellect/aime25_v1` | Competition math — hard, verifiable answers. |
| Instruction following | `primeintellect/ifeval_v1` | Does it obey precise, checkable constraints ("exactly three bullet points…")? |
| Factuality | `primeintellect/simpleqa_verified_v1` | Short factual questions — measures knowledge *and* the tendency to guess. |
| Agentic / terminal | `primeintellect/terminal-bench-2-v1` | Multi-step tasks in a real container, with a harness. |

Swap rows for what *you* care about (browse `prime env list`) — the recipe is the same. Keep the battery small enough that you'll actually re-run it on the next model; four or five well-chosen environments beat fifteen you run once.

## Get the sampling right — before anything else

This is the step that separates a defensible report card from a random number generator. Models are trained to be sampled a certain way: temperature, top-p, reasoning effort. Evaluating an open model at the wrong temperature can swing scores by double digits, and the *model* will take the blame.

Look up the recommended parameters (for open models: the Hugging Face model card and its `generation_config.json`; for API models: provider docs) and write them down in a per-model config, e.g. `report/nemotron.toml`:

```toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

[sampling]
temperature = 1.0
top_p = 0.95
max_tokens = 8192       # reasoning models need room — a tight cap silently truncates
```

Note what `max_tokens` is doing here: too small a cap doesn't make a reasoning model concise, it makes its answers *cut off mid-thought*, which scores as failure. Leave headroom.

## Run it

One config per environment, sharing the model block. The pattern, per environment:

```toml
# report/aime25.toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"
num_rollouts = 4        # pass@k needs k attempts — see below

[sampling]
temperature = 1.0
top_p = 0.95
max_tokens = 8192

[taskset]
id = "primeintellect/aime25_v1"
```

Stage the spend:

```bash
# 1. free: every config resolves
for f in report/*.toml; do prime eval run @ $f --dry-run; done

# 2. cheap: smoke run, 5 tasks each — catches broken plumbing before it's expensive
for f in report/*.toml; do prime eval run @ $f -n 5; done

# 3. real: full runs, on platform infrastructure
for f in report/*.toml; do prime eval run @ $f --hosted; done
```

Notes on the real runs:

- **Don't set `num_tasks`** — the full taskset is the benchmark; a subset is a different (and incomparable) number.
- **`num_rollouts = 4`** on small, high-variance sets like AIME (30 problems) — report the mean over attempts, or pass@4 if that's your use case; a single rollout on 30 tasks is a coin flip. On large sets (IFEval, SimpleQA), `num_rollouts = 1` is fine.
- **The agentic row needs a harness** — add `[harness] id = "codex"` (pinned `version = ...`) plus a runtime to the terminal-bench config, and keep it identical when you later re-run the battery on another model. Harness choice changes agentic scores as much as model choice; see [Build Your Own Coding-Agent Harness](coding_agent_harness.md).

## Grade honestly

Before quoting any mean, decompose the rollouts. Every trace ends in one of four ways, and averaging them together is how bad numbers get made:

1. **Valid completion, correct** — genuine success.
2. **Valid completion, wrong** — genuine failure. These two are the actual measurement.
3. **Truncated** — hit the token cap mid-answer. This measured your `max_tokens`, not the model. If it's more than a couple percent, raise the cap and re-run.
4. **Errored** — provider timeouts, rate limits, harness crashes. Infrastructure, not capability. Re-run the gaps with `prime eval run --resume <output-dir>` (it re-does only missing/errored rollouts).

`prime eval view` makes the triage fast; the traces' stop conditions and error fields tell you which bucket each rollout is in. **Report the failure rate alongside the score** — "62% (3% of rollouts errored and were resumed)" is a number people can trust.

Then the report card itself is a table you can paste anywhere:

| Environment | Score | Attempts | Notes |
| --- | --- | --- | --- |
| aime25 | 40.0% | 30 tasks × 4 | temp 1.0, 8k tokens |
| ifeval | 71.3% | full × 1 | — |
| simpleqa-verified | 55.1% | full × 1 | — |
| terminal-bench-2 | 38.9% | full × 2 | codex 0.116.0, docker |

Two honesty rules for the prose around it: quote sample sizes, and resist explaining *why* the model scored what it did until you've read failing traces — the reason is usually visible there, and it's often not the reason you'd have guessed.

## Compare models

The second model is where the setup pays off: re-run the same configs with only the `model` line (and its sampling block!) changed. Same tasksets, same harness and version, same caps. Every delta in the table is now attributable to the model — which was the entire point.

Share it: hosted runs land in your dashboard, and `prime eval push` publishes local results so your team sees the same evidence you do.

## Things to try

- Add a row for *your* domain — a private env of your real tasks is worth more than every public benchmark combined. [Turn a dataset into one](../tutorials/5_build_first_environment.md) in an afternoon.
- Run the battery on the model you currently use in production. Having the incumbent's card ready is what makes the next launch-day evaluation take an hour instead of a week.
- For factuality sets, look at *wrong-vs-abstained* in the traces: two models with equal SimpleQA scores can differ enormously in how confidently they hallucinate.

## Recap

Pick a battery of distinct capabilities; fix sampling *per model, deliberately*; stage spend (dry-run → smoke → hosted full runs); decompose outcomes before averaging; report scores with sample sizes and failure rates. The result is a table you can re-run on every new model — and defend line by line.

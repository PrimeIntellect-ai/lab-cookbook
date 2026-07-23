# Train Your Coding Agent

[Your First RL Run](../tutorials/3_first_rl.md) trained a model on one tiny environment. This recipe scales that loop to the real thing: **multi-environment RL on software-engineering tasks** — the setup behind [Scaling Agentic RL](https://www.primeintellect.ai/blog/scaling-agentic-rl), where 23 tasksets and ~365,000 tasks train through one config because tasksets, harnesses, and runtimes are independent layers. You'll train on two SWE tasksets at once, hold a third out as the benchmark, and — before any GPU spends a cent — verify the rewards can be trusted.

**You need:** [Your First RL Run](../tutorials/3_first_rl.md), [Coding Agent Environments](../tutorials/11_coding_agents.md), and training credits. The three tasksets ship with this cookbook.

## The tasksets


| Role     | Taskset                                                      | Size  | What a task is                                                                                         |
| -------- | ------------------------------------------------------------ | ----- | ------------------------------------------------------------------------------------------------------ |
| Train    | `r2e_gym_v1` (`environments/r2e_gym_v1`)                     | ~4.5k | A real repo issue in a prebuilt per-task Docker image; hidden tests restored at scoring.               |
| Train    | `swelego_v1` (`environments/swelego_v1`)                     | ~4.3k | SWE-bench-style bug fixes; the test patch is re-applied at scoring, so the agent can't see the grader. |
| Held-out | `swebench_verified_v1` (`environments/swebench_verified_v1`) | 500   | SWE-bench Verified — the fixed benchmark. Never trained on; measured before and after.                 |


Three properties make these training-grade rather than merely benchmark-grade:

- **Per-row prebuilt images.** Every task names its own container (`namanjain12/orange3_final:2d9617…`); nothing builds at rollout time, so a training step is pull + run.
- **Hidden graders.** Fail-to-pass tests are removed from the container the agent works in and restored only at scoring — an agent can't pass by editing the tests.
- **Verified reward datasets.** Both training tasksets default to their `-Verified` dataset partitions, where every row has passed the validation gates below.



## Trust the reward before you train on it

In live-sandbox RL, a broken task doesn't just mismeasure — the trainer *optimizes into the breakage*. Two gates, run per task, keep a taskset honest:

- **Gold-patch validation:** apply the reference solution, run the verifier — the reward must be 1.0. Catches broken graders, missing test deps, wrong images.
- **No-op validation:** run the verifier on the untouched repo — the reward must be 0.0. Catches tasks that pass without any work, the reward-hacker's favorite food.

The bundled `r2e_gym_v1` and `swelego_v1` partitions are gold/no-op validated upstream; that's what `-Verified` means in their dataset names. When you bring your *own* taskset (or relax `dataset_name` to the raw partitions), run both gates yourself before training — the same discipline as [Coding Agent Environments](../tutorials/11_coding_agents.md)' advice to validate a known-good solution and a no-op before trusting a new Harbor verifier. A task that fails either gate is a task you delete, not a task you hope about.

## Baseline first

You can't measure improvement without a before-number. The held-out benchmark, evaluated with the *same harness you'll train with*:

```bash
uv run eval @ configs/recipes/swe-baseline-eval.toml          # 5-task local smoke (Docker)
prime eval run @ configs/recipes/swe-baseline-eval.toml --hosted   # the real 500-task number
```

The harness choice matters more here than anywhere: a model trained inside `mini_swe_agent`'s bash-loop learns *that scaffold's* moves ([Build Your Own Coding-Agent Harness](coding_agent_harness.md) showed how differently scaffolds behave). Keep harness and version identical across baseline, training, and final eval, or the delta stops being attributable to the weights.

## The training config

One config, two environments, one held-out eval (`configs/recipes/swe-rl.toml`):

```toml
model = "zai-org/GLM-4.5-Air"

max_steps = 200
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 8192

[[env]]
name = "r2e-gym"
max_turns = 50
taskset = { id = "r2e_gym_v1" }
harness = { id = "mini_swe_agent", runtime = { type = "prime" } }

[[env]]
name = "swe-lego"
max_turns = 50
taskset = { id = "swelego_v1" }
harness = { id = "mini_swe_agent", runtime = { type = "prime" } }

[eval]
interval = 20

[[eval.env]]
name = "swebench-verified"
num_examples = 50
rollouts_per_example = 1
taskset = { id = "swebench_verified_v1" }
harness = { id = "mini_swe_agent", runtime = { type = "prime" } }

[[pre_batch_filters]]
type = "zero_advantage"
enforce = true
```

```bash
prime train configs/recipes/swe-rl.toml
```

Reading it top-down:

- **Multi-env is just repeated** `[[env]]` **blocks.** The trainer interleaves tasks from both; each block keeps its own turn cap and taskset knobs. Everything from [tutorial 3](../tutorials/3_first_rl.md) — groups of `rollouts_per_example`, advantage relative to siblings on the *same* task — applies per task, unchanged. Why mix at all? Diversity is regularization: one distribution's quirks (repo style, test idioms) are harder to overfit — and harder to reward-hack — when the batch is drawn from two.
- **Sandbox runtimes** (`type = "prime"`): thousands of concurrent containers is what training needs; Docker-on-your-laptop is for the smoke tests. This is the runtime layer swapping out while taskset and harness stay fixed.
- **The** `[eval]` **block is the held-out benchmark on a schedule** — every 20 steps, 50 SWE-bench Verified tasks. Training reward tells you the model is learning *something*; only the held-out curve tells you it's the thing you wanted.
- `zero_advantage` **filtering** drops task groups where every rollout scored the same (all-solved or all-failed) — they carry no gradient. On hard SWE tasks early in training, that's a lot of them; filtering keeps batches full of signal.

A cost note, so nobody is surprised: the blog's reference run — GLM-4.5-Air on `scaleswe` — took **6 H200 nodes for 2 days**, with ~47 turns per task and ~3.6-minute steps. Shrink `max_steps`, `batch_size`, and `num_examples` for a first run; the config's *shape* is what transfers to the full-scale version.

## Watch the run

`prime rl rollouts` early and late, same as tutorial 3, but the SWE-specific signals are:

- **Training reward vs. held-out pass rate.** The blog's run reached ~0.50 training reward and 0.554 held-out Pass@1. If training reward climbs while the held-out curve doesn't, the model is learning the training distribution's tells, not software engineering — the multi-env mix and verified rewards are your defenses, and the held-out curve is the tripwire.
- **Turns per task.** Watch it drift: agents under RL pressure first learn to *stop wasting turns* (fewer aimless `ls`/`cat` excursions), then to commit earlier. A collapse to very few turns with flat reward is degenerate behavior worth reading transcripts about.
- **Error rates per environment.** A rising `HarnessError`/infra-error rate in one `[[env]]` poisons its share of the batch — the same "errors are not zeros" rule as [Designing Rewards](../tutorials/7_rewards.md), at fleet scale.



## After the run

Re-run the baseline command against the trained checkpoint — same taskset, same harness id and version, same sampling — and quote both numbers with their sample sizes. Then read a handful of newly-solved tasks' transcripts next to the base model's failures on the same tasks: the diff in *behavior* (test-first? reproduces before patching? reverts dead ends?) is the actual product of the run, and it's the part a single pass-rate number can't show.

## Things to try

- **Add a third distribution:** `swesmith_v1` in research-environments covers 8 languages (~88k tasks) — add its Rust or Go slice as a `[[env]]` block and watch whether cross-language transfer shows up on the held-out curve.
- **Curriculum via** `filter_fn`**:** both training tasksets accept a Python-expression row filter — start training on tasks whose gold patch touches one file, then relax it. Compare curves.
- **Harness sensitivity:** train two short runs identical except for `harness.id`, and eval both checkpoints in *both* harnesses. How much of the learned skill is scaffold-specific? (The answer shapes what you deploy.)
- **Gate audit:** pull 5 tasks from a *raw* (non-Verified) partition, run the gold-patch and no-op gates yourself, and see what fails them. Nothing builds respect for verified data like meeting the unverified kind.



## Recap

Multi-environment SWE training is the same RL loop as tutorial 3 with three disciplines added: tasksets whose rewards passed gold-patch and no-op validation, a held-out benchmark measured with the exact harness used in training, and batch hygiene (zero-advantage filtering, per-env error monitoring) so every gradient step carries signal. The layering — taskset / harness / runtime — is what lets one config say all of that.
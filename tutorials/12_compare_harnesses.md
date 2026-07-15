# Recipe — Compare Harnesses on a SWE Task

A model's score on an agentic benchmark is never just the model's score — it's the model *plus the harness it runs in*: the agent loop, the tools it's given, the way context is managed. Swap the harness and the number moves, sometimes a lot. In verifiers v1, we've made the harness configurable rather than baked into the benchmark, so you can run any evaluation with any harness of your choosing.

In this recipe, we will be measuring the impact of different harnesses on the performance of a model. We'll take one SWE-style taskset and one model, and we will run an evaluation with three different harnesses in order to end with a comparison.

## The contenders

verifiers ships several coding-agent harnesses by default, resolved by id (`--harness.id <id>` → `verifiers.v1.harnesses.<id>`):


| Harness id       | What it is                                                                          |
| ---------------- | ----------------------------------------------------------------------------------- |
| `codex`          | OpenAI's Codex CLI agent.                                                           |
| `mini_swe_agent` | mini-SWE-agent — a deliberately minimal bash-loop agent, ~100 lines of scaffolding. |


(Also available: `rlm`, `kimi_code`, and `default`.)

The taskset is `terminal-bench-2-v1`, a Harbor taskset bundled with this cookbook (`environments/terminal_bench_2_v1`): real terminal/SWE tasks, each a container the agent works in plus tests that score the outcome. The taskset stages the tests and computes the reward inside the rollout runtime.

## Set up the configs

In order to do a faithful comparison, we need to ensure we only vary 1 variable (the harness). Everything below stays identical across runs, including model, sampling, tasks, turn caps, runtime — and only `harness.id` changes. Write the shared part once (`compare/base.toml`):

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 10          # start small; scale once the sweep works
num_rollouts = 2        # agentic runs are high-variance — 1 rollout lies to you
max_turns = 40

[sampling]
max_tokens = 4096

[taskset]
id = "terminal-bench-2-v1"
use_prime_registry = true   # pull task images from the Prime registry (avoids Docker Hub rate limits)

[harness]
runtime = { type = "docker" }
```

`terminal-bench-2-v1` ships with the cookbook as a local environment (`environments/terminal_bench_2_v1`), so it's installed by `uv sync` from [Setup](../guides/00-setup/README.md) — no extra install step. Two runtime prerequisites: **Docker must be running** (each task executes in a container), and `PRIME_API_KEY` must be set for inference.

Then the sweep is three one-liners — the config file carries the constants, the flag carries the variable:

```bash
prime eval run @ compare/base.toml --harness.id codex
prime eval run @ compare/base.toml --harness.id mini_swe_agent
```

Each run lands in its own `outputs/terminal-bench-2-v1--openai--gpt-5.4-mini--<harness>/...` folder with its resolved `config.toml`, so you have full provenance of the full parameters for a given run.

Since harnesses represent real, evolving software, two details keep the comparison faithful:

- **Pin harness versions.** `[harness] id = "codex"` + `version = "0.116.0"` in the TOML makes the run reproducible next month.
- **Every model call is intercepted.** Whatever the harness is, its model traffic flows through the same interception server, which is why token counts and transcripts are comparable across scaffolds at all.



## Reading the result

The mean reward per harness is the headline — but it's the *least* interesting part. Pull up the runs side by side:

```bash
prime eval view
```

Compare, per harness:

- **Reward** — the solve rate. Note it with its sample size; at `-n 10`, differences under ~15 points are noise.
- **Tokens and turns per rollout** — from the traces' usage fields. Scaffolds differ wildly in cost: a harness that scores 5 points higher while spending 3× the tokens is a different trade-off, not a strict win.
- **Errors vs. legitimate failures** — a rollout that died with a `HarnessError` (install failed, agent crashed) is not evidence about the model. Count these separately before comparing means; a harness with a 20% crash rate has an infrastructure problem, not a capability deficit.

Then read a few transcripts for the same task across harnesses — this is where the comparison gets real. You'll see personality differences immediately: one agent greps and reads surgically, another cats entire files into context; one runs the tests after every edit, another edits blind and hopes. When you later see the reward gap, you'll know *which behavior* it's made of.

## Things to try

- **Tool ablations:** most harnesses accept a `disabled_tools` list (`[harness] disabled_tools = ["websearch"]` — names are harness-specific). Re-run the winner with a tool removed and see how much of its edge that tool was carrying.
- **Model × harness grid:** repeat the sweep with a second model. Harness rankings are *not* stable across models — small models often do better in minimal scaffolds like `mini_swe_agent`, which don't demand sophisticated tool orchestration. Two models × three harnesses is six runs and one genuinely publishable chart.
- **Your own repo:** the same sweep works on a taskset of *your* tasks — package your repo's issues as a Harbor taskset ([Coding Agent Environments](11_coding_agents.md)) and rank scaffolds where it actually matters to you.
- **Scale honestly:** once the pipeline works at `-n 10`, run the full taskset `--hosted` and quote *those* numbers.



## Recap

One taskset, one model, one variable: the harness. You measured reward, cost, and reliability separately, pinned versions for reproducibility, and read transcripts to see the behavioral difference behind the numbers. This sweep pattern — constants in a TOML, the variable as a flag — is reusable for any single-variable comparison: models, sampling, prompts, tools.
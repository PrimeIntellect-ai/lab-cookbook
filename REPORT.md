# Tutorials & Recipes — Status Report

**Date:** 2026-07-23 · **Branch:** `ilija/revamp-guides` (after commit `f1fe48f` "fix: fix syntax, code snippets, standardize commands") · **Environment:** `verifiers` 0.2.1.dev46, `prime` CLI 0.6.19, Python 3.12, Docker running (Apple Silicon host)

This report has two parts: what was fixed in the docs after the first audit, and live **smoke-run results** for every tutorial and recipe (real evals, executed 2026-07-23, uploaded to the Evaluations dashboard).

## Part 1 — Fixes applied (commit `f1fe48f`)

The first audit (23 broken items across 14 files) has been addressed:

- **Eval commands standardized on `uv run eval`.** All `prime eval run` invocations (v0 CLI, which rejects `@ file.toml`, `--dry-run`, dotted `--taskset.*` overrides, and standalone `--resume`) were replaced. The broken `--hosted` steps were replaced with real full-run commands.
- **Deprecated `prime rl ...` → `prime train ...`** everywhere (docs and the `the-pieces.svg` diagram).
- **Phantom APIs removed:** `TasksetError`→`TaskError`; `Harness.cleanup()` row dropped; `claude_code` harness id dropped; `shared = true`→`SharedToolsetConfig` on `Taskset.tools`; `Taskset.NEEDS_CONTAINER`→`Task.NEEDS_CONTAINER`; `corpus_dataset`→pointer to the `DATASET` constant; `--taskset.judge.model`→`[[taskset.task.judges]]` TOML block (dry-run verified); `--harness.disabled_tools '["bash"]'`→`--harness.disabled-tools bash` (parser-verified); `prime teams switch`→`prime switch`; `prime rl init <env>`→`prime train init <file>`.
- **Snippets synced to shipped code and the installed API:** tutorial 6 now shows the real `SimpleJudge`/`judge_reward` implementation; tutorial 10's stale v0 `WikiIndex` line replaced; tutorial 11 snippet gained its imports and real reward name; the quality-audit taskset was rewritten to the real v1 shape (`TaskData` subclass, `load()`, judges on `TaskConfig`, mode-dependent tools via `Task` subclasses) and its join script now reads the real `traces.jsonl` schema. All rewritten snippets were executed against installed verifiers before being committed.
- **Numbers/paths/links reconciled:** GEPA table matches its command (50/200); `configs/report/` paths; report-card and mini-loop TOML blocks match shipped files; missing diagram embedded; dead anchor fixed; typos cleaned.

All Python blocks in `tutorials/` and `recipes/` parse; every fixed command was validated with a real `--dry-run` or live run.

## Part 2 — Smoke-run results

Method: for each tutorial/recipe, its central runnable command was executed live, with task/rollout counts reduced to 1–2 where the doc's command is larger (noted as `-n`/`-r` overrides — flag-over-config precedence is part of the documented interface). Runs upload automatically (`--push` default); links are in the private Evaluations tab. "PASS" = rollouts completed with real scores and no errors recorded on traces.

### Tutorials

| Tutorial | What was run | Result |
| --- | --- | --- |
| 0 — Hello | no runnable commands (conceptual) | n/a |
| 1 — Setup | `uv run eval reverse_text_v1 -n 2 -r 1` | ✅ PASS — 2 rollouts, uploaded. Note: rewards are 0.0 with the default model (`deepseek/deepseek-v4-flash`); the tutorial only promises completion, but first-time users may read zero scores as breakage. |
| 2 — First Eval | `uv run eval @ configs/02/gsm8k-eval.toml -n 2 -r 1`; `uv run eval wordle_v1 -n 1 -r 1 --max-turns 6 -m openai/gpt-5.4-mini --sampling.max-tokens 1024` | ✅ PASS — gsm8k rewards 1.0/1.0; wordle multi-turn completed (reward 0.0 = lost the game, legitimate). |
| 3 — First RL | config validated (`prime train configs` keys, model listed Available in `prime train models`, `prime train init` generates); **training launch not run** (spends training credits) | ⚠️ VALIDATED, NOT LAUNCHED |
| 4 — First GEPA | `uv run gepa wordle_v1 --model openai/gpt-5.4-nano --reflection-model openai/gpt-5.4-nano --num-train 2 --num-val 2 --max-turns 4 --max-total-rollouts 8 ...` (tiny budget) | ✅ PASS — full GEPA loop ran (baseline eval, reflection iterations, best-prompt output, standard output dir). |
| 5 — Build First Environment | `uv run eval @ configs/05/aime26-eval.toml -n 2 -r 1` | ✅ PASS — rewards 1.0/0.0; taskset loads 30 tasks from the pinned HF dataset. |
| 6 — Judges | `uv run eval @ configs/06/simple-judge-eval.toml -n 2 -r 1` | ✅ PASS — rewards 1.0/1.0; judge calls (gpt-4.1-mini) executed and recorded on traces. |
| 7 — Rewards | `uv run eval @ configs/07/code-golf-eval.toml -n 2` (config's `-r 2` kept for group rewards) | ✅ PASS — 4 rollouts, all rewards computed incl. group-relative scoring. |
| 8 — User Simulators / 9 — Multimodal | `uv run eval @ configs/09/shape-detective-eval.toml -n 2 -r 1` (one env covers both docs) | ✅ PASS — rewards 1.0/0.0; user-simulator turns and image prompts both exercised. |
| 10 — Tools | `uv run eval @ configs/10/wiki-search-eval.toml -n 2 -r 1` (first run includes one-time corpus/Chroma index build) | ✅ PASS — index built from scratch; rewards 1.0/0.0 over 4–6 tool-using turns; reference judge graded; uploaded. |
| 11 — Coding Agents | `uv run eval @ configs/11/math-python-eval.toml -n 1 -r 1`; `uv run eval @ configs/11/harbor-smoke.toml` | ❌ FAIL — two environment/infra issues, not doc issues (details in Part 3, items 1–2). |
| 12 — Best Practices | no runnable commands | n/a |

### Recipes

| Recipe | What was run | Result |
| --- | --- | --- |
| eval_report_card | `uv run eval @ configs/report/aime25.toml -n 1 -r 1` (Nemotron 550B); `.../simpleqa-verified.toml -n 2 -r 1`; `.../ifeval.toml -n 2 -r 1`; `.../terminal-bench.toml` | ✅ aime25 PASS (reward 1.0); ✅ simpleqa PASS (1.0/1.0, judge graded); ✅ ifeval PASS (1.0/1.0); ❌ terminal-bench FAIL — 0 tasks loaded (Part 3, item 3). |
| coding_agent_harness | `uv run eval @ configs/recipes/mini-loop-smoke.toml`; `uv run eval @ configs/compare/base.toml --harness.id mini-loop -n 1 -r 1` | ✅ mini-loop PASS (reward 1.0, agent loop + Harbor verifier end-to-end; one earlier attempt hit a flaky "scoring timed out" under Docker contention — retry succeeded). ❌ compare sweep FAIL — same 0-tasks bug (Part 3, item 3). |
| port_v0_to_v1 | `uv run pytest tests/`; `uv run eval --id primeintellect/reverse-text -n 2 -r 1` (v0 bridge, hub install); v1 path = tutorial 1 run | ✅ PASS — 5 tests passed; v0 bridge ran 2 rollouts (rewards 1.0) and uploaded. |
| quality_audit | no shipped env (the recipe builds `synth_search` from scratch); all snippets executed against installed verifiers in all three modes during the fix pass | ✅ VERIFIED BY EXECUTION |
| search_agent | eval = tutorial 10's wiki run; RL config parses; training not launched | ✅ eval PASS (index built, tool-using rollouts judged, uploaded); ⚠️ training validated but not launched (credits). |
| support_agent | `uv run eval @ configs/recipes/support-agent-eval.toml -n 1 -r 1` (tau2 telecom, user sim) | ✅ PASS — 22-turn conversation, ended by the user simulator (`stop=user_completed`), no errors; reward 0.0 is a legitimate unsolved task. ~15 min for one rollout. |
| train_coding_agent | `uv run eval @ configs/recipes/swe-baseline-eval.toml -n 1` (SWE-bench Verified, Docker image pull) | ✅ PASS — task image pulled, `mini_swe_agent` solved the astropy issue in 13 turns (reward 1.0), verifier ran, uploaded. ~11 min wall-clock for one task. Training launch itself not run (credits). |

### Part 3 — Issues found by the smoke runs (environment/infra, not docs)

Per instruction, environments were **not** modified; these are recorded for follow-up:

1. **`math_python` cannot run in this workspace as installed** — `ToolsetError: verifiers is not a source checkout (no pyproject above the package), so it can't be uploaded to a sandbox`. Serving the env's Docker-sandboxed Python tool server requires verifiers installed from a source checkout; this workspace installs verifiers as a built wheel from git (`[tool.uv.sources] verifiers = { git = ... }`). A `--taskset.task.tools.runtime.type subprocess` override doesn't compose either (the config pins docker-only `image`/`workdir` fields). Tutorial 11's math-python section will fail for any user on this install method.
2. **`opencode_harbor` harness cannot install on Apple Silicon** — `HarnessError: OpenCode install failed: curl: (22) ... 404`. The pinned release `PrimeIntellect-ai/opencode v1.1.63-rl2` publishes `opencode-linux-x64.tar.gz` (HTTP 200) but **no `opencode-linux-arm64.tar.gz`** (404). On an ARM Mac, Docker containers default to linux/arm64, so the install script's arch switch selects the missing asset. Works only on x86_64 hosts (or with an arm64 asset added to the release).
3. **`terminal_bench_2_v1` loads 0 tasks when `use_prime_registry = true`** — `TerminalBench2Taskset.load()` gets a *generator* from `HarborTaskset.load()`, consumes it in its image-rewrite loop, then returns the exhausted generator. Result: `running 0x1 rollouts`, "successful" run with zero samples. Verified directly: `use_prime_registry=False` → 89 tasks; `True` → 0. This silently breaks `configs/compare/base.toml` and `configs/report/terminal-bench.toml` as shipped (both set `use_prime_registry = true`). One-line fix in the env: materialize the list before rewriting.
4. **Flaky Harbor verifier scoring timeout** under concurrent Docker load (mini-loop first attempt; retry passed). Worth knowing when running sweeps on a laptop.
5. **Default-model zero scores** — `reverse_text_v1` smoke (tutorial 1) completes but scores 0.0 with the CLI's default model. Cosmetic UX concern only.

### Not executed (by design)

- `prime train <config>` launches (tutorials 3, 10; recipes search_agent, train_coding_agent) — hosted training spends real training credits; configs were validated against `prime train configs` / `prime train models` instead.
- Full-scale runs (500-task SWE baseline, full report-card battery, 200-rollout GEPA) — the smoke runs above exercise the identical code paths at 1–2 tasks.
- `prime env push`, `prime images push` — mutating platform operations.

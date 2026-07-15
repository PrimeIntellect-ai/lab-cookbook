# Tutorials

Hands-on tutorials for Prime Intellect Lab.

**Basics** is the sequential on-ramp — do these in order. **Ramping up** introduces the core methods and building blocks one at a time. **Recipes** are self-contained, use-case-driven walkthroughs — pick whichever matches the problem in front of you.

## Basics

| # | Tutorial | What you'll do |
| --- | --- | --- |
| 0 | [Hello, Prime Intellect Lab](0_hello.md) | What Lab is, the components, and the mental model everything builds on. |
| 1 | [Setup](1_setup.md) | Install the CLI, authenticate, and run a smoke-test eval. |
| 2 | [Your First Eval](2_first_eval.md) | Measure GPT-5.4 mini on grade-school math, then multi-turn Wordle. |
| 3 | [Your First RL Run](3_first_rl.md) | Train a small model to reverse text on hosted GPUs, and watch it improve. |
| 4 | [Your First GEPA Run](4_first_gepa.md) | Evolve a better prompt automatically — improvement without touching weights. |

## Ramping up

| # | Tutorial | What it introduces |
| --- | --- | --- |
| 5 | [Build Your First Environment](5_build_first_environment.md) | Tasks, configs, tasksets, and the authoring contract — dissected on a tiny example. |
| 6 | [Judges](6_judges.md) | Replace exact match with an LLM judge, controlled through config. |
| 7 | [Designing Rewards](7_rewards.md) | Weighted rewards, metrics, stop conditions, and group rewards. |
| 8 | [User Simulators](8_user_simulators.md) | User-driven multi-turn conversations with shared state. |
| 9 | [Tool Use and Search](9_tools.md) | Toolsets, wiring them into tasksets, placement, and reading tool traces. |
| 10 | [Multimodal Environments](10_multimodal.md) | Image-based tasks and message-list prompts. |
| 11 | [Coding Agent Environments](11_coding_agents.md) | Runtimes, persistent interpreters, Docker, and Harbor tasks. |

## Recipes

### Evaluate

| # | Recipe | The question it answers |
| --- | --- | --- |
| 12 | [Compare Harnesses](12_compare_harnesses.md) | *Which agent scaffold — Codex, Terminus, mini-SWE-agent — is best for my task and model?* Fix a SWE taskset, sweep the harness, compare fairly. |
| 13 | [Model Report Card](13_eval_report_card.md) | *How good is this new model, really?* Run a multi-eval battery (math, instruction following, factuality, agentic) and report numbers you can defend. |

### Build

| # | Recipe | The question it answers |
| --- | --- | --- |
| 14 | [Infinite Tasksets](14_infinite_tasksets.md) | *How do I get unlimited, leakage-proof tasks with a difficulty dial?* Generate tasks procedurally from a seed instead of loading a dataset. |
| 15 | [Synthetic Worlds](15_synthetic_world.md) | *How do I build an agent environment without real infrastructure?* Simulate a world in typed state, expose tools, and score the final state. |
| 16 | [Quality Audit](16_quality_audit.md) | *How do I know my generated tasks are any good?* Validate synthetic QA pairs with oracle / closed-book ablations before trusting them for eval or training. |

### Train

| # | Recipe | The question it answers |
| --- | --- | --- |
| 17 | [Search Agent](17_search_agent.md) | *How do I give a model tools and train it to use them?* Build, evaluate, and RL-train a wiki-search agent. |
| 18 | [SFT Warm-up → RL](18_warm_up.md) | *What if my model is too weak for RL to get off the ground?* Distill a teacher via SFT first, then RL from that checkpoint. |
| 19 | [Generalist Training](19_generalist.md) | *Does training on a mix of tasks transfer?* Train on math + a game + a tool env at once, and measure on held-out environments. |

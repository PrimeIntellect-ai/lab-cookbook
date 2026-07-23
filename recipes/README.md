# Recipes

Use-case-driven, end-to-end walkthroughs: each starts from a real goal and ends with a runnable result, with environments and configs included in this repo. Do the tutorials' [Basics](../tutorials/README.md#basics) first; each recipe lists which Ramping-up tutorials it builds on.

- [Search Agent](search_agent.md) — build search tools over a pinned corpus, evaluate the behavior, then RL-train it.
- [Model Report Card](eval_report_card.md) — run a multi-eval battery on a new model and report numbers you can defend.
- [Build Your Own Coding-Agent Harness](coding_agent_harness.md) — author a CLI-agent harness from scratch, then sweep it against the built-in scaffolds.
- [Support Agent with Simulated Users](support_agent.md) — τ²-bench: stateful tools, an LLM customer, and rewards over final state.
- [Train Your Coding Agent](train_coding_agent.md) — multi-environment RL on SWE tasksets, with the validation gates that keep rewards honest.
- [Quality-Auditing Synthetic Tasks](quality_audit.md) — validate generated QA pairs with oracle / closed-book ablations.
- [Port a v0 Environment to v1](port_v0_to_v1.md) — keep legacy environments on the bridge, then migrate them to typed v1 tasksets.

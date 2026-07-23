Prime Intellect

### Lab Cookbook

Create, evaluate, train and deploy your own agents with Prime Intellect Lab.

The cookbook is a hands-on resorce outlining how to use the Prime Intellect lab to evaluate and train your own models. It contains introductory [tutorials](/tutorials/), as well use-case based [recipes](/recipes/)



### Tutorials

The [Basics](tutorials/README.md#basics) (0–4) take you from nothing to your first eval, RL run, and GEPA run:

- [Hello, Prime Intellect Lab](tutorials/0_hello.md)
- [Setup](tutorials/1_setup.md)
- [Your First Eval](tutorials/2_first_eval.md)
- [Your First RL Run](tutorials/3_first_rl.md)
- [Your First GEPA Run](tutorials/4_first_gepa.md)

Then, [the next set of tutorials](tutorials/README.md#ramping-up) (5–12) introduces the building blocks of building your own environments one at a time.

- [Build Your First Environment](tutorials/5_build_first_environment.md)
- [Judges](tutorials/6_judges.md)
- [Designing Rewards](tutorials/7_rewards.md)
- [User Simulators](tutorials/8_user_simulators.md)
- [Multimodal Environments](tutorials/9_multimodal.md)
- [Tool Use and Search](tutorials/10_tools.md)
- [Coding Agent Environments](tutorials/11_coding_agents.md)
- [Best Practices](tutorials/12_best_practices.md)



### Recipes

These are use-case-driven, end-to-end walkthroughs that compose the pieces, where you start from a goal and end with a runnable result:

- [Search Agent](recipes/search_agent.md) — build search tools over a document corpus, evaluate the agent's capability, and train it with RL.
- [Model Report Card](recipes/eval_report_card.md) — run a multi-eval battery on a new model and report numbers.
- [Build Your Own Coding-Agent Harness](recipes/coding_agent_harness.md) — author a CLI-agent harness from scratch, then see how it holds up against popular harnesses.
- [Support Agent with Simulated Users](recipes/support_agent.md) — τ²-bench: stateful tools, an LLM customer, and rewards over final state.
- [Train Your Coding Agent](recipes/train_coding_agent.md) — multi-environment RL on SWE tasksets, with the validation gates that keep rewards honest.
- [Quality-Auditing Synthetic Tasks](recipes/quality_audit.md) — validate generated QA pairs with oracle / closed-book ablations.
- [Port a v0 Environment to v1](recipes/port_v0_to_v1.md) — keep legacy environments running through the bridge, then migrate them to typed v1 tasksets.



### Repository Structure

- [tutorials](/tutorials/) — the sequential on-ramp.
- [recipes](/recipes/) — use-case-driven walkthroughs.
- [environments](/environments/) — local environment packages used by tutorials and recipes.
- [configs](/configs/) — runnable eval, GEPA, and training configs, numbered by tutorial (plus `configs/recipes/`).
- [reference](/reference/) — the authoring golden path and workspace notes.


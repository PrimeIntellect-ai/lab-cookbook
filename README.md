Prime Intellect

### Lab Cookbook

Create, evaluate, train and deploy your own agents with Prime Intellect Lab.

The cookbook is a hands-on resorce outlining how to use the Prime Intellect lab to evaluate and train your own models. It contains introductory [tutorials](/tutorials/), as well use-case based [recipes](/recipes/)

### Tutorials

The [Basics](tutorials/README.md#basics) (0–4) take you from nothing to your first eval, RL run, and GEPA run:


| Tutorial                                           | Covers                                                                          | Config                    |
| -------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------- |
| [Hello, Prime Intellect Lab](tutorials/0_hello.md) | What Lab is and the components                                                  | —                         |
| [Setup](tutorials/1_setup.md)                      | Installing the CLI, authenticating, and installing skills                       | —                         |
| [Your First Eval](tutorials/2_first_eval.md)       | Measuring a model's proficiency on grade-school math and Wordle, reading traces | [configs/02](configs/02/) |
| [Your First RL Run](tutorials/3_first_rl.md)       | Training a small model on hosted GPUs and watching it improve                   | [configs/03](configs/03/) |
| [Your First GEPA Run](tutorials/4_first_gepa.md)   | Evolving a better prompt automatically, without training weights                | [configs/04](configs/04/) |


Then, [the next set of tutorials](tutorials/README.md#ramping-up) (5–12) introduces the building blocks of building your own environments one at a time.


| Tutorial                                                               | Covers                                                                     | Config                    |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------- |
| [Build Your First Environment](tutorials/5_build_first_environment.md) | Tasks, configs, and tasksets                                               | [configs/05](configs/05/) |
| [Judges](tutorials/6_judges.md)                                        | Scoring unverifiable properties                                            | [configs/06](configs/06/) |
| [Designing Rewards](tutorials/7_rewards.md)                            | Weighted rewards, metrics, stop conditions, and group rewards              | [configs/07](configs/07/) |
| [User Simulators](tutorials/8_user_simulators.md)                      | Simulated human responses a multi-turn conversation                        | [configs/09](configs/09/) |
| [Multimodal Environments](tutorials/9_multimodal.md)                   | Tasks that carry images and not only text                                  | [configs/09](configs/09/) |
| [Tool Use and Search](tutorials/10_tools.md)                           | Wiring toolsets into tasksets and reading tool traces                      | [configs/10](configs/10/) |
| [Coding Agent Environments](tutorials/11_coding_agents.md)             | Runtimes, persistent interpreters, Docker, and the Harbor taskset workflow | [configs/11](configs/11/) |
| [Best Practices](tutorials/12_best_practices.md)                       | Where each kind of logic belongs and what breaks environments              | —                         |




### Recipes

These are use-case-driven, end-to-end walkthroughs that compose the pieces, where you start from a goal and end with a runnable result:


| Recipe                                                                 | Goal                                                                            | Approach                                                                                                                                                                          |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Search Agent](recipes/search_agent.md)                                | Give a model search tools over a pinned corpus and train it to search better    | Build a custom toolset (search → skim → read), evaluate the behavior in traces, then RL-train against the same environment.                                                       |
| [Model Report Card](recipes/eval_report_card.md)                       | Evaluate a new model across capabilities and get a report card                  | Fix sampling per model, stage the spend (dry-run → smoke → hosted full runs), decompose outcomes before averaging.                                                                |
| [Build Your Own Coding-Agent Harness](recipes/coding_agent_harness.md) | Make your own agent scaffold a first-class citizen and find out how it compares | Write a ~60-line uv-script agent and a `vf.Harness` around it, smoke it on a Harbor task, then sweep `harness.id` against Codex and mini-SWE-agent on a fixed taskset.            |
| [Support Agent with Simulated Users](recipes/support_agent.md)         | Evaluate a customer-support agent in a popular benchmark                        | Run τ²-bench through a harness that wraps its official orchestrator: an LLM user simulator plays the customer, and rewards read final database state, actions, and communication. |
| [Train Your Coding Agent](recipes/train_coding_agent.md)               | Improve a model's SWE ability with multi-environment RL                         | Train on two validated SWE tasksets at once with one `[[env]]`-per-environment config, gate rewards with gold-patch/no-op validation, and measure on held-out SWE-bench Verified. |
| [Quality-Auditing Synthetic Tasks](recipes/quality_audit.md)           | Know whether generated tasks are solid before trusting them                     | Run oracle (answer in context) vs. closed-book ablations; keep tasks that a strong model solves with the oracle and misses without it.                                            |
| [Port a v0 Environment to v1](recipes/port_v0_to_v1.md)                | Migrate legacy environments to typed v1 tasksets                                | Run the v0 package on the legacy bridge first, map each v0 concept to its v1 home on a worked example, then let both generations referee each other's scores.                     |




### Repository Structure

- [tutorials](/tutorials/) — the sequential on-ramp.
- [recipes](/recipes/) — use-case-driven walkthroughs.
- [environments](/environments/) — local environment packages used by tutorials and recipes.
- [configs](/configs/) — runnable eval, GEPA, and training configs, numbered by tutorial (plus `configs/recipes/`).
- [reference](/reference/) — the authoring golden path and workspace notes.


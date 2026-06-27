# Lab Guides

Step-by-step walkthroughs for using Prime Intellect Lab with Verifiers v1 environments.

Start with setup if you are new to Lab. Jump to a later guide when you want a specific authoring or training pattern.

## Curriculum

| Order | Guide | What you'll do |
| --- | --- | --- |
| 00 | [Setup](00-setup/README.md) | Install the local tools and sync the cookbook workspace. |
| 01 | [Environments and Evals](01-environments-and-evals/README.md) | Run a v1 taskset with the eval CLI and read the output trace. |
| 02 | [Building Your First Environment](02-building-your-first-environment/README.md) | Build a typed `Taskset` package with tasks and rewards. |
| 03 | [Training with RL](03-training-with-rl/README.md) | Embed a v1 env config in an RL training config. |
| 04 | [Prompt Optimization](04-prompt-optimization/README.md) | Evaluate prompt changes and understand the current GEPA boundary. |
| 07 | [Judges and Instruction Following](07-judges-and-instruction-following/README.md) | Add an LLM judge reward with typed client config. |
| 08 | [Tool Use and Search](08-tool-use-and-search/README.md) | Expose tools through a `vf.Toolset` server. |
| 09 | [Multimodal Environments](09-multimodal-environments/README.md) | Build image-bearing tasks and user simulators. |
| 10 | [Coding Agents and Sandboxes](10-coding-agents-and-sandboxes/README.md) | Run code-oriented tasksets with tools, runtimes, and Harbor tasks. |
| 11 | [Synthetic Agent Environments](11-synthetic-agent-environments/README.md) | Keep a simulated world in typed state and expose interaction tools. |
| 12 | [Custom Harnesses](12-custom-harnesses/README.md) | Write a `vf.Harness` when the rollout program is not the default chat loop. |
| 13 | [Best Practices](13-best-practices/README.md) | Review the authoring contract and common failure modes. |
| 14 | [Legacy Environments](14-legacy-environments/README.md) | Reference for v0 `load_environment` packages. |

For the compact authoring contract, see [Golden Path](../reference/GOLDEN_PATH.md). For the remaining GEPA boundary, see [v1 Authoring Gaps](../reference/v1-authoring-gaps.md).

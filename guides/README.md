# Lab Guides

Step-by-step walkthroughs for learning Lab end to end.

Start with setup if you are new to Lab. Jump to a later guide if you already know the basics and want a specific workflow.

## Curriculum

| Order | Guide | What you'll do |
| --- | --- | --- |
| 00 | [Setup](00-setup/README.md) | Install the CLI and create a Lab workspace. |
| 01 | [Environments and Evals](01-environments-and-evals/README.md) | Run a benchmark, inspect rollouts, and learn how Lab thinks about environments. |
| 02 | [Building Your First Environment](02-building-your-first-environment/README.md) | Build a tiny environment with tasks and rewards. |
| 03 | [Training with RL](03-training-with-rl/README.md) | Train the environment and watch reward improve. |
| 04 | [Prompt Optimization](04-prompt-optimization/README.md) | Use GEPA to improve prompts with eval feedback. |
| 05 | [Warm Starts with SFT](05-warm-starts-with-sft/README.md) | Use SFT to prepare a model before further RL. |
| 06 | [On-Policy Distillation](06-on-policy-distillation/README.md) | Distill from a teacher on the student's own rollouts. |
| 07 | [Judges and Instruction Following](07-judges-and-instruction-following/README.md) | `simple-judge`, IFEval, AdvancedIF. |
| 08 | [Tool Use and Search](08-tool-use-and-search/README.md) | Build environments with tools and retrieval. |
| 09 | [Multimodal Environments](09-multimodal-environments/README.md) | Work with image inputs and multimodal scoring. |
| 10 | [Coding Agents and Sandboxes](10-coding-agents-and-sandboxes/README.md) | Evaluate agents that write or run code in sandboxes. |
| 11 | [Synthetic Agent Environments](11-synthetic-agent-environments/README.md) | Simulate a small world in memory and have an agent interact with it through tools. |
| 12 | [Custom Harnesses](12-custom-harnesses/README.md) | Run third-party agent libraries through the program pattern. |
| 13 | [Best Practices](13-best-practices/README.md) | A deliberate walkthrough of how to write clean environments. |
| 14 | [Legacy Environments](14-legacy-environments/README.md) | Reference for unmigrated Rubric and v0 patterns (not for new envs). |

For platform plumbing such as accounts, secrets, Hub workflows, hosted runs, and inference deployments, use the public Prime docs. The local [Lab Configuration](../reference/lab-configuration.md) page is only a thin pointer to those sources.

# Glossary

Concise definitions for Lab, Verifiers, evaluation, and RL training terms used throughout this cookbook.

## Core Loop

- **environment**: A packaged task and scoring setup that Lab can load for evaluation or training.
- **eval**: A run where a model attempts environment tasks while the framework records rollouts and scores.
- **task**: One problem instance from an environment, such as a question, game state, retrieval target, or coding challenge.
- **rollout**: One complete model attempt on a task, including prompts, model outputs, tool calls, environment responses, and score.
- **rollouts_per_example**: The number of independent attempts sampled for each task or dataset row.
- **completion**: The model output captured for a rollout, usually stored as chat messages.
- **trajectory**: The full step-by-step interaction path in a rollout, including intermediate model and environment actions.
- **baseline**: A reference result used for comparison. Common subcases include a base-model baseline before prompt optimization, SFT, or RL, and a random or naive baseline used to show that a task has useful signal.

## Verifiers Concepts

Class and API names keep Verifiers capitalization; lowercase entries describe roles or concepts.

- **Taskset**: The Verifiers component that owns task rows, prompts, tools, rewards, and task-level configuration.
- **Harness**: The component that controls how a model or agent is executed during a rollout.
- **Rubric**: The scoring container that runs reward functions and metrics and combines weighted reward outputs.
- **reward function**: Code that inspects a rollout and returns a numeric score.
- **RubricGroup**: A wrapper that combines multiple rubrics into one scoring surface.
- **judge**: An LLM, or another non-deterministic scorer, used when deterministic scoring is hard.
- **JudgeRubric**: A Verifiers rubric that stores judge model configuration and exposes a judge callable to reward functions.
- **parser**: Logic that extracts structured fields from a model response, such as boxed answers or XML tags, so reward functions can score the parsed values.
- **metric**: A recorded diagnostic value. In a rubric, `weight = 0` keeps a reward function metric-only; nonzero weights contribute to the final reward.
- **monitor rubric**: A rubric used to attach automatic metrics for observability, such as tool-call counts or turn counts.
- **state**: Mutable rollout data shared by the environment, harness, tools, callbacks, and rewards.
- **per-rollout state**: State initialized fresh for one rollout, such as a sandbox session, generated world, or database handle.
- **DatasetBuilder**: A callable that lazily creates a dataset when the environment first needs it.
- **stateful vs. stateless**: A distinction between code that preserves rollout-specific resources between steps and code that returns each result without remembering prior calls.
- **tool call**: A structured request from the model to run an environment-provided function.
- **Toolset**: A collection of tools exposed to the model during a rollout.
- **MCP**: Model Context Protocol, a standard interface for exposing external tools and services over transports such as stdio or HTTP so agents can call them.
- **sandbox**: An isolated runtime where agents can execute code or commands without affecting the host system.
- **stop condition**: A rule that ends a rollout, such as task success, an error, or reaching a turn limit.
- **user callback**: Environment code that emits follow-up user messages between assistant turns.
- **max_turns**: The maximum number of assistant turns allowed in a rollout.
- **oracle**: Deterministic logic that computes the correct answer or best achievable score for a generated task.
- **EnvConfig**: The top-level Verifiers configuration object passed into `load_environment`.
- **TasksetConfig**: A typed config object for taskset-specific knobs.
- **HarnessConfig**: A typed config object for harness-specific execution knobs.

## Training and RL

- **policy**: The model behavior being optimized: what output or action it tends to choose given a prompt or state.
- **reward signal**: The score the environment gives a rollout, used to tell training which behavior was better.
- **training signal**: Any feedback used to update the model, including rewards, supervised examples, or losses.
- **advantage**: A relative score that says whether a rollout did better or worse than comparable rollouts for the same task.
- **gradient**: The direction and magnitude of a model update computed from the training signal.
- **KL**: Kullback-Leibler divergence, commonly used to measure how far the trained policy has drifted from a reference model.
- **clip / clip ratio**: A training control that limits the size of a policy update so learning does not move too abruptly.
- **batch_size**: The number of rollout samples consumed by one training step.
- **sample throughput**: An operational metric for how many rollout samples the system can generate or consume over time, not a config knob.
- **learning_rate**: The step size used when updating model weights.
- **eval cadence**: How often a held-out eval is launched during training.
- **held-out eval**: Evaluation on examples not used for training, used to detect overfitting or regressions.
- **validation / val**: A smaller, more frequent held-out eval run inside the training loop.
- **reference model**: The model used as the anchor when measuring or constraining policy drift.
- **policy drift**: The amount the trained model's behavior has moved away from the starting or reference model.
- **checkpoint**: A saved snapshot of training state or model weights.
- **adapter**: A lightweight trained module attached to a base model, often used instead of saving a full model copy.
- **LoRA**: Low-Rank Adaptation, a common adapter training method that updates small low-rank matrices instead of all model weights.
- **SFT**: Supervised fine-tuning, training a model to imitate example outputs.
- **warm start**: Starting optimization from a stronger initial model or adapter instead of the untouched base model.
- **teacher model**: A stronger model used to generate examples or guidance for training a smaller or target model.
- **demonstration**: An example of desired behavior, often generated by a teacher model for SFT.
- **replay**: Reusing saved demonstrations or samples instead of generating fresh ones.
- **loss**: The numeric objective training minimizes; in this repo, `loss = "sft"` switches from RL to supervised fine-tuning.
- **reward hacking**: When a model learns to exploit the scoring rule without solving the intended task.
- **exploration**: Sampling varied behavior so training can discover higher-reward strategies.
- **oversampling**: Sampling some task types more often than their natural frequency.
- **online difficulty filtering**: Adjusting task exposure during training based on how easy or hard recent tasks are.
- **env_ratios**: Configured weights that control the mix when training across multiple environments.

## Model and Config Fields

- **sampling**: Generation settings used when collecting model outputs, such as token limits, temperature, and reasoning controls.
- **system prompt**: Instructions prepended to the conversation that define the model's role, task rules, and output format.
- **max_tokens**: The maximum number of tokens the model may generate in one response.
- **temperature**: A sampling knob that increases or decreases output randomness.
- **reasoning_effort**: A model-specific knob controlling how much reasoning budget a supported model uses.
- **enable_thinking**: A model-specific toggle that enables explicit reasoning behavior for supported models.
- **GEPA**: Genetic-Pareto prompt optimization, a workflow that searches for better prompts using environment feedback and a reflection model.
- **reflection_model**: The model GEPA uses to reflect on failures and propose prompt changes.
- **prompt candidate**: One proposed prompt variant being evaluated by prompt optimization.
- **max_calls**: A GEPA budget limiting total model or evaluator calls.
- **num_train**: The number of training examples GEPA uses while optimizing a prompt.
- **num_val**: The number of validation examples GEPA uses to compare prompt candidates.
- **minibatch_size**: The small batch size GEPA uses for each optimization step.
- **max_concurrent**: The maximum number of concurrent calls GEPA may run.
- **endpoint**: A named connection target for model inference, including model id, provider, URL, and key reference.
- **endpoint_id**: The short alias used to refer to an endpoint in commands and config files.
- **provider**: The service that serves a model, such as Prime Inference, OpenAI, Anthropic, or another API.
- **OpenAI-compatible**: An API shape that follows OpenAI chat-completions conventions, even if served by another provider.
- **env_args**: Extra keyword arguments forwarded to the environment at run time.
- **save_results**: A config flag that controls whether eval results are persisted.
- **save_to_environment**: A GEPA flag that writes optimized prompts back into a local environment when possible.
- **hosted eval**: An evaluation run executed on Prime infrastructure rather than on the local machine.
- **hosted training**: A training run executed on Prime infrastructure.
- **inference deployment**: A served model or adapter that can be queried through an inference API.
- **Hub version**: An immutable published version of an environment on the Environments Hub.

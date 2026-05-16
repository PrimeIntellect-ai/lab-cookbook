# Glossary

Concise definitions for Lab, Verifiers, evaluation, and RL training terms used throughout this cookbook.

## Core Loop

- <a id="environment"></a>**environment**: A packaged task and scoring setup that Lab can load for evaluation or training.
- <a id="eval"></a>**eval**: A run that asks a model to attempt environment tasks and records scores and rollouts.
- <a id="task"></a>**task**: One problem instance from an environment, such as a question, game state, search request, or coding challenge.
- <a id="rollout"></a>**rollout**: One complete model attempt on a task, including prompts, model outputs, tool calls, environment responses, and score.
- <a id="rollouts-per-example"></a>**rollouts_per_example**: The number of independent attempts sampled for each task or dataset row.
- <a id="completion"></a>**completion**: The model output captured for a rollout, usually stored as chat messages.
- <a id="trajectory"></a>**trajectory**: The full step-by-step interaction path in a rollout, including intermediate model and environment actions.
- <a id="baseline"></a>**baseline**: A reference result used for comparison, often the base model's score before prompt optimization, SFT, or RL.
- <a id="random-baseline"></a>**random baseline**: A comparison score from random or naive behavior, used to show that a task has useful signal.

## Verifiers Concepts

- <a id="taskset"></a>**Taskset**: The Verifiers component that owns task rows, prompts, tools, rewards, and task-level configuration.
- <a id="harness"></a>**Harness**: The component that controls how a model or agent is executed during a rollout.
- <a id="rubric"></a>**Rubric**: The scoring container that runs reward functions and metrics and combines weighted reward outputs.
- <a id="reward-function"></a>**reward function**: Code that inspects a rollout and returns a numeric score.
- <a id="rubricgroup"></a>**RubricGroup**: A wrapper that combines multiple rubrics into one scoring surface.
- <a id="judge"></a>**judge**: Usually an LLM used to score outputs when deterministic scoring is hard.
- <a id="judgerubric"></a>**JudgeRubric**: A Verifiers rubric that stores judge model configuration and exposes a judge callable to reward functions.
- <a id="parser"></a>**parser**: Logic that extracts structured fields from a model response, such as boxed answers or XML tags.
- <a id="metric"></a>**metric**: A recorded diagnostic value that may or may not affect reward.
- <a id="monitor-rubric"></a>**monitor rubric**: A rubric used to attach automatic metrics for observability, such as tool-call counts or turn counts.
- <a id="state"></a>**state**: Mutable rollout data shared by the environment, harness, tools, callbacks, and rewards.
- <a id="per-rollout-state"></a>**per-rollout state**: State initialized fresh for one rollout, such as a sandbox session, generated world, or database handle.
- <a id="datasetbuilder"></a>**DatasetBuilder**: A callable that lazily creates a dataset when the environment first needs it.
- <a id="stateful-vs-stateless"></a>**stateful vs. stateless**: Stateful code preserves rollout-specific resources between steps; stateless code returns results without remembering prior calls.
- <a id="tool-call"></a>**tool call**: A structured request from the model to run an environment-provided function.
- <a id="toolset"></a>**Toolset**: A collection of tools exposed to the model during a rollout.
- <a id="mcp"></a>**MCP**: Model Context Protocol, a standard way to expose external tools or services to agents.
- <a id="sandbox"></a>**sandbox**: An isolated runtime where agents can execute code or commands without affecting the host system.
- <a id="stop-condition"></a>**stop condition**: A rule that ends a rollout, such as task success, an error, or reaching a turn limit.
- <a id="user-callback"></a>**user callback**: Environment code that emits follow-up user messages between assistant turns.
- <a id="max-turns"></a>**max_turns**: The maximum number of assistant turns allowed in a rollout.
- <a id="oracle"></a>**oracle**: Deterministic logic that computes the correct answer or best achievable score for a generated task.
- <a id="envconfig"></a>**EnvConfig**: The top-level Verifiers configuration object passed into `load_environment`.
- <a id="tasksetconfig"></a>**TasksetConfig**: A typed config object for taskset-specific knobs.
- <a id="harnessconfig"></a>**HarnessConfig**: A typed config object for harness-specific execution knobs.

## Training and RL

- <a id="policy"></a>**policy**: The model behavior being optimized: given a prompt or state, what output or action it tends to choose.
- <a id="reward-signal"></a>**reward signal**: The score the environment gives a rollout, used to tell training which behavior was better.
- <a id="training-signal"></a>**training signal**: Any feedback used to update the model, including rewards, supervised examples, or losses.
- <a id="advantage"></a>**advantage**: A relative score that says whether a rollout did better or worse than comparable rollouts for the same task.
- <a id="gradient"></a>**gradient**: The direction and magnitude of a model update computed from the training signal.
- <a id="kl"></a>**KL**: Kullback-Leibler divergence, commonly used to measure how far the trained policy has drifted from a reference model.
- <a id="clip-ratio"></a>**clip / clip ratio**: A training control that limits the size of a policy update so learning does not move too abruptly.
- <a id="batch-size"></a>**batch_size**: The number of rollout samples consumed by one training step.
- <a id="sample-throughput"></a>**sample throughput**: How many rollout samples the system can generate or consume over time.
- <a id="learning-rate"></a>**learning_rate**: The step size used when updating model weights.
- <a id="eval-cadence"></a>**eval cadence**: How often evaluation runs during training.
- <a id="held-out-eval"></a>**held-out eval**: Evaluation on examples not used for training, used to detect overfitting or regressions.
- <a id="validation"></a>**validation / val**: A lighter-weight periodic check during training, usually on held-out examples.
- <a id="reference-model"></a>**reference model**: The model used as the anchor when measuring or constraining policy drift.
- <a id="policy-drift"></a>**policy drift**: The amount the trained model's behavior has moved away from the starting or reference model.
- <a id="checkpoint"></a>**checkpoint**: A saved snapshot of training state or model weights.
- <a id="adapter"></a>**adapter**: A lightweight trained module attached to a base model, often used instead of saving a full model copy.
- <a id="lora"></a>**LoRA**: Low-Rank Adaptation, a common adapter training method that updates small low-rank matrices instead of all model weights.
- <a id="sft"></a>**SFT**: Supervised fine-tuning, training a model to imitate example outputs.
- <a id="warm-start"></a>**warm start**: Starting optimization from a stronger initial model or adapter instead of the untouched base model.
- <a id="teacher-model"></a>**teacher model**: A stronger model used to generate examples or guidance for training a smaller or target model.
- <a id="demonstration"></a>**demonstration**: An example of desired behavior, often generated by a teacher model for SFT.
- <a id="replay"></a>**replay**: Reusing saved demonstrations or samples instead of generating fresh ones.
- <a id="loss"></a>**loss**: The numeric objective training minimizes; in this repo, `loss = "sft"` switches from RL to supervised fine-tuning.
- <a id="reward-hacking"></a>**reward hacking**: When a model learns to exploit the scoring rule without solving the intended task.
- <a id="exploration"></a>**exploration**: Sampling varied behavior so training can discover higher-reward strategies.
- <a id="oversampling"></a>**oversampling**: Sampling some task types more often than their natural frequency.
- <a id="online-difficulty-filtering"></a>**online difficulty filtering**: Adjusting task exposure during training based on how easy or hard recent tasks are.
- <a id="env-ratios"></a>**env_ratios**: Configured weights that control the mix when training across multiple environments.

## Model and Config Fields

- <a id="sampling"></a>**sampling**: Generation settings used when collecting model outputs, such as token limits, temperature, and reasoning controls.
- <a id="system-prompt"></a>**system prompt**: Instructions prepended to the conversation that define the model's role, task rules, and output format.
- <a id="max-tokens"></a>**max_tokens**: The maximum number of tokens the model may generate in one response.
- <a id="temperature"></a>**temperature**: A sampling knob that increases or decreases output randomness.
- <a id="reasoning-effort"></a>**reasoning_effort**: A model-specific knob controlling how much reasoning budget a supported model uses.
- <a id="enable-thinking"></a>**enable_thinking**: A model-specific toggle that enables explicit reasoning behavior for supported models.
- <a id="gepa"></a>**GEPA**: The prompt optimization workflow used by Lab to search for better prompts from environment feedback. The repo defines it by behavior, not by acronym expansion.
- <a id="reflection-model"></a>**reflection_model**: The model GEPA uses to reflect on failures and propose prompt changes.
- <a id="prompt-candidate"></a>**prompt candidate**: One proposed prompt variant being evaluated by prompt optimization.
- <a id="max-calls"></a>**max_calls**: A GEPA budget limiting total model or evaluator calls.
- <a id="num-train"></a>**num_train**: The number of training examples GEPA uses while optimizing a prompt.
- <a id="num-val"></a>**num_val**: The number of validation examples GEPA uses to compare prompt candidates.
- <a id="minibatch-size"></a>**minibatch_size**: The small batch size GEPA uses for each optimization step.
- <a id="max-concurrent"></a>**max_concurrent**: The maximum number of concurrent calls GEPA may run.
- <a id="endpoint"></a>**endpoint**: A named connection target for model inference, including model id, provider, URL, and key reference.
- <a id="endpoint-id"></a>**endpoint_id**: The short alias used to refer to an endpoint in commands and config files.
- <a id="provider"></a>**provider**: The service that serves a model, such as Prime Inference, OpenAI, Anthropic, or another API.
- <a id="openai-compatible"></a>**OpenAI-compatible**: An API shape that follows OpenAI chat-completions conventions, even if served by another provider.
- <a id="env-args"></a>**env_args**: Extra environment arguments passed at run time.
- <a id="save-results"></a>**save_results**: A config flag that controls whether eval results are persisted.
- <a id="save-to-environment"></a>**save_to_environment**: A GEPA flag that writes optimized prompts back into a local environment when possible.
- <a id="hosted-eval"></a>**hosted eval**: An evaluation run executed on Prime infrastructure rather than on the local machine.
- <a id="hosted-training"></a>**hosted training**: A training run executed on Prime infrastructure.
- <a id="inference-deployment"></a>**inference deployment**: A served model or adapter that can be queried through an inference API.
- <a id="hub-version"></a>**Hub version**: An immutable published version of an environment on the Environments Hub.

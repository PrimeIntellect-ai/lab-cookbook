# Environments and Evals

A Verifiers v1 environment is selected by a taskset and a harness.

- `Taskset`: typed tasks, prompts, tools, user simulator, rewards, and metrics.
- `Harness`: the rollout program that calls the model, such as the built-in `default`, `bash`, `rlm`, or a custom agent harness.
- `Runtime`: where the harness and colocated pieces run, chosen by harness config.
- `Trace`: the rollout record that scoring reads and eval saves.

Run the first eval:

```bash
uv run eval @ configs/01/first-eval.toml
```

The config is intentionally small:

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 10
num_rollouts = 2

[sampling]
max_tokens = 1024

[taskset]
id = "gsm8k"
```

The equivalent CLI form is:

```bash
uv run eval gsm8k -n 10 -r 2 --model openai/gpt-5.4-nano --sampling.max-tokens 1024
```

The eval output is a stream of serialized `Trace` objects. Rewards read the trace, not an ad hoc completion dict. The fields you will inspect most often are `trace.task`, `trace.assistant_messages`, `trace.tool_messages`, `trace.state`, `trace.info`, `trace.rewards`, `trace.metrics`, and `trace.stop_condition`.

## Multi-turn Example

Run Wordle with the default harness and a user simulator:

```bash
uv run eval @ configs/01/first-eval-suite.toml
```

This config adds a turn cap because the taskset can run multiple assistant/user turns:

```toml
max_turns = 6

[taskset]
id = "wordle"
num_tasks = 20

[harness]
id = "default"
```

Use `max_turns` at the env config level. It is enforced by the framework across harnesses.

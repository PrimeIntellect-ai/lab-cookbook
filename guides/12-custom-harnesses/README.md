# Custom Harnesses

Run third-party agent libraries inside Lab.

Most environments can use the default harness: the model receives a task, calls tools if available, and returns a final answer. Use a custom harness when the rollout is owned by another agent runtime, framework, or program.

The Taskset still owns tasks, rewards, and metrics. The Harness owns how the model or agent is executed.

## The Program Pattern

Local [opencode_harbor](../../environments/opencode_harbor/opencode_harbor.py) composes a Taskset and Harness separately:

```python
import harnesses as h
import tasksets as t
import verifiers as vf


def load_taskset(config: t.HarborTasksetConfig) -> t.HarborTaskset:
    return t.HarborTaskset(config=config)


def load_harness(config: h.OpenCodeConfig) -> h.OpenCode:
    return h.OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

Inside the harness program, route third-party model calls through the rollout endpoint:

```python
async def run_program(task: vf.Task, state: vf.State) -> vf.State:
    endpoint = state.get_endpoint_config(api="chat")
    # Build the framework client from endpoint["model"],
    # endpoint["api_base"], and endpoint["api_key"].
    ...
    return state
```

`get_endpoint_config` is appropriate here — inside an active harness program, not inside a reward function. See [Judges and Instruction Following](../07-judges-and-instruction-following/README.md) for the judge pitfall.

## Deep Agents

[primeintellect/langchain-deep-agents-env](https://app.primeintellect.ai/dashboard/environments/primeintellect/langchain-deep-agents-env) is a Hub example. The Taskset loads GSM8K rows and scores boxed answers. The Harness runs a LangChain Deep Agents program.

Run a small eval:

```bash
prime eval run primeintellect/langchain-deep-agents-env \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 1 \
  -t 2048
```

Or run with a config file:

```toml
# [configs/12/deep-agents-eval.toml](../../configs/12/deep-agents-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/langchain-deep-agents-env"
num_examples = 5
rollouts_per_example = 1
sampling_args = { max_tokens = 2048 }
```

```bash
prime eval run configs/12/deep-agents-eval.toml
```

## DSPy

[primeintellect/dspy-rlm](https://app.primeintellect.ai/dashboard/environments/primeintellect/dspy-rlm) shows the same split with DSPy.

Run a small eval:

```bash
prime eval run primeintellect/dspy-rlm \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 1 \
  -t 2048
```

Or run with a config file:

```toml
# [configs/12/dspy-rlm-eval.toml](../../configs/12/dspy-rlm-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/dspy-rlm"
num_examples = 5
rollouts_per_example = 1
sampling_args = { max_tokens = 2048 }
```

```bash
prime eval run configs/12/dspy-rlm-eval.toml
```

For a domain-specific DSPy example, use `dspy-flights`.

## When to Use One

Use a custom harness when:

- the rollout is controlled by a third-party agent library
- the agent makes its own model calls internally
- the environment needs to preserve framework-specific traces or artifacts
- you want the same Taskset to run against multiple harnesses

Do not add a custom harness just to expose a tool or change a system prompt. The default harness already handles those cases.

## Next

- [Lab Configuration](../../reference/lab-configuration.md) — accounts, secrets, Hub workflows, hosted runs, inference deployments
- [Legacy Environments](../13-legacy-environments/README.md) — older Rubric and `source()` patterns you may see in unmigrated Hub packages

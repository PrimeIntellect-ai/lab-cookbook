# Custom Harnesses

Run third-party agent libraries inside Lab.

Most environments can use the default harness: the model receives a task, calls tools if available, and returns a final answer. Use a custom harness when the rollout is owned by another agent runtime, framework, or program.

The Taskset still owns tasks, rewards, and metrics. The Harness owns how the model or agent is executed.

## The Program Pattern

Local [opencode_harbor](../../environments/opencode_harbor/opencode_harbor.py) composes a Taskset and Harness separately:

```python
import verifiers as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


class OpenCodeHarborTasksetConfig(HarborTasksetConfig):
    bundle_package: str | None = __name__


def load_taskset(config: OpenCodeHarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

The loader annotations define the concrete config types. `vf.load_taskset` and
`vf.load_harness` use those annotations to validate `config.taskset` and
`config.harness`, so environment entrypoints stay small.

Every guide so far used the framework's built-in rollout loop — the [setup → model turn → env reply → stop → render → cleanup](../04-prompt-optimization/README.md#how-a-multi-turn-rollout-runs) sequence, with the taskset attaching hooks and tools to it. A custom harness program replaces that loop wholesale. The program is one async function that owns the entire rollout: it gets the task and state, drives whatever external agent runtime it wraps, and returns the finished state. The taskset still owns tasks and rewards; the harness owns *how the turns happen*, and when you supply a program, the turns happen however the program says.

Inside the harness program, route third-party model calls through the rollout endpoint:

```python
async def run_program(task: vf.Task, state: vf.State) -> vf.State:
    endpoint = state.get_endpoint_config(api="chat")
    # Build the framework client from endpoint.model,
    # endpoint.base_url, and endpoint.api_key_var.
    ...
    return state
```

`get_endpoint_config` is appropriate here — inside an active harness program, not inside a reward function. See [Judges and Instruction Following](../07-judges-and-instruction-following/README.md) for the judge pitfall.

## Config Overrides

Tasksets and harnesses are tuned independently. For Harbor + OpenCode, the
taskset decides which tasks to load; the harness decides how many turns the CLI
agent gets; the nested program config decides how OpenCode itself is launched.

```toml
[[eval]]
env_id = "prime/opencode-harbor"

[eval.taskset]
task_names = ["regex-log"]

[eval.harness]
max_turns = 4

[eval.harness.program]
disabled_tools = ["webfetch", "question"]
```

```bash
prime eval run configs/10/opencode-harbor.toml
```

Use this split as the default rule:

- put dataset, task selection, task sandbox defaults, task tools, rewards, and
  metrics on the Taskset config
- put rollout limits, program execution, framework adapters, harness-owned
  tools, and artifact collection on the Harness config
- put command-specific settings under `harness.program`

## Deep Agents

[prime/langchain-deep-agents-wikispeedia](https://app.primeintellect.ai/dashboard/environments/prime/langchain-deep-agents-wikispeedia) is a Hub example. The Taskset owns the Wikispeedia graph, navigation tools, deterministic rewards, and metrics. The Harness adapts those tools into a LangChain Deep Agents program.

This is an advanced adapter example, not the golden shape for a first
environment. Most taskset logic should be attached directly to the Taskset or
Harness. Wikispeedia has extra module-level graph loading, adapter functions,
and metric factories because it bridges SNAP data and LangChain Deep Agents;
copy that structure only when you have a similarly large external protocol
boundary.

Run a small eval:

```bash
prime eval run prime/langchain-deep-agents-wikispeedia \
  -m openai/gpt-5.4-nano \
  -n 5 \
  -r 1 \
  -t 4096
```

Or run with a config file:

```toml
# [configs/12/deep-agents-eval.toml](../../configs/12/deep-agents-eval.toml)
model = "openai/gpt-5.4-nano"
save_results = true

[[eval]]
env_id = "prime/langchain-deep-agents-wikispeedia"
num_examples = 5
rollouts_per_example = 1
sampling_args = { max_tokens = 4096 }
```

```bash
prime eval run configs/12/deep-agents-eval.toml
```

## When to Use One

Use a custom harness when:

- the rollout is controlled by a third-party agent library
- the agent makes its own model calls internally
- the environment needs to preserve framework-specific traces or artifacts
- you want the same Taskset to run against multiple harnesses

For a new tool or system prompt, use the default harness. Custom harnesses are for rollouts owned by another runtime or program.

## Next

In [Best Practices](../13-best-practices/README.md), step back from any single environment and walk through the habits that keep environments clean.

- [Lab Configuration](../../reference/lab-configuration.md) - thin pointer to managed and public platform docs
- [Legacy Environments](../14-legacy-environments/README.md) — older Rubric and `source()` patterns you may see in unmigrated Hub packages

# Custom Harnesses

Run third-party agent libraries inside Lab.

Most environments can use the default harness: the model receives a task, calls tools if available, and returns a final answer. Use a custom harness when the rollout is owned by another agent runtime, framework, or program.

The Taskset still owns the task rows, metrics, and rewards. The Harness owns how the model or agent is executed.

## The Program Pattern

A custom harness usually starts with a program:

```python
import verifiers as vf


async def run_program(task, state):
    endpoint = state.get_endpoint_config(api="chat")

    # Build the third-party client from endpoint["model"],
    # endpoint["api_base"], and endpoint["api_key"].
    # Run the framework agent on this task.
    # Store the final answer and any useful artifacts in state.

    state["completion"] = [{"role": "assistant", "content": final_answer}]
    return state


def load_harness(config=None):
    return vf.Harness(program=run_program, config=config)
```

The important part is `state.get_endpoint_config(api="chat")`. It gives the framework the model, base URL, and API key for the current rollout, so calls made inside the third-party library are routed through Lab instead of bypassing the environment.

## Deep Agents

[`langchain-deep-agents-env`](https://app.primeintellect.ai/dashboard/environments/primeintellect/langchain-deep-agents-env) is the clearest first example. The Taskset uses GSM8K rows and a numeric-answer reward. The Harness runs a LangChain Deep Agents program with `deepagents.create_deep_agent`.

Run a small eval:

```bash
prime eval run primeintellect/langchain-deep-agents-env \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 1 \
  -t 2048
```

```text
TODO: expected output
```

Inspect:

- the task question
- the framework's final answer
- any framework trace or agent result saved into state
- whether the final answer format matches the reward
- whether the reward failure is a framework issue, answer issue, or scoring issue

In source, the package is `langchain_deep_agents_env`. It builds a LangChain chat model from `state.get_endpoint_config(api="chat")`, creates a Deep Agent, runs it on the task, and writes the final output back into Lab state.

## DSPy

[`dspy-rlm`](https://app.primeintellect.ai/dashboard/environments/primeintellect/dspy-rlm) shows the same pattern with DSPy. The Taskset again owns GSM8K rows and reward logic. The Harness runs a DSPy RLM program and routes DSPy's LM through the rollout endpoint config.

Run a small eval:

```bash
prime eval run primeintellect/dspy-rlm \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 1 \
  -t 2048
```

```text
TODO: expected output
```

In source, the package is `dspy_rlm`. The program creates a DSPy LM from the Lab endpoint config, runs the DSPy module, stores the answer in state, and lets the Taskset reward score the result.

For a more domain-specific DSPy example, use `dspy-flights`, whose source package is `dspy_flights`.

## When to Use One

Use a custom harness when:

- the rollout is controlled by a third-party agent library
- the agent makes its own model calls internally
- the environment needs to preserve framework-specific traces or artifacts
- you want the same Taskset to run against multiple harnesses

Do not add a custom harness just to expose a tool or change a system prompt. The default harness already handles those cases.

## Next

In [Advanced Tasksets and Harnesses](../11-advanced-tasksets-and-harnesses/README.md), you will go deeper on reusable Tasksets, reusable Harnesses, config sections, nested calls, and lower-level composition.

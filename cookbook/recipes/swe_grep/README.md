# SWE-grep

This recipe is inspired by Cognition’s [SWE-grep](https://cognition.ai/blog/swe-grep): a reinforcement learning setup for training a model to retrieve the right code context quickly.

Instead of optimizing for open-ended code generation, this environment optimizes for **efficient code search**. The model is rewarded for finding the right files, answering correctly, and using parallel tool calls well.

## Why this environment exists

`grep`-style search is still one of the most reliable ways to navigate a large codebase.

Compared with embedding-heavy retrieval pipelines, grep-based search has a few advantages:

- no vector database to manage
- direct access to exact code matches
- fast iteration on search patterns
- easy grounding in real file paths and line-level evidence

The challenge is that the model must learn to search **efficiently**, not just eventually. A strong agent should turn a high-level question like:

> How is the panning and zooming functionality implemented?

into a small number of targeted, parallel search operations that surface the right files quickly.

## Environment overview

The environment is implemented in `swe_grep.py` as `SweGrepEnv`, which extends `vf.SandboxEnv`.

The stack looks like this:

- `StatefulToolEnv`: gives the model tool access and preserves rollout state
- `SandboxEnv`: provisions a Prime sandbox for each rollout
- `SweGrepEnv`: customizes the sandbox and tools for grep-centric retrieval

See the Verifiers docs for more on stateful environments:
https://docs.primeintellect.ai/verifiers/environments#stateful-tool-environments

## Tools exposed to the model

`SweGrepEnv` removes the default bash tool and replaces it with three task-specific tools:

- `grep_tool`: search for text patterns with `ripgrep`
- `list_files`: inspect directory contents
- `read_file`: read bounded line ranges from a file

```python
self.remove_tool(self.bash)
self.add_tool(self.grep_tool, args_to_skip=["sandbox_id"])
self.add_tool(self.list_files, args_to_skip=["sandbox_id"])
self.add_tool(self.read_file, args_to_skip=["sandbox_id"])
```

This keeps the action space narrow and focuses learning on search behavior rather than arbitrary shell usage.

## Stateful tool pattern

Each rollout gets its own Prime sandbox. The environment injects `sandbox_id` into tool calls so the model does not have to manage sandbox state itself.

```python
def update_tool_args(self, tool_name: str, tool_args: dict[str, Any], messages, state, **kwargs):
    updated_args = dict(tool_args)
    if tool_name in ["grep_tool", "list_files", "read_file"]:
        updated_args["sandbox_id"] = state["sandbox_id"]
    return updated_args
```

This is the core `StatefulToolEnv` pattern: keep persistent rollout state in `state`, and let the environment handle internal bookkeeping.

## Sandbox setup

For each rollout, the sandbox is prepared by:

1. installing `git` and `ripgrep`
2. cloning the VS Code repository
3. verifying that the clone succeeded

The model then searches that repo to answer questions.

## Dataset

The dataset is loaded from `cdreetz/swe-grep-v2` and filtered to examples where `check == "Yes"`.

During preprocessing:

- `user_query` is renamed to `question`
- `ground_truth` is renamed to `answer`
- `file_path` and `file_path_2` are preserved for reward computation
- the dataset is split into train and eval sets

The examples are synthetic but grounded in real code from Microsoft’s VS Code repository. The goal is to train retrieval behavior on realistic developer questions paired with technical explanations and source files.

For more detail on the dataset generation pipeline, see:
https://app.primeintellect.ai/dashboard/environments/prime/swe-grep/files/frt126ew7h8p1fud3bwl9ceu/src/create_dataset.py

## Reward design

This recipe uses a `vf.JudgeRubric` with three active rewards and one tracking metric:

- **Correct answer** (`0.4`): did the model produce the right technical explanation?
- **Correct file paths** (`0.4`): did it identify the relevant file or files?
- **Parallel tool calls** (`0.2`): did it use available tool parallelism effectively?
- **Efficiency bonus** (`0.0`): among correct rollouts, reward fewer turns

```python
rubric = vf.JudgeRubric(judge_prompt=JUDGE_PROMPT)
rubric.add_reward_func(correct_answer_reward_func, weight=0.4)
rubric.add_reward_func(correct_file_paths_reward_func, weight=0.4)
rubric.add_reward_func(parallel_tool_calls_reward_func, weight=0.2)
rubric.add_reward_func(efficiency_bonus_for_correct, weight=0.0)
```

A few notable design choices:

- correctness is judged semantically, not by exact string match
- multi-file tasks are supported via `file_path` and `file_path_2`
- the environment explicitly encourages parallelism
- the default system prompt constrains the agent to **2 turns**, increasing pressure to search well

## Agent behavior being optimized

The system prompt pushes the model toward a very specific behavior profile:

- use tools aggressively
- make multiple tool calls per turn
- gather evidence from all relevant files
- return both file paths and a final answer

Expected response format:

```text
Files:
- <path/to/file1>
- <path/to/file2>
Answer: <your answer here>
```

## Quick start

From this recipe directory, install dependencies and run eval through Verifiers or Prime tooling.

### Environment entrypoint

`pyproject.toml` should point at:

```toml
[tool.verifiers.environment]
entrypoint = "swe_grep:load_environment"
```

### Eval defaults currently present

```toml
[tool.verifiers.eval]
num_examples = 5
rollouts_per_example = 3
```

### Python usage

```python
from swe_grep import load_environment

env = load_environment()
```

## Files

- `swe_grep.py`: environment, tools, prompt, dataset loading, and rewards
- `src/create_dataset.py`: dataset generation pipeline
- `src/sandbox_metrics.py`: sandbox execution metrics and retry helpers

## Notes and limitations

- The current eval defaults are very small (`5 x 3`) and seem intended for quick iteration rather than robust benchmarking.
- Reward quality depends on judge quality, so score stability may vary across judge models.
- The environment is intentionally opinionated: it trains search behavior under strict turn limits rather than general software engineering performance.

## Environment Hub

Prime Environment Hub:
https://app.primeintellect.ai/dashboard/environments/prime/swe-grep

## Summary

This recipe is a compact example of RL for retrieval behavior:

- ground the model in a real repository
- give it a small, focused tool set
- reward correctness, coverage, and speed
- encourage parallel search under tight constraints

It is not the only way to train a strong grep agent, but it is a clear and practical starting point.

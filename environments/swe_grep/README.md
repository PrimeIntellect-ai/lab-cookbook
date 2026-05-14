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

The environment is implemented in `swe_grep.py` as a v1 `Taskset` plus the default `Harness`.

The stack looks like this:

- `SweGrepTasksetConfig`: owns dataset, judge, prompt, and sandbox defaults
- `vf.Taskset`: exposes train/eval rows, toolsets, and reward signals
- `vf.Toolset`: provisions one Prime sandbox per rollout and exposes grep-specific tools
- `vf.Harness`: runs the default endpoint-backed tool loop

See the Verifiers docs for more on tasksets and harnesses:
https://docs.primeintellect.ai/verifiers/byo-harness

## Tools exposed to the model

The taskset exposes three task-specific tools:

- `grep_tool`: search for text patterns with `ripgrep`
- `list_files`: inspect directory contents
- `read_file`: read bounded line ranges from a file

This keeps the action space narrow and focuses learning on search behavior rather than arbitrary shell usage.

## Taskset tool pattern

Each rollout gets its own Prime sandbox through the v1 `Toolset` sandbox config. The model never sees or manages sandbox IDs; the runtime injects the sandbox handle into the Python tool call.

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

This recipe uses v1 reward signals owned by the taskset:

- **Correct answer** (`0.4`): did the model produce the right technical explanation?
- **Correct file paths** (`0.4`): did it identify the relevant file or files?
- **Parallel tool calls** (`0.2`): did it use available tool parallelism effectively?

A few notable design choices:

- answer correctness is judged semantically by an LLM judge, run once per rollout as an `@vf.update` and read by the reward
- file-path coverage is a cheap substring check against the agent's final message (the system prompt forces a `Files:`/`Answer:` shape, so the paths appear verbatim)
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

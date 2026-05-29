# Judges and Instruction Following

Three environments, in order:

1. [simple-judge](../../environments/simple_judge/simple_judge.py) — local Taskset, one yes/no judge criterion per task
2. [prime/ifeval](https://app.primeintellect.ai/dashboard/environments/primeintellect/ifeval) — [google/IFEval](https://huggingface.co/datasets/google/IFEval), programmatic constraint checks
3. [will/advanced-if](https://app.primeintellect.ai/dashboard/environments/will/advanced-if) — [facebook/AdvancedIF](https://huggingface.co/datasets/facebook/AdvancedIF), multiple rubric bullets per task

## Part 1: simple-judge

Each task stores a criterion in `info`. `judge_reward` calls an LLM and parses `yes` / `no`.

```bash
prime env install simple-judge
prime eval run simple-judge -m openai/gpt-4.1-mini -n 6 -r 2
```

```toml
# [configs/07/simple-judge-eval.toml](../../configs/07/simple-judge-eval.toml)
model = "openai/gpt-4.1-mini"
save_results = true

[[eval]]
env_id = "simple-judge"
num_examples = 6
rollouts_per_example = 2
sampling_args = { max_tokens = 256 }
```

Judge settings live on `TasksetConfig`:

```python
class SimpleJudgeTasksetConfig(vf.TasksetConfig):
    judge_model: str = "openai/gpt-4.1-mini"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    judge_api_key_var: str = "PRIME_API_KEY"
```

Override with `-a` and a nested `taskset` object, or a `taskset` table in eval
configs:

```bash
prime eval run simple-judge \
  -m openai/gpt-4.1-mini \
  -a '{"taskset": {"judge_model": "openai/gpt-5-mini"}}'
```

```toml
[[eval]]
env_id = "simple-judge"
taskset = { judge_model = "openai/gpt-5-mini", judge_api_key_var = "PRIME_API_KEY" }
```

Call `vf.ensure_keys(...)` in `load_taskset` if the env requires API keys.

In reward functions, use a dedicated `AsyncOpenAI` client built from those config fields — not `state.get_endpoint_config(api="chat")`, which is tied to the rollout proxy.

Implementation: [environments/simple_judge/simple_judge.py](../../environments/simple_judge/simple_judge.py).

## Part 2: IFEval

```bash
prime eval run prime/ifeval -m openai/gpt-4.1-mini -n 10 -r 1 -t 1024
```

```toml
# [configs/07/ifeval-eval.toml](../../configs/07/ifeval-eval.toml)
model = "openai/gpt-4.1-mini"
save_results = true

[[eval]]
env_id = "prime/ifeval"
num_examples = 10
rollouts_per_example = 1
sampling_args = { max_tokens = 1024 }
taskset = { mode = "strict" }
```

`mode` is `"strict"` or `"loose"`. Inspect the Hub package for reward and metric names.

## Part 3: AdvancedIF

```bash
prime eval run will/advanced-if -m openai/gpt-4.1-mini -n 5 -r 1 -t 2048
```

```toml
# [configs/07/advanced-if-eval.toml](../../configs/07/advanced-if-eval.toml)
model = "openai/gpt-4.1-mini"
save_results = true

[[eval]]
env_id = "will/advanced-if"
num_examples = 5
rollouts_per_example = 1
sampling_args = { max_tokens = 2048 }
```

Rubric lines are in `info["rubrics"]`. Inspect [will/advanced-if](https://app.primeintellect.ai/dashboard/environments/will/advanced-if) for judge call shape and aggregation.

## Next

[Tool Use and Search](../08-tool-use-and-search/README.md) — `wiki-search` uses the same direct-judge wiring on top of retrieval tools.

Unmigrated Hub packages (`JudgeRubric`, `source()`, etc.): [Legacy Environments](../13-legacy-environments/README.md).

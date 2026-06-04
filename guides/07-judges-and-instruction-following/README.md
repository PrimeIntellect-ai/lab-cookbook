# Judges and Instruction Following

This guide uses [simple-judge](../../environments/simple_judge/simple_judge.py), a local v1 Taskset with one yes/no judge criterion per task. It is the smallest useful pattern for adding LLM-judged instruction following to a Lab environment.

## Build the Judge Environment

Each task stores a criterion in `info`. `judge_reward` calls an LLM and parses `yes` / `no`.

```bash
prime env install simple-judge
prime eval run prime/simple-judge -m openai/gpt-4.1-mini -n 6 -r 2
```

```toml
# [configs/07/simple-judge-eval.toml](../../configs/07/simple-judge-eval.toml)
model = "openai/gpt-4.1-mini"
save_results = true

[[eval]]
env_id = "prime/simple-judge"
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

Override judge settings on the taskset config in eval TOML:

```toml
[[eval]]
env_id = "prime/simple-judge"

[eval.taskset]
judge_model = "openai/gpt-5-mini"
judge_api_key_var = "PRIME_API_KEY"
```

Call `vf.ensure_keys(...)` from the component that owns the dependency. For a
direct judge reward, that means the Taskset reward method; for embedding-backed
tools, that means `load_toolsets(config)`.

In reward functions, use a dedicated `AsyncOpenAI` client built from those config fields. `state.get_endpoint_config(api="chat")` belongs inside harness programs that route the agent's own model calls through the rollout proxy.

Implementation: [environments/simple_judge/simple_judge.py](../../environments/simple_judge/simple_judge.py).

## Next

[Tool Use and Search](../08-tool-use-and-search/README.md) — `wiki-search` uses the same direct-judge wiring on top of retrieval tools.

Unmigrated Hub packages (`JudgeRubric`, `source()`, etc.): [Legacy Environments](../13-legacy-environments/README.md).

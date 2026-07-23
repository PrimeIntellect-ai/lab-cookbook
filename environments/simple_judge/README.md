# simple-judge

A tiny latest-verifiers-v1 instruction-following taskset scored by an LLM judge.

- `SimpleJudgeTaskData` carries each prompt's grading criterion.
- `SimpleJudgeTask` owns the one-turn stop and judge-backed reward.
- `SimpleJudge` records its response and billed usage on the rollout trace.
- Judge settings live under the task config.

Run:

```bash
uv run eval @ configs/07/simple-judge-eval.toml
```

```toml
[taskset]
id = "simple-judge"

[taskset.task.judge]
model = "openai/gpt-4.1-mini"
```

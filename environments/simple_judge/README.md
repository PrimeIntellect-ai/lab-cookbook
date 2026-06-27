# simple-judge

A tiny v1 instruction-following taskset scored by an LLM judge.

- Task: prompt plus criterion
- Config: nested `judge` client config
- Reward: judge replies `yes` or `no`
- Stop: one model turn

Run:

```bash
uv run eval @ configs/07/simple-judge-eval.toml
```

```toml
[taskset]
id = "simple-judge"

[taskset.judge]
model = "openai/gpt-4.1-mini"
```

# gsm8k

A v1 taskset for GSM8K grade-school math.

- Config: `split = "test" | "train"`
- Task: problem prompt plus numeric final answer
- Reward: exact match against the last `#### <answer>` in the final assistant message
- Stop: one model turn

Run:

```bash
uv run eval @ configs/01/first-eval.toml
uv run eval gsm8k -n 10 -r 2
```

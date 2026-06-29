# Judges and Instruction Following

Use an LLM judge when the reward is semantic and exact matching would be too narrow. In v1, the judge configuration is a normal nested taskset config.

`environments/simple_judge/simple_judge.py` defines:

```python
class JudgeConfig(vf.BaseClientConfig):
    model: str = "openai/gpt-4.1-mini"


class SimpleJudgeConfig(vf.TasksetConfig):
    judge: JudgeConfig = JudgeConfig()
```

The reward builds a judge prompt from the typed task and the finished trace:

```python
@vf.reward(weight=1.0)
async def judge_reward(self, task: SimpleJudgeTask, trace: vf.Trace) -> float:
    response_text = trace.assistant_messages[-1].content if trace.assistant_messages else ""
    client = vf.resolve_client(self.config.judge)
    try:
        verdict = await client.get_response(...)
    finally:
        await client.close()
    return float("yes" in (verdict.message.content or "").lower())
```

Configure it through TOML:

```toml
[taskset]
id = "simple-judge"

[taskset.judge]
model = "openai/gpt-4.1-mini"
```

Judge rewards should be narrow: one rubric question, explicit allowed answers, and no hidden mutable state.

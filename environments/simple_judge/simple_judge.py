import verifiers.v1 as vf
from verifiers.v1.dialects import ChatDialect

JUDGE_PROMPT = """You are grading a short model response against one criterion.

Criterion:
```
{criterion}
```

User message:
```
{user_message}
```

Model response:
```
{response}
```

Does the response satisfy the criterion? Reply with exactly one word: yes or no.
"""

SYSTEM = "Follow the user instruction carefully. Keep answers short."

TOY_TASKS = [
    {
        "prompt": "Write one cheerful sentence about mornings.",
        "criterion": "The response sounds upbeat and enthusiastic.",
    },
    {
        "prompt": "Write one formal sentence declining a party invitation.",
        "criterion": "The response politely declines without sounding rude or overly casual.",
    },
    {
        "prompt": "Explain photosynthesis in one sentence for a five-year-old.",
        "criterion": "The response uses simple words a young child could follow.",
    },
    {
        "prompt": "Write one sentence recommending a book you love.",
        "criterion": "The response names a specific book title.",
    },
    {
        "prompt": "Apologize in one sentence for arriving late to a meeting.",
        "criterion": "The response clearly apologizes and acknowledges being late.",
    },
    {
        "prompt": "Write one sentence that refuses to help with cheating on a test.",
        "criterion": "The response refuses the request and does not provide cheating advice.",
    },
]


class SimpleJudgeTask(vf.Task):
    criterion: str


class JudgeConfig(vf.BaseClientConfig):
    model: str = "openai/gpt-4.1-mini"


class SimpleJudgeConfig(vf.TasksetConfig):
    judge: JudgeConfig = JudgeConfig()


class SimpleJudgeTaskset(vf.Taskset[SimpleJudgeTask, SimpleJudgeConfig]):
    def load_tasks(self) -> list[SimpleJudgeTask]:
        return [
            SimpleJudgeTask(
                idx=i,
                prompt=row["prompt"],
                system_prompt=SYSTEM,
                criterion=row["criterion"],
            )
            for i, row in enumerate(TOY_TASKS)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: SimpleJudgeTask, trace: vf.Trace) -> float:
        response_text = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        prompt = JUDGE_PROMPT.format(
            criterion=task.criterion,
            user_message=task.prompt,
            response=response_text or "",
        )
        client = vf.resolve_client(self.config.judge)
        try:
            verdict = await client.get_response(
                ChatDialect(),
                {"messages": [{"role": "user", "content": prompt}]},
                self.config.judge.model,
                vf.SamplingConfig(),
            )
        finally:
            await client.close()
        return float("yes" in (verdict.message.content or "").lower())


__all__ = ["SimpleJudgeTaskset"]

import os

import verifiers as vf
from openai import AsyncOpenAI

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


TOY_TASKS: list[dict] = [
    {
        "prompt": [{"role": "user", "content": "Write one cheerful sentence about mornings."}],
        "info": {"criterion": "The response sounds upbeat and enthusiastic."},
    },
    {
        "prompt": [
            {"role": "user", "content": "Write one formal sentence declining a party invitation."}
        ],
        "info": {
            "criterion": "The response politely declines without sounding rude or overly casual."
        },
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "Explain photosynthesis in one sentence for a five-year-old.",
            }
        ],
        "info": {"criterion": "The response uses simple words a young child could follow."},
    },
    {
        "prompt": [{"role": "user", "content": "Write one sentence recommending a book you love."}],
        "info": {"criterion": "The response names a specific book title."},
    },
    {
        "prompt": [
            {"role": "user", "content": "Apologize in one sentence for arriving late to a meeting."}
        ],
        "info": {"criterion": "The response clearly apologizes and acknowledges being late."},
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "Write one sentence that refuses to help with cheating on a test.",
            }
        ],
        "info": {
            "criterion": "The response refuses the request and does not provide cheating advice."
        },
    },
]


class SimpleJudgeTasksetConfig(vf.TasksetConfig):
    judge_model: str = "openai/gpt-oss-120b"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    judge_api_key_var: str = "PRIME_API_KEY"
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = (
        "Follow the user instruction carefully. Keep answers short."
    )


class SimpleJudgeTaskset(vf.Taskset[SimpleJudgeTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        _ = split
        for row in TOY_TASKS:
            yield row

    @vf.reward(weight=1.0)
    async def judge_reward(self, task: vf.Task, state: vf.State) -> float:
        response_text = ""
        for message in reversed(state.get("completion") or []):
            if isinstance(message, dict) and message.get("role") == "assistant":
                response_text = str(message.get("content") or "")
                break
        user_message = ""
        for message in task.get("prompt") or []:
            if isinstance(message, dict) and message.get("role") == "user":
                user_message = str(message.get("content") or "")
                break
        info = task.get("info") or {}
        criterion = str(info.get("criterion") or "")
        judge = AsyncOpenAI(
            api_key=os.getenv(self.config.judge_api_key_var, ""),
            base_url=self.config.judge_base_url,
        )
        try:
            response = await judge.chat.completions.create(
                model=self.config.judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            criterion=criterion,
                            user_message=user_message,
                            response=response_text,
                        ),
                    }
                ],
            )
        finally:
            await judge.close()
        text = response.choices[0].message.content or ""
        return 1.0 if "yes" in text.lower() else 0.0


def load_taskset(config: SimpleJudgeTasksetConfig) -> SimpleJudgeTaskset:
    vf.ensure_keys([config.judge_api_key_var])
    return SimpleJudgeTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )

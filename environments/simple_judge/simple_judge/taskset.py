"""A tiny single-turn instruction-following taskset scored by an LLM judge."""

import json
from pathlib import Path
from typing import cast

import verifiers.v1 as vf

TASKS_FILE = Path(__file__).parent / "tasks.json"

JUDGE_PROMPT = """You are grading a short model response against one criterion.

Criterion:
```
{criterion}
```

User message:
```
{question}
```

Model response:
```
{response}
```

Does the response satisfy the criterion? Reply with exactly one word: yes or no.
"""

SYSTEM = "Follow the user instruction carefully. Keep answers short."


class SimpleJudge(vf.Judge[bool]):
    prompt = JUDGE_PROMPT

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return response.text.strip().lower().startswith("yes")


class SimpleJudgeTaskData(vf.TaskData):
    criterion: str


class SimpleJudgeTaskConfig(vf.TaskConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig(model="openai/gpt-4.1-mini")


class SimpleJudgeTask(vf.Task[SimpleJudgeTaskData, vf.State, SimpleJudgeTaskConfig]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def judge_reward(self, trace: vf.Trace) -> float:
        config = cast(SimpleJudgeTaskConfig, self.config)
        result = await SimpleJudge(config.judge).evaluate(
            trace=trace,
            criterion=self.data.criterion,
            question=self.data.prompt_text,
            response=trace.last_reply,
        )
        return float(result.parsed is True)


class SimpleJudgeConfig(vf.TasksetConfig):
    task: SimpleJudgeTaskConfig = SimpleJudgeTaskConfig()


class SimpleJudgeTaskset(
    vf.Taskset[SimpleJudgeTask, SimpleJudgeConfig]  # ty: ignore[invalid-type-arguments]
):
    def load(self) -> list[SimpleJudgeTask]:
        rows = json.loads(TASKS_FILE.read_text())
        return [
            SimpleJudgeTask(
                SimpleJudgeTaskData(
                    idx=i,
                    prompt=row["prompt"],
                    system_prompt=SYSTEM,
                    criterion=row["criterion"],
                ),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]

import re
from typing import Literal

import verifiers.v1 as vf

SYSTEM = (
    "Solve the grade-school math problem. Reason step by step, then give the final "
    "answer as a single number on the last line, prefixed with '#### ' (for example, '#### 42')."
)
FINAL_ANSWER = re.compile(r"####\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?)")


class GSM8KTask(vf.Task):
    answer: str


class GSM8KConfig(vf.TasksetConfig):
    split: Literal["train", "test"] = "test"


class GSM8KTaskset(vf.Taskset[GSM8KTask, GSM8KConfig]):
    def load_tasks(self) -> list[GSM8KTask]:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split=self.config.split)
        return [
            GSM8KTask(
                idx=i,
                prompt=f"{SYSTEM}\n\n{row['question']}",
                answer=row["answer"].split("####")[-1].strip(),
            )
            for i, row in enumerate(rows)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(self, task: GSM8KTask, trace: vf.Trace) -> float:
        completion = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        matches = FINAL_ANSWER.findall(completion or "")
        return float(bool(matches) and matches[-1].strip() == task.answer)


__all__ = ["GSM8KTaskset"]

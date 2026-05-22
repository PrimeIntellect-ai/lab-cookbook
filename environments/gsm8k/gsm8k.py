import verifiers.v1 as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


@vf.reward(weight=1.0)
async def correct_answer(task: vf.Task, state: vf.State) -> float:
    if len(state.get("completion") or []) > 0:
        text = str(state.get("completion")[-1].get("content") or "")
        response = extract_boxed_answer(text, strict=True)
        return 1.0 if response == str(task["answer"]) else 0.0
    return 0.0


def source():
    ds = load_example_dataset("gsm8k", split="train")
    for index, row in enumerate(ds):
        assert isinstance(row, dict), "Dataset rows must be dicts."
        yield dict(
            row,
            example_id=index,
            prompt=[{"role": "user", "content": str(row["question"])}],
            answer=str(row["answer"]),
        )


def eval_source():
    ds = load_example_dataset("gsm8k", split="test")
    for index, row in enumerate(ds):
        assert isinstance(row, dict), "Dataset rows must be dicts."
        yield dict(
            row,
            example_id=index,
            prompt=[{"role": "user", "content": str(row["question"])}],
            answer=str(row["answer"]),
        )


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.Taskset(
            source=source,
            eval_source=eval_source,
            system_prompt=BOXED_SYSTEM_PROMPT,
            rewards=[correct_answer],
            config=config.taskset,
        ),
    )

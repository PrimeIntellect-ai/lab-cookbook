from difflib import SequenceMatcher

from datasets import load_dataset

import verifiers.v1 as vf


DATASET_NAME = "PrimeIntellect/Reverse-Text-RL"

SYSTEM_PROMPT = (
    "Reverse the text character-by-character. Put your answer in "
    "<reversed_text> tags."
)


def source():
    rows = []
    for row in load_dataset(DATASET_NAME, split="train"):
        text = row["prompt"]
        rows.append({
            "prompt": [{"role": "user", "content": text}],
            "answer": text[::-1],
        })
    return rows


@vf.reward(weight=1.0)
async def lcs_reward(task, state) -> float:
    text = state["completion"][-1]["content"]
    response = text.split("<reversed_text>", 1)[-1].split("</reversed_text>", 1)[0].strip()
    return SequenceMatcher(None, response, task["answer"]).ratio()


def load_taskset(config: vf.TasksetConfig) -> vf.Taskset:
    return vf.Taskset(
        source=source,
        system_prompt=SYSTEM_PROMPT,
        rewards=[lcs_reward],
        config=config,
    )


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=load_taskset(config=config.taskset))

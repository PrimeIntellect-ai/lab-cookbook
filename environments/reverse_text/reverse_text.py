from difflib import SequenceMatcher

import verifiers.v1 as vf
from datasets import load_dataset

DATASET_NAME = "PrimeIntellect/Reverse-Text-RL"
SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."


@vf.reward(weight=1.0)
async def lcs_reward(task: vf.Task, state: vf.State) -> float:
    text = ""
    for message in reversed(state.get("completion") or []):
        if message.get("role") == "assistant":
            text = str(message.get("content") or "")
            break
    response = text.split("<reversed_text>", 1)[-1].split("</reversed_text>", 1)[0].strip()
    return SequenceMatcher(None, response, str(task["answer"])).ratio()


def source():
    for index, row in enumerate(load_dataset(DATASET_NAME, split="train")):
        text = str(row["prompt"])
        yield {
            "example_id": index,
            "prompt": [{"role": "user", "content": text}],
            "answer": text[::-1],
        }


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.Taskset(
            source=source,
            system_prompt=SYSTEM_PROMPT,
            rewards=[lcs_reward],
            config=config.taskset,
        ),
    )

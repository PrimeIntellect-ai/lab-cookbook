from __future__ import annotations

from collections.abc import Mapping
from difflib import SequenceMatcher

import verifiers.v1 as vf
from datasets import load_dataset

DATASET_NAME = "PrimeIntellect/Reverse-Text-RL"

SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."


class ReverseTextTasksetConfig(vf.TasksetConfig):
    dataset_name: str = DATASET_NAME
    dataset_split: str = "train"
    system_prompt: str | None = SYSTEM_PROMPT


def response_text(state: vf.State) -> str:
    completion = state.get("completion") or []
    for message in reversed(completion):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def reversed_text(text: str) -> str:
    return text.split("<reversed_text>", 1)[-1].split("</reversed_text>", 1)[0].strip()


def source(dataset_name: str, dataset_split: str):
    for index, row in enumerate(load_dataset(dataset_name, split=dataset_split)):
        text = str(row["prompt"])
        yield {
            "example_id": index,
            "prompt": [{"role": "user", "content": text}],
            "answer": text[::-1],
        }


@vf.reward(weight=1.0)
async def lcs_reward(task: vf.Task, state: vf.State) -> float:
    response = reversed_text(response_text(state))
    return SequenceMatcher(None, response, str(task["answer"])).ratio()


def load_taskset(
    config: vf.TasksetConfig | Mapping[str, object] | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    system_prompt: str | None = None,
) -> vf.Taskset:
    taskset_config = ReverseTextTasksetConfig.from_config(
        config,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        system_prompt=system_prompt,
    )
    return vf.Taskset(
        source=lambda: source(taskset_config.dataset_name, taskset_config.dataset_split),
        system_prompt=taskset_config.system_prompt,
        rewards=[lcs_reward],
        config=taskset_config,
    )


def load_harness(
    config: vf.HarnessConfig | Mapping[str, object] | None = None,
) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(
    config: vf.EnvConfig | Mapping[str, object] | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    system_prompt: str | None = None,
) -> vf.Env:
    config = vf.EnvConfig.from_config(
        config,
        taskset=ReverseTextTasksetConfig.from_config(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            system_prompt=system_prompt,
        ),
    )
    return vf.Env(
        taskset=load_taskset(
            config=config.taskset,
        ),
        harness=load_harness(config=config.harness),
    )

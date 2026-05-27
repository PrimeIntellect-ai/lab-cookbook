from difflib import SequenceMatcher

import verifiers as vf
from datasets import load_dataset


class ReverseTextTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    # loaders
    def load_tasks(self) -> vf.Tasks:
        ds = load_dataset(self.config.dataset_name, split=self.config.dataset_split).map(
            lambda x: {"prompt": x["prompt"], "answer": x["prompt"][::-1]}
        )
        return ds

    def load_system_prompt(self) -> vf.SystemPrompt:
        return "Reverse the text character-by-character. Put your answer in <reversed_text> tags."

    # rewards
    @vf.reward(weight=1.0)
    async def lcs_reward(self, task: vf.Task, state: vf.State) -> float:
        text = ""
        for message in reversed(state.get("completion") or []):
            if message.get("role") == "assistant":
                text = str(message.get("content") or "")
                break
        response = text.split("<reversed_text>", 1)[-1].split("</reversed_text>", 1)[0].strip()
        return SequenceMatcher(None, response, str(task["answer"])).ratio()


def load_taskset(config: ReverseTextTasksetConfig) -> vf.Taskset:
    return ReverseTextTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))

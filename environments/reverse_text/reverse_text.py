from difflib import SequenceMatcher

import verifiers as vf
from datasets import load_dataset


class ReverseTextTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    train_split: str = "train"
    eval_split: str = "train"
    system_prompt: vf.SystemPrompt = (
        "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
    )


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        source_split = self.config.train_split if split == "train" else self.config.eval_split
        dataset = load_dataset(self.config.dataset_name, split=source_split)
        return dataset.rename_column("prompt", "question").map(
            lambda row: {
                "answer": row["question"][::-1],
            }
        )

    @vf.reward(weight=1.0)
    async def lcs_reward(self, task: vf.Task, state: vf.State) -> float:
        messages = vf.get_messages(state.get("completion") or [], role="assistant")
        text = str(messages[-1].content or "") if messages else ""
        response = text.split("<reversed_text>", 1)[-1].split("</reversed_text>", 1)[0].strip()
        return SequenceMatcher(None, response, str(task["answer"])).ratio()


def load_taskset(config: ReverseTextTasksetConfig) -> ReverseTextTaskset:
    return ReverseTextTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

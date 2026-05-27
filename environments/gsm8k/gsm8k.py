import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


class Gsm8kTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "gsm8k"
    train_split: str = "train"
    eval_split: str = "test"


class Gsm8kTaskset(vf.Taskset[Gsm8kTasksetConfig]):
    # loaders
    def load_tasks(self) -> vf.Tasks:
        return load_example_dataset(self.config.dataset_name, split=self.config.train_split).map(
            lambda x: {
                "prompt": [{"role": "user", "content": x["question"]}],
                "answer": x["answer"],
            }
        )

    def load_eval_tasks(self) -> vf.Tasks:
        return load_example_dataset(self.config.dataset_name, split=self.config.eval_split).map(
            lambda x: {
                "prompt": [{"role": "user", "content": x["question"]}],
                "answer": x["answer"],
            }
        )

    def load_system_prompt(self) -> vf.SystemPrompt:
        return BOXED_SYSTEM_PROMPT

    # rewards
    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        completion = state.get("completion")
        assert isinstance(completion, list)
        if not completion:
            return 0.0
        last = completion[-1]
        if not isinstance(last, vf.AssistantMessage):
            return 0.0
        content = last.content
        assert isinstance(content, str)
        response = extract_boxed_answer(content, strict=True)
        return 1.0 if response == str(task["answer"]) else 0.0


def load_taskset(config: Gsm8kTasksetConfig) -> vf.Taskset:
    return Gsm8kTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))

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
    system_prompt: vf.SystemPrompt = BOXED_SYSTEM_PROMPT


class Gsm8kTaskset(vf.Taskset[Gsm8kTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        source_split = self.config.train_split if split == "train" else self.config.eval_split
        return load_example_dataset(self.config.dataset_name, split=source_split)

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        completion = state.get("completion")
        assert isinstance(completion, list)
        assistant_messages = vf.get_messages(completion, role="assistant")
        if not assistant_messages:
            return 0.0
        content = assistant_messages[-1].content
        assert isinstance(content, str)
        response = extract_boxed_answer(content, strict=True)
        return 1.0 if response == str(task["answer"]) else 0.0


def load_taskset(config: Gsm8kTasksetConfig) -> Gsm8kTaskset:
    return Gsm8kTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

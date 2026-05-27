import re
from pathlib import Path

import verifiers as vf
from tasksets.textarena import (
    TextArenaTaskset,
    TextArenaTasksetConfig,
)

GUESS_PATTERN = re.compile(r"<guess>(.*?)</guess>", re.DOTALL)
SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""


class WordleTasksetConfig(TextArenaTasksetConfig):
    # textarena config fields
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    # taskset-specific fields
    prompt_path: str | None = None


class WordleTaskset(TextArenaTaskset[WordleTasksetConfig]):
    # loaders + overrides
    def load_system_prompt(self) -> vf.SystemPrompt:
        if self.config.prompt_path:
            return Path(self.config.prompt_path).read_text()
        return SYSTEM_PROMPT

    def format_observation(self, observation: str) -> str:
        latest_observation = observation.split("[GAME]")[-1].strip()
        if "Feedback:" in latest_observation:
            return latest_observation.split("Feedback:")[-1]
        return latest_observation

    # rewards
    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = str(task["answer"])
        completion = state.get("completion") or []
        if not isinstance(completion, list):
            return 0.0
        for message in reversed(vf.get_messages(completion)):
            if not isinstance(message, vf.AssistantMessage):
                continue
            content = message.content
            if not isinstance(content, str):
                continue
            matches = GUESS_PATTERN.findall(content)
            if matches:
                return 1.0 if matches[-1].strip() == f"[{answer}]" else 0.0
        return 0.0

    @vf.reward(weight=1.0)
    async def length_bonus(self, task: vf.Task, state: vf.State) -> float:
        answer = str(task["answer"])
        completion = state.get("completion") or []
        if not isinstance(completion, list):
            return 0.0
        guess = ""
        num_guesses = 0
        for message in vf.get_messages(completion):
            if not isinstance(message, vf.AssistantMessage):
                continue
            content = message.content
            if not isinstance(content, str):
                continue
            if GUESS_PATTERN.search(content):
                num_guesses += 1
                matches = GUESS_PATTERN.findall(content)
                if matches:
                    guess = matches[-1].strip()
        is_correct = 1.0 if guess == f"[{answer}]" else 0.0
        return is_correct / (num_guesses or 1)

    @vf.reward(weight=1.0)
    async def partial_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = str(task["answer"])
        completion = state.get("completion") or []
        if not isinstance(completion, list):
            return 0.0
        for message in reversed(vf.get_messages(completion)):
            if not isinstance(message, vf.AssistantMessage):
                continue
            content = message.content
            if not isinstance(content, str):
                continue
            matches = GUESS_PATTERN.findall(content)
            if matches and matches[-1].strip() == f"[{answer}]":
                return 0.0
            break
        for message in reversed(vf.get_messages(completion)):
            if not isinstance(message, vf.UserMessage):
                continue
            content = message.content
            if not isinstance(content, str):
                continue
            parts = content.strip().split("\n")
            if len(parts) == 3:
                scoring = parts[1].strip()
                return 0.2 * scoring.count("G") + 0.1 * scoring.count("Y")
        return 0.0

    @vf.reward(weight=0.2)
    async def format_reward(self, task: vf.Task, state: vf.State) -> float:
        del task
        completion = state.get("completion") or []
        if not isinstance(completion, list):
            return 0.0
        found = False
        for message in vf.get_messages(completion):
            if not isinstance(message, vf.AssistantMessage):
                continue
            found = True
            content = message.content
            if not isinstance(content, str):
                return 0.0
            if len(GUESS_PATTERN.findall(content)) != 1:
                return 0.0
        return 1.0 if found else 0.0


def load_taskset(config: WordleTasksetConfig) -> WordleTaskset:
    return WordleTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))

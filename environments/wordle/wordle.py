import re

import verifiers as vf
from tasksets.textarena import (
    TextArenaTaskset,
    TextArenaTasksetConfig,
    TextArenaUser,
)


class WordleTasksetConfig(TextArenaTasksetConfig):
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    system_prompt: vf.SystemPrompt = None


class WordleUser(TextArenaUser):
    async def get_response(
        self, task: vf.Task, state: vf.State, messages: list[vf.Message]
    ) -> list[vf.UserMessage]:
        response = await super().get_response(task, state, messages)
        if state.get("done") is True:
            return response
        if not response:
            return []
        content = str(response[-1].content or "")
        latest_feedback = content.split("[GAME]")[-1].strip()
        if "Feedback:" in latest_feedback:
            latest_feedback = latest_feedback.split("Feedback:")[-1]
        return [vf.UserMessage(content=latest_feedback)]


class WordleTaskset(TextArenaTaskset[WordleTasksetConfig]):
    guess_pattern = r"<guess>(.*?)</guess>"
    config: WordleTasksetConfig

    def load_user(self, config: vf.UserConfig) -> WordleUser:
        return WordleUser(config=config)

    def load_system_prompt(self, config: WordleTasksetConfig) -> vf.SystemPrompt:
        if config.system_prompt is not None:
            return config.system_prompt
        return vf.SystemPromptConfig(path="prompts/system_prompt.txt")

    def guesses(self, content: str) -> list[str]:
        return re.findall(self.guess_pattern, content, re.DOTALL)

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = str(task["answer"])
        completion = state.get("completion") or []
        for message in reversed(vf.get_messages(completion, role="assistant")):
            matches = self.guesses(str(message.content or ""))
            if matches:
                return 1.0 if matches[-1].strip() == f"[{answer}]" else 0.0
        return 0.0

    @vf.reward(weight=1.0)
    async def length_bonus(self, task: vf.Task, state: vf.State) -> float:
        answer = str(task["answer"])
        completion = state.get("completion") or []
        guess = ""
        num_guesses = 0
        for message in vf.get_messages(completion, role="assistant"):
            content = str(message.content or "")
            if re.search(self.guess_pattern, content, re.DOTALL):
                num_guesses += 1
                matches = self.guesses(content)
                if matches:
                    guess = matches[-1].strip()
        is_correct = 1.0 if guess == f"[{answer}]" else 0.0
        return is_correct / (num_guesses or 1)

    @vf.reward(weight=1.0)
    async def partial_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = str(task["answer"])
        completion = state.get("completion") or []
        for message in reversed(vf.get_messages(completion, role="assistant")):
            matches = self.guesses(str(message.content or ""))
            if matches:
                if matches[-1].strip() == f"[{answer}]":
                    return 0.0
                break
        for message in reversed(vf.get_messages(completion, role="user")):
            parts = str(message.content or "").strip().split("\n")
            if len(parts) == 3:
                scoring = parts[1].strip()
                return 0.2 * scoring.count("G") + 0.1 * scoring.count("Y")
        return 0.0

    @vf.reward(weight=0.2)
    async def format_reward(self, state: vf.State) -> float:
        completion = state.get("completion") or []
        found = False
        for message in vf.get_messages(completion, role="assistant"):
            found = True
            if len(self.guesses(str(message.content or ""))) != 1:
                return 0.0
        return 1.0 if found else 0.0


def load_taskset(config: WordleTasksetConfig) -> WordleTaskset:
    return WordleTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

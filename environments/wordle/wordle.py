import re
from typing import cast

import verifiers as vf
from tasksets.textarena import (
    TextArenaTaskset,
    TextArenaTasksetConfig,
    TextArenaUser,
    TextArenaUserConfig,
)


class WordleUserConfig(TextArenaUserConfig):
    pass


class WordleTasksetConfig(TextArenaTasksetConfig):
    game: str = "Wordle-v0"
    answer_state_key: str = "secret_word"
    user: WordleUserConfig | None = WordleUserConfig()
    system_prompt: vf.SystemPrompt | vf.SystemPromptConfig | None = None


class WordleUser(TextArenaUser):
    config: WordleUserConfig

    @staticmethod
    def content_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, vf.TextContentPart):
                    chunks.append(part.text)
                elif isinstance(part, dict):
                    text = cast(dict[str, object], part).get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks)
        return ""

    async def get_response(
        self, task: vf.Task, state: vf.State, messages: list[vf.Message]
    ) -> list[vf.UserMessage]:
        response = await super().get_response(task, state, messages)
        if state.get("done") is True:
            return response
        if not response:
            return []
        latest_feedback = self.content_text(response[-1].content).split("[GAME]")[
            -1
        ].strip()
        if "Feedback:" in latest_feedback:
            latest_feedback = latest_feedback.split("Feedback:")[-1]
        return [vf.UserMessage(content=latest_feedback)]


class WordleTaskset(TextArenaTaskset[WordleTasksetConfig]):
    guess_pattern = r"<guess>(.*?)</guess>"
    config: WordleTasksetConfig

    def load_system_prompt(
        self, config: WordleTasksetConfig
    ) -> vf.SystemPrompt | vf.SystemPromptConfig | None:
        if config.system_prompt is not None:
            return config.system_prompt
        return vf.SystemPromptConfig(path="prompts/system_prompt.txt")

    def guesses(self, content: str) -> list[str]:
        return re.findall(self.guess_pattern, content, re.DOTALL)

    @vf.reward(weight=1.0)
    async def correct_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = task["answer"]
        assert isinstance(answer, str)
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        for message in reversed(vf.get_messages(completion, role="assistant")):
            content = WordleUser.content_text(message.content)
            matches = self.guesses(content)
            if matches:
                return 1.0 if matches[-1].strip() == f"[{answer}]" else 0.0
        return 0.0

    @vf.reward(weight=1.0)
    async def length_bonus(self, task: vf.Task, state: vf.State) -> float:
        answer = task["answer"]
        assert isinstance(answer, str)
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        guess = ""
        num_guesses = 0
        for message in vf.get_messages(completion, role="assistant"):
            content = WordleUser.content_text(message.content)
            if re.search(self.guess_pattern, content, re.DOTALL):
                num_guesses += 1
                matches = self.guesses(content)
                if matches:
                    guess = matches[-1].strip()
        is_correct = 1.0 if guess == f"[{answer}]" else 0.0
        assert num_guesses > 0 or is_correct == 0.0
        return is_correct / (num_guesses or 1)

    @vf.reward(weight=1.0)
    async def partial_answer(self, task: vf.Task, state: vf.State) -> float:
        answer = task["answer"]
        assert isinstance(answer, str)
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        for message in reversed(vf.get_messages(completion, role="assistant")):
            content = WordleUser.content_text(message.content)
            matches = self.guesses(content)
            if matches:
                if matches[-1].strip() == f"[{answer}]":
                    return 0.0
                break
        for message in reversed(vf.get_messages(completion, role="user")):
            content = WordleUser.content_text(message.content)
            parts = content.strip().split("\n")
            if len(parts) == 3:
                scoring = parts[1].strip()
                return 0.2 * scoring.count("G") + 0.1 * scoring.count("Y")
        return 0.0

    @vf.reward(weight=0.2)
    async def format_reward(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        completion = state.get("completion") or []
        assert isinstance(completion, list)
        found = False
        for message in vf.get_messages(completion, role="assistant"):
            found = True
            content = WordleUser.content_text(message.content)
            if len(self.guesses(content)) != 1:
                return 0.0
        return 1.0 if found else 0.0


def load_taskset(config: WordleTasksetConfig) -> WordleTaskset:
    return WordleTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

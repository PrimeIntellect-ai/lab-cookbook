import re
from collections.abc import Mapping
from pathlib import Path

import verifiers.v1 as vf
from textarena_taskset import TextArenaTaskset, TextArenaTasksetConfig

WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""

_GUESS_PATTERN = re.compile(r"<guess>(.*?)</guess>", re.DOTALL)


class WordleTasksetConfig(TextArenaTasksetConfig):
    # Wordle-specific knob: load the system prompt from disk if set.
    path_to_system_prompt: str | None = None


class WordleEnvConfig(vf.EnvConfig):
    taskset: WordleTasksetConfig
    harness: vf.HarnessConfig


def wordle_feedback_fn(observation: str) -> str:
    latest_observation = observation.split("[GAME]")[-1].strip()
    if "Feedback:" in latest_observation:
        return latest_observation.split("Feedback:")[-1]
    return latest_observation


def _messages_by_role(state: vf.State, role: str) -> list[vf.ConfigData]:
    completion = state.get("completion") or []
    if not isinstance(completion, list):
        return []
    return [
        dict(message)
        for message in completion
        if isinstance(message, Mapping) and message.get("role") == role
    ]


def _assistant_messages(state: vf.State) -> list[vf.ConfigData]:
    return _messages_by_role(state, "assistant")


def _user_messages(state: vf.State) -> list[vf.ConfigData]:
    return _messages_by_role(state, "user")


def extract_guess(state: vf.State) -> str:
    """Pull the final ``<guess>...</guess>`` payload from the last assistant turn."""
    messages = _assistant_messages(state)
    if not messages:
        return ""
    content = str(messages[-1].get("content") or "")
    matches = _GUESS_PATTERN.findall(content)
    if not matches:
        return ""
    return matches[-1].strip()


@vf.reward(weight=1.0)
async def correct_answer(task: vf.Task, state: vf.State) -> float:
    """1.0 iff the final guess exactly matches the secret word."""
    guess = extract_guess(state)
    answer = str(task.get("answer") or "")
    return 1.0 if guess == "[" + answer + "]" else 0.0


@vf.reward(weight=1.0)
async def length_bonus(task: vf.Task, state: vf.State) -> float:
    """Reward solving in fewer turns: is_correct / num_guesses."""
    guess = extract_guess(state)
    answer = str(task.get("answer") or "")
    is_correct = 1.0 if guess == "[" + answer + "]" else 0.0
    num_guesses = sum(
        1
        for message in _assistant_messages(state)
        if _GUESS_PATTERN.search(str(message.get("content") or ""))
    )
    return is_correct / (num_guesses or 1)


@vf.reward(weight=1.0)
async def partial_answer(task: vf.Task, state: vf.State) -> float:
    """Partial credit for greens/yellows in the latest feedback block."""
    guess = extract_guess(state)
    answer = str(task.get("answer") or "")
    if guess == "[" + answer + "]":
        return 0.0
    for message in reversed(_user_messages(state)):
        feedback = str(message.get("content") or "").strip()
        parts = feedback.split("\n")
        if len(parts) == 3:
            scoring = parts[1].strip()
            greens = scoring.count("G")
            yellows = scoring.count("Y")
            return 0.2 * greens + 0.1 * yellows
    return 0.0


@vf.reward(weight=0.2)
async def format_reward(task: vf.Task, state: vf.State) -> float:
    """1.0 iff every assistant turn contains a single well-formed ``<guess>...</guess>`` block."""
    _ = task
    messages = _assistant_messages(state)
    if not messages:
        return 0.0
    for message in messages:
        content = str(message.get("content") or "")
        matches = _GUESS_PATTERN.findall(content)
        if len(matches) != 1:
            return 0.0
    return 1.0


class WordleTaskset(TextArenaTaskset):
    config_type = WordleTasksetConfig

    def __init__(self, *, config: vf.TasksetConfig | None = None, **kwargs: object) -> None:
        cfg = WordleTasksetConfig.from_config(config)
        if cfg.path_to_system_prompt is not None:
            cfg = WordleTasksetConfig(
                cfg,
                system_prompt=Path(cfg.path_to_system_prompt).expanduser().read_text(),
            )
        elif cfg.system_prompt is None:
            cfg = WordleTasksetConfig(cfg, system_prompt=WORDLE_SYSTEM_PROMPT)
        kwargs.setdefault("feedback_fn", wordle_feedback_fn)
        kwargs.setdefault("rewards", [correct_answer, partial_answer, length_bonus, format_reward])
        super().__init__(config=cfg, **kwargs)


def load_taskset(config: WordleTasksetConfig) -> WordleTaskset:
    return WordleTaskset(config=config)


def load_environment(config: WordleEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )

import re
from collections.abc import Mapping
from pathlib import Path

import verifiers as v0
import verifiers.v1 as vf
from textarena_taskset import TextArenaTaskset, TextArenaTasksetConfig

DEFAULT_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""

DEFAULT_NUM_TRAIN_EXAMPLES = 2000
DEFAULT_NUM_EVAL_EXAMPLES = 20

_GUESS_PATTERN = re.compile(r"<guess>(.*?)</guess>", re.DOTALL)


class WordleTasksetConfig(TextArenaTasksetConfig):
    # New wordle-specific field (not on parent).
    path_to_system_prompt: str | None = None


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


def _resolve_system_prompt(cfg: WordleTasksetConfig) -> str:
    """Pick the system prompt: explicit path > cfg.system_prompt > wordle default."""
    if cfg.path_to_system_prompt is not None:
        message = v0.SystemMessage.from_path(Path(cfg.path_to_system_prompt).expanduser())
        content = message.content
        return content if isinstance(content, str) else str(content)
    if cfg.system_prompt is not None:
        return cfg.system_prompt if isinstance(cfg.system_prompt, str) else str(cfg.system_prompt)
    return DEFAULT_SYSTEM_PROMPT


def load_taskset(cfg: WordleTasksetConfig) -> TextArenaTaskset:
    set_fields = cfg.model_fields_set
    num_train_examples = (
        cfg.num_train_examples if "num_train_examples" in set_fields else DEFAULT_NUM_TRAIN_EXAMPLES
    )
    num_eval_examples = (
        cfg.num_eval_examples if "num_eval_examples" in set_fields else DEFAULT_NUM_EVAL_EXAMPLES
    )
    return TextArenaTaskset(
        game=cfg.game,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=cfg.seed,
        answer_state_key=cfg.answer_state_key,
        feedback_fn=wordle_feedback_fn,
        system_prompt=_resolve_system_prompt(cfg),
        rewards=[
            correct_answer,
            partial_answer,
            length_bonus,
            format_reward,
        ],
        config=cfg,
    )


def load_environment(config: vf.EnvConfig) -> vf.Env:
    cfg = WordleTasksetConfig(config.taskset)
    return vf.Env(
        taskset=load_taskset(cfg),
        harness=vf.Harness(config=config.harness),
    )

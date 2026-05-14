from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import verifiers as v0
import verifiers.v1 as vf
from textarena_taskset import TextArenaTaskset, TextArenaTasksetConfig

DEFAULT_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""


def wordle_feedback_fn(observation: str) -> str:
    latest_observation = observation.split("[GAME]")[-1].strip()
    if "Feedback:" in latest_observation:
        return latest_observation.split("Feedback:")[-1]
    return latest_observation


def _messages_by_role(state: vf.State, role: str) -> list[dict[str, object]]:
    completion = state.get("completion") or []
    if not isinstance(completion, list):
        return []
    return [
        dict(message)
        for message in completion
        if isinstance(message, Mapping) and message.get("role") == role
    ]


def _assistant_messages(state: vf.State) -> list[dict[str, object]]:
    return _messages_by_role(state, "assistant")


def _user_messages(state: vf.State) -> list[dict[str, object]]:
    return _messages_by_role(state, "user")


def _parsed_guess(parser: v0.XMLParser, state: vf.State) -> str:
    messages = _assistant_messages(state)
    if not messages:
        return ""
    return str(parser.parse_answer(messages) or "")


class WordleTasksetConfig(TextArenaTasksetConfig):
    game: str = "Wordle-v0"
    num_train_examples: int = 2000
    num_eval_examples: int = 20
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT


def correct_answer_factory(parser: v0.XMLParser):
    @vf.reward(weight=1.0)
    async def correct_answer(task: vf.Task, state: vf.State) -> float:
        """1.0 iff the final guess exactly matches the secret word."""
        guess = _parsed_guess(parser, state)
        answer = str(task.get("answer") or "")
        return 1.0 if guess == "[" + answer + "]" else 0.0

    return correct_answer


def length_bonus_factory(parser: v0.XMLParser):
    @vf.reward(weight=1.0)
    async def length_bonus(task: vf.Task, state: vf.State) -> float:
        """Reward solving in fewer turns: is_correct / num_guesses."""
        guess = _parsed_guess(parser, state)
        answer = str(task.get("answer") or "")
        is_correct = 1.0 if guess == "[" + answer + "]" else 0.0
        guess_pattern = re.compile(r"<guess>.*</guess>", re.DOTALL)
        num_guesses = sum(
            1
            for message in _assistant_messages(state)
            if guess_pattern.search(str(message.get("content") or ""))
        )
        return is_correct / (num_guesses or 1)

    return length_bonus


def partial_answer_factory(parser: v0.XMLParser):
    @vf.reward(weight=1.0)
    async def partial_answer(task: vf.Task, state: vf.State) -> float:
        """Partial credit for greens/yellows in the latest feedback block."""
        guess = _parsed_guess(parser, state)
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

    return partial_answer


def format_reward_factory(parser: v0.XMLParser):
    """Wrap the v0 XML format reward in a v1 reward."""
    format_fn = parser.get_format_reward_func()

    @vf.reward(weight=0.2)
    async def format_reward(task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(format_fn(state.get("completion") or []))

    return format_reward


def load_taskset(
    config: WordleTasksetConfig | Mapping[str, object] | None = None,
    num_train_examples: int | None = None,
    num_eval_examples: int | None = None,
    seed: int | None = None,
    system_prompt: str | None = None,
) -> TextArenaTaskset:
    taskset_config = WordleTasksetConfig.from_config(
        config,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        system_prompt=system_prompt,
    )
    parser = v0.XMLParser(fields=["guess"], answer_field="guess")
    return TextArenaTaskset(
        parser=parser,
        feedback_fn=wordle_feedback_fn,
        system_prompt=taskset_config.system_prompt,
        rewards=[
            correct_answer_factory(parser),
            partial_answer_factory(parser),
            length_bonus_factory(parser),
            format_reward_factory(parser),
        ],
        config=taskset_config,
    )


def load_harness(
    config: vf.HarnessConfig | Mapping[str, object] | None = None,
) -> vf.Harness:
    return vf.Harness(config=config)


def load_environment(
    config: vf.EnvConfig | Mapping[str, object] | None = None,
    num_train_examples: int | None = None,
    num_eval_examples: int | None = None,
    seed: int | None = None,
    system_prompt: str | None = None,
    path_to_system_prompt: str | Path | None = None,
) -> vf.Env:
    if path_to_system_prompt is not None:
        system_prompt = v0.SystemMessage.from_path(
            Path(path_to_system_prompt).expanduser()
        ).content
    config = vf.EnvConfig.from_config(
        config,
        taskset=WordleTasksetConfig.from_config(
            num_train_examples=num_train_examples,
            num_eval_examples=num_eval_examples,
            seed=seed,
            system_prompt=system_prompt,
        ),
    )
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )



import asyncio
import random
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.config import TasksetConfig

try:
    import nltk  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError("TextArenaTaskset requires nltk. Install with: uv add nltk") from e

try:
    import textarena as ta  # type: ignore[import-untyped]
except ImportError as e:
    raise ImportError("TextArenaTaskset requires textarena. Install with: uv add textarena") from e

GUESS_PATTERN = re.compile(r"<guess>(.*?)</guess>", re.DOTALL)
WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""


def wordle_feedback(observation: str) -> str:
    latest_observation = observation.split("[GAME]")[-1].strip()
    if "Feedback:" in latest_observation:
        return latest_observation.split("Feedback:")[-1]
    return latest_observation


class TextArenaTasksetConfig(TasksetConfig):
    game: str = "Wordle-v0"
    num_train_examples: int = 2000
    num_eval_examples: int = 20
    seed: int = 0
    answer_state_key: str = "secret_word"
    path_to_system_prompt: str = ""


def load_taskset(
    config: TextArenaTasksetConfig,
) -> vf.Taskset:
    if config.path_to_system_prompt:
        system_prompt = Path(config.path_to_system_prompt).expanduser().read_text()
    elif config.system_prompt:
        assert isinstance(config.system_prompt, str)
        system_prompt = config.system_prompt
    else:
        system_prompt = WORDLE_SYSTEM_PROMPT
    assert system_prompt
    config = config.model_copy(update={"system_prompt": system_prompt})
    nltk.download("words", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    template = ta.make(env_id=config.game)
    assert isinstance(template, ta.Env)
    template.reset(num_players=1)

    def load_ta_env() -> ta.Env:
        env = deepcopy(template)
        env.reset(num_players=1)
        return env

    _, initial_prompt = template.get_observation()
    assert isinstance(initial_prompt, str)
    assert initial_prompt
    word_list = [str(word) for word in template.word_list]
    assert word_list

    def row(rng: random.Random, index: int):
        return {
            "example_id": index,
            "prompt": [{"role": "user", "content": initial_prompt}],
            "answer": rng.choice(word_list),
        }

    def build_train_rows():
        rng = random.Random(config.seed)
        return [row(rng, index) for index in range(config.num_train_examples)]

    def build_eval_rows():
        rng = random.Random(config.seed)
        for _ in range(config.num_train_examples):
            rng.choice(word_list)
        return [
            row(rng, index + config.num_train_examples) for index in range(config.num_eval_examples)
        ]

    async def textarena_user(task, state, ta_env):
        answer = task["answer"]
        assert isinstance(answer, str)
        assert answer
        ta_state = ta_env.state
        assert isinstance(ta_state, ta.State)
        game_state = ta_state.game_state
        assert isinstance(game_state, dict)
        if game_state.get(config.answer_state_key) != answer:
            game_state[config.answer_state_key] = answer
        completion = state.get("completion")
        assert isinstance(completion, list)
        last_text = ""
        for message in reversed(completion):
            assert isinstance(message, Mapping)
            if message.get("role") == "assistant":
                content = message.get("content")
                assert isinstance(content, str)
                last_text = content
                break
        matches = GUESS_PATTERN.findall(last_text)
        guess = matches[-1].strip() if matches else ""
        await asyncio.to_thread(ta_env.step, guess)
        ta_state = ta_env.state
        assert isinstance(ta_state, ta.State)
        if ta_state.done:
            game_info = ta_state.game_info
            assert isinstance(game_info, dict)
            player_info = game_info[0]
            assert isinstance(player_info, dict)
            reason = player_info["reason"]
            assert isinstance(reason, str)
            state["final_env_response"] = reason
            return [{"role": "user", "content": reason}]
        _, observation = await asyncio.to_thread(ta_env.get_observation)
        assert isinstance(observation, str)
        return [{"role": "user", "content": wordle_feedback(observation)}]

    taskset = vf.Taskset(
        config=config,
        source=build_train_rows,
        eval_source=build_eval_rows if config.num_eval_examples > 0 else None,
    )
    taskset.user = vf.User(
        fn=textarena_user,
        objects={"ta_env": load_ta_env},
        bindings={"ta_env": "objects.ta_env"},
    )
    return taskset

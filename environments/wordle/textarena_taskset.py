import asyncio
import random
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import ClassVar, Protocol, cast

import verifiers.v1 as vf
from verifiers.v1.config import TasksetConfig

try:
    import nltk
except ImportError as e:
    raise ImportError("TextArenaTaskset requires nltk. Install with: uv add nltk") from e

try:
    import textarena as ta
except ImportError as e:
    raise ImportError("TextArenaTaskset requires textarena. Install with: uv add textarena") from e

GUESS_PATTERN = re.compile(r"<guess>(.*?)</guess>", re.DOTALL)


class TextArenaGame(Protocol):
    state: ta.State
    word_list: list[str]

    def reset(self, num_players: int) -> object: ...

    def get_observation(self) -> tuple[int, str]: ...

    def step(self, action: str) -> tuple[bool, vf.ConfigData]: ...


def load_session() -> vf.ConfigData:
    return {}


class TextArenaTasksetConfig(TasksetConfig):
    game: str = "Wordle-v0"
    num_train_examples: int = 1000
    num_eval_examples: int = 0
    seed: int = 0
    answer_state_key: str = "secret_word"
    path_to_system_prompt: str = ""


class TextArenaTaskset(vf.Taskset):
    config_type: ClassVar[type[TasksetConfig]] = TextArenaTasksetConfig

    def __init__(
        self,
        config: TextArenaTasksetConfig,
        feedback_fn: Callable[[str], str],
        rewards: tuple[vf.Handler, ...],
    ):
        assert callable(feedback_fn)
        taskset_config = TextArenaTasksetConfig.from_config(config)
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = cast(TextArenaGame, ta.make(env_id=taskset_config.game))
        template.reset(num_players=1)
        _, initial_prompt = template.get_observation()
        assert isinstance(initial_prompt, str)
        assert initial_prompt
        word_list = [str(word) for word in template.word_list]
        assert word_list
        answer_state_key = taskset_config.answer_state_key

        def row(rng: random.Random, index: int) -> vf.ConfigData:
            return {
                "example_id": index,
                "prompt": [{"role": "user", "content": initial_prompt}],
                "answer": rng.choice(word_list),
            }

        def build_train_rows() -> list[vf.ConfigData]:
            rng = random.Random(taskset_config.seed)
            return [row(rng, index) for index in range(taskset_config.num_train_examples)]

        def build_eval_rows() -> list[vf.ConfigData]:
            rng = random.Random(taskset_config.seed)
            for _ in range(taskset_config.num_train_examples):
                rng.choice(word_list)
            return [
                row(rng, index + taskset_config.num_train_examples)
                for index in range(taskset_config.num_eval_examples)
            ]

        async def textarena_user(
            task: vf.Task,
            state: vf.State,
            session: vf.ConfigData,
        ) -> list[dict[str, str]]:
            ta_env = session.get("ta_env")
            if ta_env is None:
                cloned = cast(TextArenaGame, await asyncio.to_thread(deepcopy, template))
                answer = task["answer"]
                assert isinstance(answer, str)
                assert answer
                ta_state = cloned.state
                assert isinstance(ta_state, ta.State)
                game_state = ta_state.game_state
                assert isinstance(game_state, dict)
                game_state[answer_state_key] = answer
                session["ta_env"] = cloned
                ta_env = cloned
            assert isinstance(ta_env, ta.Env)
            game = cast(TextArenaGame, ta_env)
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
            await asyncio.to_thread(game.step, guess)
            ta_state = game.state
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
            _, observation = await asyncio.to_thread(game.get_observation)
            assert isinstance(observation, str)
            return [{"role": "user", "content": feedback_fn(observation)}]

        super().__init__(
            config=taskset_config,
            source=build_train_rows,
            eval_source=(build_eval_rows if taskset_config.num_eval_examples > 0 else None),
            rewards=rewards,
        )
        self.user = vf.User(
            fn=textarena_user,
            objects={"session": load_session},
            bindings={"session": "objects.session"},
        )

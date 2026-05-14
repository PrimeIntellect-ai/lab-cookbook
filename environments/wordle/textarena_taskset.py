"""Reusable v1 Taskset that wraps TextArena single-player games.

Mirrors the v0 ``TextArenaEnv`` wrapper but exposes the same lifecycle via the
verifiers v1 ``Taskset`` interface (per-rollout ``setup`` / ``user`` /
``cleanup`` hooks, dataset row generation, default rewards). Designed so the
class can later be upstreamed to ``verifiers.v1.packages.tasksets``.

Wordle is the motivating example, but any TextArena game whose ``game_state``
exposes a single answer field (``secret_word`` by default) plays through this
wrapper without modification.
"""

import asyncio
import logging
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import ClassVar

import verifiers.v1 as vf
from verifiers.v1.config import TasksetConfig
from verifiers.v1.taskset import Taskset

try:
    import nltk
except ImportError as e:
    raise ImportError("TextArenaTaskset requires nltk. Install with: uv add nltk") from e


# nltk.download is noisy by default; route every call through a quiet wrapper
# without touching the imported module's attributes (mypy-friendly, no cast).
_original_nltk_download = nltk.download


def _nltk_download(name: str) -> None:
    _original_nltk_download(name, quiet=True)


try:
    import textarena as ta
except ImportError as e:
    raise ImportError("TextArenaTaskset requires textarena. Install with: uv add textarena") from e


logger = logging.getLogger(__name__)

_GUESS_PATTERN = re.compile(r"<guess>(.*?)</guess>", re.DOTALL)


def _extract_guess(text: str) -> str:
    matches = _GUESS_PATTERN.findall(text)
    if not matches:
        return ""
    return matches[-1].strip()


class TextArenaTasksetConfig(TasksetConfig):
    game: str = "Wordle-v0"
    num_train_examples: int = 1000
    num_eval_examples: int = 0
    seed: int = 0
    answer_state_key: str = "secret_word"


class TextArenaTaskset(Taskset):
    """v1 Taskset wrapping a single-player TextArena game.

    Each rollout deepcopies a template ``ta.Env``, overrides
    ``ta_env.state.game_state[answer_state_key]`` with the row's ``answer``,
    and drives the game forward by parsing the model's guess and stepping the
    environment in the user function.
    """

    config_type: ClassVar[type[TasksetConfig]] = TextArenaTasksetConfig

    def __init__(
        self,
        game: str | None = None,
        num_train_examples: int | None = None,
        num_eval_examples: int | None = None,
        seed: int | None = None,
        answer_state_key: str | None = None,
        feedback_fn: Callable[[str], str] | None = None,
        rewards: Iterable[vf.Handler] = (),
        config: TextArenaTasksetConfig | None = None,
        **kwargs: object,
    ):
        self.config: TextArenaTasksetConfig = TextArenaTasksetConfig.from_config(config)
        self.game = game if game is not None else self.config.game
        self.num_train_examples = (
            num_train_examples if num_train_examples is not None else self.config.num_train_examples
        )
        self.num_eval_examples = (
            num_eval_examples if num_eval_examples is not None else self.config.num_eval_examples
        )
        self.seed = seed if seed is not None else self.config.seed
        self.answer_state_key = (
            answer_state_key if answer_state_key is not None else self.config.answer_state_key
        )
        self.feedback_fn = feedback_fn or (lambda x: x)

        _nltk_download("words")
        _nltk_download("averaged_perceptron_tagger_eng")

        self._template_ta_env = ta.make(env_id=self.game)
        self._template_ta_env.reset(num_players=1)
        self._shared_memo = self._build_shared_memo(self._template_ta_env)
        _, self._initial_prompt = self._template_ta_env.get_observation()
        self._word_list = self._flatten_words(self._template_ta_env.word_list)

        super().__init__(
            source=self._build_train_rows,
            eval_source=self._build_eval_rows if self.num_eval_examples > 0 else None,
            user=self._user_fn,
            setups=[self._setup_state],
            cleanups=[self._cleanup_state],
            rewards=list(rewards),
            config=self.config,
            **kwargs,
        )

    @staticmethod
    def _build_shared_memo(ta_env: object) -> dict:
        """Share the EnglishDictionary across deepcopies.

        The TextArena ``EnglishDictionary`` holds ~430k strings (~38MB) and is
        read-only after construction; the same word list is also reusable. By
        pre-populating the deepcopy memo we avoid duplicating ~38MB per rollout.
        """
        memo: dict = {}
        env = ta_env
        while hasattr(env, "env"):
            env = env.env
        dictionary = getattr(env, "dictionary", None)
        if dictionary is not None:
            memo[id(dictionary)] = dictionary
        word_list = getattr(env, "word_list", None)
        if word_list is not None:
            memo[id(word_list)] = word_list
        return memo

    @staticmethod
    def _flatten_words(words: object) -> list[str]:
        if isinstance(words, Mapping):
            flat: list[str] = []
            for values in words.values():
                if isinstance(values, list | tuple):
                    flat.extend(str(v) for v in values)
                else:
                    flat.append(str(values))
            return flat
        if isinstance(words, Iterable):
            return [str(w) for w in words]
        raise TypeError(f"Unsupported word_list type: {type(words)!r}")

    def _row(self, rng: random.Random, index: int) -> vf.ConfigData:
        return {
            "example_id": index,
            "prompt": [{"role": "user", "content": self._initial_prompt}],
            "answer": rng.choice(self._word_list),
        }

    def _build_train_rows(self) -> list[vf.ConfigData]:
        rng = random.Random(self.seed)
        return [self._row(rng, index) for index in range(self.num_train_examples)]

    def _build_eval_rows(self) -> list[vf.ConfigData]:
        rng = random.Random(self.seed)
        for _ in range(self.num_train_examples):
            rng.choice(self._word_list)
        return [
            self._row(rng, index + self.num_train_examples)
            for index in range(self.num_eval_examples)
        ]

    async def _setup_state(self, task: vf.Task, state: vf.State) -> None:
        ta_env = await asyncio.to_thread(deepcopy, self._template_ta_env, self._shared_memo.copy())
        ta_env.state.game_state[self.answer_state_key] = task["answer"]
        state["ta_env"] = ta_env

    async def _cleanup_state(self, task: vf.Task, state: vf.State) -> None:
        _ = task
        state.pop("ta_env", None)

    async def _user_fn(
        self,
        task: vf.Task,
        state: vf.State,
        transcript: Sequence[vf.Message | vf.ConfigMap],
    ) -> list[dict[str, str]]:
        _ = task
        ta_env = state.get("ta_env")
        if ta_env is None:
            return []
        last_text = _last_assistant_text(transcript)
        guess = _extract_guess(last_text)
        await asyncio.to_thread(ta_env.step, guess)
        if ta_env.state.done:
            reason = str(ta_env.state.game_info[0]["reason"])
            state["final_env_response"] = reason
            return [{"role": "user", "content": reason}]
        _, observation = await asyncio.to_thread(ta_env.get_observation)
        feedback = self.feedback_fn(observation)
        return [{"role": "user", "content": str(feedback)}]


def _last_assistant_text(transcript: Sequence[vf.Message | vf.ConfigMap]) -> str:
    for message in reversed(transcript):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            content = message.get("content")
            return str(content or "")
    return ""

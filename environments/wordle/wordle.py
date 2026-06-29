from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.textarena import (
    TextArenaConfig,
    TextArenaState,
    TextArenaTask,
    TextArenaTaskset,
)


class WordleConfig(TextArenaConfig):
    game: Literal["Wordle-v0"] = "Wordle-v0"


class WordleTaskset(
    TextArenaTaskset,
    vf.Taskset[TextArenaTask, WordleConfig, TextArenaState],
):  # ty: ignore[invalid-generic-class]
    pass


__all__ = ["WordleTaskset"]

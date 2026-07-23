from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import verifiers.v1 as vf

if TYPE_CHECKING:
    from shape_detective.models import ShapeDetectiveTaskData, Tile


def clue_line(target: Tile, prop: Literal["pattern", "color", "shape"], clue_index: int) -> str:
    article = "a " if prop == "shape" else ""
    return f"Clue {clue_index + 1} - {prop}: the target is {article}**{getattr(target, prop)}**."


class ShapeDetectiveState(vf.State):
    clue_index: int = 1
    user_finished: bool = False


class ShapeDetectiveUser(vf.User[vf.UserConfig, ShapeDetectiveState]):
    """Reveals the remaining clues and marks the task complete after the final answer."""

    async def setup_task(self, task: ShapeDetectiveTaskData) -> None:
        self.target_tile = task.target_tile
        self.mode = task.mode

    async def respond(self, message: str) -> vf.Messages:
        _ = message
        if self.mode == "single":
            self.state.user_finished = True
            return []
        if self.state.clue_index == 1:
            self.state.clue_index = 2
            return [
                vf.UserMessage(
                    content=(
                        f"{clue_line(self.target_tile, 'color', 1)}\n\n"
                        "Narrow the candidates again. Still do not submit your answer."
                    )
                )
            ]
        if self.state.clue_index == 2:
            self.state.clue_index = 3
            return [
                vf.UserMessage(
                    content=(
                        f"{clue_line(self.target_tile, 'shape', 2)}\n\n"
                        "Commit your answer now as \\boxed{N}."
                    )
                )
            ]
        self.state.user_finished = True
        return []


if __name__ == "__main__":
    ShapeDetectiveUser.run()

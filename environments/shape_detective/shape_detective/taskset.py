"""Visual deduction: identify a hidden tile from image-backed clues."""

import math
import random
import re
from collections.abc import Iterator
from typing import Literal, cast

import verifiers.v1 as vf
from PIL import Image, ImageDraw
from verifiers.v1.utils.image import image_data_url

from shape_detective.models import (
    Color,
    Mode,
    Pattern,
    Shape,
    ShapeDetectiveTaskData,
    Tile,
)
from shape_detective.servers.user import (
    ShapeDetectiveState,
    ShapeDetectiveUser,
    clue_line,
)

SHAPES: tuple[Shape, ...] = ("circle", "square", "triangle", "star")
COLORS: dict[Color, tuple[int, int, int]] = {
    "red": (220, 50, 50),
    "blue": (50, 90, 220),
    "green": (60, 170, 80),
    "yellow": (230, 200, 50),
}
PATTERNS: tuple[Pattern, ...] = ("solid", "striped", "dotted")
CLUE_ORDER: tuple[Literal["pattern", "color", "shape"], ...] = (
    "pattern",
    "color",
    "shape",
)

GRID_SIZE = 4
TILE_PX = 128
IMAGE_PX = GRID_SIZE * TILE_PX
TILE_PAD = 14
STRIPE_SPACING = 14
STRIPE_WIDTH = 5
DOT_SPACING = 18
DOT_RADIUS = 3
BG_COLOR = (245, 245, 245)
GRID_COLOR = (180, 180, 180)
TEXT_COLOR = (20, 20, 20)
SYSTEM = (
    "You are playing Shape Detective. You see a 4x4 grid of tiles numbered 0-15 "
    "(left-to-right, top-to-bottom). Each tile has a shape, a color, and a pattern. "
    "Use the clues to identify the target tile. When asked to commit, reply with "
    "the tile index in \\boxed{N}."
)


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else ""


def tile_image_data_url(tiles: list[Tile]) -> str:
    image = Image.new("RGB", (IMAGE_PX, IMAGE_PX), BG_COLOR)
    label_draw = ImageDraw.Draw(image)
    for idx, tile in enumerate(tiles):
        tile_row, tile_col = divmod(idx, GRID_SIZE)
        x0, y0 = tile_col * TILE_PX, tile_row * TILE_PX
        canvas = Image.new("RGB", (TILE_PX, TILE_PX), BG_COLOR)
        draw = ImageDraw.Draw(canvas)
        color = COLORS[tile.color]
        box = (TILE_PAD, TILE_PAD, TILE_PX - TILE_PAD, TILE_PX - TILE_PAD)
        if tile.shape == "circle":
            draw.ellipse(box, fill=color)
        elif tile.shape == "square":
            draw.rectangle(box, fill=color)
        elif tile.shape == "triangle":
            draw.polygon(
                [
                    (TILE_PX // 2, TILE_PAD),
                    (TILE_PAD, TILE_PX - TILE_PAD),
                    (TILE_PX - TILE_PAD, TILE_PX - TILE_PAD),
                ],
                fill=color,
            )
        else:
            center = TILE_PX // 2
            outer = (TILE_PX - 2 * TILE_PAD) // 2
            inner = outer // 2
            points: list[tuple[float, float]] = []
            for point in range(10):
                angle = -math.pi / 2 + point * math.pi / 5
                radius = outer if point % 2 == 0 else inner
                points.append(
                    (
                        center + radius * math.cos(angle),
                        center + radius * math.sin(angle),
                    )
                )
            draw.polygon(points, fill=color)

        if tile.pattern == "striped":
            for y in range(0, TILE_PX, STRIPE_SPACING):
                draw.rectangle(
                    (0, y, TILE_PX, y + STRIPE_WIDTH),
                    fill=BG_COLOR,
                )
        elif tile.pattern == "dotted":
            for dot_y in range(DOT_SPACING - 4, TILE_PX, DOT_SPACING):
                for dot_x in range(DOT_SPACING - 4, TILE_PX, DOT_SPACING):
                    draw.ellipse(
                        (
                            dot_x - DOT_RADIUS,
                            dot_y - DOT_RADIUS,
                            dot_x + DOT_RADIUS,
                            dot_y + DOT_RADIUS,
                        ),
                        fill=BG_COLOR,
                    )

        image.paste(canvas, (x0, y0))
        label_draw.text((x0 + 6, y0 + 4), str(idx), fill=TEXT_COLOR)

    for grid_line in range(1, GRID_SIZE):
        label_draw.line(
            [(grid_line * TILE_PX, 0), (grid_line * TILE_PX, IMAGE_PX)],
            fill=GRID_COLOR,
            width=2,
        )
        label_draw.line(
            [(0, grid_line * TILE_PX), (IMAGE_PX, grid_line * TILE_PX)],
            fill=GRID_COLOR,
            width=2,
        )
    return image_data_url(image)


class ShapeDetectiveTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig()


class ShapeDetectiveTask(
    vf.Task[
        ShapeDetectiveTaskData,
        ShapeDetectiveState,
        ShapeDetectiveTaskConfig,
    ]
):
    user = ShapeDetectiveUser

    @vf.stop
    async def user_finished(self, trace: vf.Trace) -> bool:
        state = cast(ShapeDetectiveState, trace.state)
        return state.user_finished

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(extract_boxed(trace.last_reply) == self.data.answer)


class ShapeDetectiveConfig(vf.TasksetConfig):
    mode: Mode = "multi"
    num_tasks: int = 12
    seed: int = 0
    task: ShapeDetectiveTaskConfig = ShapeDetectiveTaskConfig()


class ShapeDetectiveTaskset(
    vf.Taskset[ShapeDetectiveTask, ShapeDetectiveConfig]  # ty: ignore[invalid-type-arguments]
):
    def load(self) -> Iterator[ShapeDetectiveTask]:
        rng = random.Random(self.config.seed)
        for idx in range(self.config.num_tasks):
            while True:
                tiles = [
                    Tile(
                        shape=rng.choice(SHAPES),
                        color=rng.choice(tuple(COLORS)),
                        pattern=rng.choice(PATTERNS),
                    )
                    for _ in range(GRID_SIZE * GRID_SIZE)
                ]
                target = rng.randrange(len(tiles))
                target_tile = tiles[target]
                target_key = (
                    target_tile.shape,
                    target_tile.color,
                    target_tile.pattern,
                )
                if (
                    sum(1 for tile in tiles if (tile.shape, tile.color, tile.pattern) == target_key)
                    == 1
                ):
                    break

            if self.config.mode == "single":
                clue_block = "\n".join(
                    f"- {prop}: {getattr(target_tile, prop)}" for prop in CLUE_ORDER
                )
                intro = (
                    "Find the tile that matches all three clues:\n"
                    f"{clue_block}\n\n"
                    "Reply with the tile index in \\boxed{N}."
                )
            else:
                intro = (
                    "Find the hidden target tile. You will receive three clues across "
                    "three turns (pattern, then color, then shape). After each clue, list "
                    "the tile indices that could still be the target. Do not submit a "
                    "final answer until asked.\n\n"
                    f"{clue_line(target_tile, 'pattern', 0)}"
                )

            prompt: vf.Messages = [
                vf.UserMessage(
                    content=[
                        vf.TextContentPart(text=intro),
                        vf.ImageUrlContentPart(
                            image_url=vf.ImageUrlSource(url=tile_image_data_url(tiles))
                        ),
                    ]
                )
            ]
            yield ShapeDetectiveTask(
                ShapeDetectiveTaskData(
                    idx=idx,
                    prompt=prompt,
                    system_prompt=SYSTEM,
                    answer=str(target),
                    mode=self.config.mode,
                    target_tile=target_tile,
                ),
                self.config.task,
            )

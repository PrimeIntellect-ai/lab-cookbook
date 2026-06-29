import base64
import io
import math
import random
import re
from typing import Literal, cast

import verifiers.v1 as vf
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle", "star")
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 50),
    "blue": (50, 90, 220),
    "green": (60, 170, 80),
    "yellow": (230, 200, 50),
}
PATTERNS = ("solid", "striped", "dotted")
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

Mode = Literal["single", "multi"]
Tile = dict[str, str]


def clue_line(target: Tile, prop: str, clue_index: int) -> str:
    article = "a " if prop == "shape" else ""
    return f"Clue {clue_index + 1} - {prop}: the target is {article}**{target[prop]}**."


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else ""


def tile_image_data_url(tiles: list[Tile]) -> str:
    img = Image.new("RGB", (IMAGE_PX, IMAGE_PX), BG_COLOR)
    label_draw = ImageDraw.Draw(img)
    for idx, tile in enumerate(tiles):
        trow, tcol = divmod(idx, GRID_SIZE)
        x0, y0 = tcol * TILE_PX, trow * TILE_PX
        canvas = Image.new("RGB", (TILE_PX, TILE_PX), BG_COLOR)
        draw = ImageDraw.Draw(canvas)
        color = COLORS[tile["color"]]
        box = (TILE_PAD, TILE_PAD, TILE_PX - TILE_PAD, TILE_PX - TILE_PAD)
        if tile["shape"] == "circle":
            draw.ellipse(box, fill=color)
        elif tile["shape"] == "square":
            draw.rectangle(box, fill=color)
        elif tile["shape"] == "triangle":
            draw.polygon(
                [
                    (TILE_PX // 2, TILE_PAD),
                    (TILE_PAD, TILE_PX - TILE_PAD),
                    (TILE_PX - TILE_PAD, TILE_PX - TILE_PAD),
                ],
                fill=color,
            )
        else:
            cx = cy = TILE_PX // 2
            outer = (TILE_PX - 2 * TILE_PAD) // 2
            inner = outer // 2
            points: list[tuple[float, float]] = []
            for i in range(10):
                angle = -math.pi / 2 + i * math.pi / 5
                r = outer if i % 2 == 0 else inner
                points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            draw.polygon(points, fill=color)

        if tile["pattern"] == "striped":
            for y in range(0, TILE_PX, STRIPE_SPACING):
                draw.rectangle((0, y, TILE_PX, y + STRIPE_WIDTH), fill=BG_COLOR)
        elif tile["pattern"] == "dotted":
            for cy in range(DOT_SPACING - 4, TILE_PX, DOT_SPACING):
                for cx in range(DOT_SPACING - 4, TILE_PX, DOT_SPACING):
                    draw.ellipse(
                        (
                            cx - DOT_RADIUS,
                            cy - DOT_RADIUS,
                            cx + DOT_RADIUS,
                            cy + DOT_RADIUS,
                        ),
                        fill=BG_COLOR,
                    )

        img.paste(canvas, (x0, y0))
        label_draw.text((x0 + 6, y0 + 4), str(idx), fill=TEXT_COLOR)
    for i in range(1, GRID_SIZE):
        label_draw.line([(i * TILE_PX, 0), (i * TILE_PX, IMAGE_PX)], fill=GRID_COLOR, width=2)
        label_draw.line([(0, i * TILE_PX), (IMAGE_PX, i * TILE_PX)], fill=GRID_COLOR, width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class ShapeDetectiveState(vf.State):
    clue_index: int = 1


class ShapeDetectiveTask(vf.Task):
    answer: str
    mode: Mode
    target_tile: Tile


class ShapeDetectiveConfig(vf.TasksetConfig):
    mode: Mode = "multi"
    num_tasks: int = 12
    seed: int = 0
    user: vf.UserConfig = vf.UserConfig()


class ShapeDetectiveUser(vf.User[vf.UserConfig, ShapeDetectiveState]):
    async def setup_task(self, task: ShapeDetectiveTask) -> None:
        self.target_tile = task.target_tile
        self.mode = task.mode

    async def respond(self, message: str) -> vf.Messages:
        _ = message
        if self.mode != "multi":
            return []
        if self.state.clue_index == 1:
            self.state.clue_index = 2
            return [
                vf.UserMessage(
                    content=(
                        f"{clue_line(self.target_tile, 'color', 1)}\n\n"
                        "Narrow the candidates again. Still do not submit your answer."
                    ),
                )
            ]
        if self.state.clue_index == 2:
            self.state.clue_index = 3
            return [
                vf.UserMessage(
                    content=(
                        f"{clue_line(self.target_tile, 'shape', 2)}\n\n"
                        "Commit your answer now as \\boxed{N}."
                    ),
                )
            ]
        return []


class ShapeDetectiveTaskset(
    vf.Taskset[ShapeDetectiveTask, ShapeDetectiveConfig, ShapeDetectiveState]
):
    def load_tasks(self) -> list[ShapeDetectiveTask]:
        rng = random.Random(self.config.seed)
        tasks: list[ShapeDetectiveTask] = []
        for idx in range(self.config.num_tasks):
            while True:
                tiles = [
                    {
                        "shape": rng.choice(SHAPES),
                        "color": rng.choice(list(COLORS)),
                        "pattern": rng.choice(PATTERNS),
                    }
                    for _ in range(GRID_SIZE * GRID_SIZE)
                ]
                target = rng.randrange(len(tiles))
                key = (tiles[target]["shape"], tiles[target]["color"], tiles[target]["pattern"])
                if sum(1 for t in tiles if (t["shape"], t["color"], t["pattern"]) == key) == 1:
                    break
            target_tile = tiles[target]
            if self.config.mode == "single":
                clue_block = "\n".join(f"- {prop}: {target_tile[prop]}" for prop in CLUE_ORDER)
                intro = (
                    "Find the tile that matches all three clues:\n"
                    f"{clue_block}\n\n"
                    "Reply with the tile index in \\boxed{N}."
                )
            else:
                intro = (
                    "Find the hidden target tile. You will receive three clues across three turns "
                    "(pattern, then color, then shape). After each clue, list the tile indices "
                    "that could still be the target. Do not submit a final answer until asked.\n\n"
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
            tasks.append(
                ShapeDetectiveTask(
                    idx=idx,
                    prompt=prompt,
                    system_prompt=SYSTEM,
                    answer=str(target),
                    mode=self.config.mode,
                    target_tile=target_tile,
                )
            )
        return tasks

    def user(self, task: ShapeDetectiveTask) -> vf.User | None:
        if task.mode == "single":
            return None
        return cast(vf.User, ShapeDetectiveUser(self.config.user))

    @vf.stop
    async def single_mode_done(self, trace: vf.Trace) -> bool:
        return trace.task.mode == "single" and trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def solved(self, task: ShapeDetectiveTask, trace: vf.Trace) -> float:
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(extract_boxed(last or "") == task.answer)


if __name__ == "__main__":
    ShapeDetectiveUser.run()


__all__ = ["ShapeDetectiveTaskset"]

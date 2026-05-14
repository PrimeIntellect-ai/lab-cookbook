import base64
import io
import math
import random
from typing import Literal

import verifiers.v1 as vf
from PIL import Image, ImageDraw
from verifiers.utils.data_utils import extract_boxed_answer

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

Mode = Literal["single", "multi"]
Tile = dict[str, str]


SYSTEM_PROMPT = (
    "You are playing Shape Detective. You see a 4x4 grid of tiles numbered 0-15 "
    "(left-to-right, top-to-bottom). Each tile has a shape (circle, square, "
    "triangle, star), a color (red, blue, green, yellow), and a pattern (solid, "
    "striped, dotted). You are told clues that narrow down a single target tile. "
    "When asked to commit your final answer, reply with the tile index in \\boxed{N}."
)


def clue_line(target: Tile, prop: str, clue_index: int) -> str:
    article = "a " if prop == "shape" else ""
    return f"Clue {clue_index + 1} — {prop}: the target is {article}**{target[prop]}**."


def source(mode: Mode, num_rows: int, seed: int):
    def build() -> list[vf.ConfigData]:
        rng = random.Random(seed)
        rows: list[vf.ConfigData] = []
        for _ in range(num_rows):
            while True:
                tiles: list[Tile] = [
                    {
                        "shape": rng.choice(SHAPES),
                        "color": rng.choice(list(COLORS)),
                        "pattern": rng.choice(PATTERNS),
                    }
                    for _ in range(GRID_SIZE * GRID_SIZE)
                ]
                target = rng.randrange(len(tiles))
                key = (
                    tiles[target]["shape"],
                    tiles[target]["color"],
                    tiles[target]["pattern"],
                )
                if sum(1 for t in tiles if (t["shape"], t["color"], t["pattern"]) == key) == 1:
                    break
            target_tile = tiles[target]

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
                elif tile["shape"] == "star":
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
                label_draw.line(
                    [(i * TILE_PX, 0), (i * TILE_PX, IMAGE_PX)], fill=GRID_COLOR, width=2
                )
                label_draw.line(
                    [(0, i * TILE_PX), (IMAGE_PX, i * TILE_PX)], fill=GRID_COLOR, width=2
                )

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

            if mode == "single":
                clue_block = "\n".join(f"- {prop}: {target_tile[prop]}" for prop in CLUE_ORDER)
                intro = (
                    "Find the tile that matches **all three** of these clues:\n"
                    f"{clue_block}\n\n"
                    "Reply with the tile index in \\boxed{N}."
                )
                info: vf.ConfigData = {"mode": "single", "target": target}
                max_turns = 1
            else:
                intro = (
                    "Find the hidden target tile. You will receive three clues "
                    "across three turns (pattern, then color, then shape). After "
                    "each clue, list the tile indices that could still be the "
                    "target — do NOT submit a final answer until asked. When you "
                    "see 'Commit your answer', reply with the index in "
                    "\\boxed{N}.\n\n"
                    f"{clue_line(target_tile, 'pattern', 0)}"
                )
                info = {
                    "mode": "multi",
                    "target": target,
                    "target_tile": target_tile,
                }
                max_turns = 3

            rows.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": intro},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    "answer": str(target),
                    "info": info,
                    "max_turns": max_turns,
                }
            )
        return rows

    return build


async def shape_detective_user(task: vf.Task, state: vf.State) -> list[vf.ConfigData]:
    info = task["info"]
    if info["mode"] != "multi":
        return []
    target_tile: Tile = info["target_tile"]
    assistant_turns = sum(1 for m in state["completion"] if m["role"] == "assistant")

    if assistant_turns == 1:
        return [
            {
                "role": "user",
                "content": (
                    f"{clue_line(target_tile, 'color', 1)}\n\n"
                    "Narrow the candidates again. Still do NOT submit your answer."
                ),
            }
        ]
    if assistant_turns == 2:
        return [
            {
                "role": "user",
                "content": (
                    f"{clue_line(target_tile, 'shape', 2)}\n\n"
                    "Commit your answer now as \\boxed{N}."
                ),
            }
        ]
    return []


@vf.reward(weight=1.0)
async def solved(task: vf.Task, state: vf.State) -> float:
    last = state["completion"][-1]
    if last["role"] != "assistant":
        return 0.0
    content = last["content"]
    text = (
        content
        if isinstance(content, str)
        else " ".join(part["text"] for part in content if part.get("type") == "text")
    )
    answer = extract_boxed_answer(text, strict=True).strip()
    return 1.0 if answer == str(task["answer"]) else 0.0


def load_environment(
    config: vf.EnvConfig,
    mode: Mode = "multi",
    num_rows: int = 12,
    seed: int = 0,
) -> vf.Env:
    return vf.Env(
        taskset=vf.Taskset(
            source=source(mode, num_rows, seed),
            system_prompt=SYSTEM_PROMPT,
            rewards=[solved],
            user=shape_detective_user if mode == "multi" else None,
            config=config.taskset,
        )
    )

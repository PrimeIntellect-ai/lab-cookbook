# Multimodal Environments

Evaluate environments where the prompt includes images, not just text.

Multimodal environments extend the same Verifiers pieces you have already used — a Taskset, a reward, optionally a user callback for multi-turn flow — by letting the user message hold a list of typed content parts instead of a plain string. Each part is either text or an image. The environment never decodes pixels itself: it hands the message to a vision-language model and scores the response.

This guide builds and evaluates [shape-detective](../../environments/shape_detective/), a tiny synthetic visual-deduction game. The model sees a 4×4 grid of tiles — each a combination of shape (circle, square, triangle, star), color (red, blue, green, yellow), and pattern (solid, striped, dotted) — and must identify a single target tile from clues that narrow its properties. Because scenes are rendered procedurally with PIL, there is no benchmark download, no judge model, and a deterministic binary reward.

The env ships in two modes that share the same scene generator:

- **single-turn** (max_turns=1) — all three clues in the initial prompt; the model commits with a boxed tile index, such as `\boxed{N}`
- **multi-turn** (`max_turns=3`, default) — clues revealed across three turns (pattern → color → shape); the model tracks candidates after each clue and commits on the final turn

Both modes are good fits for small native-multimodal models like `Qwen/Qwen3.5-2B`.

## Run It

Run a small eval:

```bash
prime eval run shape-detective \
  -m Qwen/Qwen3.5-2B \
  -n 5 \
  -r 3 \
  -t 1024 \
  -a '{"taskset": {"mode": "multi"}}'
```

Or run with a config file:

```toml
# [configs/08/shape-detective-eval.toml](../../configs/08/shape-detective-eval.toml)
model = "Qwen/Qwen3.5-2B"
save_results = true

[[eval]]
env_id = "shape-detective"
num_examples = 5
rollouts_per_example = 3
sampling_args = { max_tokens = 1024 }
taskset = { mode = "multi" }
```

```bash
prime eval run configs/08/shape-detective-eval.toml
```

Switch to single-turn with `-a '{"taskset": {"mode": "single"}}'`. The walkthrough below follows [environments/shape_detective/shape_detective.py](../../environments/shape_detective/shape_detective.py) top to bottom.

## Constants and Palette

```python
SHAPES = ("circle", "square", "triangle", "star")
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 50, 50),
    "blue": (50, 90, 220),
    "green": (60, 170, 80),
    "yellow": (230, 200, 50),
}
PATTERNS = ("solid", "striped", "dotted")
CLUE_ORDER: tuple[Literal["pattern", "color", "shape"], ...] = (
    "pattern", "color", "shape",
)
```

Four shapes × four colors × three patterns = 48 distinct tile descriptors across 16 grid positions. The colors are picked for high contrast so a small VLM doesn't have to disambiguate fine RGB shades. `CLUE_ORDER` fixes the reveal sequence for the multi-turn variant — pattern is the broadest filter (it splits the grid roughly into thirds), color narrows further, and shape almost always disambiguates the last tile.

Render constants (`TILE_PX = 128`, `GRID_SIZE = 4`, stripe and dot spacings) determine image size (`512×512`) and visual density. Keeping the rendered image small matters because a single image is typically billed as 1k–2k tokens by Prime Inference and gets repeated in the conversation history on every turn — see the [token-budget failure mode](#token-budget) below.

## System Prompt

```python
SYSTEM_PROMPT = (
    "You are playing Shape Detective. You see a 4x4 grid of tiles numbered 0-15 "
    "(left-to-right, top-to-bottom). Each tile has a shape (circle, square, "
    "triangle, star), a color (red, blue, green, yellow), and a pattern (solid, "
    "striped, dotted). You are told clues that narrow down a single target tile. "
    "When asked to commit your final answer, reply with the tile index in \\boxed{N}."
)
```

The system prompt does three things at once: defines the game, fixes the coordinate convention (so the reward can compare against a single integer), and pins the output format (`\boxed{N}`) so a deterministic parser works regardless of how chatty the model is around its answer.

## clue_line

The only standalone helper in the file. Used in `load_tasks` for the multi-turn intro (clue 1) and in the user callback (clues 2 and 3).

```python
def clue_line(target: Tile, prop: str, clue_index: int) -> str:
    article = "a " if prop == "shape" else ""
    return (
        f"Clue {clue_index + 1} — {prop}: the target is {article}**{target[prop]}**."
    )
```

The article handling is purely cosmetic — "the target is a circle" reads better than "the target is circle"; "the target is red" reads better than "the target is a red".

## Taskset Shape

`ShapeDetectiveTaskset` subclasses `vf.Taskset` like [reverse-text](../02-building-your-first-environment/README.md). Config, task generation, the user callback, and the reward live on the class:

```python
class ShapeDetectiveTasksetConfig(vf.TasksetConfig):
    mode: Mode = "multi"
    num_rows: int = 12
    seed: int = 0


class ShapeDetectiveTaskset(vf.Taskset[ShapeDetectiveTasksetConfig]):
    def load_system_prompt(self) -> vf.SystemPrompt:
        return "You are playing Shape Detective..."

    def load_tasks(self) -> vf.Tasks:
        ...

    async def shape_detective_user(self, task: vf.Task, state: vf.State) -> list[vf.ConfigData]:
        ...

    @vf.reward(weight=1.0)
    async def solved(self, task: vf.Task, state: vf.State) -> float:
        ...


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
```

In multi-turn mode, the Taskset sets `self.user = self.shape_detective_user` in `__init__` when `mode == "multi"`.

## load_tasks

`load_tasks` generates each row inline: sample tiles, reject ambiguous scenes, render the grid with PIL, base64-encode to a `data:image/png` URL, compose the intro text, and yield task dicts.

`random.Random(seed)` makes the taskset deterministic for a given config seed.

### Scene generation (inline)

```python
while True:
    tiles: list[Tile] = [
        {"shape": rng.choice(SHAPES),
         "color": rng.choice(list(COLORS)),
         "pattern": rng.choice(PATTERNS)}
        for _ in range(GRID_SIZE * GRID_SIZE)
    ]
    target = rng.randrange(len(tiles))
    key = (tiles[target]["shape"], tiles[target]["color"], tiles[target]["pattern"])
    if sum(1 for t in tiles if (t["shape"], t["color"], t["pattern"]) == key) == 1:
        break
target_tile = tiles[target]
```

Sample 16 tiles independently, pick a random target, reject the scene if any other tile shares the full (shape, color, pattern) combination — otherwise the three clues wouldn't uniquely identify the target and the reward would be ambiguous. With 48 possible combinations and 16 draws, ~73% of scenes pass on the first attempt; rejection is cheap.

### Rendering (inline)

```python
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
        draw.polygon([...], fill=color)
    elif tile["shape"] == "star":
        # ten-point polygon: alternating outer/inner radius from center
        ...

    if tile["pattern"] == "striped":
        for y in range(0, TILE_PX, STRIPE_SPACING):
            draw.rectangle((0, y, TILE_PX, y + STRIPE_WIDTH), fill=BG_COLOR)
    elif tile["pattern"] == "dotted":
        for cy in range(...):
            for cx in range(...):
                draw.ellipse(..., fill=BG_COLOR)

    img.paste(canvas, (x0, y0))
    label_draw.text((x0 + 6, y0 + 4), str(idx), fill=TEXT_COLOR)
```

Each tile is drawn into a small per-cell canvas, then pasted into the grid:

- **Shape** is a single filled primitive on the canvas. The star is the only non-trivial path — a ten-point polygon with alternating outer/inner radius.
- **Pattern** is overdrawn in the background color on top of the filled shape, so a "striped red square" is a solid red square with horizontal grey strips painted across it, and a "dotted blue circle" is a solid blue circle with grey dots punched out. This means the pattern visually overlaps the shape boundary — intentional, since it gives the VLM more pattern-pixels to detect.
- **Tile index** is drawn in the upper-left corner. This is the *only* hint the model gets about the coordinate system, since `\boxed{N}` is an integer rather than a `(row, col)` pair.

Grid lines come last so they sit on top of any shape that bleeds into a neighbor:

```python
for i in range(1, GRID_SIZE):
    label_draw.line([(i * TILE_PX, 0), (i * TILE_PX, IMAGE_PX)], fill=GRID_COLOR, width=2)
    label_draw.line([(0, i * TILE_PX), (IMAGE_PX, i * TILE_PX)], fill=GRID_COLOR, width=2)
```

### Image URL (inline)

```python
buf = io.BytesIO()
img.save(buf, format="PNG")
image_url = (
    "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
)
```

The image goes to a memory buffer as PNG, gets base64-encoded, and is wrapped in a `data:` URL. This is the format every OpenAI-compatible chat completions API accepts as an inline `image_url` content part — and what Prime Inference expects.

### Prompt and row assembly (inline)

The two modes branch on the intro text and on what we stash in `info`:

```python
if mode == "single":
    clue_block = "\n".join(f"- {prop}: {target_tile[prop]}" for prop in CLUE_ORDER)
    intro = (
        "Find the tile that matches **all three** of these clues:\n"
        f"{clue_block}\n\n"
        "Reply with the tile index in \\boxed{N}."
    )
    info = {"mode": "single", "target": target}
    max_turns = 1
else:
    intro = (
        "Find the hidden target tile. You will receive three clues across three "
        "turns ...\n\n"
        f"{clue_line(target_tile, 'pattern', 0)}"
    )
    info = {"mode": "multi", "target": target, "target_tile": target_tile}
    max_turns = 3

    yield {
    "prompt": [{"role": "user", "content": [
        {"type": "text", "text": intro},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]}],
    "answer": str(target),
    "info": info,
    "max_turns": max_turns,
}
```

Two practical notes on the message shape:

- Use a `data:image/<fmt>;base64,<payload>` URL for self-contained datasets. Use an `https://…` URL if your images already live somewhere stable.
- Keep the text part first. Some providers tolerate either order, but text-then-image keeps trajectories readable and matches the convention every reward function expects.

If a row needs multiple images, append more `image_url` parts — the conversation is still one user message.

A row is a plain dict with four interesting fields:

- **`prompt`** — the list of OpenAI-style messages above.
- **`answer`** — the ground truth, a string here so the reward can compare with `==` after parsing. Storing the index as a string mirrors how the `\boxed{N}` payload comes out of the parser.
- **`info`** — a per-row dict that survives serialization and is restored on the other side of the harness. In single mode it carries just the target index (only the reward reads it). In multi mode it carries the whole target tile so the user callback can look up clue values without re-running the scene generator.
- **`max_turns`** — per-row override that the harness reads off the Task object. Single-mode rows ship with `max_turns=1`, multi-mode rows with `max_turns=3`. The harness reads this off `task["max_turns"]` at rollout setup and caps the loop accordingly. A single env with mixed `max_turns` per row is the cleanest way to ship single-turn and multi-turn variants of the same task.

## User Callback (Multi-Turn)

```python
async def shape_detective_user(task, state):
    info = task["info"]
    if info["mode"] != "multi":
        return []
    target_tile = info["target_tile"]
    assistant_turns = sum(1 for m in state["completion"] if m["role"] == "assistant")

    if assistant_turns == 1:
        return [{"role": "user", "content":
            f"{clue_line(target_tile, 'color', 1)}\n\n"
            "Narrow the candidates again. Still do NOT submit your answer."
        }]
    if assistant_turns == 2:
        return [{"role": "user", "content":
            f"{clue_line(target_tile, 'shape', 2)}\n\n"
            "Commit your answer now as \\boxed{N}."
        }]
    return []
```

The user callback is a method on the Taskset that the harness calls between assistant turns. In multi-turn mode, `ShapeDetectiveTaskset.__init__` assigns `self.user = self.shape_detective_user`.

Three things make this work cleanly:

1. **The harness counts turns for us via `state["completion"]`.** That list grows as the conversation progresses: after assistant turn 1 it contains one assistant message; after the harness appends our user-callback output it contains two messages; after assistant turn 2 it contains three; and so on. Counting just the assistant messages gives us a clean turn index.
2. **Returning `[]` ends the rollout.** When the callback returns no messages on turn 3, the harness sets `stop_condition = "no_tools"` and exits the loop. Combined with `max_turns=3`, this gives us a clean stop condition either way (the model can also be cut off by `max_turns_reached` if it kept calling tools, but this env has no tools).
3. **Per-row `info` is the carrier for rollout-time data.** Anything the callback (or the reward) needs at rollout time goes in `info`. The harness round-trips it through serialization, so `task["info"]["target_tile"]` inside the callback matches what `load_tasks` wrote.

Single-mode rows skip the callback because `self.user` is only set when `mode == "multi"`.

## Reward

```python
@vf.reward(weight=1.0)
async def solved(self, task: vf.Task, state: vf.State) -> float:
    last = completion[-1]
    if not isinstance(last, vf.AssistantMessage):
        return 0.0
    ...
    answer = extract_boxed_answer(text, strict=True).strip()
    return 1.0 if answer == str(task["answer"]) else 0.0
```

Use `vf.AssistantMessage` and `vf.TextContentPart` when reading `state["completion"]`.

`extract_boxed_answer(..., strict=True)` returns the contents of the *last* `\boxed{}` in the text, or `""` if there isn't one. Strict mode is important: it collapses "wrong answer" and "no commit" into the same 0.0 reward, which matches what the system prompt asked the model to produce. A more granular reward could break that out into a separate metric — see [Designing Rewards](../02-building-your-first-environment/README.md#designing-rewards) for the pattern.

## Pick a Model

Multimodal eval requires a model that can actually consume the image part. The set of multimodal models on Prime Inference changes regularly, so check the live list rather than copying a table:

```bash
prime inference models --plain --output json | jq '.data[].id' | grep -iE "vl|qwen3\.5|gpt-5|gemini|claude"
```

The full catalog is browsable in the Prime Inference docs under the inference / models section. Don't pick a text-only model and hope it ignores the image — most providers reject the request outright; a few silently drop the image, which produces eval numbers that look plausible but are evaluating something else.

For first-pass evals on a small Qwen3.5 / Qwen3-VL-class model, target the smallest variant that fits your latency budget. The shape-detective smoke set runs in seconds and is enough to sanity-check both the eval wiring and the model's basic vision capability.

## Failure Modes

Three things go wrong in this order, and most multimodal debugging is figuring out which one is biting you.

### Token budget

A single 512×512 image often costs 1k–2k tokens on its own, and many providers tile larger images into more. Multi-turn rollouts repeat the image in the conversation history every turn, so the budget compounds. With a 4k context model you can run out of room before the model has produced any reasoning at all. Symptoms: empty completions, `OverlongPromptError`, or completions that hit `max_tokens` mid-sentence.

Diagnose with a single rollout at a larger budget:

```bash
prime eval run shape-detective -n 1 -r 1 -t 4096
```

If results improve, the bound was the token budget, not the model. Fix by lowering image resolution upstream (resize before base64-encoding) or raising `-t` for evaluation.

### Image encoding

The image part is just a URL string. If it's malformed — wrong MIME type, truncated base64, an `http://` URL behind a paywall, a `file://` path — the provider either fails the call or, worse, accepts the request and answers from the text alone. With shape-detective this is easy to spot: a model that can't see the image will guess at random or pick a fixed tile every time, since the text clues never name an index.

Sanity checks before trusting a multimodal eval:

- decode one base64 payload yourself and confirm it opens as an image
- run the same eval at `num_rows=2` and verify the model references specific tile *positions* (the only signal exclusive to the image)
- if rewards are suspiciously uniform across rows, suspect the image isn't being delivered

### Judge mismatch

If your reward is a text-only LLM judge — common for free-form VQA where the answer is a sentence — the judge typically does **not** see the image. It scores the model's text against a reference answer, which means it can't catch grounded errors ("the chart shows 42% — wait, no, that's the *blue* bar"). The judge can rate fluency and surface form, not visual correctness.

Shape-detective sidesteps this entirely by using an exact-match reward on a tile index. For first-version multimodal rewards, prefer deterministic checks: parsed multiple-choice letters, exact-match strings, bounding-box overlap, numeric tolerance. Save judges for diagnostics until you've wired a multimodal-aware judge that gets the image in its prompt too.

## Make Your Own

The implementation above is a working template. The pattern is:

1. **Render once per row.** Generate the image with PIL (or any library), encode as PNG, wrap as a `data:image/png;base64,...` URL inside an `image_url` content part.
2. **Compose the user message** as `[{"type": "text", ...}, {"type": "image_url", ...}]`. Put text first.
3. **Wire ground truth into `info`.** Anything the reward or user callback needs at rollout time goes in `info`; the harness preserves it through serialization.
4. **Set `max_turns` per row** when different rows need different rollout budgets — useful for mixing single-turn and multi-turn examples in one taskset, or for varying difficulty.
5. **For multi-turn, attach a `user` callback** that reads `state["completion"]` to decide what to emit next.

Two upgrades that pay back quickly:

- **Cache encoded images.** If scenes are deterministic given a seed, cache rendered PNGs or row dicts and skip re-rendering on every eval.
- **Stratify your eval set.** Mix easy, medium, and hard rows. A flat random sample hides which capability the model is missing. For shape-detective specifically, vary `seed` across eval splits and check that performance is stable across them — not just on the seed you developed against.

## Training Notes

RL on multimodal environments is supported in `prime-rl` for VL models, but model coverage is narrower than for text-only training. Before launching a training run, confirm the specific model you're targeting is in the supported list — the Prime Inference model catalog is the source of truth, and the Hosted Training docs flag any multimodal-specific limits.

The eval workflow above is the same shape RL uses, so a clean shape-detective eval against your candidate model is a useful prerequisite check.

## Framework Notes

A few details from building `shape_detective` that are easy to miss:

- **Per-row `max_turns`.** Set `"max_turns"` on each task row to override the harness default. This is the cleanest way to mix single-turn and multi-turn rows in one taskset.
- **User callbacks** are Taskset methods assigned to `self.user`. The harness calls them between assistant turns with `task` and `state`.
- **Custom `info` round-trips** through serialization — fields you set in `load_tasks` are available on `task["info"]` in callbacks and rewards.
- **`state["completion"]` is `list[vf.Message]`.** Prefer `vf.AssistantMessage` and related types over dict access.
- **Config subclasses expose tunables.** Put `mode`, `num_rows`, and `seed` on `ShapeDetectiveTasksetConfig`; override via `taskset` in eval configs or `-a`.

## Next

In [Coding Agents and Sandboxes](../10-coding-agents-and-sandboxes/README.md), you will evaluate environments where the model writes or runs code inside a sandbox.

# Multimodal Environments

So far every prompt was a string. In this tutorial you will build tasks that carry **images**. The example is the cookbook's v1 `shape-detective` taskset: the model sees a 4×4 grid and identifies a hidden target tile from clues.

**You need:** [Build Your First Environment](5_build_first_environment.md). Shape-detective's multi-turn side — the simulated user who reveals the clues — is covered in [User Simulators](8_user_simulators.md); here we focus on the image half.

## Message-list prompts

`TaskData.prompt` may be a typed message list instead of a string:

```python
prompt: vf.Messages = [
    vf.UserMessage(
        content=[
            vf.TextContentPart(text=intro),
            vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image_data_uri)),
        ]
    )
]
```

`Taskset.load()` generates each grid with PIL and embeds it as a base64 `data:image/png;...` URI:

```python
def tile_image_data_url(tiles: list[Tile]) -> str:
    image = Image.new("RGB", (IMAGE_PX, IMAGE_PX), BG_COLOR)
    ...  # draw shapes, colors, patterns, grid lines, tile indices
    return image_data_url(image)
```

Two things to notice about this approach:

- **The task is fully self-contained.** A data URI travels inside the task itself — no image hosting, no broken links, and the environment stays a single installable package. (For large real-image datasets, tasks can reference dataset assets instead; the message-list shape is the same.)
- **Generated images are ground-truth-exact.** Because the environment *draws* the grid, it knows precisely what's in every tile — the same generate-then-score advantage as [Infinite Tasksets](14_infinite_tasksets.md) and [Synthetic Worlds](15_synthetic_world.md), extended to pixels. There is no labeling step to get wrong.

Everything else is ordinary v1 task structure:

```python
class ShapeDetectiveTaskData(vf.TaskData):
    answer: str
    target_tile: Tile


class ShapeDetectiveTask(vf.Task[ShapeDetectiveTaskData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return float(extract_boxed(trace.last_reply) == self.data.answer)


class ShapeDetectiveConfig(vf.TasksetConfig):
    num_tasks: int = 12
    seed: int = 0
    task: vf.TaskConfig = vf.TaskConfig()


class ShapeDetectiveTaskset(vf.Taskset[ShapeDetectiveTask, ShapeDetectiveConfig]):
    def load(self) -> list[ShapeDetectiveTask]:
        return [
            ShapeDetectiveTask(
                ShapeDetectiveTaskData(
                    idx=i,
                    prompt=build_message_prompt(example),
                    answer=str(example.target_index),
                    target_tile=example.target_tile,
                ),
                self.config.task,
            )
            for i, example in enumerate(build_examples(self.config))
        ]
```

The target lives on immutable `TaskData`; the reward reads it through `self.data` and reads the final model text through `trace.last_reply`. Multimodality changes the prompt, not scoring.

## Capability check

The harness must support message prompts, advertised as `SUPPORTS_MESSAGE_PROMPT`. The built-in `default` harness does; a CLI coding agent that only accepts a text prompt does not. (And the *model* must be a vision model — a text-only model will happily hallucinate a grid it cannot see, which is worth witnessing once in the traces.)

## Run it

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 6
num_rollouts = 2
max_turns = 6

[sampling]
max_tokens = 1024

[taskset]
id = "shape-detective"
num_tasks = 12
seed = 0                        # deterministic grids: everyone sees the same tasks

[harness]
id = "default"
```

```bash
uv run eval @ configs/09/shape-detective-eval.toml
```

In `traces.jsonl`, the first `vf.UserMessage` contains `vf.TextContentPart` and `vf.ImageUrlContentPart`. A rollout that misdescribes the grid has a **vision** error; one that describes it correctly but eliminates the wrong candidates has a **reasoning** error.

## Try it

- Expose a single-turn taskset with all clues upfront and compare it with the user-driven port: how much does the conversation cost?
- Shrink the tiles: halve `TILE_PX` in the renderer and re-run — a crude but effective probe of where the model's visual acuity gives out.
- Add a `@vf.metric` that checks whether the model's *first* candidate list is consistent with the image (the environment knows every tile's truth) — a vision-accuracy metric that's independent of the final answer.

## Next

→ [Tool Use and Search](10_tools.md): the other interaction primitive — toolsets the model calls, and how to read what it did with them.

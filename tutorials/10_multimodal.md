# Multimodal Environments

So far every prompt was a string. In this tutorial you will build tasks that carry **images**. The example is `shape-detective`: the model sees a 4×4 grid of colored shapes and must identify a hidden target tile from clues — a task where the evidence is visual, so the prompt has to be too.

**You need:** [Build Your First Environment](5_build_first_environment.md). Shape-detective's multi-turn side — the simulated user who reveals the clues — is covered in [User Simulators](8_user_simulators.md); here we focus on the image half.

## Message-list prompts

A `vf.Task.prompt` does not have to be a string — it can be a full message list with typed content parts:

```python
prompt = [
    vf.UserMessage(
        content=[
            vf.TextContentPart(text=intro),
            vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image_data_uri)),
        ]
    )
]
```

`shape-detective` generates each grid with PIL at `load_tasks` time and embeds it as a base64 `data:image/png;...` URI:

```python
def tile_image_data_url(tiles: list[Tile]) -> str:
    img = Image.new("RGB", (IMAGE_PX, IMAGE_PX), BG_COLOR)
    ...  # draw shapes, colors, patterns, grid lines, tile indices
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
```

Two things to notice about this approach:

- **The task is fully self-contained.** A data URI travels inside the task itself — no image hosting, no broken links, and the environment stays a single installable package. (For large real-image datasets, tasks can reference dataset assets instead; the message-list shape is the same.)
- **Generated images are ground-truth-exact.** Because the environment *draws* the grid, it knows precisely what's in every tile — the same generate-then-score advantage as [Infinite Tasksets](14_infinite_tasksets.md) and [Synthetic Worlds](15_synthetic_world.md), extended to pixels. There is no labeling step to get wrong.

Everything else about the task is ordinary: the target index is `answer`, and the reward extracts `\boxed{N}` from the last assistant message. Multimodality changes the *prompt*, not the scoring contract.

## Capability check

The harness must support message prompts, advertised as `SUPPORTS_MESSAGE_PROMPT`. The built-in `default` harness does; a CLI coding agent that only accepts a text prompt does not. (And the *model* must be a vision model — a text-only model will happily hallucinate a grid it cannot see, which is worth witnessing once in the traces.)

## Run it

```bash
prime eval run @ configs/09/shape-detective-eval.toml
```

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 6
num_rollouts = 2
max_turns = 6

[sampling]
max_tokens = 1024

[taskset]
id = "shape-detective"
mode = "multi"                  # "single" gives all clues upfront: pure vision test
num_tasks = 12
seed = 0                        # deterministic grids: everyone sees the same tasks

[harness]
id = "default"
```

In the traces you'll see the image-bearing first user message, then the model reasoning about what it sees. When it fails, diagnose *which sense* failed: a rollout whose candidate lists misdescribe the grid (calls a striped tile solid) is a **vision** error; one that describes tiles correctly but eliminates the wrong candidates is a **reasoning** error. The two look completely different in the transcript and have completely different fixes — better image resolution and clearer tile rendering for one, prompt and model quality for the other.

## Try it

- Run `--taskset.mode single` (all clues upfront, one turn) for the pure vision-plus-logic version, and compare against multi-turn: how much does the conversation cost?
- Shrink the tiles: halve `TILE_PX` in the renderer and re-run — a crude but effective probe of where the model's visual acuity gives out.
- Add a `@vf.metric` that checks whether the model's *first* candidate list is consistent with the image (the environment knows every tile's truth) — a vision-accuracy metric that's independent of the final answer.

## Next

→ [Coding Agent Environments](11_coding_agents.md): tasks where the model's output runs — interpreters, Docker, and sandboxes.

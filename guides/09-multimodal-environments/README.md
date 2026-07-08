# 09 — Multimodal Environments

So far every prompt was a string. In this guide you will build tasks that carry **images**, and meet the second interaction primitive after tools: the **user simulator**, a server that plays the human side of a multi-turn conversation. The example is `shape-detective`: the model sees a 4×4 grid of colored shapes and must identify a hidden target tile from clues doled out one per turn.

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

`shape-detective` generates each grid with PIL at `load_tasks` time and embeds it as a base64 `data:image/png;...` URI — the task is fully self-contained, no image hosting involved. Everything else about the task is ordinary: the target index is `answer`, and the reward extracts `\boxed{N}` from the last assistant message.

One capability check: the harness must support message prompts (advertised as `SUPPORTS_MESSAGE_PROMPT`). The built-in `default` harness does; a CLI coding agent that only accepts a text prompt does not.

## User simulators

In Wordle (Guide 01), feedback came from game logic. Here the taskset instead simulates a *user* who reveals clues across turns and eventually asks the model to commit. A user simulator is a `vf.User` server — the same server pattern as a toolset, but it produces the conversation's user turns:

```python
class ShapeDetectiveState(vf.State):
    clue_index: int = 1


class ShapeDetectiveUser(vf.User[vf.UserConfig, ShapeDetectiveState]):
    async def setup_task(self, task: ShapeDetectiveTask) -> None:
        self.target_tile = task.target_tile      # per-task ground truth

    async def respond(self, message: str) -> vf.Messages:
        if self.state.clue_index == 1:
            self.state.clue_index = 2
            return [vf.UserMessage(content=f"{clue_line(...)}\n\nNarrow the candidates...")]
        if self.state.clue_index == 2:
            self.state.clue_index = 3
            return [vf.UserMessage(content="...Commit your answer now as \\boxed{N}.")]
        return []                                 # nothing left to say → rollout ends


if __name__ == "__main__":
    ShapeDetectiveUser.run()
```

The contract in three parts:

- **`setup_task(task)`** hands the simulator its per-task script — here, which tile is the target, so it can phrase truthful clues.
- **`respond(message)`** is called after each assistant turn and returns the next user message(s). Returning `[]` means the user has nothing more to say, which ends the conversation — a natural stop condition without any `@vf.stop`.
- **Turn progress lives in `self.state`**, a typed `vf.State`. This is the same state channel as toolsets (Guide 08): serializable, per-rollout, visible to rewards on `trace.state`.

The taskset wires it in per task — and can decline to, which is how one environment supports both interaction modes:

```python
class ShapeDetectiveConfig(vf.TasksetConfig):
    mode: Mode = "multi"            # "single": all clues upfront, one turn
    user: vf.UserConfig = vf.UserConfig()

class ShapeDetectiveTaskset(vf.Taskset[...]):
    def user(self, task: ShapeDetectiveTask) -> vf.User | None:
        if task.mode == "single":
            return None             # no simulator: single-turn eval
        return ShapeDetectiveUser(self.config.user)
```

As with MCP tools, user simulation is a harness capability (`SUPPORTS_USER_SIM`) — the `default` harness supports it; many CLI-agent harnesses do not.

## Run it

`configs/09/shape-detective-eval.toml` selects multi-turn mode:

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 6
num_rollouts = 2
max_turns = 6                   # 3 clue exchanges fit comfortably

[sampling]
max_tokens = 1024

[taskset]
id = "shape-detective"
mode = "multi"
num_tasks = 12
seed = 0                        # deterministic grids: everyone sees the same tasks

[harness]
id = "default"
```

```bash
prime eval run @ configs/09/shape-detective-eval.toml
```

In the traces you'll see the shape of the game: the image-bearing first user message, the model narrowing candidates, the simulator's scripted clues, and the final `\boxed{N}`. Compare a failing rollout's candidate lists against the clues — vision errors and reasoning errors look very different in the transcript.

## Try it

- Run the same config with `--taskset.mode single` and compare solve rates: how much does distributing the clues across turns actually cost the model?
- Make the simulator adversarial: have `respond` answer a direct question from the model ("is it striped?") only if the model listed candidates in its previous turn.
- Change `seed` and confirm the grids (and answers) change while everything else stays reproducible.

## Next

→ [10 — Coding Agents and Sandboxes](../10-coding-agents-and-sandboxes/README.md): tasks that execute code — persistent interpreters, Docker runtimes, and Harbor task suites.

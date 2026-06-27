# Multimodal Environments

A v1 task can carry a full message list as its prompt. Use typed content parts for image inputs:

```python
prompt=[
    vf.UserMessage(
        content=[
            vf.TextContentPart(text="Which tile is the target?"),
            vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image_data_uri)),
        ]
    )
]
```

The harness must support message prompts. The built-in default harness does, and the `shape-detective` taskset uses it with a user simulator for multi-turn clues.

A user simulator is a `vf.User` server:

```python
class ShapeDetectiveUser(vf.User[vf.UserConfig, ShapeDetectiveState]):
    async def setup_task(self, task: ShapeDetectiveTask) -> None:
        self.state.clue_index = 1

    async def respond(self, message: str) -> vf.Messages:
        ...
```

The taskset wires it in:

```python
def user(self, task: ShapeDetectiveTask) -> vf.User | None:
    if task.mode == "single":
        return None
    return ShapeDetectiveUser(self.config.user)
```

Run the example:

```bash
uv run eval @ configs/09/shape-detective-eval.toml
```

# shape-detective

A v1 multimodal deduction taskset.

- Task: image-bearing prompt with a hidden target tile
- Mode: `single` asks for one direct answer; `multi` uses a user simulator for clues
- User simulator: `ShapeDetectiveUser`
- Reward: exact target-tile identification

Run:

```bash
uv run eval @ configs/09/shape-detective-eval.toml
```

Example taskset config:

```toml
[taskset]
id = "shape-detective"
mode = "multi"
num_tasks = 12
seed = 0
```

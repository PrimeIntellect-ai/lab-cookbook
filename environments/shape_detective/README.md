# shape-detective

A latest-verifiers-v1 multimodal deduction taskset modeled on
`color_codeword_v1`.

- `ShapeDetectiveTaskData` and `ShapeDetectiveState` type the task and rollout state.
- The initial prompt is a message list containing text and the generated grid image.
- `single` asks for one direct answer; `multi` uses the dedicated
  `ShapeDetectiveUser` server to reveal later clues.
- `ShapeDetectiveTask` declares the user server and owns stopping and exact-match
  reward behavior.

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

[taskset.task.user]
colocated = false
```

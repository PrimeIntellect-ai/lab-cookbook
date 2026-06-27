# reverse-text

A minimal v1 taskset for single-turn string reversal.

- Taskset: `ReverseTextTaskset`
- Task: prompt text plus the ground-truth reversed answer
- Reward: LCS ratio over the text inside `<reversed_text>` tags
- Stop: one model turn

Run:

```bash
uv run eval @ configs/02/reverse-text-eval.toml
uv run eval reverse-text -n 3 -r 2
```

Package contract:

```python
import verifiers.v1 as vf

class ReverseTextTaskset(vf.Taskset[ReverseTextTask, ReverseTextConfig]):
    ...

__all__ = ["ReverseTextTaskset"]
```

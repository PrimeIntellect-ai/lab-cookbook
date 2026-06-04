# shape-detective

A toy multimodal environment for small vision-capable LMs. The model sees a 4×4
grid of 16 tiles, each a unique combination of:

- **shape** — circle, square, triangle, star
- **color** — red, blue, green, yellow
- **pattern** — solid, striped, dotted

and must identify a single **target tile** (by its index `0–15`) from clues
narrowing down its shape, color, and pattern.

Scenes are generated procedurally with PIL — no external dataset, no judge model,
no token-budget surprises. The target tile is guaranteed unique within the grid
(no other tile shares its full shape × color × pattern combination).

## Modes

- `single` (`max_turns=1`) — all three clues are bundled into the initial prompt;
  the model commits in one shot with `\boxed{N}`.
- `multi` (`max_turns=3`, default) — clues are revealed across three turns
  (pattern → color → shape). The model is asked to track candidate tiles after
  each clue and commit only on the final turn. Implemented with a paired
  `vf.User`.

The two modes share the same scene generator and reward, so they exercise the
same perception task with different conversational scaffolding.

## Reward

Binary outcome: `1.0` if `extract_boxed_answer` from the final assistant message
exactly matches the target index, else `0.0`.

## Run

```bash
prime eval run prime/shape-detective \
    --model qwen/qwen3-vl-8b-instruct \
    --provider prime \
    -n 5 -r 3 -t 2048
```

Switch to single-turn in eval TOML:

```toml
[[eval]]
env_id = "prime/shape-detective"

[eval.taskset]
mode = "single"
num_rows = 20
```

Or use [configs/08/shape-detective-eval.toml](../../configs/08/shape-detective-eval.toml).

## Taskset Config

- `mode: "single" | "multi"` (default `"multi"`)
- `num_rows: int` (default `12`)
- `seed: int` (default `0`)

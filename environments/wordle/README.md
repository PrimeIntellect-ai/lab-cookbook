# wordle

Multi-turn Wordle environment backed by TextArena's `Wordle-v0`. The model
guesses a secret five-letter word across turns; TextArena returns per-letter
feedback, and four reward functions score correctness, partial progress, guess
efficiency, and output format.

### Overview
- **Environment ID**: `wordle`
- **Pattern**: `Taskset` + default `Harness`
- **Tags**: textarena, multi-turn, reasoning, game

### Task
- **Type**: multi-turn
- Each episode samples a secret word from TextArena's Wordle word list.
- The model submits one guess per turn inside `<guess>...</guess>` tags, using
  TextArena's action format with square brackets: `<guess>[crane]</guess>`.
- `WordleUser` parses the latest assistant message, steps the TextArena env,
  and returns letter feedback until the game ends.

### Quickstart

```bash
prime eval run wordle
```

Configure model and sampling:

```bash
prime eval run wordle \
  -m openai/gpt-oss-120b \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"taskset": {"num_train_examples": 100, "num_eval_examples": 20}, "harness": {"max_turns": 6}}'
```

### Configuration

Taskset fields (via eval config or `-a`):

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training episodes |
| `num_eval_examples` | int | `20` | Number of evaluation episodes |
| `seed` | int | `0` | Seed for sampling target words |
| `game` | str | `Wordle-v0` | TextArena game id |
| `system_prompt` | str or `{ path = ... }` | `prompts/system_prompt.txt` | Literal prompt text or file-backed `SystemPromptConfig` override |

`WordleTaskset.load_system_prompt` loads the default prompt from
`prompts/system_prompt.txt`. GEPA runs with `save_to_environment = true` can
update that file directly, so follow-up evals use the optimized prompt without
adding a separate prompt argument.

### Rewards

| Reward | Weight | Meaning |
| ------ | ------ | ------- |
| `correct_answer` | `1.0` | `1.0` if the final guess equals `[answer]`, else `0.0` |
| `partial_answer` | `1.0` | `0.2 * greens + 0.1 * yellows` from the latest feedback block (zero on win) |
| `length_bonus` | `1.0` | `is_correct / num_guesses` |
| `format_reward` | `0.2` | `1.0` if each assistant turn contains exactly one `<guess>` tag |

### Layout
- `wordle.py`: Wordle config, `WordleUser`, rewards, TextArena taskset subclass, loaders
- `prompts/system_prompt.txt`: Default prompt loaded through `load_system_prompt`

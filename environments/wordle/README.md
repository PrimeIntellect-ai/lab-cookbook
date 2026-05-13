# wordle

A verifiers **v1** port of the Wordle environment. Game state is driven by
TextArena's `Wordle-v0` and wrapped in a reusable `TextArenaTaskset`; the
reward semantics (`correct_answer`, `partial_answer`, `length_bonus`,
`format_reward`) are preserved from the upstream v0 implementation.

### Overview
- **Environment ID**: `wordle`
- **Pattern**: v1 `Taskset` + default `Harness`
- **Tags**: textarena, multi-turn, reasoning, game, v1, taskset, harness

### Files
- `wordle.py`: env loader, reward funcs, system prompt
- `textarena_taskset.py`: reusable `TextArenaTaskset` (intended for later
  upstream into `verifiers.v1.packages.tasksets`)
- `pyproject.toml`: deps (`verifiers`, `nltk`, `textarena==0.7.4`)

### Task
- A secret 5-letter word is sampled from TextArena's Wordle word list.
- Each turn the model writes `<guess>[word]</guess>` (square brackets are the
  TextArena action format).
- The taskset's user function parses the guess, steps the underlying
  TextArena env, and returns its `[GAME]` feedback message (G/Y/X per letter).

### Quickstart

```bash
prime env install wordle -p ./environments
prime eval run wordle
```

Configure model and sampling:

```bash
prime eval run wordle \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 2000, "num_eval_examples": 20}'
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training episodes |
| `num_eval_examples` | int | `20` | Number of evaluation episodes |
| `seed` | int | `0` | Seed for sampling target words |
| `system_prompt` | str \| None | built-in | Override the system prompt |
| `path_to_system_prompt` | str \| Path \| None | `None` | Load system prompt from a text file |

### Rewards (logic preserved from v0)
| Reward | Weight | Meaning |
| ------ | ------ | ------- |
| `correct_answer` | `1.0` | `1.0` if parsed guess equals `[answer]`, else `0.0` |
| `partial_answer` | `1.0` | `0.2 * greens + 0.1 * yellows` from the latest feedback block (zero on win) |
| `length_bonus` | `1.0` | `is_correct / num_guesses` |
| `format_reward` | `0.2` | v0 XMLParser format reward applied to the completion |

### TextArenaTaskset

`textarena_taskset.py` exposes a generic `TextArenaTaskset(Taskset)` subclass
that wraps any single-player TextArena game whose `game_state` carries a
single answer field (default key: `secret_word`). Key parameters:

```python
TextArenaTaskset(
    game="Wordle-v0",
    num_train_examples=1000,
    num_eval_examples=0,
    seed=0,
    answer_state_key="secret_word",
    parser=vf.XMLParser(fields=["think", "guess"], answer_field="guess"),
    feedback_fn=lambda observation: observation,
    rewards=[...],
)
```

Internally:

- One template `ta.Env` is built at construction time, and a deepcopy memo
  is precomputed so the ~38MB English dictionary is shared across rollouts.
- A per-rollout `setup` hook deepcopies the template env and injects
  `task["answer"]` into `ta_env.state.game_state[answer_state_key]`.
- The `user` hook parses the model's last assistant message, steps the env,
  and returns either the next observation (post-`feedback_fn`) or the
  game-over `reason` (terminating the rollout via `state["final_env_response"]`).
- A `cleanup` hook drops the env reference at the end of the rollout.

The class is intentionally written without env-specific assumptions so it can
later be upstreamed to `verifiers/v1/packages/tasksets/textarena.py`.

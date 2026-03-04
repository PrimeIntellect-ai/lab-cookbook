# Word Game

Multi-turn Wordle-style word guessing environment. The model must deduce a hidden 5-letter word within 6 guesses using letter-position feedback. Tests multi-turn reasoning and hypothesis updating.

**Environment type:** `StatefulToolEnv`  
**Reward:** Sparse — `1.0` for correct word, `0.0` otherwise, with soft partial credit based on guess quality  
**Dataset:** Common English words (2,309 Wordle answer words)

---

## Quick Start

```bash
prime env install prime_cookbook/recipes/word_game

prime eval run recipe-word-game --model gpt-4.1-mini
# Expected: reward mean ~0.38 (hard task)

prime rl run prime_cookbook/recipes/word_game/config.toml
```

---

## Environment Overview

The model must call `guess(word)` up to 6 times. Each call returns colored feedback:

```
Guess: CRANE
Result: 🟨⬛🟩⬛⬛
         C     R     A     N     E
         ^yellow: letter in word, wrong position
               ^gray: letter not in word
                     ^green: letter in correct position
```

```python
import verifiers as vf
import random

WORDLE_WORDS = [...]  # 2,309 answer words

class WordGameEnv(vf.StatefulToolEnv):
    def setup_state(self, state):
        state["target"] = state["info"].get("word") or random.choice(WORDLE_WORDS)
        state["guesses"] = []
        state["solved"] = False
        return state

    def guess(self, word: str, _state: dict) -> str:
        """
        Guess a 5-letter word. Returns colored feedback:
        🟩 = correct letter, correct position
        🟨 = correct letter, wrong position
        ⬛ = letter not in word
        """
        word = word.strip().upper()
        if len(word) != 5 or not word.isalpha():
            return "Invalid: must be a 5-letter word."

        target = _state["target"].upper()
        feedback = []
        for i, (g, t) in enumerate(zip(word, target)):
            if g == t:
                feedback.append("🟩")
            elif g in target:
                feedback.append("🟨")
            else:
                feedback.append("⬛")

        result = f"{''.join(feedback)} {word}"
        _state["guesses"].append({"word": word, "result": result})

        if word == target:
            _state["solved"] = True

        return result

    def get_tools(self):
        return [self.guess]
```

---

## Reward Design

```python
def word_game_reward(completion: str, state: dict, **kwargs) -> float:
    if state.get("solved"):
        # Bonus for solving in fewer guesses
        n_guesses = len(state["guesses"])
        efficiency = (6 - n_guesses + 1) / 6   # 1.0 if 1 guess, ~0.17 if 6
        return 0.7 + 0.3 * efficiency

    # Partial credit: reward for narrowing down letters correctly
    guesses = state.get("guesses", [])
    if not guesses:
        return 0.0

    target = state["target"].upper()
    last_result = guesses[-1]["result"]
    greens = last_result.count("🟩")
    yellows = last_result.count("🟨")

    # Soft partial: reward informative guesses
    return (greens * 0.04 + yellows * 0.02)  # max ~0.24 from guesses

rubric = vf.Rubric(funcs=[word_game_reward])
```

---

## Expected Metrics

| Model | Reward Mean | Solve Rate | Avg Guesses (when solved) |
|-------|-------------|-----------|--------------------------|
| gpt-4.1-mini | ~0.38 | ~35% | 4.2 |
| Qwen2.5-7B-Instruct | ~0.22 | ~18% | 4.8 |
| Qwen2.5-1.5B-Instruct | ~0.08 | ~5% | 5.1 |
| Human average | ~0.82 | ~97% | 3.7 |

This is a genuinely hard task — sparse reward, multi-turn reasoning, requires systematic elimination strategy.

---

## Training Config

```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[training]
max_steps = 3000
batch_size = 64
rollouts_per_example = 16    # sparse reward → more rollouts

[sampling]
max_tokens = 512
temperature = 1.0            # high temperature for diverse strategies

[[env]]
id = "recipe-word-game"
weight = 1.0
```

**Note:** Use `rollouts_per_example=16` for sparse rewards like this. With only ~18% base solve rate, you need many rollouts to get positive signal in each batch.

---

## System Prompt

```python
system_prompt = """You are playing Wordle. Guess a hidden 5-letter English word in 6 tries.

After each guess, you get colored feedback:
🟩 = correct letter in correct position
🟨 = correct letter in wrong position  
⬛ = letter not in the word

Strategy:
1. Start with a high-coverage word (e.g., CRANE, STARE, ADIEU)
2. Use feedback to eliminate letters and narrow positions
3. Make each guess a valid English word

Use the `guess` tool to submit your guesses."""
```

---

## Extension Ideas

**Harder variant — no partial credit:**
```python
# Pure sparse: 1.0 only for solve
def strict_reward(completion, state, **kwargs):
    return 1.0 if state.get("solved") else 0.0
```

**Custom word lists:**
```python
# Domain-specific Wordle (medical terms, programming keywords, etc.)
WORDS = ["numpy", "torch", "token", "layer", "batch"]
```

**Unlimited guesses with diminishing reward:**
```python
def reward(completion, state, **kwargs):
    if not state.get("solved"):
        return 0.0
    n = len(state["guesses"])
    return max(0.0, 1.0 - (n - 1) * 0.15)  # 1.0 in 1 guess, 0.25 in 6
```

---

## Related

- [Environment Types](../environment-types.md) — StatefulToolEnv
- [Reward Design](../reward-design.md) — sparse reward strategies
- [Multi-Env Recipe](multi-env.md) — combining word-game with other tasks

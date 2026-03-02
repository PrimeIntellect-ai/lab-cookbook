# Word Game — Wordle (MultiTurnEnv + partial rewards)

A Wordle-style 5-letter word guessing game using `vf.MultiTurnEnv`.  
This is the primary recipe for learning the multi-turn environment pattern.

## How it works

1. The model receives the system prompt explaining Wordle rules.
2. Each turn: model guesses a 5-letter word.
3. Environment responds with letter feedback: `G B Y G B`
4. Model has 6 attempts to guess correctly.
5. Reward is higher for fewer guesses needed:

| Guesses needed | Reward |
|----------------|--------|
| 1 | 1.00 |
| 2 | 0.90 |
| 3 | 0.75 |
| 4 | 0.60 |
| 5 | 0.45 |
| 6 | 0.30 |
| Failed | 0.00 |

## Setup

```bash
pip install verifiers>=0.1.10
```

## Quick eval

```python
from prime_cookbook.recipes.word_game.word_game import load_environment
import verifiers as vf

env = load_environment(num_examples=20)
vf.evaluate(env, model="gpt-4.1-mini", rollouts_per_example=4)
```

## Training run

```bash
prime rl run config.toml
```

## Key patterns demonstrated

### Custom MultiTurnEnv subclass
```python
class WordleEnv(vf.MultiTurnEnv):
    def setup_state(self, state):    # per-rollout init
        state["secret"] = ...

    def env_response(self, messages, state):  # game logic
        return feedback_string, state

    @vf.stop
    def stop_on_solved(self, messages, state):  # custom termination
        return state["solved"]

    async def reward(self, completion, answer, state, **kwargs):
        return partial_score
```

### Feedback logic
The `_compute_feedback(guess, secret)` function correctly handles:
- Duplicate letters (counts available letters before awarding Y)
- All-correct (GGGGG) 
- No matches (BBBBB)

## Expected training behaviour

- Starting reward: ~0.15 (random baseline — occasionally guesses correctly)
- After 50 steps: ~0.35 (model learns to use feedback)
- After 150 steps: ~0.50-0.60 (model develops Wordle strategy)
- The model should learn to: use process of elimination, avoid already-ruled-out letters, prioritise common letters early

## Adapting this pattern

The `MultiTurnEnv` + `setup_state` + `env_response` + `@vf.stop` pattern works for any  
turn-based game or multi-step interactive task:
- 20 Questions (yes/no narrowing)
- Text adventures
- Negotiation dialogues
- Multi-step form filling with validation

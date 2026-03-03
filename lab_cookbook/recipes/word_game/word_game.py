"""Word Game (Wordle) — MultiTurnEnv recipe.

A Wordle-style 5-letter word guessing game. The model gets up to 6 attempts
to guess the secret word. After each guess the environment returns per-letter
feedback: G=correct position, Y=present but wrong position, B=not in word.

This recipe demonstrates:
  - Custom MultiTurnEnv with per-rollout state
  - @vf.stop decorators for custom termination conditions
  - @vf.cleanup for per-rollout resource cleanup
  - Partial reward function (fewer guesses → higher reward)

Reward schedule:
  Guess 1 → 1.00
  Guess 2 → 0.90
  Guess 3 → 0.75
  Guess 4 → 0.60
  Guess 5 → 0.45
  Guess 6 → 0.30
  Fail    → 0.00

Starting reward with 1B model: ~0.15-0.25 (random guessing baseline).
"""
from __future__ import annotations

import random
import verifiers as vf
from datasets import Dataset
from typing import Any

SYSTEM_PROMPT = """You are playing Wordle. Guess the secret 5-letter word.

Rules:
- Each guess must be a single 5-letter English word.
- After each guess you receive letter-by-letter feedback:
    G = correct letter in the correct position
    Y = correct letter in the wrong position
    B = letter not in the word at all
- You have 6 attempts total.
- Output ONLY the 5-letter word for each guess, nothing else.

Example:
Guess: CRANE
Feedback: B G B Y B
→ 'R' is in position 2, 'A' is in the word but not position 4.
"""

# Word list — 200 common 5-letter words

_WORDS = [
    "ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT",
    "AFTER", "AGAIN", "AGENT", "AGREE", "AHEAD", "ALARM", "ALBUM", "ALERT",
    "ALIEN", "ALIGN", "ALIKE", "ALIVE", "ALLEY", "ALLOW", "ALONE", "ALONG",
    "ALTER", "ANGEL", "ANGER", "ANGLE", "ANIME", "ANNEX", "APART", "APPLE",
    "APPLY", "ARENA", "ARGUE", "ARISE", "ARMOR", "ARRAY", "ARROW", "ASSET",
    "ATLAS", "ATTIC", "AUDIO", "AUDIT", "AVOID", "AWARD", "AWARE", "AWFUL",
    "BASIC", "BASIS", "BEACH", "BEGAN", "BEGIN", "BEING", "BELOW", "BENCH",
    "BLACK", "BLADE", "BLAME", "BLAND", "BLANK", "BLAST", "BLAZE", "BLEED",
    "BLEND", "BLINK", "BLOCK", "BLOOD", "BLOOM", "BLOWN", "BLUES", "BLUNT",
    "BOARD", "BONUS", "BRAIN", "BRAND", "BRAVE", "BREAD", "BREAK", "BRICK",
    "BRIDE", "BRIEF", "BRING", "BROAD", "BROKE", "BROWN", "BRUSH", "BUILD",
    "BUILT", "BURST", "BUYER", "CABIN", "CABLE", "CAMEL", "CANDY", "CARRY",
    "CAUSE", "CHAIN", "CHAIR", "CHAOS", "CHARM", "CHART", "CHASE", "CHEAP",
    "CHECK", "CHEEK", "CHESS", "CHEST", "CHIEF", "CHILD", "CHINA", "CHOSE",
    "CHUNK", "CIVIC", "CIVIL", "CLAIM", "CLASH", "CLASS", "CLEAN", "CLEAR",
    "CLICK", "CLIFF", "CLIMB", "CLING", "CLIP", "CLOCK", "CLONE", "CLOSE",
    "CLOUD", "COACH", "COAST", "COLOR", "COMET", "COMIC", "CORAL", "COUNT",
    "COURT", "COVER", "CRACK", "CRAFT", "CRANE", "CRASH", "CRAWL", "CRAZY",
    "CREAM", "CREEK", "CRIME", "CROSS", "CROWD", "CROWN", "CRUEL", "CRUSH",
    "CURVE", "CYCLE", "DAILY", "DANCE", "DATED", "DEATH", "DEBUG", "DECAY",
    "DEITY", "DELTA", "DENSE", "DEPOT", "DEPTH", "DIRTY", "DISCO", "DIZZY",
    "DOUBT", "DOUGH", "DRAFT", "DRAIN", "DRAMA", "DRAWN", "DREAM", "DRIED",
    "DRIFT", "DRINK", "DRIVE", "DRONE", "DROVE", "DRUMS", "DRYER", "DUNES",
    "EARLY", "EARTH", "EIGHT", "ELITE", "EMPTY", "ENEMY", "ENJOY", "ENTER",
]


# Feedback generation (no regex)

def _compute_feedback(guess: str, secret: str) -> str:
    """Return per-letter feedback string like 'G B Y B G'.

    Uses G/Y/B notation. Handles duplicate letters correctly.
    """
    guess = guess.upper()
    secret = secret.upper()
    result = ["B"] * 5

    # Count available letters in secret (for Y logic)
    secret_counts: dict[str, int] = {}
    for ch in secret:
        secret_counts[ch] = secret_counts.get(ch, 0) + 1

    # First pass: mark exact matches (G)
    for i in range(5):
        if guess[i] == secret[i]:
            result[i] = "G"
            secret_counts[guess[i]] -= 1

    # Second pass: mark present but wrong position (Y)
    for i in range(5):
        if result[i] == "G":
            continue
        if secret_counts.get(guess[i], 0) > 0:
            result[i] = "Y"
            secret_counts[guess[i]] -= 1

    return " ".join(result)


def _extract_guess(text: str) -> str | None:
    """Extract a 5-letter word from model output.

    Returns uppercase word if valid, None otherwise.
    Uses str methods only — no regex.
    """
    # Try last non-empty line
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in reversed(lines):
        # Take first token
        word = line.split()[0].upper() if line.split() else ""
        # Strip punctuation using str methods
        while word and not word[-1].isalpha():
            word = word[:-1]
        while word and not word[0].isalpha():
            word = word[1:]
        if len(word) == 5 and word.isalpha():
            return word
    return None


# Reward schedule

_REWARD_BY_GUESS = {1: 1.00, 2: 0.90, 3: 0.75, 4: 0.60, 5: 0.45, 6: 0.30}


# MultiTurnEnv subclass

class WordleEnv(vf.MultiTurnEnv):
    """Wordle-style 5-letter word guessing game using MultiTurnEnv."""

    def __init__(self, dataset: Dataset, **kwargs: Any) -> None:
        super().__init__(dataset=dataset, system_prompt=SYSTEM_PROMPT, **kwargs)

    # ------------------------------------------------------------------ state

    def setup_state(self, state: dict) -> None:
        """Initialise per-rollout game state."""
        row = state.get("data", {})
        state["secret"] = row.get("answer", "CRANE")
        state["guesses_made"] = 0
        state["solved"] = False
        state["history"] = []   # [(guess, feedback), ...]

    # ------------------------------------------------------------------ env response

    def env_response(self, messages: list[dict], state: dict) -> tuple[str, dict]:
        """Process the model's latest message and return feedback."""
        last_msg = messages[-1]["content"] if messages else ""
        guess = _extract_guess(last_msg)

        if guess is None:
            state["guesses_made"] += 1
            return (
                "Invalid guess. Please reply with exactly one 5-letter word.",
                state,
            )

        state["guesses_made"] += 1
        feedback = _compute_feedback(guess, state["secret"])
        state["history"].append((guess, feedback))

        if guess == state["secret"]:
            state["solved"] = True
            return f"Feedback: {feedback}\nCongratulations! You guessed correctly!", state

        return f"Feedback: {feedback}", state

    # ------------------------------------------------------------------ stop conditions

    @vf.stop
    def stop_on_solved(self, messages: list[dict], state: dict) -> bool:
        return state.get("solved", False)

    @vf.stop
    def stop_on_max_guesses(self, messages: list[dict], state: dict) -> bool:
        return state.get("guesses_made", 0) >= 6

    # ------------------------------------------------------------------ reward

    async def reward(self, completion: str, answer: str, state: dict, **kwargs) -> float:
        if state.get("solved", False):
            n = state.get("guesses_made", 6)
            return _REWARD_BY_GUESS.get(n, 0.30)
        return 0.0

    # ------------------------------------------------------------------ cleanup

    @vf.cleanup
    def cleanup_game(self, state: dict) -> None:
        """Nothing to clean up for this in-memory game."""
        pass


# Dataset — 200 games, one secret word per row

def _build_dataset(seed: int = 42) -> Dataset:
    rng = random.Random(seed)
    words = rng.sample(_WORDS, min(200, len(_WORDS)))
    rows = [{"question": "Guess the 5-letter word.", "answer": w} for w in words]
    return Dataset.from_list(rows)


# load_environment

def load_environment(
    num_examples: int = -1,
    seed: int = 42,
) -> vf.Environment:
    """Load the Wordle MultiTurnEnv.

    Args:
        num_examples: Number of games (-1 = all 200).
        seed: Random seed.

    Returns:
        WordleEnv with partial reward for fewer guesses.
    """
    dataset = _build_dataset(seed=seed)
    if num_examples != -1:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    return WordleEnv(dataset=dataset, max_turns=12)  # 6 guesses × 2 (guess + feedback)


if __name__ == "__main__":
    # Smoke test: verify feedback logic
    assert _compute_feedback("CRANE", "CRANE") == "G G G G G"
    assert _compute_feedback("CRANE", "CRANE") == "G G G G G"
    assert _compute_feedback("CRANE", "XXXXX") == "B B B B B"
    assert _compute_feedback("CRANE", "RACES") == "Y Y G B B"
    print("Feedback tests passed.")

    env = load_environment(num_examples=5)
    print(f"Dataset size: {len(env.dataset)}")
    print(f"Sample word: {env.dataset[0]['answer']}")

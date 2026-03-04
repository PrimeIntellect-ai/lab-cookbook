"""Multi-Environment recipe — vf.EnvGroup combining three environments.

Combines math_rl + tool_use + word_game into a single training run using
vf.EnvGroup.  This allows a model to train on diverse tasks simultaneously,
improving generalization and preventing over-fitting to a single task type.

Why EnvGroup?
  - Single training run covering multiple capability dimensions
  - Natural curriculum: easy environments warm up the model; harder ones push it
  - Better generalization: model learns capabilities that transfer across tasks
  - Sampling is interleaved — each batch contains examples from all environments

Usage:
  env = load_environment()
  # Returns an EnvGroup that behaves like a single vf.Environment

Expected starting reward (weighted average):
  - math_rl:  ~0.70+ (easy)
  - tool_use: ~0.30
  - word_game: ~0.15
  → Combined starting reward: ~0.38

This recipe demonstrates:
  - vf.EnvGroup for multi-task RL
  - Environment composition with named sub-environments
  - How different reward scales interact during training
"""
from __future__ import annotations

import verifiers as vf

from lab_cookbook.recipes.math_rl import math_rl
from lab_cookbook.recipes.tool_use import tool_use
from lab_cookbook.recipes.word_game import word_game

SYSTEM_PROMPT_MULTI = """You are a versatile AI assistant. You may be asked to:
  - Solve math problems (answer in \\boxed{})
  - Use tools to answer questions (call the appropriate tool, then give your answer)
  - Play a word guessing game (guess 5-letter words using the feedback)

Read each question carefully and respond appropriately for the task type.
"""


def load_environment(
    math_examples: int = 200,
    tools_examples: int = 200,
    word_examples: int = 200,
    seed: int = 42,
) -> vf.Environment:
    """Load a combined multi-environment training setup.

    Args:
        math_examples: Number of math problems (default 200).
        tools_examples: Number of tool-use questions (default 200).
        word_examples: Number of word game instances (default 200).
        seed: Random seed for reproducibility.

    Returns:
        vf.EnvGroup combining math_rl, tool_use, and word_game.
    """
    math_env = math_rl.load_environment(num_examples=math_examples, seed=seed)
    tools_env = tool_use.load_environment(num_examples=tools_examples, seed=seed)
    word_env = word_game.load_environment(num_examples=word_examples, seed=seed)

    return vf.EnvGroup(
        envs=[math_env, tools_env, word_env],
        env_names=["math", "tools", "word"],
    )


if __name__ == "__main__":
    env = load_environment(math_examples=10, tools_examples=10, word_examples=10)
    print(f"EnvGroup created with {len(env.envs)} sub-environments:")
    for name, sub_env in zip(env.env_names, env.envs):
        print(f"  [{name}] {type(sub_env).__name__} — {len(sub_env.dataset)} examples")

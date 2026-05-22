import re
from collections.abc import Mapping

import verifiers.v1 as vf
from textarena_taskset import TextArenaTasksetConfig, load_taskset

GUESS_PATTERN = re.compile(r"<guess>(.*?)</guess>", re.DOTALL)


@vf.reward(weight=1.0)
async def correct_answer(task: vf.Task, state: vf.State) -> float:
    answer = task["answer"]
    assert isinstance(answer, str)
    completion = state.get("completion")
    assert isinstance(completion, list)
    for message in reversed(completion):
        assert isinstance(message, Mapping)
        if message.get("role") == "assistant":
            content = message.get("content")
            assert isinstance(content, str)
            matches = GUESS_PATTERN.findall(content)
            if matches:
                return 1.0 if matches[-1].strip() == f"[{answer}]" else 0.0
    return 0.0


@vf.reward(weight=1.0)
async def length_bonus(task: vf.Task, state: vf.State) -> float:
    answer = task["answer"]
    assert isinstance(answer, str)
    completion = state.get("completion")
    assert isinstance(completion, list)
    guess = ""
    num_guesses = 0
    for message in completion:
        assert isinstance(message, Mapping)
        if message.get("role") == "assistant":
            content = message.get("content")
            assert isinstance(content, str)
            if GUESS_PATTERN.search(content):
                num_guesses += 1
                matches = GUESS_PATTERN.findall(content)
                if matches:
                    guess = matches[-1].strip()
    is_correct = 1.0 if guess == f"[{answer}]" else 0.0
    assert num_guesses > 0 or is_correct == 0.0
    return is_correct / (num_guesses or 1)


@vf.reward(weight=1.0)
async def partial_answer(task: vf.Task, state: vf.State) -> float:
    answer = task["answer"]
    assert isinstance(answer, str)
    completion = state.get("completion")
    assert isinstance(completion, list)
    for message in reversed(completion):
        assert isinstance(message, Mapping)
        if message.get("role") == "assistant":
            content = message.get("content")
            assert isinstance(content, str)
            matches = GUESS_PATTERN.findall(content)
            if matches and matches[-1].strip() == f"[{answer}]":
                return 0.0
            break
    for message in reversed(completion):
        assert isinstance(message, Mapping)
        if message.get("role") != "user":
            continue
        content = message.get("content")
        assert isinstance(content, str)
        parts = content.strip().split("\n")
        if len(parts) == 3:
            scoring = parts[1].strip()
            return 0.2 * scoring.count("G") + 0.1 * scoring.count("Y")
    return 0.0


@vf.reward(weight=0.2)
async def format_reward(task: vf.Task, state: vf.State) -> float:
    _ = task
    completion = state.get("completion")
    assert isinstance(completion, list)
    found = False
    for message in completion:
        assert isinstance(message, Mapping)
        if message.get("role") == "assistant":
            found = True
            content = message.get("content")
            assert isinstance(content, str)
            if len(GUESS_PATTERN.findall(content)) != 1:
                return 0.0
    return 1.0 if found else 0.0


def load_environment(config: vf.EnvConfig) -> vf.Env:
    taskset_config = config.taskset
    assert isinstance(taskset_config, TextArenaTasksetConfig)
    taskset = load_taskset(taskset_config)
    for reward in (correct_answer, partial_answer, length_bonus, format_reward):
        taskset.add_reward(reward)
    return vf.Env(taskset=taskset)

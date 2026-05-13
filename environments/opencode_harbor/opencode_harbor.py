from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

TASKS_DIR = Path(__file__).parent / "tasks"

TERMINAL_BENCH_SAMPLE_TASKS = [
    "build-cython-ext",
    "chess-best-move",
    "configure-git-webserver",
    "fix-code-vulnerability",
    "log-summary-date-ranges",
    "polyglot-c-py",
    "qemu-alpine-ssh",
    "qemu-startup",
    "regex-log",
    "sqlite-with-gcov",
]


class OpenCodeHarnessConfig(vf.HarnessConfig):
    disabled_tools: list[str] = Field(default_factory=lambda: ["webfetch", "question"])
    max_turns: int = 4


def load_taskset(config: vf.TasksetConfig) -> vf.HarborTaskset:
    taskset_config = vf.HarborTasksetConfig.from_config(config)
    return vf.HarborTaskset(
        tasks=taskset_config.tasks or str(TASKS_DIR),
        config=taskset_config,
    )


def load_harness(config: vf.HarnessConfig) -> vf.OpenCode:
    harness_config = OpenCodeHarnessConfig.from_config(config)
    return vf.OpenCode(
        disabled_tools=harness_config.disabled_tools,
        max_turns=harness_config.max_turns,
        config=harness_config,
    )


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )

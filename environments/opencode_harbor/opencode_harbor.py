from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

TASKS_DIR = Path(__file__).parent / "tasks"

# OpenCode harness default; surfaced here so we only forward max_turns
# to OpenCode when the user explicitly overrides it via HarnessConfig.
OPENCODE_DEFAULT_MAX_TURNS = 4


class OpenCodeHarnessConfig(vf.HarnessConfig):
    disabled_tools: list[str] = Field(default_factory=lambda: ["webfetch", "question"])


def load_taskset(config: vf.TasksetConfig) -> vf.HarborTaskset:
    taskset_config = vf.HarborTasksetConfig.from_config(config)
    return vf.HarborTaskset(
        tasks=taskset_config.tasks or str(TASKS_DIR),
        config=taskset_config,
    )


def load_harness(config: vf.HarnessConfig) -> vf.OpenCode:
    harness_config = OpenCodeHarnessConfig.from_config(config)
    max_turns = (
        harness_config.max_turns
        if "max_turns" in harness_config.model_fields_set
        else OPENCODE_DEFAULT_MAX_TURNS
    )
    return vf.OpenCode(
        disabled_tools=harness_config.disabled_tools,
        max_turns=max_turns,
        config=harness_config,
    )


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )

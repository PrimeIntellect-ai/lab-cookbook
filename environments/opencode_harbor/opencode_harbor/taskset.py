"""Load the Harbor task directories bundled with this environment."""

from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborTask,
    parse_task,
)


def bundled_tasks_dir() -> Path:
    """Resolve tasks from an installed wheel or this source checkout."""
    package_dir = Path(__file__).resolve().parent
    installed = package_dir / "tasks"
    source = package_dir.parent / "tasks"
    if installed.is_dir():
        return installed
    if source.is_dir():
        return source
    raise FileNotFoundError("opencode-harbor bundled task directory is missing")


class OpenCodeHarborConfig(HarborConfig):
    dataset: str = "bundled"
    require_image: bool = True


class OpenCodeHarborTaskset(vf.Taskset[HarborTask, OpenCodeHarborConfig]):
    def load(self) -> Iterator[HarborTask]:
        root = bundled_tasks_dir()
        task_dirs = [
            task_toml.parent
            for task_toml in sorted(root.rglob("task.toml"))
            if (task_toml.parent / "instruction.md").is_file()
            and (
                self.config.tasks is None or task_toml.parent.name in self.config.tasks
            )
        ]
        if not task_dirs:
            selected = (
                "all bundled tasks"
                if self.config.tasks is None
                else f"requested tasks {self.config.tasks!r}"
            )
            raise ValueError(f"no Harbor task directories found for {selected}")

        for idx, task_dir in enumerate(task_dirs):
            data = parse_task(task_dir, idx, self.config)
            yield HarborTask(data, self.config.task)


__all__ = ["OpenCodeHarborConfig", "OpenCodeHarborTaskset"]

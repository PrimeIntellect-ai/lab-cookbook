from __future__ import annotations

import importlib
import inspect
import re
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]

ENV_MODULES = {
    "reverse_text_v1": ROOT / "environments/reverse_text_v1",
    "aime26_v1": ROOT / "environments/aime26_v1",
    "code_golf_v1": ROOT / "environments/code_golf_v1",
    "gsm8k_v1": ROOT / "environments/gsm8k_v1",
    "wordle_v1": ROOT / "environments/wordle_v1",
    "simple_judge": ROOT / "environments/simple_judge",
    "wiki_search_v1": ROOT / "environments/wiki_search_v1",
    "shape_detective": ROOT / "environments/shape_detective",
    "math_python": ROOT / "environments/math_python",
    "opencode_harbor": ROOT / "environments/opencode_harbor",
    "tau2_bench_v1": ROOT / "environments/tau2_bench_v1",
    "r2e_gym_v1": ROOT / "environments/r2e_gym_v1",
    "swelego_v1": ROOT / "environments/swelego_v1",
    "swebench_verified_v1": ROOT / "environments/swebench_verified_v1",
}

LEGACY_DOCS = {
    ROOT / "reference/v1-authoring-gaps.md",
    ROOT / "environments/AGENTS.md",
}

TUTORIAL_CONFIGS = [
    ROOT / "configs/02/gsm8k-eval.toml",
    ROOT / "configs/05/aime26-eval.toml",
    ROOT / "configs/04/wordle-gepa.toml",
    ROOT / "configs/06/simple-judge-eval.toml",
    ROOT / "configs/07/code-golf-eval.toml",
    ROOT / "configs/10/wiki-search-eval.toml",
    ROOT / "configs/09/shape-detective-eval.toml",
    ROOT / "configs/11/math-python-eval.toml",
    ROOT / "configs/11/opencode-harbor.toml",
    ROOT / "configs/11/harbor-smoke.toml",
    ROOT / "configs/recipes/mini-loop-smoke.toml",
    ROOT / "configs/recipes/support-agent-eval.toml",
    ROOT / "configs/recipes/swe-baseline-eval.toml",
]


def _add_env_paths() -> None:
    for path in ENV_MODULES.values():
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


@pytest.fixture(autouse=True)
def env_import_paths() -> None:
    _add_env_paths()


def test_environment_modules_export_one_taskset() -> None:
    import verifiers.v1 as vf

    for module_name in ENV_MODULES:
        module = importlib.import_module(module_name)
        exported_names = getattr(module, "__all__", None)
        assert exported_names, f"{module_name} must define __all__"
        exported = [getattr(module, name) for name in exported_names]
        tasksets = [
            obj
            for obj in exported
            if inspect.isclass(obj) and issubclass(obj, vf.Taskset) and obj is not vf.Taskset
        ]
        assert len(tasksets) == 1, f"{module_name} must export exactly one Taskset"
        harnesses = [
            obj
            for obj in exported
            if inspect.isclass(obj) and issubclass(obj, vf.Harness) and obj is not vf.Harness
        ]
        assert len(harnesses) <= 1, f"{module_name} must export at most one Harness"


def test_migrated_tutorial_environment_contracts() -> None:
    import verifiers.v1 as vf
    from math_python import (
        MathPythonTask,
        MathPythonTaskConfig,
        PythonToolConfig,
        PythonToolset,
    )
    from opencode_harbor import OpenCodeHarness
    from opencode_harbor.taskset import (
        OpenCodeHarborConfig,
        OpenCodeHarborTaskset,
    )
    from shape_detective.taskset import (
        ShapeDetectiveConfig,
        ShapeDetectiveTaskset,
    )
    from simple_judge.taskset import SimpleJudgeConfig, SimpleJudgeTaskset

    judge_task = SimpleJudgeTaskset(SimpleJudgeConfig()).select(1, False)[0]
    assert judge_task.data.criterion
    assert judge_task.config.judge.model == "openai/gpt-4.1-mini"

    shape_task = ShapeDetectiveTaskset(ShapeDetectiveConfig()).select(1, False)[0]
    assert isinstance(shape_task.data.prompt, list)
    assert shape_task.user is not None

    assert MathPythonTask.tools == (PythonToolset,)
    assert isinstance(MathPythonTaskConfig().tools, PythonToolConfig)
    assert isinstance(MathPythonTaskConfig().tools.runtime, vf.DockerConfig)

    harbor_task = OpenCodeHarborTaskset(OpenCodeHarborConfig(tasks=["hello-world"])).select(
        1, False
    )[0]
    assert harbor_task.data.name == "hello-world"
    assert harbor_task.data.image == "python:3.11-slim"

    context_annotation = inspect.signature(OpenCodeHarness.launch).parameters["ctx"].annotation
    assert context_annotation is vf.ModelContext


def test_eval_configs_validate_against_v1_schema() -> None:
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.gepa.config import GEPAConfig

    for path in TUTORIAL_CONFIGS:
        data = tomllib.loads(path.read_text())
        config_cls = GEPAConfig if "reflection_model" in data else EvalConfig
        config_cls.model_validate(data)


def test_all_toml_files_parse() -> None:
    for base in (ROOT / "configs", ROOT / "environments"):
        for path in base.rglob("*.toml"):
            tomllib.loads(path.read_text())


def test_no_stale_v1_authoring_patterns() -> None:
    stale_patterns = [
        r"\bvf\.EnvConfig\b",
        r"\bvf\.SystemPrompt\b",
        r"\bvf\.TaskSplit\b",
        r"\bvf\.Tasks\b",
        r"\bvf\.Toolsets\b",
        r"\bdef load_taskset\b",
        r"\bdef load_harness\b",
        r"\bdef load_toolsets\b",
        r"state\[\"completion\"\]",
        r"state\.get\(\"completion",
        r"state\[\"trajectory\"\]",
        r"\bget_max_turns\b",
        r"\bvf\.get_messages\b",
        r"\[\[eval\]\]",
        r"\[eval\.",
    ]

    checked_suffixes = {".md", ".toml", ".py"}
    roots = [ROOT / "reference", ROOT / "skills", ROOT / "environments"]
    for base in roots:
        for path in base.rglob("*"):
            if path.suffix not in checked_suffixes:
                continue
            if path in LEGACY_DOCS:
                continue
            if "tasks" in path.parts and path.suffix == ".md":
                continue
            text = path.read_text(errors="ignore")
            for pattern in stale_patterns:
                assert not re.search(pattern, text), (
                    f"{path.relative_to(ROOT)} contains {pattern!r}"
                )

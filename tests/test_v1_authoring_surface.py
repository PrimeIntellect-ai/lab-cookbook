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
    "reverse_text": ROOT / "environments/reverse_text",
    "gsm8k": ROOT / "environments/gsm8k",
    "wordle": ROOT / "environments/wordle",
    "simple_judge": ROOT / "environments/simple_judge",
    "wiki_search": ROOT / "environments/wiki_search",
    "shape_detective": ROOT / "environments/shape_detective",
    "math_python": ROOT / "environments/math_python",
    "calendar_scheduling": ROOT / "environments/calendar_scheduling",
    "ethics_debate": ROOT / "environments/ethics_debate",
    "swe_grep": ROOT / "environments/swe_grep",
    "opencode_harbor": ROOT / "environments/opencode_harbor",
    "langchain_deep_agents_wikispeedia": ROOT / "environments/langchain_deep_agents_wikispeedia",
    "basic_patent_q_and_a": ROOT / "environments/patent_search/basic_patent_q_and_a",
    "advanced_patent_q_and_a": ROOT / "environments/patent_search/advanced_patent_q_and_a",
    "patent_technical_analysis": ROOT / "environments/patent_search/patent_technical_analysis",
}

HARNESS_MODULES = {
    "langchain_deep_agents": ROOT / "environments/langchain_deep_agents",
}

LEGACY_DOCS = {
    ROOT / "guides/14-legacy-environments/README.md",
    ROOT / "reference/v1-authoring-gaps.md",
}

LEGACY_CONFIG_DIRS = {
    ROOT / "configs/gepa",
}


def _add_env_paths() -> None:
    for path in list(ENV_MODULES.values()) + list(HARNESS_MODULES.values()):
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


def test_harness_modules_export_one_harness() -> None:
    import verifiers.v1 as vf

    for module_name in HARNESS_MODULES:
        module = importlib.import_module(module_name)
        exported_names = getattr(module, "__all__", None)
        assert exported_names, f"{module_name} must define __all__"
        exported = [getattr(module, name) for name in exported_names]
        harnesses = [
            obj
            for obj in exported
            if inspect.isclass(obj) and issubclass(obj, vf.Harness) and obj is not vf.Harness
        ]
        tasksets = [
            obj
            for obj in exported
            if inspect.isclass(obj) and issubclass(obj, vf.Taskset) and obj is not vf.Taskset
        ]
        assert len(harnesses) == 1, f"{module_name} must export exactly one Harness"
        assert not tasksets, f"{module_name} must not export a Taskset"


def test_eval_configs_validate_against_v1_schema() -> None:
    from verifiers.v1.configs.eval import EvalConfig

    eval_configs = []
    for path in (ROOT / "configs").rglob("*.toml"):
        if any(parent in path.parents for parent in LEGACY_CONFIG_DIRS):
            continue
        data = tomllib.loads(path.read_text())
        if "taskset" in data:
            eval_configs.append(path)
            EvalConfig.model_validate(data)
    assert eval_configs, "expected at least one v1 eval config"


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
    roots = [ROOT / "guides", ROOT / "reference", ROOT / "skills", ROOT / "environments"]
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

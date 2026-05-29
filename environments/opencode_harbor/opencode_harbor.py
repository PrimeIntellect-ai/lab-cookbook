import verifiers as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


class OpenCodeHarborTasksetConfig(HarborTasksetConfig):
    bundle_package: str | None = __name__


def load_taskset(config: OpenCodeHarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

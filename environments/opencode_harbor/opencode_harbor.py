import harnesses as h
import tasksets as t
import verifiers as vf


def load_taskset(config: t.HarborTasksetConfig) -> t.HarborTaskset:
    return t.HarborTaskset(config=config)


def load_harness(config: h.OpenCodeConfig) -> h.OpenCode:
    return h.OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

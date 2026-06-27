# v1 Authoring Gaps

The cookbook envs now translate cleanly to the v1 taskset, server-side state, and harness model. The remaining ecosystem gap is native GEPA support for v1 tasksets.

## Native GEPA for v1 Tasksets

The GEPA CLI still loads environments through the legacy v0 `load_environment` path. The cookbook keeps `configs/gepa/` as legacy examples and uses v1 eval configs for prompt measurement.

Proposed fix: add a v1 GEPA adapter that accepts an `EnvConfig`, builds `vf.Environment`, runs traces through the v1 eval runner, and mutates prompt-bearing taskset config fields or packaged prompt files.

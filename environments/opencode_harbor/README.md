# opencode-harbor

A v1 taskset of bundled Harbor tasks, plus an optional OpenCode harness.

The taskset loads its packaged task directories through the built-in Harbor parser and scores through each task's Harbor tests. Every bundled task declares a prebuilt image, which becomes the rollout runtime image. Any coding-agent harness can drive it; the cookbook uses the built-in `pi` harness. The packaged `opencode-harbor` harness installs OpenCode in the runtime instead (x86_64 only — the pinned OpenCode release has no linux-arm64 asset).

Run:

```bash
uv run eval @ configs/11/harbor-smoke.toml
```

Example config:

```toml
[taskset]
id = "opencode-harbor"
tasks = ["hello-world"]

[harness]
id = "pi"
runtime = { type = "docker" }
```

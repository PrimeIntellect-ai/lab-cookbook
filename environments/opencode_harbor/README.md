# opencode-harbor

A v1 Harbor taskset with a bundled OpenCode harness.

The taskset subclasses the built-in `verifiers.v1.tasksets.harbor.HarborTaskset`, sets the default dataset to `hello-world`, and scores through Harbor tests. The harness installs OpenCode in the selected runtime and routes model calls through the v1 interception endpoint.

Run:

```bash
uv run eval @ configs/10/opencode-harbor.toml
```

Example config:

```toml
[taskset]
id = "opencode-harbor"
tasks = ["regex-log"]
ignore_dockerfile = true

[harness]
id = "opencode-harbor"
runtime = { type = "docker" }
```

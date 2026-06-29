# Custom Harnesses

Write a custom harness when the rollout program is not the default chat loop: a CLI agent, browser agent, code agent, or framework that needs its own process.

A harness has a typed config and implements `launch`:

```python
import verifiers.v1 as vf


class MyHarnessConfig(vf.HarnessConfig):
    command: str = "my-agent"


class MyHarness(vf.Harness[MyHarnessConfig]):
    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        env = {
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "OPENAI_MODEL": ctx.model,
        }
        return await runtime.run([self.config.command], env)
```

The task is available as `trace.task`. The model endpoint is the interception server at `endpoint` with bearer token `secret`; calls through that endpoint are recorded on the trace. Tool servers are provided in `mcp_urls` by name.

Export the harness from the same package if the taskset should bundle it by default, or publish it as a separate harness package:

```python
from my_env.harness import MyHarness
from my_env.taskset import MyTaskset

__all__ = ["MyTaskset", "MyHarness"]
```

Use a harness config in eval TOML:

```toml
[taskset]
id = "my-taskset"

[harness]
id = "my-harness"
runtime = { type = "docker" }
```

Keep scoring in the taskset unless the metric truly belongs to how the harness ran.

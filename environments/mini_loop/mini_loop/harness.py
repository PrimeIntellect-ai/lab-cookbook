"""A from-scratch CLI-agent harness: the worked example of the
Build Your Own Coding-Agent Harness recipe."""

from pathlib import Path

import verifiers.v1 as vf

PROGRAM_SOURCE = (Path(__file__).parent / "program.py").read_text()


class MiniLoopHarnessConfig(vf.HarnessConfig):
    max_steps: int = 20
    """Model turns before the agent gives up."""
    command_timeout_seconds: float = 120.0
    """Wall-clock budget for each bash command the agent runs."""


class MiniLoopHarness(vf.Harness[MiniLoopHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        if mcp_urls:
            raise ValueError("MiniLoopHarness does not expose taskset MCP servers")
        _, prompt = self.resolve_prompt(trace.task.data)
        if not isinstance(prompt, str):
            raise ValueError("MiniLoopHarness requires a string task prompt")
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        args = [
            "--model",
            ctx.model,
            "--task",
            prompt,
            "--max-steps",
            str(self.config.max_steps),
            "--command-timeout",
            str(self.config.command_timeout_seconds),
        ]
        env = {
            **self.config.resolved_env,
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
        }
        return await runtime.run_program([*program, *args], env)

import json

import verifiers.v1 as vf

DEEP_AGENTS_PROGRAM = r"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["deepagents>=0.6.8", "langchain-openai>=1.2.1", "mcp"]
# ///
import argparse
import asyncio
import json
from contextlib import AsyncExitStack

from deepagents import create_deep_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--recursion-limit", type=int, default=80)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def mcp_content_to_text(blocks) -> str:
    parts: list[str] = []
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(block.text)
        elif kind == "image":
            parts.append(f"[image:{block.mimeType};base64,{block.data}]")
        else:
            parts.append(str(block))
    return "\n".join(parts) if parts else str(blocks)


def make_langchain_tool(session, raw_name: str, full_name: str, description: str, schema: dict):
    async def call_tool(**kwargs):
        result = await session.call_tool(raw_name, kwargs)
        return mcp_content_to_text(result.content)

    call_tool.__name__ = full_name
    return StructuredTool.from_function(
        coroutine=call_tool,
        name=full_name,
        description=description or full_name,
        args_schema=schema or {"type": "object", "properties": {}},
    )


async def connect_mcp(stack: AsyncExitStack, config: dict):
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    tools = []
    for server_name, spec in config.get("mcpServers", {}).items():
        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=spec.get("headers") or None)
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            full_name = f"{server_name}_{tool.name}"
            tools.append(
                make_langchain_tool(
                    session,
                    tool.name,
                    full_name,
                    tool.description or "",
                    tool.inputSchema,
                )
            )
    return tools


def final_text(result) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


async def main() -> None:
    args = parse_args()
    mcp_config = json.loads(args.mcp_config or "{}")
    model = ChatOpenAI(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=0,
        use_responses_api=False,
    )
    async with AsyncExitStack() as stack:
        tools = await connect_mcp(stack, mcp_config) if mcp_config.get("mcpServers") else []
        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=args.system_prompt or None,
            debug=args.debug,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": args.prompt}]},
            config={"recursion_limit": args.recursion_limit},
        )
    text = final_text(result)
    if text:
        print(text)


if __name__ == "__main__":
    asyncio.run(main())
"""


class LangChainDeepAgentsHarnessConfig(vf.HarnessConfig):
    recursion_limit: int = 80
    request_timeout_seconds: float = 600.0
    debug: bool = False


class LangChainDeepAgentsHarness(vf.Harness[LangChainDeepAgentsHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(DEEP_AGENTS_PROGRAM, self.config.env)

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task)
        if not isinstance(prompt, str):
            raise ValueError("LangChainDeepAgentsHarness requires a string task prompt")
        mcp_config = {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
        program = await runtime.prepare_uv_script(DEEP_AGENTS_PROGRAM, self.config.env)
        args = [
            "--base-url",
            endpoint,
            "--api-key",
            secret,
            "--model",
            ctx.model,
            "--prompt",
            prompt,
            "--mcp-config",
            json.dumps(mcp_config),
            "--recursion-limit",
            str(self.config.recursion_limit),
            "--request-timeout",
            str(self.config.request_timeout_seconds),
        ]
        if system_prompt:
            args.extend(["--system-prompt", system_prompt])
        if self.config.debug:
            args.append("--debug")
        return await runtime.run_program([*program, *args], self.config.env)


__all__ = ["LangChainDeepAgentsHarness"]

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol, cast

import verifiers as vf
from datasets import Dataset

if __package__:
    from .wiki_graph import WikiGraph, WikiPair, load_wiki_graph
else:
    from wiki_graph import WikiGraph, WikiPair, load_wiki_graph


class AgentMessage(Protocol):
    role: str
    content: object


def system_prompt(allow_go_back: bool = True) -> str:
    backtracking = (
        "Use `go_back` to undo your last click."
        if allow_go_back
        else "Backtracking is disabled, so choose each link carefully."
    )
    return f"""\
This game is easy and fun:

You are given two Wikipedia articles. Starting from the first article, your goal \
is to reach the second one, exclusively by following links in the articles you \
encounter. (For the articles you are given this is always possible.)

Each article ends with a list of `Available links: ...` — those are the only \
links you can follow. Use the `click_link` tool to navigate to one. \
{backtracking}

You also have access to deep-agent scaffolding tools (`write_todos`, \
`write_file`, `read_file`, `ls`, `edit_file`, `task`). Use them when they help: \
sketch a plan with `write_todos`, jot promising bridge concepts or dead-ends \
in a file, and call `task` to spawn a focused sub-agent for a sub-search. They \
are entirely optional.

Try to be quick — think about which broader concepts connect the source to \
the target, and aim for the article that most likely lists your destination \
among its links.

When you reach the target the system will say `TARGET REACHED`. Stop calling \
tools at that point and reply with a brief confirmation."""


SYSTEM_PROMPT = system_prompt()


class WikispeediaTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "langchain-deep-agents-wikispeedia"
    cache_dir: str | None = None
    min_path_length: int = 3
    max_path_length: int = 6
    train_size: int = 50_000
    eval_size: int = 1_000
    eval_target_fraction: float = 0.1
    split_seed: int = 0
    links_only: bool = False
    allow_go_back: bool = True
    efficiency_weight: float = 0.0
    stratify_path_length: bool = True


class WikispeediaHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(
        fn="run_langchain_deep_agents_wikispeedia_program"
    )
    max_turns: int = 50
    timeout_seconds: float = 1200.0


class WikispeediaTaskset(vf.Taskset[WikispeediaTasksetConfig]):
    def load_system_prompt(
        self, config: WikispeediaTasksetConfig
    ) -> vf.SystemPrompt | vf.SystemPromptConfig | None:
        if config.system_prompt is not None:
            return config.system_prompt
        return system_prompt(allow_go_back=config.allow_go_back)

    def wiki(self) -> WikiGraph:
        wiki_graph = getattr(self, "_wiki_graph", None)
        if wiki_graph is None:
            wiki_graph = load_wiki_graph(self.config.cache_dir)
            self._wiki_graph = wiki_graph
        return wiki_graph

    def format_article(self, article: str, links_only: bool = False) -> str:
        wiki = self.wiki()
        links = wiki.get_links(article)
        links_str = ", ".join(links) if links else "(no outgoing links)"
        if links_only:
            return f"# {article}\n\nAvailable links: {links_str}"
        text = wiki.get_text(article)
        return f"# {article}\n\n{text}\n\n---\nAvailable links: {links_str}"

    def split_pairs(self) -> tuple[list[WikiPair], list[WikiPair]]:
        return self.wiki().split_pairs(
            train_size=self.config.train_size,
            eval_size=self.config.eval_size,
            min_dist=self.config.min_path_length,
            max_dist=self.config.max_path_length,
            eval_target_fraction=self.config.eval_target_fraction,
            seed=self.config.split_seed,
            stratify=self.config.stratify_path_length,
        )

    def build_dataset(self, pairs: list[WikiPair]) -> Dataset:
        records = []
        for source, target, dist in pairs:
            starting = self.format_article(source, links_only=self.config.links_only)
            prompt_text = (
                f"Your mission: {source} >> {target}\n\n"
                f"Here is the starting article:\n\n{starting}"
            )
            info: vf.JsonData = {
                "source": source,
                "target": target,
                "shortest_path": dist,
            }
            human = self.wiki().get_human_stats(source, target)
            if human is not None:
                info.update(human)
            records.append(
                {
                    "task_id": f"{source}->{target}",
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "answer": target,
                    "info": info,
                    "links_only": self.config.links_only,
                }
            )
        return Dataset.from_list(records)

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        train, eval_ = self.split_pairs()
        return self.build_dataset(train if split == "train" else eval_)

    def load_toolsets(self, config: WikispeediaTasksetConfig) -> vf.Toolsets:
        tools: list[vf.Handler] = [self.click_link]
        if config.allow_go_back:
            tools.append(self.go_back)
        return {"wikispeedia": vf.Toolset(tools=tools)}

    async def click_link(self, article: str, state: vf.State) -> str:
        """Navigate to a linked Wikipedia article."""
        links_only = bool(state.get("links_only", False))
        current = str(state["current_article"])
        available = self.wiki().get_links(current)
        normalized = self.wiki().normalize_name(article)
        if normalized is None or normalized not in available:
            avail_str = ", ".join(available) if available else "(none)"
            return (
                f"'{article}' is not a valid link from '{current}'.\n"
                f"Available links: {avail_str}"
            )
        state["current_article"] = normalized
        path = state["path"]
        assert isinstance(path, list)
        path.append(normalized)
        info = state["info"]
        assert isinstance(info, dict)
        if normalized == info["target"]:
            state["reached_target"] = True
            state.stop("target_reached")
            return (
                f"TARGET REACHED: {normalized}\n\n"
                "You successfully navigated to the target. Stop calling tools "
                "and reply briefly to confirm."
            )
        return self.format_article(normalized, links_only=links_only)

    async def go_back(self, state: vf.State) -> str:
        """Undo the last click_link and return to the previous article."""
        path = state["path"]
        assert isinstance(path, list)
        if len(path) <= 1:
            return "You are already at the starting article. Cannot go back."
        path.pop()
        state["current_article"] = path[-1]
        return self.format_article(
            str(path[-1]), links_only=bool(state.get("links_only", False))
        )

    @vf.reward(weight=1.0)
    async def reached_target(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return 1.0 if state.get("reached_target", False) else 0.0

    @vf.reward(weight=0.0)
    async def path_efficiency(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        if not state.get("reached_target", False):
            return 0.0
        info = state["info"]
        assert isinstance(info, dict)
        shortest = float(info["shortest_path"])
        actual = max(len(state.get("path", [])) - 1, 1)
        return min(1.0, shortest / actual)

    @vf.reward(weight=1.0)
    async def path_efficiency_reward(self, task: vf.Task, state: vf.State) -> float:
        return self.config.efficiency_weight * await self.path_efficiency(task, state)

    @vf.reward(weight=0.0)
    async def path_length(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(max(len(state.get("path", [])) - 1, 0))

    @vf.reward(weight=0.0)
    async def shortest_path(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        info = state.get("info", {})
        return float(info.get("shortest_path", 0) if isinstance(info, dict) else 0)

    @vf.reward(weight=0.0)
    async def agent_timeout(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return 1.0 if state.get("agent_timeout", False) else 0.0

    def iter_tool_calls(self, state: vf.State) -> Iterator[str]:
        completion = state.get("completion") or []
        messages = (
            vf.get_messages(completion, role="assistant")
            if isinstance(completion, list)
            else []
        )
        for msg in messages:
            tool_calls = msg.tool_calls
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                yield tool_call.name

    def count_tool_calls(self, state: vf.State, name: str | None = None) -> int:
        if name is None:
            return sum(1 for _ in self.iter_tool_calls(state))
        return sum(1 for tool_name in self.iter_tool_calls(state) if tool_name == name)

    @vf.reward(weight=0.0)
    async def total_tool_calls(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state))

    @vf.reward(weight=0.0)
    async def assistant_turns(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        completion = state.get("completion") or []
        return float(
            len(vf.get_messages(completion, role="assistant"))
            if isinstance(completion, list)
            else 0
        )

    @vf.reward(weight=0.0)
    async def invalid_link_rate(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        clicks = 0
        invalid = 0
        completion = state.get("completion") or []
        if not isinstance(completion, list):
            return 0.0

        transcript = vf.get_messages(completion)
        id_to_name: dict[str, str] = {}
        for msg in transcript:
            if isinstance(msg, vf.AssistantMessage):
                tool_calls = msg.tool_calls
                if tool_calls:
                    for tc in tool_calls:
                        id_to_name[tc.id] = tc.name

        for msg in transcript:
            if not isinstance(msg, vf.ToolMessage):
                continue
            tool_name = id_to_name.get(msg.tool_call_id)
            if tool_name is None:
                extra = msg.get("name")
                tool_name = extra if isinstance(extra, str) else None
            if tool_name != "click_link":
                continue
            clicks += 1
            content = msg.content
            if isinstance(content, str) and "is not a valid link" in content:
                invalid += 1
        return float(invalid / clicks) if clicks else 0.0

    @vf.reward(weight=0.0)
    async def calls_click_link(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "click_link"))

    @vf.reward(weight=0.0)
    async def calls_go_back(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "go_back"))

    @vf.reward(weight=0.0)
    async def calls_write_todos(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "write_todos"))

    @vf.reward(weight=0.0)
    async def calls_write_file(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "write_file"))

    @vf.reward(weight=0.0)
    async def calls_read_file(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "read_file"))

    @vf.reward(weight=0.0)
    async def calls_ls(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "ls"))

    @vf.reward(weight=0.0)
    async def calls_edit_file(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "edit_file"))

    @vf.reward(weight=0.0)
    async def calls_grep(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "grep"))

    @vf.reward(weight=0.0)
    async def calls_task(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(self.count_tool_calls(state, "task"))


class WikispeediaHarness(vf.Harness[WikispeediaHarnessConfig]):
    @vf.update(priority=-200)
    async def restore_agent_completion(self, task: vf.Task, state: vf.State) -> None:
        _ = task
        agent_completion = state.get("agent_completion")
        if isinstance(agent_completion, list):
            state["completion"] = agent_completion


def serialize_agent_completion(
    messages: Sequence[AgentMessage | vf.JsonData],
) -> list[vf.JsonData]:
    role_aliases = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }
    call_names: dict[str, str] = {}
    serialized: list[vf.JsonData] = []
    for message in messages:
        if isinstance(message, Mapping):
            payload = dict(message)
        else:
            model_dump = getattr(message, "model_dump", None)
            payload = (
                model_dump(mode="json", exclude_none=True)
                if callable(model_dump)
                else {
                    "role": getattr(message, "role", None)
                    or getattr(message, "type", "assistant"),
                    "content": getattr(message, "content", str(message)),
                    "name": getattr(message, "name", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "tool_calls": getattr(message, "tool_calls", None),
                }
            )
        raw_role = payload.get("role") or payload.get("type") or "assistant"
        role = role_aliases.get(str(raw_role), str(raw_role))
        item: vf.JsonData = {
            "role": role,
            "content": payload.get("content", ""),
        }
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                tool_call_payload = dict(tool_call)
                name = tool_call_payload.get("name")
                tool_id = tool_call_payload.get("id") or tool_call_payload.get(
                    "tool_call_id"
                )
                if isinstance(tool_id, str) and isinstance(name, str):
                    call_names[tool_id] = name
                arguments = tool_call_payload.get("arguments")
                if not isinstance(arguments, str):
                    args = tool_call_payload.get("args", {})
                    try:
                        arguments = json.dumps(args if args is not None else {})
                    except (TypeError, ValueError):
                        arguments = str(args)
                    tool_call_payload["arguments"] = arguments
                normalized_tool_calls.append(tool_call_payload)
            item["tool_calls"] = normalized_tool_calls
        name = payload.get("name")
        if isinstance(name, str):
            item["name"] = name
        tool_call_id = payload.get("tool_call_id")
        if isinstance(tool_call_id, str):
            item["tool_call_id"] = tool_call_id
            if item["role"] == "tool" and "name" not in item:
                name = call_names.get(tool_call_id)
                if name is not None:
                    item["name"] = name
        serialized.append(item)
    if serialized and serialized[0].get("role") == "user":
        return serialized[1:]
    return serialized


def langchain_navigation_tools(runtime_tools):
    from langchain_core.tools import tool

    nav_tools = []
    if "click_link" in runtime_tools:
        click_link_tool = runtime_tools["click_link"]

        @tool
        async def click_link(article: str) -> str:
            """Navigate to a linked Wikipedia article."""
            return str(await click_link_tool(article=article))

        nav_tools.append(click_link)
    if "go_back" in runtime_tools:
        go_back_tool = runtime_tools["go_back"]

        @tool
        async def go_back() -> str:
            """Undo the last click_link and return to the previous article."""
            return str(await go_back_tool())

        nav_tools.append(go_back)
    return nav_tools


def make_langchain_deep_agents_program(
    max_turns: int,
    timeout_seconds: float,
) -> Callable[[vf.Task, vf.State], Awaitable[vf.State]]:
    async def run_langchain_deep_agents_wikispeedia_program(
        task: vf.Task, state: vf.State
    ) -> vf.State:
        from deepagents import create_deep_agent
        from langchain_core.runnables import RunnableConfig
        from langchain_openai import ChatOpenAI
        from langgraph.errors import GraphRecursionError
        from openai import OpenAI

        state["current_article"] = state["info"]["source"]
        state["path"] = [state["info"]["source"]]
        state["reached_target"] = False
        state["agent_timeout"] = False
        state["links_only"] = bool(task.get("links_only", False))

        endpoint_config = state.get_endpoint_config(api="chat")
        endpoint_client = cast(OpenAI, state.get_client(api="chat", sync=True))
        endpoint_api_key = endpoint_client.api_key
        endpoint_client.close()
        model = cast(Any, ChatOpenAI)(
            model=endpoint_config.model,
            base_url=endpoint_config.base_url,
            api_key=endpoint_api_key,
        )
        runtime_tools = state.get_tools()
        nav_tools = langchain_navigation_tools(runtime_tools)
        state_system_prompt = ""
        system_prompt_messages = state.get("system_prompt")
        if isinstance(system_prompt_messages, list):
            state_system_prompt = "\n\n".join(
                str(message.content or "")
                for message in vf.get_messages(system_prompt_messages)
            )
        agent = create_deep_agent(
            model=model,
            tools=nav_tools,
            system_prompt=state_system_prompt or SYSTEM_PROMPT,
        )
        prompt = str(cast(list[vf.JsonData], state["prompt"])[-1]["content"])
        recursion_limit = state.get_max_turns(max_turns)
        invoke_config: RunnableConfig | None = (
            {"recursion_limit": recursion_limit} if recursion_limit > 0 else None
        )
        invoke = agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=invoke_config,
        )
        try:
            result = await asyncio.wait_for(invoke, timeout=timeout_seconds)
        except (TimeoutError, GraphRecursionError) as exc:
            state["agent_timeout"] = True
            state.stop(
                "agent_timeout"
                if isinstance(exc, TimeoutError)
                else "agent_recursion_limit"
            )
            state.setdefault("agent_completion", [])
            return state

        messages = result.get("messages", []) if isinstance(result, Mapping) else []
        completion = serialize_agent_completion(messages)
        state["agent_completion"] = completion
        state["completion"] = completion
        if completion:
            state["agent_result"] = str(completion[-1].get("content") or "")
        return state

    return run_langchain_deep_agents_wikispeedia_program


async def run_langchain_deep_agents_wikispeedia_program(
    task: vf.Task, state: vf.State, harness: WikispeediaHarness
) -> vf.State:
    return await make_langchain_deep_agents_program(
        max_turns=harness.config.max_turns,
        timeout_seconds=harness.config.timeout_seconds,
    )(task, state)


def load_taskset(config: WikispeediaTasksetConfig) -> WikispeediaTaskset:
    return WikispeediaTaskset(config=config)


def load_harness(config: WikispeediaHarnessConfig) -> WikispeediaHarness:
    return WikispeediaHarness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

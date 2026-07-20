# Tool Use and Search

Some tasks are unanswerable from parametric memory — the model has to *look things up*. In this tutorial you will study `wiki_search_v1`, an environment where the model answers trivia questions by searching a Wikipedia corpus through custom tools, and learn how v1 packages tools as `vf.Toolset` MCP servers.

Before writing a tool, check whether you need one: many harnesses ship with bash, file, and web-search tools already, and a taskset that relies on harness-provided tools runs everywhere. Custom toolsets are for capabilities the task itself owns — here, search over a *specific pinned corpus*, so the eval measures search behavior rather than whatever the open web returns. Note that custom tools are installed via MCP, and not every harness supports that (the harness advertises it with `SUPPORTS_MCP`; the `default` harness does).

**You need:** [Build Your First Environment](5_build_first_environment.md). For the use-case view of this same environment — evaluating and *training* the search behavior — see the [Search Agent](17_search_agent.md) recipe.

## The anatomy of a toolset

A `vf.Toolset` bundles tools the way a `vf.Taskset` bundles tasks: a typed config, lifecycle hooks, and `@vf.tool` methods (`environments/wiki_search_v1/wiki_search_v1/servers/wiki.py`, abbreviated):

```python
class WikiSearchToolset(vf.Toolset[vf.SharedToolsetConfig]):
    TOOL_PREFIX = "wiki"

    async def setup(self) -> None:
        # runs once per server: load the corpus, build/open the Chroma index
        ...
        self.wiki = WikiIndex(collection, page_id_to_title, page_id_to_content)

    @vf.tool
    def search_pages(self, query: str) -> list[dict]:
        """Search for relevant articles using title embedding similarity."""
        ...

    @vf.tool
    def view_sections(self, page_id: str) -> list[dict]:
        """View the sections of a page."""
        ...

    @vf.tool
    def read_section(self, section_id: str) -> str:
        """Read a section of a page."""
        ...


if __name__ == "__main__":
    WikiSearchToolset.run()
```

The pieces:

- **`@vf.tool` methods become MCP tools.** The docstring is the model-facing description and the signature is the schema. Methods may be sync or async. `TOOL_PREFIX = "wiki"` namespaces them (`wiki_search_pages`, ...).
- **`setup()` runs once per server**, before any rollout uses it. Expensive, task-agnostic work belongs here — this one loads the corpus and builds an embedding index. Because this toolset uses `SharedToolsetConfig` and is declared on the taskset, one server is shared by an environment worker.
- **`setup_task(task)` receives `TaskData`.** Use it only on task-scoped servers that need row data; shared taskset tools are intentionally task-agnostic.
- **The `__main__` block is required.** The tool server runs as a standalone process; `Toolset.run()` is its entrypoint.
- Notice the tools' design: search returns *ids*, sections are read *one at a time*. The model must navigate — search, skim, read — instead of receiving the answer in one blob. Tool granularity is task design.

## Wiring tools into the taskset

The taskset declares this expensive read-only server once at worker scope:

```python
class WikiSearchConfig(vf.TasksetConfig):
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: WikiSearchTaskConfig = WikiSearchTaskConfig()


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    tools = (WikiSearchToolset,)
```

Scoring is supplied by a reference judge in `WikiSearchTaskConfig`, comparing `trace.last_reply` against ground truth. The model emits typed `vf.ToolCall` entries on assistant messages; tool results are typed `vf.ToolMessage` objects in `trace.tool_messages`, with `tool_call_id`, optional `name`, and `content`.

## Placement: where does the server run?

Tool placement is config, not code. The options, roughly in order of preference:

- **Task-scoped** — declare the class on `Task.tools`; a fresh server runs per rollout.
- **Taskset-scoped shared** — declare the class on `Taskset.tools` with `SharedToolsetConfig`, as wiki search does.
- **Colocated** — set `colocated = true` on a task-scoped tool config when it needs the harness filesystem.
- **Remote** — set `url` to connect to an existing MCP service.

## Run it

The eval config drives everything you just read, including the nested tool config (`configs/10/wiki-search-eval.toml`):

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 5
num_rollouts = 2
max_turns = 8                     # search takes turns: leave room

[sampling]
max_tokens = 2048

[taskset]
id = "wiki_search_v1"

[harness]
id = "default"
```

```bash
uv run eval @ configs/10/wiki-search-eval.toml
```

The first run pays the index build; subsequent runs reuse `~/.cache/wiki_search` (override it with `WIKI_SEARCH_CACHE`). In `traces.jsonl`, follow the assistant tool calls and matching tool messages: a good trajectory searches, opens a page's sections, reads one, and answers.

This environment trains as-is — `configs/10/wiki-search-rl.toml` embeds it in a training config exactly as [tutorial 3](3_first_rl.md) described, tools included. The [Search Agent](17_search_agent.md) recipe walks that full loop.

## Try it

- Cap the model's search budget: define a custom shared tool config with `max_searches`, track per-rollout usage in typed state, and have `search_pages` refuse beyond it. Watch how behavior changes in the traces.
- Add a `@vf.metric` counting tool calls per rollout, and compare tool usage between a strong and a weak model on the same tasks.

## Next

→ [Coding Agent Environments](11_coding_agents.md): tasks where the model's output runs — interpreters, Docker, and sandboxes.

# 08 — Tool Use and Search

Some tasks are unanswerable from parametric memory — the model has to *look things up*. In this guide you will study `wiki-search`, an environment where the model answers trivia questions by searching a Wikipedia corpus through custom tools, and learn how v1 packages tools as `vf.Toolset` MCP servers.

Before writing a tool, check whether you need one: many harnesses ship with bash, file, and web-search tools already, and a taskset that relies on harness-provided tools runs everywhere. Custom toolsets are for capabilities the task itself owns — here, search over a *specific pinned corpus*, so the eval measures search behavior rather than whatever the open web returns. Note that custom tools are installed via MCP, and not every harness supports that (the harness advertises it with `SUPPORTS_MCP`; the `default` harness does).

## The anatomy of a toolset

A `vf.Toolset` bundles tools the way a `vf.Taskset` bundles tasks: a typed config, lifecycle hooks, and `@vf.tool` methods (`environments/wiki_search/wiki_search.py`, abbreviated):

```python
class WikiToolConfig(vf.ToolsetConfig):
    corpus_dataset: str = "willcb/rare-wiki-pages"
    chroma_db_dir: str = ".chroma_db/wiki-search"
    shared: bool = True


class WikiSearchToolset(vf.Toolset[WikiToolConfig]):
    TOOL_PREFIX = "wiki"

    async def setup(self) -> None:
        # runs once per server: load the corpus, build/open the Chroma index
        ...
        self.wiki = WikiIndex(collection, page_id_to_title, page_id_to_content)

    @vf.tool
    async def search_pages(self, query: str) -> dict[str, list[dict[str, str]]]:
        """Search for relevant articles using title embedding similarity."""
        ...

    @vf.tool
    async def view_sections(self, page_id: str) -> dict[str, list[dict[str, str]]]:
        """View the sections of a page."""
        ...

    @vf.tool
    async def read_section(self, section_id: str) -> str:
        """Read a section of a page."""
        ...


if __name__ == "__main__":
    WikiSearchToolset.run()
```

The pieces:

- **`@vf.tool` methods become MCP tools.** The docstring is the tool description the model sees, the signature is the schema — write both for the model, not for yourself. `TOOL_PREFIX = "wiki"` namespaces them (`wiki_search_pages`, ...), so multiple toolsets can coexist without collisions.
- **`setup()` runs once per server**, before any rollout uses it. Expensive, task-agnostic work belongs here — this one loads the corpus and builds an embedding index. There is also `setup_task(task)` for per-task preparation (Guide 10 uses it).
- **The `__main__` block is required.** The tool server runs as a standalone process; `Toolset.run()` is its entrypoint.
- Notice the tools' design: search returns *ids*, sections are read *one at a time*. The model must navigate — search, skim, read — instead of receiving the answer in one blob. Tool granularity is task design.

## Wiring tools into the taskset

The taskset exposes tool servers per task; the default is none, so you wire it explicitly:

```python
class WikiSearchConfig(vf.TasksetConfig):
    max_examples: int | None = None
    judge: JudgeConfig = JudgeConfig()
    tools: WikiToolConfig = WikiToolConfig()   # nested config, TOML-visible


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    def tools(self, task: TriviaTask) -> list[vf.Toolset]:
        return [WikiSearchToolset(self.config.tools)]
```

Scoring is a judge reward (Guide 07) comparing the model's final answer against the trivia ground truth — tools change *how* the model works, not how it is scored. The tool calls themselves land in `trace.tool_messages`, so you can audit search behavior in the traces.

## Placement: where does the server run?

Tool placement is config, not code. The options, roughly in order of preference:

- **per-rollout** (the default) — a fresh server per rollout; right for cheap setup.
- **`shared = true`** — one server for all rollouts; right for expensive read-mostly setup like this Chroma index. Per-rollout mutable state must then live in `self.state` (a typed `vf.State`, serialized through the state channel — see Guide 10), never on `self`, or rollouts will trample each other.
- **`shared = true` with `fork = true`** — a shared warm parent forked per rollout; for process-local state that cannot be serialized.
- **`colocated`** — inside the harness runtime, when the tool needs the agent's filesystem.
- **`url`** — point at an MCP service you run elsewhere.

## Run it

The eval config drives everything you just read, including the nested tool config (`configs/08/wiki-search-eval.toml`):

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 5
num_rollouts = 2
max_turns = 8                     # search takes turns: leave room

[sampling]
max_tokens = 2048

[taskset]
id = "wiki-search"
max_examples = 250

[taskset.tools]
shared = true
chroma_db_dir = ".chroma_db/wiki-search-small"

[harness]
id = "default"
```

```bash
prime eval run @ configs/08/wiki-search-eval.toml
```

The first run pays the index build; subsequent runs reuse the Chroma directory. In the traces, follow one rollout's `tool_messages`: a good trajectory searches, opens a page's sections, reads one, and answers.

This environment trains as-is — `configs/08/wiki-search-rl.toml` embeds it under `[[orchestrator.train.env]]` exactly as Guide 03 described, tools included.

## Try it

- Cap the model's search budget: add a `max_searches` field to `WikiToolConfig` and have `search_pages` refuse beyond it. Watch how behavior changes in the traces.
- Add a `@vf.metric` counting tool calls per rollout, and compare tool usage between a strong and a weak model on the same tasks.

## Next

→ [09 — Multimodal Environments](../09-multimodal-environments/README.md): tasks that carry images, and user simulators that drive multi-turn games.

# Tool Use and Search

Tools in v1 are MCP servers authored as `vf.Toolset` classes. A taskset exposes tool server instances through `tools(self, task)`.

`wiki-search` uses a shared tool server because the Chroma index is expensive to initialize:

```python
class WikiSearchState(vf.State):
    searches: int = 0


class WikiToolConfig(vf.ToolsetConfig):
    corpus_dataset: str = "willcb/rare-wiki-pages"
    chroma_db_dir: str = ".chroma_db"
    shared: bool = True


class WikiSearchToolset(vf.Toolset[WikiToolConfig, WikiSearchState]):
    TOOL_PREFIX = "wiki"

    async def setup(self) -> None:
        ...

    @vf.tool
    async def search_pages(self, query: str) -> dict[str, list[dict[str, str]]]:
        self.state.searches += 1
        ...


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    def tools(self, task: TriviaTask) -> list[vf.Toolset]:
        return [WikiSearchToolset(self.config.tools)]
```

The tool server must be runnable as a standalone module:

```python
if __name__ == "__main__":
    WikiSearchToolset.run()
```

Configure tool placement under the taskset field that owns the tool config:

```toml
[taskset]
id = "wiki-search"
max_examples = 250

[taskset.tools]
shared = true
chroma_db_dir = ".chroma_db/wiki-search-small"
```

Use `shared = true` for read-mostly expensive setup. Each rollout still gets isolated `self.state` through the state channel. Use a per-rollout server, or `shared = true` with `fork = true`, when the tool needs private process-local state that cannot be represented in `self.state`.

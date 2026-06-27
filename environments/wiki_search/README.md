# wiki-search

A v1 tool-use taskset for answering trivia with a searchable wiki corpus.

- Taskset: loads questions from `willcb/wiki-trivia-questions-v4`
- Toolset: `WikiSearchToolset` with `wiki_search_pages`, `wiki_view_sections`, and `wiki_read_section`
- Tool placement: shared by default so the Chroma index is initialized once
- Reward: LLM judge over the final assistant answer

Run:

```bash
uv run eval @ configs/08/wiki-search-eval.toml
```

Tool config lives under `[taskset.tools]`:

```toml
[taskset]
id = "wiki-search"
max_examples = 250

[taskset.tools]
shared = true
chroma_db_dir = ".chroma_db/wiki-search-small"
```

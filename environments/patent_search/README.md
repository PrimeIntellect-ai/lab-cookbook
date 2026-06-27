# patent-search

Three v1 patent-search tasksets built around a shared pattern: a searchable patent corpus toolset plus an LLM judge.

- `basic-patent-q-and-a`: basic patent QA
- `advanced-patent-q-and-a`: metadata, abstract, section, and read tools
- `patent-technical-analysis`: abstract, claims, description, and rubric-style judging

Each package exports one `vf.Taskset` subclass and uses a shared tool config shape:

```toml
[taskset.tools]
shared = true
chroma_db_dir = ".chroma_db"
```

The environment-local `config.toml` files are hosted-training examples using `[[orchestrator.train.env]]`.

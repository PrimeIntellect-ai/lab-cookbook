# langchain-deep-agents-wikispeedia

A v1 Wikispeedia taskset with native tools and trace-based scoring.

- Task: navigate from a source article to a target article
- Toolset: read article text and follow article links through server-side state
- State: records visited path and completion
- Reward: success and path-quality metrics

Run:

```bash
uv run eval @ configs/12/deep-agents-eval.toml
```

The example config pairs this taskset with the reusable `langchain-deep-agents` harness.

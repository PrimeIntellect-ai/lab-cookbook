# langchain-deep-agents

A reusable v1 harness for running LangChain Deep Agents against any taskset that exposes MCP tools.

The harness owns the Deep Agents runtime loop. Task data, tool servers, server-side state, and scoring stay on the selected taskset.

Use it from eval config:

```toml
[taskset]
id = "my-tool-taskset"

[harness]
id = "langchain-deep-agents"
```

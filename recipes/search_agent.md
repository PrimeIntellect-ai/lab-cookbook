# A Search Agent

Some questions can't be answered from memory — the model has to *look things up*. In this recipe you'll work with `wiki_search_v1`, an environment where the model answers obscure trivia questions by searching a pinned Wikipedia corpus through custom tools. You'll see how tools are built and wired, evaluate a model's search behavior, and then — the payoff — train a model with RL to search *better*.

**You need:** tutorials [1](../tutorials/1_setup.md)–[3](../tutorials/3_first_rl.md). The official environment ships in this repo at `environments/wiki_search_v1/`.

## Why custom tools at all?

First, the question to ask before building any tool: does the harness already provide it? Many harnesses ship with bash, file access, and web search built in — and a taskset that relies on harness tools runs everywhere. Custom tools earn their keep when the *task owns the capability*. Here, that's the point: the model must search a **specific pinned corpus** (rare Wikipedia pages), so the eval measures search skill — not whatever the open web happens to return that day. Same reason a chemistry exam supplies the periodic table it wants you to use.

Custom tools are served to the model over MCP, and the harness must support that (the built-in `default` harness does).

## The tools

A `vf.Toolset` bundles tools the way a taskset bundles tasks. Wiki search exposes three, deliberately shaped like a research workflow (`environments/wiki_search_v1/wiki_search_v1/servers/wiki.py`, abbreviated):

```python
class WikiSearchToolset(vf.Toolset[vf.SharedToolsetConfig]):
    TOOL_PREFIX = "wiki"

    async def setup(self) -> None:
        # runs once per server: load the corpus, build a Chroma embedding index
        ...

    @vf.tool
    async def search_pages(self, query: str) -> dict:
        """Search for relevant articles using title embedding similarity."""

    @vf.tool
    async def view_sections(self, page_id: str) -> dict:
        """View the sections of a page."""

    @vf.tool
    async def read_section(self, section_id: str) -> str:
        """Read a section of a page."""


if __name__ == "__main__":
    WikiSearchToolset.run()   # the tool server runs as its own process
```

What to notice:

- `@vf.tool` **methods become the model's tools.** The docstring is the description the model reads; the signature is the schema. Write both *for the model*.
- **The tools force navigation.** Search returns page *ids*; sections are read *one at a time*. The model must search → skim → read, like a person would — tool granularity is task design. One "give me the answer" tool would measure nothing.
- `setup()` **runs once per server** and holds the expensive part (building the embedding index). Because this server is costly and read-only, the taskset declares it with `SharedToolsetConfig`: one server serves an environment worker's rollouts. (The full menu of placement options is in [Tool Use and Search](../tutorials/10_tools.md).)

The taskset declares the shared tool server, while each task is configured with a reference judge:

```python
class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    tools = (WikiSearchToolset,)

class WikiSearchTaskConfig(vf.TaskConfig):
    judges: vf.Judges = [vf.ReferenceJudgeConfig(...)]
```

Tools change *how* the model works, never *what counts as correct*.

## Evaluate the agent

```bash
uv run eval @ configs/10/wiki-search-eval.toml
```

The config, with the parts that matter:

```toml
model = "openai/gpt-5.4-nano"
num_tasks = 5
num_rollouts = 2
max_turns = 8               # search takes turns — leave room

[sampling]
max_tokens = 2048

[taskset]
id = "wiki_search_v1"

[harness]
id = "default"
```

The first run pays the one-time index build; later runs reuse `~/.cache/wiki_search` (or the directory selected by `WIKI_SEARCH_CACHE`).

Now read the traces (`prime eval view`) — with tools, the transcript grows a third voice. Alongside the model's messages you'll see `trace.tool_messages`: every search it issued, every section it read. A good rollout reads like research — search, open a page's sections, read the right one, answer. The failures are more instructive, and they cluster into visibly different species:

- **Never searched** — answered from parametric memory, wrong. (On this deliberately obscure corpus, that's the point.)
- **Searched badly** — reasonable queries, wrong pages, gave up.
- **Found it, fumbled it** — the right section is *in the trace* and the final answer still doesn't use it.

Diagnose which species dominates before changing anything — they have different fixes (prompt, tools, model), and only one of them is trainable away.

## Train the behavior

Which brings us to the payoff. Search is a *behavior*, not a fact — exactly what RL trains. This environment is training-ready:

```bash
prime train init wiki-search.toml   # generates a template config; point its [[env]] at wiki_search_v1
```

The essentials of the training config (`configs/10/wiki-search-rl.toml` shows a complete open-source-trainer variant):

```toml
model = "openai/gpt-oss-20b"
max_steps = 100
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 2048

[[env]]
name = "wiki-search"
max_turns = 8
taskset = { id = "wiki_search_v1" }
harness = { id = "default" }
```

Everything from [tutorial 3](../tutorials/3_first_rl.md) applies — groups, sometimes-solvable tasks, watching reward climb. What's new is *what improvement looks like*: check `prime train rollouts` early and late in the run. Early: skipped searches, one-shot queries, answers ungrounded in what was read. Later: consistent search-first behavior, query reformulation after a miss, answers quoting the retrieved section. The reward only said "right answer or not" — the *strategy* is what the model discovered under it.

## Things to try

- Compare two models on the same 50 tasks and count tool calls per rollout from the traces: stronger models often search *less but better* — fewer, sharper queries.
- Tighten `max_turns` from 8 to 4 and watch the reward drop: how much of the score was persistence? (For training, that same cap is pressure toward *efficient* search.)
- Break it on purpose: swap the judge to a weaker model — add a `[[taskset.task.judges]]` block to the eval config (`id = "reference"`, `question_field = "question"`, and a weaker `model`) — and audit 10 verdicts by hand: a search agent's eval is only as good as its judge ([Judges](../tutorials/6_judges.md)).
- Swap the corpus: the page corpus is pinned as the `DATASET` constant in `wiki_search_v1/servers/wiki.py` — point your own copy at your internal docs instead of rare wiki pages, and the same environment evaluates *your* retrieval task.



## Recap

Custom tools are MCP servers the taskset owns: docstrings are the model's documentation, granularity is task design, and expensive read-only setup is shared across rollouts. Evaluation gains a third voice in the trace — read the tool calls, not just the answer, and classify failures before fixing them. And because search is a behavior under a verifiable reward, the same environment that measures it can train it.
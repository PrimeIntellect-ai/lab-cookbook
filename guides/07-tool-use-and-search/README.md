# Tool Use and Search

Build an environment where the model has to search before it answers.

In the first environments, each task gives the model everything it needs in the prompt. Search environments add another layer: the task asks a question, and the environment gives the model tools for finding evidence in a corpus.

This guide uses [primeintellect/wiki-search](https://app.primeintellect.ai/dashboard/environments/primeintellect/wiki-search), a Wikipedia search environment on the Environments Hub. The model gets a trivia question, searches a small Wikipedia corpus, reads relevant sections, and answers from the evidence it finds.

## Evaluate the Hub Environment

Run a small eval:

```bash
prime eval run primeintellect/wiki-search \
  -m openai/gpt-5-nano \
  -n 5 \
  -r 2 \
  -t 2048
```

Open the eval results:

```bash
prime lab view --evals
```

For tool-use environments, inspect the full trajectory. The final answer matters, but the tool calls explain how the model got there.

Look for:

- the first search query
- which pages the model opens
- which sections it reads
- whether the final answer uses retrieved evidence
- whether failed rollouts failed at search, reading, reasoning, or answer formatting
- whether the reward matches what you see in the trajectory

## The Search Pattern

A search environment usually has four pieces:

- **Tasks**: questions and answers, plus any metadata needed for scoring.
- **Corpus**: the documents, pages, files, or records the model can search.
- **Tools**: the interface the model uses to explore the corpus.
- **Metrics**: the scoring signal computed from the completed rollout.

In `wiki-search`, the tools are intentionally small:

- `search_pages(query)` returns matching Wikipedia page IDs and titles.
- `view_sections(page_id)` lists sections inside a selected page.
- `read_section(section_id)` returns the text for one section.

This gives the model a natural retrieval path: search broadly, choose a page, inspect its structure, read the useful section, then answer.

## The Taskset Shape

The `wiki-search` implementation follows the v1 Taskset pattern. The Taskset is the place where tasks, tools, prompts, and rewards come together.

Conceptually, it looks like this:

```python
import verifiers.v1 as vf


def load_taskset(...):
    return vf.Taskset(
        source=build_source(max_turns=max_turns),
        system_prompt=SYSTEM_PROMPT,
        toolsets=[load_toolset(...)],
        rewards=[judge_reward_factory(...)],
    )


def load_v1_environment(...) -> vf.Env:
    return vf.Env(taskset=load_taskset(...))
```

The source yields task rows with a user prompt, answer, example ID, and turn limit. The toolset exposes the search tools and binds them to a loaded Wikipedia index. The reward checks the model's final answer against the reference answer.

You do not need a custom harness for this pattern. The default tool-use loop is enough: the model calls tools, receives tool results, and eventually responds with a final answer.

## Design the Tools

Good search tools should give the model enough control to solve the task without giving away the answer.

For a document search environment, start with a small tool surface:

- one tool to search or filter the corpus
- one tool to inspect candidate results
- one tool to read the selected content

Avoid tools that do the reasoning for the model. In `wiki-search`, the environment can retrieve pages and sections, but the model still has to decide what evidence matters and compose the answer.

## Design the Reward

Search rewards need to separate retrieval quality from answer quality.

For a first version, score the completed rollout by final-answer correctness. That is what `wiki-search` does with a judge reward: it gives the judge the question, reference answer, and model response, then asks whether the response is correct and coherent.

As the environment matures, add diagnostics around the trajectory:

- Did the model call search?
- Did it read any sections before answering?
- Did it cite or use information from the retrieved text?
- Did it stop too early or hit the turn limit?

Keep those as metrics until you are confident they should affect training. A metric that is useful for debugging is not always a good reward.

## Train on Search

Once the eval shows that the environment is healthy, train against the same Hub environment.

Create `configs/rl/wiki-search.toml`:

```toml
model = "openai/gpt-oss-20b"
max_steps = 100

batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 2048
reasoning_effort = "medium"

[[env]]
id = "primeintellect/wiki-search"
```

Launch training:

```bash
prime train configs/rl/wiki-search.toml
```

During training, watch both reward and trajectories. A good run should show the model searching more directly, reading fewer irrelevant sections, and answering from evidence more consistently.

## Next

In [Code and Sandboxes](../08-code-and-sandboxes/README.md), you will work with environments where the model writes code and the environment checks behavior by running it.

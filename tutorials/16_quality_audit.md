# Recipe — Quality-Auditing Synthetic Tasks

Generating tasks is easy — an LLM will happily turn any corpus into ten thousand QA pairs overnight. The hard question comes after: **are they any good?** A synthetic taskset looks exactly like a real one from the outside; the aggregate score won't tell you that a third of the questions are unanswerable and another fifth are guessable without doing the task. In this recipe you'll build the audit that catches both — demonstrated on a synthetic *search* environment, where the technique is at its cleanest: **a QA pair is sound if a strong model scores high with the oracle content in context, and low without it.**

**You need:** tutorials [1](1_setup.md)–[2](2_first_eval.md); [Search Agent](17_search_agent.md) for context on search environments, and [Infinite Tasksets](14_infinite_tasksets.md) if generated tasks are new to you.

## The two ways a synthetic task lies

Say you generate QA pairs from a document corpus for a search environment: pick a section, have a generator LLM write a question answerable from it, keep the answer and the source section. Every generated pair can fail in one of two *opposite* directions:

- **Unanswerable.** The generator hallucinated the answer, the question is ambiguous, the gold is subtly wrong, or the judge can't recognize a correct response. Even a *perfect* search agent scores 0 on these — they put a hard ceiling on measurable performance, and in RL they're worse than useless: they punish correct behavior.
- **Guessable.** The answer is famous, leaks into the question itself ("In which year did the 1969 moon landing occur?"), or sits comfortably in every model's parametric memory. Agents score on these *without searching* — the task no longer measures retrieval, and RL will learn precisely the wrong lesson: skip the tools.

Both are invisible in the mean score. Both are common in generated data. And they have to be caught **per task**, because filtering is the fix.

## The audit: bracket every task between two ablations

The insight is that a search task makes an implicit two-part claim — *the answer is in the content* (answerable) and *the answer requires the content* (retrieval-necessary). Each part can be tested directly by manipulating what's in context, using a strong model as the probe:

| Run | What the model gets | A sound task's result |
| --- | --- | --- |
| **Oracle** | The question **plus the source section** pasted into the prompt. No tools. | **High** — if a strong model can't answer *with the evidence in hand*, the pair (or the judge) is broken. |
| **Closed-book** | The question alone. No tools, no content. | **Low** — if the model answers *without* the evidence, search was never required. |

A task is **sound** when it passes both: high with oracle, low without. Notice what makes this powerful: it needs no human labels, it audits the *judge* along the way (oracle failures with a correct-looking response are judge failures), and it produces a per-task verdict you can filter on.

## Build the modes into the environment

The audit should run through the exact pipeline the real eval uses — same taskset, same judge, same reward — with only the information varied. So the mode is taskset *config*, not a separate script:

```python
Mode = Literal["search", "oracle", "closed_book"]

ORACLE_TEMPLATE = """Answer the question using the reference content below.

Reference:
\"\"\"{oracle_content}\"\"\"

Question: {question}"""


class SynthSearchTask(vf.Task):
    question: str
    answer: str
    oracle_content: str          # the section this QA pair was generated from


class SynthSearchConfig(vf.TasksetConfig):
    mode: Mode = "search"
    pairs_file: str = "data/generated_pairs.jsonl"
    judge: JudgeConfig = JudgeConfig()
    tools: SearchToolConfig = SearchToolConfig()


class SynthSearchTaskset(vf.Taskset[SynthSearchTask, SynthSearchConfig]):
    def load_tasks(self) -> list[SynthSearchTask]:
        tasks = []
        for i, row in enumerate(load_jsonl(self.config.pairs_file)):
            if self.config.mode == "oracle":
                prompt = ORACLE_TEMPLATE.format(
                    oracle_content=row["oracle_content"], question=row["question"]
                )
            else:  # search and closed_book both see only the question
                prompt = row["question"]
            tasks.append(SynthSearchTask(
                idx=i, prompt=prompt,
                question=row["question"], answer=row["answer"],
                oracle_content=row["oracle_content"],
            ))
        return tasks

    def tools(self, task: SynthSearchTask) -> list[vf.Toolset]:
        if self.config.mode != "search":
            return []                       # ablations run tool-free
        return [SearchToolset(self.config.tools)]

    @vf.reward(weight=1.0)
    async def judged(self, task, trace) -> float:
        ...  # same judge in every mode — that's the point
```

The generation step itself (corpus section → generator LLM → `{question, answer, oracle_content}` JSONL) is a one-time script; keep the source section with every pair — it *is* the oracle, and pairs without provenance can't be audited.

## Run the audit

Two runs, differing in exactly one flag. Use a **strong** model — the probe should be at least as capable as any model you'll ever evaluate here, so that "the strong model couldn't do it with the answer in hand" means *broken task*, not *weak prober*. And use multiple rollouts: per-task verdicts from a single sample are coin flips.

```bash
prime eval run synth-search --taskset.mode oracle \
  -r 4 --model openai/gpt-5.4 -o outputs/audit-oracle

prime eval run synth-search --taskset.mode closed_book \
  -r 4 --model openai/gpt-5.4 -o outputs/audit-closed
```

Then join the two runs per task — each `results.jsonl` holds every rollout's trace with its task index and reward:

```python
import json
from collections import defaultdict

def pass_rates(path):
    rates = defaultdict(list)
    for line in open(path):
        t = json.loads(line)
        rates[t["task"]["idx"]].append(t["reward"])
    return {i: sum(r) / len(r) for i, r in rates.items()}

oracle = pass_rates("outputs/audit-oracle/results.jsonl")
closed = pass_rates("outputs/audit-closed/results.jsonl")

sound = [i for i in oracle if oracle[i] >= 0.75 and closed.get(i, 0) <= 0.25]
print(f"{len(sound)}/{len(oracle)} pairs are sound")
json.dump(sound, open("data/sound_ids.json", "w"))
```

(Adjust the field access to your trace schema; thresholds are dials — 0.75/0.25 is a sane start.)

## Read the quadrants

Every task now sits in one of four cells, and each cell has a different diagnosis:

| | **Closed-book low** | **Closed-book high** |
| --- | --- | --- |
| **Oracle high** | ✅ **Sound** — answerable, and search is required. Keep. | ⚠️ **Guessable** — real question, but the world already knows the answer. Drop for a search env (it trains tool-skipping); fine for a plain QA set. |
| **Oracle low** | ❌ **Broken** — bad gold, ambiguous question, or a judge that can't recognize correctness. Read a few traces before dropping: if the model's oracle answer *looks* right, your **judge** is the broken part — fix it and re-audit, because a broken judge poisons all four cells. | 🚨 **Inverted** — the model answers "correctly" from memory but fails *with the evidence*? Almost always a gold answer that contradicts the source, or the question and section drifted apart during generation. Inspect by hand. |

Spend ten minutes reading traces from the two failure cells — generated-task failures are highly patterned (the generator writes ambiguous questions in one recognizable style, leaks answers in another), and one prompt fix to the *generator* often clears a whole cluster on the next batch. The audit isn't just a filter; it's the feedback loop for your generation prompt.

## Ship the filtered taskset

Make soundness a config field, so the filter is part of the environment rather than tribal knowledge:

```python
class SynthSearchConfig(vf.TasksetConfig):
    ...
    sound_ids_file: str | None = None    # audit output; None = unfiltered

# in load_tasks:
if self.config.sound_ids_file:
    keep = set(json.load(open(self.config.sound_ids_file)))
    tasks = [t for t in tasks if t.idx in keep]
```

Now the real environment runs filtered by default, and the audit trail is reproducible: anyone can re-run the two ablations and regenerate `sound_ids.json`. Record the yield in the env README — *"412/500 generated pairs passed the oracle/closed-book audit (gpt-5.4, 4 rollouts, 0.75/0.25 thresholds)"* — that one sentence is what makes a synthetic benchmark trustworthy to someone who didn't build it.

Two habits to go with it: **re-audit whenever anything upstream changes** (generator prompt, corpus, judge — the audit is cheap; a poisoned training run is not), and treat the aggregate closed-book pass rate as your generator's **leakage score**, a number to drive down over generator iterations.

## Beyond search

The bracket generalizes to any synthetic task with hidden ground truth: the oracle run gives the model whatever information makes the task *trivially checkable* (the relevant document, the constraint list, the intermediate result), the closed-book run withholds it, and soundness is the gap between them. Note what each half audits: the oracle side validates your **data and judge**; the closed-book side validates that the task measures the **capability you built it for**. Environments with generated worlds get the first half free — a validated generator ([Synthetic Worlds](15_synthetic_world.md)) can't produce unanswerable tasks — but the leakage half still applies to anything whose answers exist in pretraining data.

## Things to try

- **Add a distractor leg:** oracle content plus three irrelevant sections. High performance with a clean oracle but low with distractors reveals questions that are answerable but *fragile* — a difficulty signal the two-run audit can't see.
- **Two-prober agreement:** run the audit with two different strong models and keep only tasks where both agree. Disagreement clusters are almost always ambiguity.
- **Audit an existing taskset:** run the closed-book leg alone against any search environment you already use — the guessable fraction of established QA sets is routinely surprising.
- **Close the training loop:** RL-train a search agent ([Search Agent](17_search_agent.md)) on the unfiltered vs. the filtered taskset and compare tool-use rates — the filtered run should search *more*, because guessing no longer pays.

## Recap

Synthetic tasks fail in two opposite directions — unanswerable and guessable — and both are invisible in aggregate scores. Bracket every task between an oracle run (evidence in context; must be high) and a closed-book run (no evidence; must be low), using a strong model, multiple rollouts, and the environment's own judge. Filter to the sound quadrant, read the failure cells to fix your generator and judge, ship the filter as config, and re-audit whenever anything upstream changes.

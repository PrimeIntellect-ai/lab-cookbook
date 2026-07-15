# Recipe — Synthetic Worlds

Agent benchmarks usually mean infrastructure: containers, real APIs, flaky external services. But an "agent" is just a model acting through tools against some state — and nothing says the state has to be real. In this recipe you'll study a **synthetic world**: an environment that simulates its universe entirely in typed, in-memory state, exposes it through tools, and scores *the final state of the world* rather than the transcript. All the agency, none of the infrastructure — fast, deterministic, and free to reset.

The worked example is `calendar-scheduling`, which ships in this repo (`environments/calendar_scheduling/`): the model plays a scheduling assistant that must find the best meeting window for several attendees whose calendars it can only see through tools. Hub environments push the same pattern much further — `emerge/feishu-office-v1` simulates an entire office workspace, "scored by deterministic checks on final workspace state."

**You need:** tutorials [1](1_setup.md)–[2](2_first_eval.md); [Search Agent](17_search_agent.md) is a good warm-up if tools are new to you.

## The anatomy of a world

A synthetic world is four decisions. Here's each, with calendar-scheduling's answer.

### 1. The world lives in generated task data

Each task *is* a small world: attendees, their calendars, their constraints, and — because the environment generated all of it — the ground-truth best answer:

```python
class CalendarSchedulingTask(vf.Task):
    answer: str                 # the oracle-optimal window
    calendar_task: JsonObject   # the full hidden world: calendars, constraints
```

Worlds are generated procedurally (seed + difficulty + validation — the [Infinite Tasksets](14_infinite_tasksets.md) pattern), so there are unlimited fresh ones and every instance is guaranteed solvable. This is the quiet superpower of synthetic worlds: **the generator knows the optimum**, which will make scoring exact rather than judged.

### 2. The model sees the world only through tools

The task prompt states the goal; the *facts* are locked behind a toolset. Calendar-scheduling exposes four tools:

```python
class CalendarToolset(vf.Toolset[CalendarToolConfig, CalendarState]):
    @vf.tool
    async def check_attendee_calendar(self, attendee_id: str, day_index: int) -> str: ...
    @vf.tool
    async def view_attendee_constraints(self, attendee_id: str) -> str: ...
    @vf.tool
    async def check_proposal(self, ...) -> str:   # scores a candidate — budgeted!
    @vf.tool
    async def submit_window(self, ...) -> str:    # commits the final answer
```

This *information asymmetry* is what makes it an agent task instead of a reading-comprehension task: the model must decide what to look up, in what order, and when it knows enough. Note the two special tools — `check_proposal` lets the model test a candidate answer but only a **budgeted number of times** (a `score_checks_remaining` counter), and `submit_window` is an explicit, one-shot commit. Budgets and commit-tools are the levers that force real decision-making: unlimited checking would turn the task into brute-force search.

### 3. World changes live in typed per-rollout state

Everything the rollout mutates goes in a `vf.State` subclass, not on the tool object:

```python
class CalendarState(vf.State):
    score_checks_remaining: int = -1
    score_checks_used: int = 0
    proposal_checks: list[JsonObject] = Field(default_factory=list)
    submitted: bool = False
    submitted_valid: bool = False
    submitted_score: float = 0.0
```

Two reasons this discipline pays. Isolation: each rollout gets its own state, so a thousand concurrent rollouts can't contaminate each other. Visibility: the final state is serialized onto `trace.state`, where scoring — and you, debugging — can read it. Keep state JSON-serializable; live handles and caches belong on the tool server instance itself.

### 4. Score the end state, not the transcript

The rollout ends when the world reaches a terminal condition — here, the moment of submission:

```python
@vf.stop(priority=50)
async def has_submission(self, trace: vf.Trace) -> bool:
    return trace.state.submitted

@vf.reward(weight=1.0)
async def final_score_from_submission(self, trace: vf.Trace) -> float:
    if not trace.state.submitted_valid:
        return 0.0
    return trace.state.submitted_score
```

The reward never parses the conversation. It asks one question: *what state is the world in now?* This is the design principle to steal even when nothing else here fits your domain — end-state scoring is robust to *how* the agent got there (weird phrasing, unexpected-but-valid strategies, retries), where transcript-parsing rewards break on anything you didn't anticipate. It's the same reason coding environments run the tests instead of grading the diff.

And because the generator knew the optimum (decision 1), the environment doesn't stop at pass/fail — it ships *diagnostic metrics* computed against the oracle:

```python
@vf.metric
async def optimality_gap(self, task, trace) -> float: ...     # how far from the best answer?
@vf.metric
async def score_checks_used(self, trace) -> float: ...        # how much budget did it burn?
```

Metrics don't affect the reward; they make the traces explain themselves.

## Run it

```bash
prime eval run @ configs/11/calendar-scheduling-eval.toml
```

The config shows the world's dials — all of them taskset config, no code:

```toml
max_turns = 8

[taskset]
id = "calendar-scheduling"
difficulty = "medium"
seed = 7
num_tasks = 32
```

Open the run in `prime eval view` and read one full rollout. The shape to look for: the model probes calendars → narrows candidates → spends a `check_proposal` or two → commits. Then look at a failure and diagnose it *from the state and metrics*: submitted invalid? burned the whole check budget probing? committed early with a large `optimality_gap`? Each is a different failure mode, and each suggests a different fix — which is precisely the observability that end-state scoring plus metrics buys you.

## It trains, too

Synthetic worlds are ideal RL environments — cheap rollouts, exact rewards, unlimited fresh instances (no memorization), and a difficulty dial. This one is ready to go: `configs/11/calendar-scheduling-rl.toml` embeds it for training exactly as [tutorial 3](3_first_rl.md) did with reverse-text. The graded reward (score of the submitted window, not just success) gives the smooth signal RL needs.

## Design checklist for your own world

Building one — a simulated inbox, a CRM, a warehouse, a filesystem — comes down to the same four decisions:

1. **World in the task** — generate it (seed, difficulty, validation), and have the generator record ground truth / the oracle answer.
2. **Tools as the only window** — information asymmetry makes it agentic; budgets and an explicit commit-tool make it strategic.
3. **Mutations in `vf.State`** — serializable, per-rollout, visible to scoring.
4. **Reward from the final state** — plus oracle-based metrics for diagnosis.

## Things to try

- Run the difficulty ladder (`--taskset.difficulty easy|medium|hard`) and watch `optimality_gap` — models often keep *succeeding* while quietly getting *further from optimal* as worlds grow.
- Tighten the information budget: fewer allowed proposal checks per task raises the strategic pressure without touching the tasks. Does the model adapt or waste its budget faster?
- Sketch your own world: pick a workflow you know (ticket triage, inventory, travel booking), write its `State` class and four tools on paper, and decide what "final state correct" means. That one-page design is 80% of the environment; [Guide 11](../guides/11-synthetic-agent-environments/README.md) covers the authoring contract for the rest.

## Recap

Simulate the world in generated, typed state; expose it only through tools (with budgets and a commit step); keep all mutations in per-rollout `vf.State`; score the final state against oracle ground truth, with metrics for everything else you want to see. You get agent evaluation and training with zero infrastructure — and rewards that can't be argued with.

# A Support Agent with Simulated Users

Every environment so far had one model in the room. A support agent has two: the **agent** you're evaluating, and the **customer** it serves — someone who reveals information gradually, changes their mind, gets things slightly wrong, and eventually confirms or abandons. You can't script that side with canned messages; you simulate it with a second LLM, and you score the *outcome* — did the database end up in the right state? did the agent take the required actions, say the required things? — not the transcript's vibes.

This recipe runs [τ²-bench](https://github.com/sierra-research/tau2-bench), the customer-service benchmark, as a native v1 environment: `tau2-bench-v1`, bundled with this cookbook. It composes everything the Ramping-up series taught — multi-turn tool use ([10](../tutorials/10_tools.md)), a user simulator ([8](../tutorials/8_user_simulators.md)), rewards over final state rather than text ([7](../tutorials/7_rewards.md)) — into one realistic task, and adds one new pattern: wrapping an external orchestrator in a harness.

**You need:** tutorials [1](../tutorials/1_setup.md)–[2](../tutorials/2_first_eval.md), [User Simulators](../tutorials/8_user_simulators.md), and [Tool Use and Search](../tutorials/10_tools.md). Everything runs on inference credits — no Docker, no GPUs.

## The task shape: dual control

A τ²-bench task drops the agent into a domain — `telecom`, `retail`, `airline` — with a policy manual, a customer database, and domain tools. The simulated user has a scenario (their persona, what they want, what they know) that the agent never sees. Crucially, τ² is **dual-control**: some tools belong to the agent (look up the account, apply a fix), some to the *user* (restart their router, read a code off their screen) — the agent has to *instruct the human* through the user-side steps, not just call its own tools. That coordination is most of the difficulty.

One task's ground truth is not an answer string — it's the official evaluation bundle: the expected database state, required agent actions, environment assertions, and required communication to the user. The reward reads all four:

```python
class Tau2Task(vf.Task[Tau2Data]):
    @vf.reward
    async def tau2_reward(self, trace: vf.Trace) -> float:
        simulation = SimulationRun.model_validate(trace.info["tau2"]["simulation"])
        reward = simulation.reward_info
        return float(reward.reward) if reward else 0.0
```

The score is τ²'s official evaluation, computed from the finished simulation and recorded — with its full breakdown — in `trace.info["tau2"]`.

## The pattern: wrap the orchestrator in a harness

[Build Your Own Coding-Agent Harness](coding_agent_harness.md) built an agent program from scratch. This environment shows the complementary move: τ² already *has* a battle-tested orchestrator (agent ↔ user ↔ tools ↔ database), so `Tau2Harness` runs it whole, and routes only the evaluated agent's model calls through the verifiers interception endpoint:

- **The agent** is τ²'s `llm_agent`, but its LLM points at `endpoint` — so turn caps, token accounting, and trace recording work exactly as in every other environment.
- **The user simulator** is a separate model (default `gpt-4.1`, billed via your Prime credentials) that plays the customer. It is *not* the model under evaluation — same principle as a judge: the measuring instrument should not be the thing measured.
- **The simulation result** — every message, tool call, and the evaluation breakdown — lands in `trace.info["tau2"]`, so the trace stays self-contained.

When a domain already ships a good simulator and scorer, wrapping beats reimplementing: you inherit the official semantics, and verifiers still owns interception, concurrency, and traces.

## Run it

```bash
uv run eval @ configs/recipes/support-agent-eval.toml
```

```toml
model = "openai/gpt-5.4-mini"
num_tasks = 5
num_rollouts = 2

[taskset]
id = "tau2-bench-v1"
domain = "telecom"

[harness]
id = "tau2-bench-v1"
```

The taskset downloads τ²'s pinned task data on first load (into `~/.cache/tau2-bench-v1/`). `domain` selects the scenario family — `telecom` is the hardest of the classic three; `telecom-workflow` swaps in a procedural troubleshooting policy for the same tasks.

## Read the traces like a support-team lead

The reward is binary per task, but the *transcript* is where the evaluation lives. In `trace.info["tau2"]` you'll find the full simulation; triage failures into the species that actually occur:

- **Policy violations** — the agent did something the manual forbids (skipped identity verification, refunded beyond limits). The action checks catch this even when the customer ends up happy.
- **Coordination failures** — the agent fixed everything on its side but never walked the user through *their* step, or gave the instruction and didn't confirm the result before moving on.
- **Communication misses** — the fix was right, but the agent never told the user the thing the task requires them to be told; τ² scores required communication explicitly.
- **User-sim drift** — occasionally the *simulator* behaves oddly. Spot-check a few transcripts before blaming the agent; a user simulator is a measuring instrument with its own error bars.

Compare `num_rollouts = 2` siblings on the same task: support conversations are high-variance, and reading a solved/failed pair for the same scenario is the fastest way to see what actually separates them.

## Things to try

- **Swap domains:** `--taskset.domain retail` (or `airline`) — same agent, different policy manual and tools. Domain transfer is a real capability question: does your model follow *the manual in front of it* or the support-agent prior from pretraining?
- **Harden the policy pressure:** compare `telecom` and `telecom-workflow` scores for the same model. A procedural policy helps weaker models (less planning) and can *hurt* stronger ones (less flexibility) — one of the more interesting model-capability signatures this benchmark produces.
- **Read one dual-control task end-to-end** and count how many turns are the agent instructing the user vs. acting itself — then check whether failures cluster in the instructing half. (They usually do.)
- **Vary the model under test, not the simulator.** Keep the user simulator fixed across all runs you intend to compare — changing it changes the exam, not just the student.

## Recap

A support agent is evaluated as a *system*: agent model, simulated customer, tools on both sides of the counter, and a reward computed from final state, required actions, and required communication — not from how helpful the transcript sounds. The environment-building lesson is the wrap-don't-reimplement pattern: when an external orchestrator already owns the domain semantics, a harness can run it intact while verifiers owns interception and traces. And the evaluation lesson is the same one as everywhere in this cookbook, sharpened: with two models in the room, read the transcript knowing either one of them can be the reason a task failed.

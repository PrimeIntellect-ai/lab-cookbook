# Ethics Debate Cookbook

This cookbook demonstrates a simple **multi-agent reinforcement learning environment** built with Verifiers.

The setup trains a single shared policy to alternate between two roles:

- **Arguer**: produces and refines an argument about an ethical question
- **Critic**: identifies weaknesses, missing perspectives, and logical gaps

The goal is to improve final-answer quality through iterative critique and revision.

## Why this environment is interesting

Multi-agent orchestration is becoming an increasingly important pattern in modern agent systems.

Examples include:

- coding agents that delegate to subagents
- research workflows that split planning, execution, and review
- training setups where one model instance critiques or improves another

This recipe explores a simple version of that idea inside Verifiers: instead of generating one argument in a single pass, the policy alternates between producing an argument and critiquing it.

The hope is that RL can improve both behaviors at once:

- better critiques
- better revisions
- stronger final arguments

The README presents this as a scaffold rather than a final canonical recipe.

## Environment structure

The environment is implemented as a v1 `Taskset` plus the default `Harness`.

The taskset owns:

- dataset loading from `ergotts/ethics_questions`
- the arguer/critic role prompts
- a `user` function that schedules the next role after each model turn
- the LLM judge reward for the final argument

The default `vf.Harness` runs the endpoint-backed multi-turn loop. No custom harness subclass is needed because the role alternation can be expressed as taskset-owned user messages and rollout state.

Relevant docs:

- Tasksets and harnesses: https://docs.primeintellect.ai/verifiers/byo-harness

## Debate flow

The taskset uses the default harness loop to alternate roles.

The intended workflow is:

1. the **arguer** presents a position
2. the **critic** points out weaknesses
3. the **arguer** revises and strengthens the argument
4. the loop continues for a fixed number of rounds
5. the final arguer output is scored

### Actor definitions

The two roles are defined with distinct role instructions:

```python
ARGUER_PROMPT = (
    "You are arguing an ethics question. "
    "Present a clear position with reasoning. "
    "When critiqued, address the weaknesses and strengthen your argument."
)

CRITIC_PROMPT = (
    "You are critiquing an ethical argument. "
    "Identify gaps, logical fallacies, and missing perspectives. "
    "Be specific and constructive."
)
```

Even though the roles differ, they share the same underlying policy. The difference comes from prompting and taskset state.

### Scheduling logic

The rollout alternates between arguer and critic in the taskset `user` function:

```python
async def debate_user(task, state, transcript):
    actor = state.get("debate_actor", ARGUER)
    handoff = parse_handoff(actor, last_assistant_text(transcript))
    ...
```

This is the simplest multi-agent schedule possible, which makes it a good starting scaffold.

### Handoffs and shared state

Each actor emits a role-specific XML handoff, and the user function stores it in shared state:

```python
def handoff_tag(actor: str) -> str:
    return "argument" if actor == ARGUER else "critique"
```

This keeps the environment logic explicit and easy to extend.

## Reward design

The final argument is graded by an LLM judge.

The judge prompt scores the argument across five dimensions:

1. **Thesis clarity**
2. **Logical structure**
3. **Counterargument handling**
4. **Nuance**
5. **Depth**

Each category is scored from `0` to `2`, for a total possible score of `0` to `10`.

```python
judge_user_template = (
    "Question: {question}\n\n"
    "Argument:\n{argument}\n\n"
    "Score each dimension 0-2 (0=absent, 1=superficial, 2=thorough):\n\n"
    "1. THESIS CLARITY: Does it state a specific, defensible position?\n"
    "   A vague 'it depends' or generic both-sidesism = 0. Clear stance with defined scope = 2.\n\n"
    "2. LOGICAL STRUCTURE: Are claims warranted by reasoning, not just asserted?\n"
    "   Deduct for logical fallacies (slippery slope, false dichotomy, appeal to authority).\n\n"
    "3. COUNTERARGUMENT HANDLING: Does it engage the STRONGEST version of opposing views\n"
    "   (steel-man), or only weak caricatures (straw-man)? Explains WHY they fail, not just asserts it?\n\n"
    "4. NUANCE: Acknowledges edge cases, limitations, or conditions where the position might not hold?\n"
    "   Distinguishes between related but different concepts?\n\n"
    "5. DEPTH: Goes beyond common platitudes? Engages specific ethical frameworks\n"
    "   (utilitarianism, deontology, virtue ethics), historical examples, or thought experiments?\n\n"
    "Sum the five scores (0-10). Respond with ONLY the total number."
)
```

This gives a richer signal than simple exact-match evaluation, which would be inappropriate for open-ended ethical arguments.

## Dataset

The cookbook points to the [ergotts/ethics_questions](https://huggingface.co/datasets/ergotts/ethics_questions) dataset.

That makes sense for broad ethical prompts, though in practice you would likely want to think carefully about:

- diversity of question types
- ambiguity and stance plurality
- calibration of judge behavior across political and moral framings
- whether your reward should favor strong argumentation, balanced analysis, or both

## What this recipe is optimizing

This environment is not trying to train the model to arrive at a single objectively correct moral answer.

Instead, it optimizes for producing a **stronger final argument** according to the judge’s rubric:

- clear thesis
- coherent reasoning
- engagement with objections
- nuance
- depth

That is an important distinction. The target is rhetorical and analytical quality, not moral truth.

## Why the pattern is useful

This recipe is valuable even beyond ethics tasks because it demonstrates a reusable pattern for RL with structured internal roles.

Possible extensions include:

- writer / editor
- proposer / verifier
- planner / executor
- researcher / critic
- solution generator / adversarial tester

The core lesson is that RL can be applied not just to tool use or single-turn answers, but also to **interaction structure** inside the rollout itself.

## Suggested workflow

A reasonable workflow for building on this recipe would be:

1. define the actors and their roles clearly
2. implement the scheduling and handoff logic
3. decide what artifact gets judged at the end
4. create a strict judge rubric
5. evaluate baseline reward before training
6. iterate on prompts, number of rounds, and judge calibration

## Summary

This cookbook provides a clear starting point for multi-agent RL in Verifiers.

It shows how to:

- model multiple roles inside one rollout
- alternate those roles through shared state
- use critique and revision to improve a final artifact
- score the result with an LLM judge

It is intentionally simple, but that simplicity is a strength: the pattern is easy to understand, easy to extend, and useful for many agent-training settings beyond ethics debate.

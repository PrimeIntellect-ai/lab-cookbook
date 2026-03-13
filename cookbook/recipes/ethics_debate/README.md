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

The cookbook introduces a generic `MultiAgentEnv`.

It subclasses `StatefulToolEnv` to reuse Verifiers’ tool-calling and multi-turn rollout infrastructure, though a similar pattern could also be built directly on `MultiTurnEnv`.

Relevant docs:

- Stateful tool environments: https://docs.primeintellect.ai/verifiers/environments#stateful-tool-environments
- Custom multi-turn environments: https://docs.primeintellect.ai/verifiers/environments#custom-multi-turn-environments

## Generic multi-agent pattern

`MultiAgentEnv` is designed around a round-robin multi-agent orchestration loop.

At a high level, you need to define:

- the set of actors or roles
- the scheduling logic for who speaks next
- the handoff format used to move information between actors through shared state

The key methods are:

- `get_all_actors`: map each `actor_id` to its system prompt
- `get_initial_actor_id`: choose the first actor in the rollout
- `get_next_actor_id`: choose who acts next
- `get_handoff_tag`: define how each actor packages its output
- `apply_handoff`: update shared state after each turn

This makes the environment reusable beyond ethics debate. The same scaffold could support planner/executor, proposer/verifier, or researcher/reviewer patterns.

## EthicsDebateEnv

`EthicsDebateEnv` is the concrete specialization of `MultiAgentEnv` for ethical argument refinement.

The intended workflow is:

1. the **arguer** presents a position
2. the **critic** points out weaknesses
3. the **arguer** revises and strengthens the argument
4. the loop continues for a fixed number of rounds
5. the final arguer output is scored

### Actor definitions

The two actors are defined with distinct system prompts:

```python
def get_all_actors(self, state: State) -> dict[str, str]:
    return {
        ARGUER: (
            "You are arguing an ethics question. "
            "Present a clear position with reasoning. "
            "When critiqued, address the weaknesses and strengthen your argument."
        ),
        CRITIC: (
            "You are critiquing an ethical argument. "
            "Identify gaps, logical fallacies, and missing perspectives. "
            "Be specific and constructive."
        ),
    }
```

Even though the roles differ, they currently share the same underlying policy. The difference comes from prompting and state.

### Scheduling logic

The rollout alternates between arguer and critic:

```python
def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
    return ARGUER

def get_next_actor_id(self, state: State) -> str:
    return CRITIC if state["trajectory_id"] == ARGUER else ARGUER
```

This is the simplest multi-agent schedule possible, which makes it a good starting scaffold.

### Handoffs and shared state

Each actor emits a role-specific handoff payload, and the environment stores it in shared state:

```python
def get_handoff_tag(self, actor_id: str, state: State) -> str:
    return "argument" if actor_id == ARGUER else "critique"

async def apply_handoff(
    self, actor_id: str, handoff: dict[str, Any], state: State
) -> str | None:
    if actor_id == ARGUER:
        state["current_argument"] = handoff["argument"]
        # Final arguer turn: all prior rounds complete
        if len(state["handoff_history"]) == 2 * self.num_rounds:
            state["final_argument"] = handoff["argument"]
            state["final_env_response"] = "Debate complete."
            return state["final_env_response"]
    else:
        state["current_critique"] = handoff["critique"]
    return None
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

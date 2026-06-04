import re
from typing import cast

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

ARGUER = "arguer"
CRITIC = "critic"

JUDGE_SYSTEM = (
    "You are a philosophy professor grading ethics arguments. "
    "Be rigorous and critical. A generic, well-intentioned argument scores 3-5. "
    "Only arguments showing genuine philosophical rigor score 7+."
)

JUDGE_USER_TEMPLATE = """Question: {question}

Argument:
{argument}

Score each dimension 0-2 (0=absent, 1=superficial, 2=thorough):

1. THESIS CLARITY: Does it state a specific, defensible position?
   A vague 'it depends' or generic both-sidesism = 0. Clear stance with defined scope = 2.

2. LOGICAL STRUCTURE: Are claims warranted by reasoning, not just asserted?
   Deduct for logical fallacies (slippery slope, false dichotomy, appeal to authority).

3. COUNTERARGUMENT HANDLING: Does it engage the STRONGEST version of opposing views
   (steel-man), or only weak caricatures (straw-man)? Explains WHY they fail, not just asserts it?

4. NUANCE: Acknowledges edge cases, limitations, or conditions where the position might not hold?
   Distinguishes between related but different concepts?

5. DEPTH: Goes beyond common platitudes? Engages specific ethical frameworks
   (utilitarianism, deontology, virtue ethics), historical examples, or thought experiments?

Sum the five scores (0-10). Respond with ONLY the total number."""


ARGUER_PROMPT = """You are arguing an ethics question. Present a clear position with reasoning. \
When critiqued, address the weaknesses and strengthen your argument.

Question: {question}

IMPORTANT: When you are done, wrap your final response in <argument>...</argument> tags. \
Do NOT write anything after the closing </argument> tag."""

CRITIC_PROMPT = """You are critiquing an ethical argument. Identify gaps, logical fallacies, \
and missing perspectives. Be specific and constructive.

Question: {question}

Argument to critique:
{argument}

IMPORTANT: When you are done, wrap your final response in <critique>...</critique> tags. \
Do NOT write anything after the closing </critique> tag."""

REFINE_PROMPT = """You are arguing an ethics question. Present a clear position with reasoning. \
When critiqued, address the weaknesses and strengthen your argument.

Critique received:
{critique}

Refine your argument.

IMPORTANT: When you are done, wrap your final response in <argument>...</argument> tags. \
Do NOT write anything after the closing </argument> tag."""


class EthicsDebateTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "ergotts/ethics_questions"
    train_split: str = "train"
    eval_split: str = "train"
    num_rounds: int = 2
    judge_model: str = "openai/gpt-4.1-mini"
    judge_base_url: str = "https://api.pinference.ai/api/v1"
    judge_api_key_var: str = "PRIME_API_KEY"
    user: vf.UserConfig | None = vf.UserConfig()
    system_prompt: vf.SystemPrompt = (
        "You are participating in a structured ethics debate. Follow the current "
        "role assignment in each user message and end every turn with the requested "
        "XML tag."
    )


class EthicsDebateUser(vf.User):
    async def get_response(
        self,
        task: vf.Task,
        state: vf.State,
        messages: list[vf.Message],
    ) -> list[vf.UserMessage]:
        actor = str(state.get("debate_actor") or ARGUER)
        tag = "argument" if actor == ARGUER else "critique"
        assistant_messages = vf.get_messages(messages, role="assistant")
        last_text = str(assistant_messages[-1].content or "") if assistant_messages else ""
        match = re.search(rf"<{tag}>(.*?)</{tag}>", last_text, re.DOTALL)
        handoff = match.group(1).strip() if match else ""
        if not handoff:
            state["rollout_completed_cleanly"] = False
            state["malformed_handoff"] = {
                "actor": actor,
                "reason": f"Expected <{tag}>...</{tag}>.",
            }
            return []

        history = list(state.get("handoff_history") or [])
        history.append({"actor": actor, "handoff": {tag: handoff}})
        state["handoff_history"] = history
        question = str(task["question"])
        num_rounds = int(task["num_rounds"])

        if actor == ARGUER:
            state["current_argument"] = handoff
            if len(history) >= 2 * num_rounds + 1:
                state["final_argument"] = handoff
                state["final_env_response"] = "Debate complete."
                return []
            state["debate_actor"] = CRITIC
            return [
                vf.UserMessage(content=CRITIC_PROMPT.format(question=question, argument=handoff))
            ]

        state["current_critique"] = handoff
        state["debate_actor"] = ARGUER
        return [vf.UserMessage(content=REFINE_PROMPT.format(critique=handoff))]


class EthicsDebateTaskset(vf.Taskset[EthicsDebateTasksetConfig]):
    def load_user(self, config: vf.UserConfig) -> EthicsDebateUser:
        return EthicsDebateUser(config=config)

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        source_split = self.config.train_split if split == "train" else self.config.eval_split
        dataset = load_dataset(self.config.dataset_name, split=source_split)
        num_rounds = self.config.num_rounds
        return dataset.map(
            lambda row: {
                "question": str(row["question"]),
                "num_rounds": num_rounds,
                "prompt": [
                    {
                        "role": "user",
                        "content": ARGUER_PROMPT.format(question=str(row["question"])),
                    }
                ],
            }
        )

    @vf.setup
    async def setup_debate(self, state: vf.State) -> None:
        state["debate_actor"] = ARGUER
        state["handoff_history"] = []
        state["current_argument"] = None
        state["current_critique"] = None
        state["final_argument"] = None
        state["rollout_completed_cleanly"] = True

    @vf.reward(weight=1.0)
    async def argument_quality(self, task: vf.Task, state: vf.State) -> float:
        final_argument = state.get("final_argument")
        if not final_argument:
            return 0.0
        client = cast(AsyncOpenAI, state.get_client(api="chat"))
        endpoint = state.get_endpoint_config(api="chat")
        response = await client.chat.completions.create(
            model=endpoint.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        question=task["question"],
                        argument=final_argument,
                    ),
                },
            ],
        )
        text = response.choices[0].message.content or ""
        numbers = re.findall(r"\b(10|[0-9])\b", text)
        return float(numbers[0]) / 10.0 if numbers else 0.0


def load_taskset(config: EthicsDebateTasksetConfig) -> EthicsDebateTaskset:
    return EthicsDebateTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

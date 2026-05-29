import os
import re

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


class EthicsDebateUserConfig(vf.UserConfig):
    pass


class EthicsDebateTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "ergotts/ethics_questions"
    dataset_split: str = "train"
    num_rounds: int = 2
    user: EthicsDebateUserConfig | None = EthicsDebateUserConfig()
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = (
        "You are participating in a structured ethics debate. Follow the current "
        "role assignment in each user message and end every turn with the requested "
        "XML tag."
    )


def content_text(content: str | list | None, separator: str = "\n") -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if part.get("type") == "text" and isinstance(text, str):
                    chunks.append(text)
        return separator.join(chunks).strip()
    return ""


def last_assistant_text(transcript: list[vf.Message]) -> str:
    for message in reversed(transcript):
        if message.role == "assistant":
            return content_text(message.content)
    return ""


def handoff_tag(actor: str) -> str:
    return "argument" if actor == ARGUER else "critique"


def parse_handoff(actor: str, text: str) -> str | None:
    tag = handoff_tag(actor)
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    if match is None:
        return None
    content = match.group(1).strip()
    return content or None


def turn_contract(tag: str) -> str:
    return (
        f"IMPORTANT: When you are done, wrap your final response in <{tag}>...</{tag}> tags. "
        f"Do NOT write anything after the closing </{tag}> tag."
    )


def arguer_message(question: str) -> str:
    return (
        "You are arguing an ethics question. Present a clear position with reasoning. "
        "When critiqued, address the weaknesses and strengthen your argument.\n\n"
        f"Question: {question}\n\n{turn_contract('argument')}"
    )


def critic_message(question: str, argument: str) -> str:
    return (
        "You are critiquing an ethical argument. Identify gaps, logical fallacies, "
        "and missing perspectives. Be specific and constructive.\n\n"
        f"Question: {question}\n\nArgument to critique:\n{argument}\n\n"
        f"{turn_contract('critique')}"
    )


def refine_message(critique: str) -> str:
    return (
        "You are arguing an ethics question. Present a clear position with reasoning. "
        "When critiqued, address the weaknesses and strengthen your argument.\n\n"
        f"Critique received:\n{critique}\n\nRefine your argument.\n\n"
        f"{turn_contract('argument')}"
    )


class EthicsDebateUser(vf.User[EthicsDebateUserConfig]):
    async def get_response(
        self,
        task: vf.Task,
        state: vf.State,
        messages: list[vf.Message],
    ) -> list[vf.UserMessage]:
        actor = str(state.get("debate_actor") or ARGUER)
        handoff = parse_handoff(actor, last_assistant_text(messages))
        if handoff is None:
            tag = handoff_tag(actor)
            state["rollout_completed_cleanly"] = False
            state["malformed_handoff"] = {
                "actor": actor,
                "reason": f"Expected <{tag}>...</{tag}>.",
            }
            return []

        history = list(state.get("handoff_history") or [])
        history.append({"actor": actor, "handoff": {handoff_tag(actor): handoff}})
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
            return [vf.UserMessage(content=critic_message(question, handoff))]

        state["current_critique"] = handoff
        state["debate_actor"] = ARGUER
        return [vf.UserMessage(content=refine_message(handoff))]


class EthicsDebateTaskset(vf.Taskset[EthicsDebateTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        _ = split
        dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        num_rounds = self.config.num_rounds
        for index, row in enumerate(dataset):
            if not isinstance(row, dict):
                raise TypeError("Dataset rows must be dicts.")
            question = str(row["question"])
            yield {
                "example_id": index,
                "question": question,
                "num_rounds": num_rounds,
                "prompt": [{"role": "user", "content": arguer_message(question)}],
                "max_turns": 2 * num_rounds + 1,
            }

    @vf.setup
    async def setup_debate(self, task: vf.Task, state: vf.State) -> None:
        _ = task
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
        endpoint = state.get_endpoint_config(api="chat")
        client = AsyncOpenAI(
            api_key=os.getenv(endpoint.api_key_var, ""),
            base_url=endpoint.base_url,
        )
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
        harness=vf.Harness(config=config.harness),
    )

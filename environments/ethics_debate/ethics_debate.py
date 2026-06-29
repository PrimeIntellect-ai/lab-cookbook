import re
from typing import cast

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.dialects import ChatDialect

ARGUER = "arguer"
CRITIC = "critic"

SYSTEM = (
    "You are participating in a structured ethics debate. Follow the current role "
    "assignment in each user message and end every turn with the requested XML tag."
)
JUDGE_SYSTEM = (
    "You are a philosophy professor grading ethics arguments. Be rigorous and critical. "
    "A generic, well-intentioned argument scores 3-5. Only arguments showing genuine "
    "philosophical rigor score 7+."
)
JUDGE_USER_TEMPLATE = """Question: {question}

Argument:
{argument}

Score each dimension 0-2 (0=absent, 1=superficial, 2=thorough):

1. THESIS CLARITY
2. LOGICAL STRUCTURE
3. COUNTERARGUMENT HANDLING
4. NUANCE
5. DEPTH

Sum the five scores (0-10). Respond with only the total number."""

ARGUER_PROMPT = """You are arguing an ethics question. Present a clear position with reasoning.

Question: {question}

When you are done, wrap your final response in <argument>...</argument> tags. Do not write anything after the closing tag."""

CRITIC_PROMPT = """You are critiquing an ethical argument. Identify gaps, logical fallacies, and missing perspectives. Be specific and constructive.

Question: {question}

Argument to critique:
{argument}

When you are done, wrap your final response in <critique>...</critique> tags. Do not write anything after the closing tag."""

REFINE_PROMPT = """You are arguing an ethics question. Address the critique and strengthen your argument.

Critique received:
{critique}

When you are done, wrap your final response in <argument>...</argument> tags. Do not write anything after the closing tag."""

HandoffRecord = dict[str, str | dict[str, str]]


class EthicsDebateState(vf.State):
    debate_actor: str = ARGUER
    handoff_history: list[HandoffRecord] = Field(default_factory=list)
    current_argument: str = ""
    current_critique: str = ""
    final_argument: str = ""
    rollout_completed_cleanly: bool = True
    malformed_handoff: dict[str, str] | None = None


class EthicsDebateTask(vf.Task):
    question: str
    num_rounds: int


class JudgeConfig(vf.BaseClientConfig):
    model: str = "openai/gpt-4.1-mini"


class EthicsDebateConfig(vf.TasksetConfig):
    dataset_name: str = "ergotts/ethics_questions"
    dataset_split: str = "train"
    num_rounds: int = 2
    judge: JudgeConfig = JudgeConfig()
    user: vf.UserConfig = vf.UserConfig()


class EthicsDebateUser(vf.User[vf.UserConfig, EthicsDebateState]):
    async def setup_task(self, task: EthicsDebateTask) -> None:
        self.question = task.question
        self.num_rounds = task.num_rounds

    async def respond(self, message: str) -> vf.Messages:
        actor = self.state.debate_actor
        tag = "argument" if actor == ARGUER else "critique"
        match = re.search(rf"<{tag}>(.*?)</{tag}>", message, re.DOTALL)
        handoff = match.group(1).strip() if match else ""
        if not handoff:
            self.state.rollout_completed_cleanly = False
            self.state.malformed_handoff = {
                "actor": actor,
                "reason": f"Expected <{tag}>...</{tag}>.",
            }
            return []

        self.state.handoff_history.append({"actor": actor, "handoff": {tag: handoff}})
        if actor == ARGUER:
            self.state.current_argument = handoff
            if len(self.state.handoff_history) >= 2 * self.num_rounds + 1:
                self.state.final_argument = handoff
                return []
            self.state.debate_actor = CRITIC
            return [
                vf.UserMessage(
                    content=CRITIC_PROMPT.format(question=self.question, argument=handoff)
                )
            ]

        self.state.current_critique = handoff
        self.state.debate_actor = ARGUER
        return [vf.UserMessage(content=REFINE_PROMPT.format(critique=handoff))]


class EthicsDebateTaskset(vf.Taskset[EthicsDebateTask, EthicsDebateConfig, EthicsDebateState]):
    def load_tasks(self) -> list[EthicsDebateTask]:
        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        return [
            EthicsDebateTask(
                idx=i,
                prompt=ARGUER_PROMPT.format(question=str(row["question"])),
                system_prompt=SYSTEM,
                question=str(row["question"]),
                num_rounds=self.config.num_rounds,
            )
            for i, row in enumerate(rows)
        ]

    def user(self, task: EthicsDebateTask) -> vf.User:
        _ = task
        return cast(vf.User, EthicsDebateUser(self.config.user))

    @vf.reward(weight=1.0)
    async def argument_quality(
        self,
        task: EthicsDebateTask,
        trace: vf.Trace[EthicsDebateTask, EthicsDebateState],
    ) -> float:
        if not trace.state.final_argument:
            return 0.0
        prompt = JUDGE_USER_TEMPLATE.format(
            question=task.question,
            argument=trace.state.final_argument,
        )
        client = vf.resolve_client(self.config.judge)
        try:
            verdict = await client.get_response(
                ChatDialect(),
                {
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ]
                },
                self.config.judge.model,
                vf.SamplingConfig(),
            )
        finally:
            await client.close()
        numbers = re.findall(r"\b(10|[0-9])\b", verdict.message.content or "")
        return float(numbers[0]) / 10.0 if numbers else 0.0


if __name__ == "__main__":
    EthicsDebateUser.run()


__all__ = ["EthicsDebateTaskset"]

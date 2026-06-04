import json

import verifiers as vf
from calendar_problem import (
    CalendarTask,
    GenerationOverrides,
    JsonObject,
    build_example,
    day_label,
    evaluate_proposal,
    format_blocks_for_day,
    generate_validated_task,
    make_proposal,
)
from pydantic import Field

SYSTEM_PROMPT = """You are scheduling a meeting in a constrained calendar environment.

Use tools deliberately:
- inspect attendee calendars and attendee constraints,
- check a small number of candidate windows,
- submit exactly one final window with submit_window.

Important:
- hard constraints must be satisfied or final score is 0.0,
- soft constraints lower attendee utilities,
- score-check calls are budgeted, so avoid brute-force probing,
- tool responses include remaining_turns.
- if you do not call submit_window before turns end, final reward is 0.0.
- when remaining_turns <= 2, immediately submit your best candidate.

Call signature reminder:
- check_proposal(day_index=<int>, start_time_utc=<HH:MM>, duration_minutes=<int>, room_id=<str>)
- submit_window(day_index=<int>, start_time_utc=<HH:MM>, duration_minutes=<int>, room_id=<str>)
"""


class CalendarSchedulingTasksetConfig(vf.TasksetConfig):
    difficulty: str = "medium"
    seed: int = 7
    num_train: int = 512
    num_eval: int = 128
    generator_overrides: GenerationOverrides = Field(default_factory=GenerationOverrides)
    system_prompt: vf.SystemPrompt = SYSTEM_PROMPT


class CalendarSchedulingTaskset(vf.Taskset[CalendarSchedulingTasksetConfig]):
    @vf.setup
    async def setup_calendar(self, task: vf.Task, state: vf.State) -> None:
        calendar_task = CalendarTask.from_task(task)
        state["score_checks_remaining"] = int(calendar_task.score_check_budget)
        state["score_checks_used"] = 0
        state["proposal_checks"] = []
        state["submitted"] = False
        state["submitted_valid"] = False
        state["submitted_score"] = 0.0
        state["submitted_payload"] = None

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "train":
            num_examples = self.config.num_train
            seed = self.config.seed
        else:
            num_examples = self.config.num_eval
            seed = self.config.seed + 1_000_003
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")
        tasks: list[vf.Task] = []
        for index in range(num_examples):
            task_seed = seed + (index * 1009)
            task, summary, config = generate_validated_task(
                seed=task_seed,
                difficulty=self.config.difficulty,
                overrides=self.config.generator_overrides,
            )
            tasks.append(build_example(task, summary, config))
        return tasks

    def load_toolsets(self, config: CalendarSchedulingTasksetConfig) -> vf.Toolsets:
        _ = config
        return {"calendar": CalendarSchedulingToolset.create()}

    @vf.stop(priority=50)
    async def has_submission(self, state: vf.State) -> bool:
        return bool(state.get("submitted", False))

    @vf.reward(weight=1.0)
    async def final_score_from_submission(self, state: vf.State) -> float:
        if not bool(state.get("submitted_valid", False)):
            return 0.0
        return float(state.get("submitted_score", 0.0))

    @vf.metric
    async def submission_made(self, state: vf.State) -> float:
        return 1.0 if bool(state.get("submitted", False)) else 0.0

    @vf.metric
    async def submission_valid(self, state: vf.State) -> float:
        return 1.0 if bool(state.get("submitted_valid", False)) else 0.0

    @vf.metric
    async def oracle_optimal_score(self, answer: object) -> float:
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                return 0.0
        if not isinstance(answer, dict):
            return 0.0
        score = {str(key): value for key, value in answer.items()}.get("optimal_score")
        if not isinstance(score, (int, float)):
            return 0.0
        return float(score)

    @vf.metric
    async def submitted_to_optimal_ratio(self, state: vf.State, answer: object) -> float:
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                return 0.0
        if not isinstance(answer, dict):
            return 0.0
        score = {str(key): value for key, value in answer.items()}.get("optimal_score")
        if not isinstance(score, (int, float)):
            return 0.0
        optimal = float(score)
        if optimal <= 0:
            return 0.0
        submitted_score = float(state.get("submitted_score", 0.0))
        return min(1.0, max(0.0, submitted_score / optimal))

    @vf.metric
    async def optimality_gap(self, state: vf.State, answer: object) -> float:
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                return 0.0
        if not isinstance(answer, dict):
            return 0.0
        score = {str(key): value for key, value in answer.items()}.get("optimal_score")
        if not isinstance(score, (int, float)):
            return 0.0
        submitted_score = float(state.get("submitted_score", 0.0))
        return max(0.0, float(score) - submitted_score)

    @vf.metric
    async def score_checks_used(self, state: vf.State) -> float:
        return float(state.get("score_checks_used", 0))

    @vf.metric
    async def score_checks_remaining(self, state: vf.State) -> float:
        return float(state.get("score_checks_remaining", 0))


class CalendarSchedulingToolset(vf.Toolset):
    @classmethod
    def create(cls) -> "CalendarSchedulingToolset":
        return cls(
            tools=[
                cls.check_attendee_calendar,
                cls.view_attendee_constraints,
                cls.check_proposal,
                cls.submit_window,
            ]
        )

    @staticmethod
    async def check_attendee_calendar(
        attendee_id: str,
        day_index: int,
        task: vf.Task,
        state: vf.State,
    ) -> str:
        """Inspect one attendee's busy calendar blocks.

        Args:
            attendee_id: Attendee identifier (for example attendee_1).
            day_index: Day index to inspect. Use -1 to return all days.

        Returns:
            JSON with busy intervals and turn/check budgets.
        """

        calendar_task = CalendarTask.from_task(task)
        attendee = next(
            (
                attendee
                for attendee in calendar_task.attendees
                if attendee.attendee_id == attendee_id.strip()
            ),
            None,
        )
        trajectory = state.get("trajectory", [])
        if not isinstance(trajectory, list):
            raise ValueError("Rollout state trajectory must be a list")
        remaining_turns = max(0, state.get_max_turns(10) - len(trajectory))
        payload: JsonObject = {
            "remaining_turns": remaining_turns,
            "score_checks_remaining": int(state.get("score_checks_remaining", 0)),
            "tool": "check_attendee_calendar",
            "attendee_id": attendee_id,
        }
        if not bool(state.get("submitted", False)) and remaining_turns <= 2:
            payload["warning"] = (
                "Submit now with submit_window. No submission before turns end gives score 0.0"
            )

        if attendee is None:
            payload["error"] = "Unknown attendee_id"
            payload["known_attendees"] = [
                attendee.attendee_id for attendee in calendar_task.attendees
            ]
            return json.dumps(payload, indent=2, sort_keys=True)

        payload["attendee"] = {
            "display_name": attendee.display_name,
            "timezone_offset_hours": attendee.timezone_offset_hours,
        }

        if day_index == -1:
            payload["days"] = {
                str(index): {
                    "label": day_label(index),
                    "busy_windows": format_blocks_for_day(
                        attendee.busy_blocks,
                        day_index=index,
                        slots_per_hour=calendar_task.slots_per_hour,
                    ),
                }
                for index in range(calendar_task.num_days)
            }
            return json.dumps(payload, indent=2, sort_keys=True)

        if day_index < 0 or day_index >= calendar_task.num_days:
            payload["error"] = f"day_index must be in [0, {calendar_task.num_days - 1}] or -1"
            return json.dumps(payload, indent=2, sort_keys=True)

        payload["day"] = {
            "day_index": day_index,
            "label": day_label(day_index),
            "busy_windows": format_blocks_for_day(
                attendee.busy_blocks,
                day_index=day_index,
                slots_per_hour=calendar_task.slots_per_hour,
            ),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    async def view_attendee_constraints(
        attendee_id: str,
        task: vf.Task,
        state: vf.State,
    ) -> str:
        """View hard and soft constraints for one attendee.

        Args:
            attendee_id: Attendee identifier (for example attendee_3).

        Returns:
            JSON with weighted importance and preference penalties.
        """

        calendar_task = CalendarTask.from_task(task)
        attendee = next(
            (
                attendee
                for attendee in calendar_task.attendees
                if attendee.attendee_id == attendee_id.strip()
            ),
            None,
        )
        trajectory = state.get("trajectory", [])
        if not isinstance(trajectory, list):
            raise ValueError("Rollout state trajectory must be a list")
        remaining_turns = max(0, state.get_max_turns(10) - len(trajectory))
        payload: JsonObject = {
            "remaining_turns": remaining_turns,
            "score_checks_remaining": int(state.get("score_checks_remaining", 0)),
            "tool": "view_attendee_constraints",
            "attendee_id": attendee_id,
        }
        if not bool(state.get("submitted", False)) and remaining_turns <= 2:
            payload["warning"] = (
                "Submit now with submit_window. No submission before turns end gives score 0.0"
            )

        if attendee is None:
            payload["error"] = "Unknown attendee_id"
            payload["known_attendees"] = [
                attendee.attendee_id for attendee in calendar_task.attendees
            ]
            return json.dumps(payload, indent=2, sort_keys=True)

        payload["attendee"] = {
            "display_name": attendee.display_name,
            "required": attendee.required,
            "weight": round(attendee.weight, 4),
            "timezone_offset_hours": attendee.timezone_offset_hours,
            "preferred_day": {
                "day_index": attendee.preferred_day,
                "label": day_label(attendee.preferred_day),
            },
            "preferred_local_time": {
                "start_hour": round(attendee.preferred_start_hour, 2),
                "end_hour": round(attendee.preferred_end_hour, 2),
            },
            "hard_local_time": {
                "start_hour": (
                    None if attendee.hard_start_hour is None else round(attendee.hard_start_hour, 2)
                ),
                "end_hour": (
                    None if attendee.hard_end_hour is None else round(attendee.hard_end_hour, 2)
                ),
            },
            "soft_penalties": {
                "early_penalty_per_hour": round(attendee.early_penalty_per_hour, 3),
                "late_penalty_per_hour": round(attendee.late_penalty_per_hour, 3),
                "day_penalty_per_day": round(attendee.day_penalty_per_day, 3),
                "back_to_back_penalty": round(attendee.back_to_back_penalty, 3),
                "optional_absence_penalty": round(attendee.optional_absence_penalty, 3),
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    async def check_proposal(
        day_index: int,
        start_time_utc: str,
        duration_minutes: int,
        room_id: str,
        task: vf.Task,
        state: vf.State,
    ) -> str:
        """Check score and hard-constraint status for a candidate window.

        Args:
            day_index: Day index in the planning window.
            start_time_utc: UTC start time in HH:MM format.
            duration_minutes: Proposed duration in minutes.
            room_id: Room identifier.

        Returns:
            JSON with validity, score, attendee utility details, and budgets.
        """

        calendar_task = CalendarTask.from_task(task)
        trajectory = state.get("trajectory", [])
        if not isinstance(trajectory, list):
            raise ValueError("Rollout state trajectory must be a list")
        remaining_turns = max(0, state.get_max_turns(10) - len(trajectory))
        payload: JsonObject = {
            "remaining_turns": remaining_turns,
            "score_checks_remaining": int(state.get("score_checks_remaining", 0)),
            "tool": "check_proposal",
            "proposal_input": {
                "day_index": day_index,
                "start_time_utc": start_time_utc,
                "duration_minutes": duration_minutes,
                "room_id": room_id,
            },
        }
        if not bool(state.get("submitted", False)) and remaining_turns <= 2:
            payload["warning"] = (
                "Submit now with submit_window. No submission before turns end gives score 0.0"
            )

        if start_time_utc.strip() == "":
            payload["error"] = "start_time_utc is required (HH:MM)"
            payload["valid"] = False
            payload["score"] = 0.0
            return json.dumps(payload, indent=2, sort_keys=True)
        if duration_minutes <= 0:
            payload["error"] = "duration_minutes must be positive"
            payload["valid"] = False
            payload["score"] = 0.0
            return json.dumps(payload, indent=2, sort_keys=True)
        if room_id.strip() == "":
            payload["error"] = "room_id is required"
            payload["valid"] = False
            payload["score"] = 0.0
            return json.dumps(payload, indent=2, sort_keys=True)

        proposal, error = make_proposal(
            task=calendar_task,
            day_index=day_index,
            start_time_utc=start_time_utc,
            duration_minutes=duration_minutes,
            room_id=room_id,
        )
        if proposal is None:
            payload["error"] = error
            payload["valid"] = False
            payload["score"] = 0.0
            return json.dumps(payload, indent=2, sort_keys=True)

        remaining_checks = int(state.get("score_checks_remaining", 0))
        if remaining_checks <= 0:
            payload["error"] = "score-check budget exhausted"
            payload["valid"] = False
            payload["score"] = 0.0
            return json.dumps(payload, indent=2, sort_keys=True)

        evaluation = evaluate_proposal(calendar_task, proposal)
        state["score_checks_remaining"] = remaining_checks - 1
        state["score_checks_used"] = int(state.get("score_checks_used", 0)) + 1
        checks = state.get("proposal_checks")
        if isinstance(checks, list):
            checks.append(
                {
                    "day_index": day_index,
                    "start_time_utc": start_time_utc,
                    "duration_minutes": duration_minutes,
                    "room_id": room_id,
                    "score": evaluation.score,
                    "valid": evaluation.valid,
                }
            )

        payload["score_checks_remaining"] = int(state["score_checks_remaining"])
        payload.update(evaluation.to_dict(calendar_task))
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    async def submit_window(
        day_index: int,
        start_time_utc: str,
        duration_minutes: int,
        room_id: str,
        task: vf.Task,
        state: vf.State,
    ) -> str:
        """Submit the final meeting window for scoring.

        Args:
            day_index: Day index in the planning window.
            start_time_utc: UTC start time in HH:MM format.
            duration_minutes: Proposed duration in minutes.
            room_id: Room identifier.

        Returns:
            JSON with the final accepted/invalid result and score.
        """

        calendar_task = CalendarTask.from_task(task)
        trajectory = state.get("trajectory", [])
        if not isinstance(trajectory, list):
            raise ValueError("Rollout state trajectory must be a list")
        remaining_turns = max(0, state.get_max_turns(10) - len(trajectory))
        payload: JsonObject = {
            "remaining_turns": remaining_turns,
            "score_checks_remaining": int(state.get("score_checks_remaining", 0)),
            "tool": "submit_window",
            "proposal_input": {
                "day_index": day_index,
                "start_time_utc": start_time_utc,
                "duration_minutes": duration_minutes,
                "room_id": room_id,
            },
        }

        error = ""
        proposal = None
        if start_time_utc.strip() == "":
            error = "start_time_utc is required (HH:MM)"
        elif duration_minutes <= 0:
            error = "duration_minutes must be positive"
        elif room_id.strip() == "":
            error = "room_id is required"
        else:
            proposal, error = make_proposal(
                task=calendar_task,
                day_index=day_index,
                start_time_utc=start_time_utc,
                duration_minutes=duration_minutes,
                room_id=room_id,
            )

        state["submitted"] = True
        state.stop("submitted")

        if proposal is None:
            error = error or "Invalid proposal"
            state["submitted_valid"] = False
            state["submitted_score"] = 0.0
            state["submitted_payload"] = {
                "valid": False,
                "score": 0.0,
                "hard_violations": [error],
            }
            payload["submitted"] = True
            payload["valid"] = False
            payload["score"] = 0.0
            payload["hard_violations"] = [error]
            return json.dumps(payload, indent=2, sort_keys=True)

        evaluation = evaluate_proposal(calendar_task, proposal)
        state["submitted_valid"] = evaluation.valid
        state["submitted_score"] = evaluation.score if evaluation.valid else 0.0
        state["submitted_payload"] = evaluation.to_dict(calendar_task, rounded=False)

        payload["submitted"] = True
        payload.update(evaluation.to_dict(calendar_task))
        return json.dumps(payload, indent=2, sort_keys=True)


def load_taskset(config: CalendarSchedulingTasksetConfig) -> CalendarSchedulingTaskset:
    return CalendarSchedulingTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )

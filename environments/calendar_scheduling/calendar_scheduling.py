import json
from typing import cast

import verifiers.v1 as vf
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

SYSTEM = """You are scheduling a meeting in a constrained calendar environment.

Use tools deliberately:
- inspect attendee calendars and attendee constraints,
- check a small number of candidate windows,
- submit exactly one final window with submit_window.

Hard constraints must be satisfied or final score is 0.0. Score-check calls are budgeted, so avoid brute-force probing. If you do not call submit_window before turns end, final reward is 0.0.
"""


class CalendarSchedulingTask(vf.Task):
    answer: str
    calendar_task: JsonObject


class CalendarState(vf.State):
    score_checks_remaining: int = -1
    score_checks_used: int = 0
    proposal_checks: list[JsonObject] = Field(default_factory=list)
    submitted: bool = False
    submitted_valid: bool = False
    submitted_score: float = 0.0
    submitted_payload: JsonObject | None = None


class CalendarToolConfig(vf.ToolsetConfig):
    pass


class CalendarSchedulingConfig(vf.TasksetConfig):
    difficulty: str = "medium"
    seed: int = 7
    num_tasks: int = 512
    generator_overrides: GenerationOverrides = Field(default_factory=GenerationOverrides)
    tools: CalendarToolConfig = CalendarToolConfig()


def optimal_score(answer: str) -> float:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return 0.0
    score = payload.get("optimal_score")
    return float(score) if isinstance(score, int | float) else 0.0


class CalendarToolset(vf.Toolset[CalendarToolConfig, CalendarState]):
    TOOL_PREFIX = "calendar"

    async def setup_task(self, task: CalendarSchedulingTask) -> None:
        self.calendar_task = CalendarTask.from_task(task)

    def _task(self) -> CalendarTask:
        return self.calendar_task

    def _ensure_state(self) -> None:
        if self.state.score_checks_remaining < 0:
            self.state.score_checks_remaining = int(self._task().score_check_budget)

    @vf.tool
    async def check_attendee_calendar(self, attendee_id: str, day_index: int) -> str:
        """Inspect one attendee's busy calendar blocks. Use day_index=-1 for all days."""
        self._ensure_state()
        calendar_task = self._task()
        attendee = next(
            (
                attendee
                for attendee in calendar_task.attendees
                if attendee.attendee_id == attendee_id.strip()
            ),
            None,
        )
        payload: JsonObject = {
            "score_checks_remaining": self.state.score_checks_remaining,
            "tool": "check_attendee_calendar",
            "attendee_id": attendee_id,
        }
        if attendee is None:
            payload["error"] = "Unknown attendee_id"
            payload["known_attendees"] = [a.attendee_id for a in calendar_task.attendees]
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

    @vf.tool
    async def view_attendee_constraints(self, attendee_id: str) -> str:
        """View hard and soft constraints for one attendee."""
        self._ensure_state()
        calendar_task = self._task()
        attendee = next(
            (
                attendee
                for attendee in calendar_task.attendees
                if attendee.attendee_id == attendee_id.strip()
            ),
            None,
        )
        payload: JsonObject = {
            "score_checks_remaining": self.state.score_checks_remaining,
            "tool": "view_attendee_constraints",
            "attendee_id": attendee_id,
        }
        if attendee is None:
            payload["error"] = "Unknown attendee_id"
            payload["known_attendees"] = [a.attendee_id for a in calendar_task.attendees]
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

    @vf.tool
    async def check_proposal(
        self,
        day_index: int,
        start_time_utc: str,
        duration_minutes: int,
        room_id: str,
    ) -> str:
        """Check score and hard-constraint status for a candidate window."""
        self._ensure_state()
        calendar_task = self._task()
        payload: JsonObject = {
            "score_checks_remaining": self.state.score_checks_remaining,
            "tool": "check_proposal",
            "proposal_input": {
                "day_index": day_index,
                "start_time_utc": start_time_utc,
                "duration_minutes": duration_minutes,
                "room_id": room_id,
            },
        }
        if not start_time_utc.strip() or duration_minutes <= 0 or not room_id.strip():
            payload["error"] = "start_time_utc, positive duration_minutes, and room_id are required"
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
        if self.state.score_checks_remaining <= 0:
            payload["error"] = "score-check budget exhausted"
            payload["valid"] = False
            payload["score"] = 0.0
            return json.dumps(payload, indent=2, sort_keys=True)
        evaluation = evaluate_proposal(calendar_task, proposal)
        self.state.score_checks_remaining -= 1
        self.state.score_checks_used += 1
        self.state.proposal_checks.append(
            {
                "day_index": day_index,
                "start_time_utc": start_time_utc,
                "duration_minutes": duration_minutes,
                "room_id": room_id,
                "score": evaluation.score,
                "valid": evaluation.valid,
            }
        )
        payload["score_checks_remaining"] = self.state.score_checks_remaining
        payload.update(evaluation.to_dict(calendar_task))
        return json.dumps(payload, indent=2, sort_keys=True)

    @vf.tool
    async def submit_window(
        self,
        day_index: int,
        start_time_utc: str,
        duration_minutes: int,
        room_id: str,
    ) -> str:
        """Submit the final meeting window for scoring."""
        self._ensure_state()
        calendar_task = self._task()
        payload: JsonObject = {
            "score_checks_remaining": self.state.score_checks_remaining,
            "tool": "submit_window",
            "proposal_input": {
                "day_index": day_index,
                "start_time_utc": start_time_utc,
                "duration_minutes": duration_minutes,
                "room_id": room_id,
            },
        }
        proposal, error = make_proposal(
            task=calendar_task,
            day_index=day_index,
            start_time_utc=start_time_utc,
            duration_minutes=duration_minutes,
            room_id=room_id,
        )
        self.state.submitted = True
        if proposal is None:
            self.state.submitted_valid = False
            self.state.submitted_score = 0.0
            self.state.submitted_payload = {
                "valid": False,
                "score": 0.0,
                "hard_violations": [error or "Invalid proposal"],
            }
            payload.update(self.state.submitted_payload)
            return json.dumps(payload, indent=2, sort_keys=True)
        evaluation = evaluate_proposal(calendar_task, proposal)
        self.state.submitted_valid = evaluation.valid
        self.state.submitted_score = evaluation.score if evaluation.valid else 0.0
        self.state.submitted_payload = evaluation.to_dict(calendar_task, rounded=False)
        payload["submitted"] = True
        payload.update(evaluation.to_dict(calendar_task))
        return json.dumps(payload, indent=2, sort_keys=True)


class CalendarSchedulingTaskset(
    vf.Taskset[CalendarSchedulingTask, CalendarSchedulingConfig, CalendarState]
):
    def load_tasks(self) -> list[CalendarSchedulingTask]:
        tasks: list[CalendarSchedulingTask] = []
        if self.config.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        for index in range(self.config.num_tasks):
            task_seed = self.config.seed + (index * 1009)
            task, summary, config = generate_validated_task(
                seed=task_seed,
                difficulty=self.config.difficulty,
                overrides=self.config.generator_overrides,
            )
            raw = build_example(task, summary, config)
            info = raw["info"]
            if not isinstance(info, dict):
                raise TypeError("calendar task info must be a dict")
            tasks.append(
                CalendarSchedulingTask(
                    idx=index,
                    prompt=str(raw["prompt"]),
                    system_prompt=SYSTEM,
                    answer=str(raw["answer"]),
                    calendar_task=info["calendar_task"],
                )
            )
        return tasks

    def tools(self, task: CalendarSchedulingTask) -> list[vf.Toolset]:
        _ = task
        return [cast(vf.Toolset, CalendarToolset(self.config.tools))]

    @vf.stop(priority=50)
    async def has_submission(self, trace: vf.Trace[CalendarSchedulingTask, CalendarState]) -> bool:
        return trace.state.submitted

    @vf.reward(weight=1.0)
    async def final_score_from_submission(
        self, trace: vf.Trace[CalendarSchedulingTask, CalendarState]
    ) -> float:
        if not trace.state.submitted_valid:
            return 0.0
        return trace.state.submitted_score

    @vf.metric
    async def submission_made(
        self, trace: vf.Trace[CalendarSchedulingTask, CalendarState]
    ) -> float:
        return float(trace.state.submitted)

    @vf.metric
    async def submission_valid(
        self, trace: vf.Trace[CalendarSchedulingTask, CalendarState]
    ) -> float:
        return float(trace.state.submitted_valid)

    @vf.metric
    async def oracle_optimal_score(self, task: CalendarSchedulingTask) -> float:
        return optimal_score(task.answer)

    @vf.metric
    async def submitted_to_optimal_ratio(
        self,
        task: CalendarSchedulingTask,
        trace: vf.Trace[CalendarSchedulingTask, CalendarState],
    ) -> float:
        optimal = optimal_score(task.answer)
        if optimal <= 0:
            return 0.0
        return min(1.0, max(0.0, trace.state.submitted_score / optimal))

    @vf.metric
    async def optimality_gap(
        self,
        task: CalendarSchedulingTask,
        trace: vf.Trace[CalendarSchedulingTask, CalendarState],
    ) -> float:
        return max(0.0, optimal_score(task.answer) - trace.state.submitted_score)

    @vf.metric
    async def score_checks_used(
        self, trace: vf.Trace[CalendarSchedulingTask, CalendarState]
    ) -> float:
        return float(trace.state.score_checks_used)

    @vf.metric
    async def score_checks_remaining(
        self, trace: vf.Trace[CalendarSchedulingTask, CalendarState]
    ) -> float:
        return float(max(trace.state.score_checks_remaining, 0))


if __name__ == "__main__":
    CalendarToolset.run()


__all__ = ["CalendarSchedulingTaskset"]

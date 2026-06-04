import json
import math
import random
import statistics
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable, cast

import verifiers as vf

BASE_DATE = date(2026, 1, 5)
SLOTS_PER_HOUR = 2
MINUTES_PER_SLOT = 60 // SLOTS_PER_HOUR
DAY_SLOTS = 24 * SLOTS_PER_HOUR
EPSILON = 1e-9

JsonObject = vf.JsonData


def _object(payload: object, label: str) -> JsonObject:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(JsonObject, {str(key): value for key, value in payload.items()})


def _required(data: JsonObject, key: str) -> object:
    if key not in data:
        raise ValueError(f"Missing required field '{key}'")
    return data[key]


def _str(data: JsonObject, key: str) -> str:
    value = _required(data, key)
    if not isinstance(value, str):
        raise ValueError(f"Field '{key}' must be a string")
    return value


def _int(data: JsonObject, key: str) -> int:
    value = _required(data, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Field '{key}' must be an integer")
    return value


def _float(data: JsonObject, key: str) -> float:
    value = _required(data, key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Field '{key}' must be numeric")
    return float(value)


def _optional_float(data: JsonObject, key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Field '{key}' must be numeric or null")
    return float(value)


def _bool(data: JsonObject, key: str) -> bool:
    value = _required(data, key)
    if not isinstance(value, bool):
        raise ValueError(f"Field '{key}' must be a boolean")
    return value


def _int_pair(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{label} must be a pair of integers")
    start, end = value
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        raise ValueError(f"{label} must be a pair of integers")
    return (start, end)


def _int_pairs(data: JsonObject, key: str) -> tuple[tuple[int, int], ...]:
    value = _required(data, key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"Field '{key}' must be a list of integer pairs")
    return tuple(_int_pair(item, f"{key}[{index}]") for index, item in enumerate(value))


def _objects(data: JsonObject, key: str) -> tuple[JsonObject, ...]:
    value = _required(data, key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"Field '{key}' must be a list of objects")
    return tuple(_object(item, f"{key}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class AttendeeSpec:
    attendee_id: str
    display_name: str
    timezone_offset_hours: int
    required: bool
    weight: float
    busy_blocks: tuple[tuple[int, int], ...]
    preferred_day: int
    preferred_start_hour: float
    preferred_end_hour: float
    hard_start_hour: float | None
    hard_end_hour: float | None
    early_penalty_per_hour: float
    late_penalty_per_hour: float
    day_penalty_per_day: float
    back_to_back_penalty: float
    optional_absence_penalty: float

    def to_dict(self) -> JsonObject:
        return {
            "attendee_id": self.attendee_id,
            "display_name": self.display_name,
            "timezone_offset_hours": self.timezone_offset_hours,
            "required": self.required,
            "weight": self.weight,
            "busy_blocks": [list(block) for block in self.busy_blocks],
            "preferred_day": self.preferred_day,
            "preferred_start_hour": self.preferred_start_hour,
            "preferred_end_hour": self.preferred_end_hour,
            "hard_start_hour": self.hard_start_hour,
            "hard_end_hour": self.hard_end_hour,
            "early_penalty_per_hour": self.early_penalty_per_hour,
            "late_penalty_per_hour": self.late_penalty_per_hour,
            "day_penalty_per_day": self.day_penalty_per_day,
            "back_to_back_penalty": self.back_to_back_penalty,
            "optional_absence_penalty": self.optional_absence_penalty,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "AttendeeSpec":
        data = _object(payload, "attendee")
        return cls(
            attendee_id=_str(data, "attendee_id"),
            display_name=_str(data, "display_name"),
            timezone_offset_hours=_int(data, "timezone_offset_hours"),
            required=_bool(data, "required"),
            weight=_float(data, "weight"),
            busy_blocks=_int_pairs(data, "busy_blocks"),
            preferred_day=_int(data, "preferred_day"),
            preferred_start_hour=_float(data, "preferred_start_hour"),
            preferred_end_hour=_float(data, "preferred_end_hour"),
            hard_start_hour=_optional_float(data, "hard_start_hour"),
            hard_end_hour=_optional_float(data, "hard_end_hour"),
            early_penalty_per_hour=_float(data, "early_penalty_per_hour"),
            late_penalty_per_hour=_float(data, "late_penalty_per_hour"),
            day_penalty_per_day=_float(data, "day_penalty_per_day"),
            back_to_back_penalty=_float(data, "back_to_back_penalty"),
            optional_absence_penalty=_float(data, "optional_absence_penalty"),
        )


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    display_name: str
    unavailable_blocks: tuple[tuple[int, int], ...]

    def to_dict(self) -> JsonObject:
        return {
            "room_id": self.room_id,
            "display_name": self.display_name,
            "unavailable_blocks": [list(block) for block in self.unavailable_blocks],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "RoomSpec":
        data = _object(payload, "room")
        return cls(
            room_id=_str(data, "room_id"),
            display_name=_str(data, "display_name"),
            unavailable_blocks=_int_pairs(data, "unavailable_blocks"),
        )


@dataclass(frozen=True)
class CalendarTask:
    task_id: str
    seed: int
    difficulty: str
    num_days: int
    slots_per_hour: int
    search_start_hour: int
    search_end_hour: int
    meeting_duration_slots: int
    score_check_budget: int
    attendees: tuple[AttendeeSpec, ...]
    rooms: tuple[RoomSpec, ...]

    def to_dict(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "num_days": self.num_days,
            "slots_per_hour": self.slots_per_hour,
            "search_start_hour": self.search_start_hour,
            "search_end_hour": self.search_end_hour,
            "meeting_duration_slots": self.meeting_duration_slots,
            "score_check_budget": self.score_check_budget,
            "attendees": [attendee.to_dict() for attendee in self.attendees],
            "rooms": [room.to_dict() for room in self.rooms],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "CalendarTask":
        data = _object(payload, "calendar_task")
        return cls(
            task_id=_str(data, "task_id"),
            seed=_int(data, "seed"),
            difficulty=_str(data, "difficulty"),
            num_days=_int(data, "num_days"),
            slots_per_hour=_int(data, "slots_per_hour"),
            search_start_hour=_int(data, "search_start_hour"),
            search_end_hour=_int(data, "search_end_hour"),
            meeting_duration_slots=_int(data, "meeting_duration_slots"),
            score_check_budget=_int(data, "score_check_budget"),
            attendees=tuple(
                AttendeeSpec.from_dict(attendee) for attendee in _objects(data, "attendees")
            ),
            rooms=tuple(RoomSpec.from_dict(room) for room in _objects(data, "rooms")),
        )

    @classmethod
    def from_task(cls, task: vf.Task) -> "CalendarTask":
        info = _object(task.get("info"), "task.info")
        if "calendar_task" not in info:
            raise ValueError("Task info is missing expected 'calendar_task' payload")
        return cls.from_dict(info["calendar_task"])


@dataclass(frozen=True)
class MeetingProposal:
    day_index: int
    start_slot_in_day: int
    duration_slots: int
    room_id: str

    def start_slot_absolute(self, task: CalendarTask) -> int:
        return self.day_index * (24 * task.slots_per_hour) + self.start_slot_in_day

    def to_dict(self, task: CalendarTask) -> JsonObject:
        return {
            "day_index": self.day_index,
            "day_label": day_label(self.day_index),
            "start_time_utc": slot_to_time(self.start_slot_in_day, task.slots_per_hour),
            "duration_minutes": self.duration_slots * (60 // task.slots_per_hour),
            "room_id": self.room_id,
            "window": proposal_label(task, self),
        }


@dataclass(frozen=True)
class ProposalEvaluation:
    proposal: MeetingProposal
    valid: bool
    score: float
    hard_violations: tuple[str, ...]
    attendee_utilities: dict[str, float]
    attendee_notes: dict[str, tuple[str, ...]]
    attendees_who_can_attend: tuple[str, ...]

    def to_dict(self, task: CalendarTask, rounded: bool = True) -> JsonObject:
        attendee_utilities = {
            key: (round(value, 4) if rounded else value)
            for key, value in self.attendee_utilities.items()
        }
        return {
            "proposal": self.proposal.to_dict(task),
            "valid": self.valid,
            "score": round(self.score, 4) if rounded else self.score,
            "hard_violations": list(self.hard_violations),
            "attendee_utilities": attendee_utilities,
            "attendee_notes": {key: list(notes) for key, notes in self.attendee_notes.items()},
            "attendees_who_can_attend": list(self.attendees_who_can_attend),
        }


@dataclass(frozen=True)
class SolverSummary:
    total_candidates: int
    valid_candidates: int
    best_score: float
    best_proposals: tuple[MeetingProposal, ...]
    random_baseline_score: float
    near_optimal_count: int

    def to_dict(self, task: CalendarTask, top_k: int = 3) -> JsonObject:
        return {
            "total_candidates": self.total_candidates,
            "valid_candidates": self.valid_candidates,
            "best_score": self.best_score,
            "random_baseline_score": self.random_baseline_score,
            "near_optimal_count": self.near_optimal_count,
            "best_proposals": [proposal.to_dict(task) for proposal in self.best_proposals[:top_k]],
        }


@dataclass(frozen=True)
class GenerationConfig:
    difficulty: str
    attendee_count_range: tuple[int, int]
    window_days_range: tuple[int, int]
    search_start_hour_range: tuple[int, int]
    search_end_hour_range: tuple[int, int]
    meeting_duration_choices: tuple[int, ...]
    timezone_offset_range: tuple[int, int]
    required_fraction_range: tuple[float, float]
    busy_blocks_per_attendee_range: tuple[int, int]
    busy_block_length_range: tuple[int, int]
    room_count_range: tuple[int, int]
    room_unavailable_blocks_range: tuple[int, int]
    room_unavailable_block_length_range: tuple[int, int]
    hard_time_probability: float
    hard_time_tolerance_hours_range: tuple[float, float]
    preferred_window_hours_range: tuple[float, float]
    early_penalty_range: tuple[float, float]
    late_penalty_range: tuple[float, float]
    day_penalty_range: tuple[float, float]
    back_to_back_penalty_range: tuple[float, float]
    optional_absence_penalty_range: tuple[float, float]
    score_check_budget: int
    min_valid_candidates: int
    max_valid_ratio: float
    max_random_baseline_score: float
    near_optimal_delta: float
    max_near_optimal_count: int
    min_best_score: float
    max_generation_attempts: int


class GenerationOverrides(vf.Config):
    attendee_count_range: tuple[int, int] | None = None
    window_days_range: tuple[int, int] | None = None
    search_start_hour_range: tuple[int, int] | None = None
    search_end_hour_range: tuple[int, int] | None = None
    meeting_duration_choices: tuple[int, ...] | None = None
    timezone_offset_range: tuple[int, int] | None = None
    required_fraction_range: tuple[float, float] | None = None
    busy_blocks_per_attendee_range: tuple[int, int] | None = None
    busy_block_length_range: tuple[int, int] | None = None
    room_count_range: tuple[int, int] | None = None
    room_unavailable_blocks_range: tuple[int, int] | None = None
    room_unavailable_block_length_range: tuple[int, int] | None = None
    hard_time_probability: float | None = None
    hard_time_tolerance_hours_range: tuple[float, float] | None = None
    preferred_window_hours_range: tuple[float, float] | None = None
    early_penalty_range: tuple[float, float] | None = None
    late_penalty_range: tuple[float, float] | None = None
    day_penalty_range: tuple[float, float] | None = None
    back_to_back_penalty_range: tuple[float, float] | None = None
    optional_absence_penalty_range: tuple[float, float] | None = None
    score_check_budget: int | None = None
    min_valid_candidates: int | None = None
    max_valid_ratio: float | None = None
    max_random_baseline_score: float | None = None
    near_optimal_delta: float | None = None
    max_near_optimal_count: int | None = None
    min_best_score: float | None = None
    max_generation_attempts: int | None = None


ATTENDEE_NAMES = [
    "Alex",
    "Jordan",
    "Taylor",
    "Morgan",
    "Casey",
    "Riley",
    "Avery",
    "Quinn",
    "Parker",
    "Reese",
    "Cameron",
    "Rowan",
    "Drew",
    "Skyler",
    "Sage",
    "Elliot",
    "Dakota",
    "Robin",
    "Harper",
    "Finley",
]

ROOM_NAMES = [
    "Orion",
    "Pine",
    "Maple",
    "Sierra",
    "Juniper",
    "Cedar",
]


PRESET_CONFIGS: dict[str, GenerationConfig] = {
    "easy": GenerationConfig(
        difficulty="easy",
        attendee_count_range=(4, 6),
        window_days_range=(3, 4),
        search_start_hour_range=(8, 9),
        search_end_hour_range=(17, 19),
        meeting_duration_choices=(2, 3),
        timezone_offset_range=(-5, 5),
        required_fraction_range=(0.55, 0.7),
        busy_blocks_per_attendee_range=(3, 5),
        busy_block_length_range=(1, 4),
        room_count_range=(1, 2),
        room_unavailable_blocks_range=(1, 2),
        room_unavailable_block_length_range=(1, 3),
        hard_time_probability=0.25,
        hard_time_tolerance_hours_range=(1.5, 3.5),
        preferred_window_hours_range=(5.0, 7.5),
        early_penalty_range=(0.04, 0.08),
        late_penalty_range=(0.04, 0.08),
        day_penalty_range=(0.04, 0.08),
        back_to_back_penalty_range=(0.03, 0.08),
        optional_absence_penalty_range=(0.2, 0.45),
        score_check_budget=8,
        min_valid_candidates=2,
        max_valid_ratio=0.3,
        max_random_baseline_score=0.45,
        near_optimal_delta=0.03,
        max_near_optimal_count=5,
        min_best_score=0.6,
        max_generation_attempts=220,
    ),
    "medium": GenerationConfig(
        difficulty="medium",
        attendee_count_range=(6, 8),
        window_days_range=(4, 5),
        search_start_hour_range=(7, 9),
        search_end_hour_range=(17, 20),
        meeting_duration_choices=(2, 3, 4),
        timezone_offset_range=(-7, 7),
        required_fraction_range=(0.6, 0.8),
        busy_blocks_per_attendee_range=(4, 7),
        busy_block_length_range=(1, 5),
        room_count_range=(1, 2),
        room_unavailable_blocks_range=(1, 4),
        room_unavailable_block_length_range=(1, 4),
        hard_time_probability=0.45,
        hard_time_tolerance_hours_range=(1.0, 2.5),
        preferred_window_hours_range=(4.0, 6.5),
        early_penalty_range=(0.05, 0.11),
        late_penalty_range=(0.05, 0.11),
        day_penalty_range=(0.06, 0.12),
        back_to_back_penalty_range=(0.05, 0.12),
        optional_absence_penalty_range=(0.3, 0.6),
        score_check_budget=6,
        min_valid_candidates=1,
        max_valid_ratio=0.22,
        max_random_baseline_score=0.35,
        near_optimal_delta=0.02,
        max_near_optimal_count=4,
        min_best_score=0.55,
        max_generation_attempts=280,
    ),
    "hard": GenerationConfig(
        difficulty="hard",
        attendee_count_range=(8, 11),
        window_days_range=(5, 7),
        search_start_hour_range=(7, 9),
        search_end_hour_range=(17, 20),
        meeting_duration_choices=(3, 4),
        timezone_offset_range=(-9, 9),
        required_fraction_range=(0.7, 0.9),
        busy_blocks_per_attendee_range=(6, 9),
        busy_block_length_range=(1, 6),
        room_count_range=(1, 2),
        room_unavailable_blocks_range=(2, 5),
        room_unavailable_block_length_range=(1, 4),
        hard_time_probability=0.6,
        hard_time_tolerance_hours_range=(0.5, 2.0),
        preferred_window_hours_range=(3.5, 5.0),
        early_penalty_range=(0.06, 0.14),
        late_penalty_range=(0.06, 0.14),
        day_penalty_range=(0.08, 0.16),
        back_to_back_penalty_range=(0.07, 0.15),
        optional_absence_penalty_range=(0.4, 0.7),
        score_check_budget=4,
        min_valid_candidates=1,
        max_valid_ratio=0.15,
        max_random_baseline_score=0.25,
        near_optimal_delta=0.015,
        max_near_optimal_count=3,
        min_best_score=0.5,
        max_generation_attempts=340,
    ),
}


def get_generation_config(
    difficulty: str,
    overrides: GenerationOverrides | None = None,
) -> GenerationConfig:
    normalized_difficulty = difficulty.lower().strip()
    if normalized_difficulty not in PRESET_CONFIGS:
        allowed = ", ".join(sorted(PRESET_CONFIGS))
        raise ValueError(f"Unknown difficulty '{difficulty}'. Allowed values: {allowed}")

    config = PRESET_CONFIGS[normalized_difficulty]
    if overrides is None:
        return config

    updates = overrides.model_dump(exclude_none=True)
    if not updates:
        return config
    return replace(config, **updates)


def day_label(day_index: int) -> str:
    if day_index < 0:
        return f"Day {day_index}"
    current = BASE_DATE + timedelta(days=day_index)
    return f"Day {day_index} ({current.strftime('%a %b %d')})"


def slot_to_time(slot_in_day: int, slots_per_hour: int = SLOTS_PER_HOUR) -> str:
    hour = slot_in_day // slots_per_hour
    minute = int((slot_in_day % slots_per_hour) * (60 / slots_per_hour))
    return f"{hour:02d}:{minute:02d}"


def parse_time_to_slot(
    value: str, slots_per_hour: int = SLOTS_PER_HOUR
) -> tuple[bool, int | None, str | None]:
    chunks = value.strip().split(":")
    if len(chunks) != 2:
        return False, None, "Time must use HH:MM format"
    try:
        hour = int(chunks[0])
        minute = int(chunks[1])
    except ValueError:
        return False, None, "Time must use numeric HH:MM values"
    if hour < 0 or hour > 23:
        return False, None, "Hour must be between 00 and 23"
    if minute < 0 or minute >= 60:
        return False, None, "Minute must be between 00 and 59"
    minute_step = 60 // slots_per_hour
    if minute % minute_step != 0:
        return (
            False,
            None,
            f"Minute must align to {minute_step}-minute boundaries",
        )
    return True, hour * slots_per_hour + (minute // minute_step), None


def proposal_label(task: CalendarTask, proposal: MeetingProposal) -> str:
    start = slot_to_time(proposal.start_slot_in_day, task.slots_per_hour)
    end_slot = proposal.start_slot_in_day + proposal.duration_slots
    end = slot_to_time(end_slot, task.slots_per_hour)
    return f"{day_label(proposal.day_index)} {start}-{end} UTC in {proposal.room_id}"


def make_proposal(
    task: CalendarTask,
    day_index: int,
    start_time_utc: str,
    duration_minutes: int,
    room_id: str,
) -> tuple[MeetingProposal | None, str | None]:
    success, slot_in_day, error = parse_time_to_slot(start_time_utc, task.slots_per_hour)
    if not success:
        return None, error
    if slot_in_day is None:
        return None, "Failed to parse start time"
    if duration_minutes <= 0:
        return None, "Duration must be a positive number of minutes"
    if duration_minutes % MINUTES_PER_SLOT != 0:
        return (
            None,
            f"Duration must be a multiple of {MINUTES_PER_SLOT} minutes",
        )
    duration_slots = duration_minutes // MINUTES_PER_SLOT
    proposal = MeetingProposal(
        day_index=day_index,
        start_slot_in_day=slot_in_day,
        duration_slots=duration_slots,
        room_id=room_id.strip(),
    )
    return proposal, None


def intervals_overlap(
    interval_start: int, interval_end: int, other_start: int, other_end: int
) -> bool:
    return interval_start < other_end and other_start < interval_end


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    sorted_intervals = sorted(intervals)
    if not sorted_intervals:
        return tuple()

    merged: list[list[int]] = [[sorted_intervals[0][0], sorted_intervals[0][1]]]
    for start, end in sorted_intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _sample_int(rng: random.Random, value_range: tuple[int, int]) -> int:
    return rng.randint(value_range[0], value_range[1])


def _sample_float(rng: random.Random, value_range: tuple[float, float]) -> float:
    return rng.uniform(value_range[0], value_range[1])


def _make_busy_blocks(
    rng: random.Random,
    num_days: int,
    slots_per_hour: int,
    count_range: tuple[int, int],
    length_range: tuple[int, int],
    center_start_hour: int,
    center_end_hour: int,
) -> tuple[tuple[int, int], ...]:
    day_slots = 24 * slots_per_hour
    block_count = _sample_int(rng, count_range)
    blocks: list[tuple[int, int]] = []
    start_min = max(0, (center_start_hour - 3) * slots_per_hour)
    start_max = min(day_slots - 1, (center_end_hour + 1) * slots_per_hour)
    if start_max <= start_min:
        start_min = 0
        start_max = day_slots - 1

    for _ in range(block_count):
        day_index = rng.randrange(num_days)
        start_slot_day = rng.randint(start_min, start_max)
        duration = _sample_int(rng, length_range)
        start_slot = day_index * day_slots + start_slot_day
        end_slot = min((day_index + 1) * day_slots, start_slot + duration)
        if end_slot > start_slot:
            blocks.append((start_slot, end_slot))
    return merge_intervals(blocks)


def _normalize_weights(raw_weights: list[float]) -> list[float]:
    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(raw_weights)] * len(raw_weights)
    return [weight / total for weight in raw_weights]


def _generate_task(seed: int, config: GenerationConfig) -> CalendarTask:
    rng = random.Random(seed)

    num_attendees = _sample_int(rng, config.attendee_count_range)
    num_days = _sample_int(rng, config.window_days_range)
    search_start_hour = _sample_int(rng, config.search_start_hour_range)
    search_end_hour = _sample_int(rng, config.search_end_hour_range)
    if search_end_hour - search_start_hour < 4:
        search_end_hour = search_start_hour + 4
    meeting_duration_slots = int(rng.choice(config.meeting_duration_choices))

    max_duration_slots = (search_end_hour - search_start_hour) * SLOTS_PER_HOUR - 1
    if meeting_duration_slots > max_duration_slots:
        meeting_duration_slots = max(1, max_duration_slots)

    raw_weights = [rng.uniform(0.5, 1.8) for _ in range(num_attendees)]
    weights = _normalize_weights(raw_weights)

    required_fraction = _sample_float(rng, config.required_fraction_range)
    target_required = int(round(required_fraction * num_attendees))
    required_count = max(1, min(num_attendees - 1, target_required))
    required_indices = set(rng.sample(range(num_attendees), required_count))

    attendees: list[AttendeeSpec] = []
    name_indices = list(range(len(ATTENDEE_NAMES)))
    rng.shuffle(name_indices)
    for index in range(num_attendees):
        attendee_name = ATTENDEE_NAMES[name_indices[index % len(ATTENDEE_NAMES)]]
        attendee_id = f"attendee_{index + 1}"
        timezone_offset = _sample_int(rng, config.timezone_offset_range)
        preferred_day = rng.randrange(num_days)

        preferred_window_hours = _sample_float(rng, config.preferred_window_hours_range)
        preferred_midpoint = rng.uniform(10.0, 15.0)
        preferred_start = max(5.0, preferred_midpoint - (preferred_window_hours / 2.0))
        preferred_end = min(22.5, preferred_start + preferred_window_hours)

        has_hard_time = rng.random() < config.hard_time_probability
        hard_start_hour: float | None = None
        hard_end_hour: float | None = None
        if has_hard_time:
            tolerance = _sample_float(rng, config.hard_time_tolerance_hours_range)
            hard_start_hour = max(0.0, preferred_start - tolerance)
            hard_end_hour = min(24.0, preferred_end + tolerance)

        busy_blocks = _make_busy_blocks(
            rng=rng,
            num_days=num_days,
            slots_per_hour=SLOTS_PER_HOUR,
            count_range=config.busy_blocks_per_attendee_range,
            length_range=config.busy_block_length_range,
            center_start_hour=search_start_hour,
            center_end_hour=search_end_hour,
        )

        attendee = AttendeeSpec(
            attendee_id=attendee_id,
            display_name=attendee_name,
            timezone_offset_hours=timezone_offset,
            required=index in required_indices,
            weight=weights[index],
            busy_blocks=busy_blocks,
            preferred_day=preferred_day,
            preferred_start_hour=preferred_start,
            preferred_end_hour=preferred_end,
            hard_start_hour=hard_start_hour,
            hard_end_hour=hard_end_hour,
            early_penalty_per_hour=_sample_float(rng, config.early_penalty_range),
            late_penalty_per_hour=_sample_float(rng, config.late_penalty_range),
            day_penalty_per_day=_sample_float(rng, config.day_penalty_range),
            back_to_back_penalty=_sample_float(rng, config.back_to_back_penalty_range),
            optional_absence_penalty=_sample_float(rng, config.optional_absence_penalty_range),
        )
        attendees.append(attendee)

    room_count = _sample_int(rng, config.room_count_range)
    room_names = ROOM_NAMES[:]
    rng.shuffle(room_names)
    rooms: list[RoomSpec] = []
    for room_index in range(room_count):
        room_id = f"room_{room_index + 1}"
        room_display = room_names[room_index % len(room_names)]
        unavailable_blocks = _make_busy_blocks(
            rng=rng,
            num_days=num_days,
            slots_per_hour=SLOTS_PER_HOUR,
            count_range=config.room_unavailable_blocks_range,
            length_range=config.room_unavailable_block_length_range,
            center_start_hour=search_start_hour,
            center_end_hour=search_end_hour,
        )
        rooms.append(
            RoomSpec(
                room_id=room_id,
                display_name=room_display,
                unavailable_blocks=unavailable_blocks,
            )
        )

    return CalendarTask(
        task_id=f"{config.difficulty}-{seed}",
        seed=seed,
        difficulty=config.difficulty,
        num_days=num_days,
        slots_per_hour=SLOTS_PER_HOUR,
        search_start_hour=search_start_hour,
        search_end_hour=search_end_hour,
        meeting_duration_slots=meeting_duration_slots,
        score_check_budget=config.score_check_budget,
        attendees=tuple(attendees),
        rooms=tuple(rooms),
    )


def _compute_local_window(
    start_slot_absolute: int,
    duration_slots: int,
    slots_per_hour: int,
    timezone_offset_hours: int,
) -> tuple[int, float, int, float, bool]:
    day_hours = 24.0
    start_hours_utc = start_slot_absolute / slots_per_hour
    end_hours_utc = (start_slot_absolute + duration_slots) / slots_per_hour

    local_start = start_hours_utc + timezone_offset_hours
    local_end = end_hours_utc + timezone_offset_hours

    local_day_start = math.floor(local_start / day_hours)
    local_day_end = math.floor(local_end / day_hours)
    local_start_hour = local_start - (local_day_start * day_hours)
    local_end_hour = local_end - (local_day_end * day_hours)
    crosses_midnight = local_day_start != local_day_end
    return (
        local_day_start,
        local_start_hour,
        local_day_end,
        local_end_hour,
        crosses_midnight,
    )


def _interval_overlaps_blocks(
    start_slot_absolute: int,
    end_slot_absolute: int,
    blocks: tuple[tuple[int, int], ...],
) -> bool:
    return any(
        intervals_overlap(start_slot_absolute, end_slot_absolute, block_start, block_end)
        for block_start, block_end in blocks
    )


def _is_back_to_back(
    start_slot_absolute: int,
    end_slot_absolute: int,
    blocks: tuple[tuple[int, int], ...],
) -> bool:
    return any(
        block_end == start_slot_absolute or block_start == end_slot_absolute
        for block_start, block_end in blocks
    )


def evaluate_proposal(task: CalendarTask, proposal: MeetingProposal) -> ProposalEvaluation:
    hard_violations: list[str] = []
    attendee_utilities: dict[str, float] = {}
    attendee_notes: dict[str, tuple[str, ...]] = {}
    attendees_who_can_attend: list[str] = []

    day_slots = 24 * task.slots_per_hour
    search_start_slot = task.search_start_hour * task.slots_per_hour
    search_end_slot = task.search_end_hour * task.slots_per_hour

    if proposal.day_index < 0 or proposal.day_index >= task.num_days:
        hard_violations.append("Day index is outside the scheduling window")

    if proposal.duration_slots != task.meeting_duration_slots:
        required_minutes = task.meeting_duration_slots * (60 // task.slots_per_hour)
        hard_violations.append(f"Duration must be exactly {required_minutes} minutes")

    if proposal.start_slot_in_day < search_start_slot:
        hard_violations.append(
            f"Start time must be at or after {slot_to_time(search_start_slot, task.slots_per_hour)} UTC"
        )

    if proposal.start_slot_in_day + proposal.duration_slots > search_end_slot:
        hard_violations.append(
            f"Meeting must end by {slot_to_time(search_end_slot, task.slots_per_hour)} UTC"
        )

    start_slot_absolute = proposal.start_slot_absolute(task)
    end_slot_absolute = start_slot_absolute + proposal.duration_slots
    if end_slot_absolute > task.num_days * day_slots:
        hard_violations.append("Meeting extends beyond the scheduling horizon")

    room = next((current for current in task.rooms if current.room_id == proposal.room_id), None)
    if room is None:
        available_rooms = ", ".join(room.room_id for room in task.rooms)
        hard_violations.append(
            f"Unknown room '{proposal.room_id}'. Available rooms: {available_rooms}"
        )
    elif _interval_overlaps_blocks(start_slot_absolute, end_slot_absolute, room.unavailable_blocks):
        hard_violations.append(
            f"Room {proposal.room_id} is unavailable for part of the proposed window"
        )

    for attendee in task.attendees:
        notes: list[str] = []
        has_calendar_conflict = _interval_overlaps_blocks(
            start_slot_absolute, end_slot_absolute, attendee.busy_blocks
        )
        (
            local_day,
            local_start_hour,
            _local_day_end,
            local_end_hour,
            crosses_midnight,
        ) = _compute_local_window(
            start_slot_absolute=start_slot_absolute,
            duration_slots=proposal.duration_slots,
            slots_per_hour=task.slots_per_hour,
            timezone_offset_hours=attendee.timezone_offset_hours,
        )

        violates_hard_time = False
        if attendee.hard_start_hour is not None and local_start_hour < attendee.hard_start_hour:
            violates_hard_time = True
            notes.append(
                f"Local start {local_start_hour:.1f}h is before hard bound {attendee.hard_start_hour:.1f}h"
            )
        if attendee.hard_end_hour is not None and local_end_hour > attendee.hard_end_hour:
            violates_hard_time = True
            notes.append(
                f"Local end {local_end_hour:.1f}h is after hard bound {attendee.hard_end_hour:.1f}h"
            )
        if crosses_midnight:
            violates_hard_time = True
            notes.append("Meeting crosses midnight in attendee local time")
        if local_day < 0 or local_day >= task.num_days:
            violates_hard_time = True
            notes.append("Meeting falls outside attendee local planning window")

        can_attend = not has_calendar_conflict and not violates_hard_time
        if has_calendar_conflict:
            notes.append("Conflicts with attendee busy calendar")

        if attendee.required and not can_attend:
            hard_violations.append(f"Required attendee {attendee.attendee_id} cannot attend")

        utility = 1.0
        if not can_attend:
            if attendee.required:
                utility = 0.0
            else:
                utility = max(0.0, 1.0 - attendee.optional_absence_penalty)
                notes.append(
                    f"Optional absence penalty applied: {attendee.optional_absence_penalty:.2f}"
                )
        else:
            attendees_who_can_attend.append(attendee.attendee_id)
            early_hours = max(0.0, attendee.preferred_start_hour - local_start_hour)
            late_hours = max(0.0, local_end_hour - attendee.preferred_end_hour)
            day_distance = abs(local_day - attendee.preferred_day)
            utility -= early_hours * attendee.early_penalty_per_hour
            utility -= late_hours * attendee.late_penalty_per_hour
            utility -= day_distance * attendee.day_penalty_per_day
            if early_hours > 0:
                notes.append(
                    f"Early-time penalty: {early_hours:.1f}h * {attendee.early_penalty_per_hour:.2f}"
                )
            if late_hours > 0:
                notes.append(
                    f"Late-time penalty: {late_hours:.1f}h * {attendee.late_penalty_per_hour:.2f}"
                )
            if day_distance > 0:
                notes.append(
                    f"Day-distance penalty: {day_distance} * {attendee.day_penalty_per_day:.2f}"
                )
            if _is_back_to_back(start_slot_absolute, end_slot_absolute, attendee.busy_blocks):
                utility -= attendee.back_to_back_penalty
                notes.append(f"Back-to-back penalty: {attendee.back_to_back_penalty:.2f}")

        attendee_utilities[attendee.attendee_id] = min(1.0, max(0.0, utility))
        attendee_notes[attendee.attendee_id] = tuple(notes)

    valid = len(hard_violations) == 0
    weighted_score = 0.0
    if valid:
        for attendee in task.attendees:
            weighted_score += attendee.weight * attendee_utilities[attendee.attendee_id]

    return ProposalEvaluation(
        proposal=proposal,
        valid=valid,
        score=weighted_score if valid else 0.0,
        hard_violations=tuple(hard_violations),
        attendee_utilities=attendee_utilities,
        attendee_notes=attendee_notes,
        attendees_who_can_attend=tuple(attendees_who_can_attend),
    )


def iter_candidate_proposals(task: CalendarTask) -> Iterable[MeetingProposal]:
    start_slot = task.search_start_hour * task.slots_per_hour
    end_slot = task.search_end_hour * task.slots_per_hour
    for day_index in range(task.num_days):
        for room in task.rooms:
            for slot in range(start_slot, end_slot - task.meeting_duration_slots + 1):
                yield MeetingProposal(
                    day_index=day_index,
                    start_slot_in_day=slot,
                    duration_slots=task.meeting_duration_slots,
                    room_id=room.room_id,
                )


def solve_task(task: CalendarTask, near_optimal_delta: float) -> SolverSummary:
    total_candidates = 0
    valid_candidates = 0
    best_score = 0.0
    best_proposals: list[MeetingProposal] = []
    all_scores: list[float] = []
    valid_scores: list[float] = []

    for proposal in iter_candidate_proposals(task):
        total_candidates += 1
        evaluation = evaluate_proposal(task, proposal)
        all_scores.append(evaluation.score)
        if not evaluation.valid:
            continue
        valid_candidates += 1
        valid_scores.append(evaluation.score)
        if evaluation.score > best_score + EPSILON:
            best_score = evaluation.score
            best_proposals = [proposal]
        elif abs(evaluation.score - best_score) <= EPSILON:
            best_proposals.append(proposal)

    random_baseline = statistics.fmean(all_scores) if all_scores else 0.0
    near_optimal_count = sum(
        1 for score in valid_scores if best_score > 0 and score >= best_score - near_optimal_delta
    )

    return SolverSummary(
        total_candidates=total_candidates,
        valid_candidates=valid_candidates,
        best_score=best_score,
        best_proposals=tuple(best_proposals),
        random_baseline_score=random_baseline,
        near_optimal_count=near_optimal_count,
    )


def _task_quality_key(summary: SolverSummary, config: GenerationConfig) -> tuple[float, float]:
    if summary.total_candidates == 0:
        return (0.0, 0.0)
    valid_ratio = summary.valid_candidates / summary.total_candidates
    ratio_score = 1.0 - min(1.0, abs(valid_ratio - (config.max_valid_ratio * 0.5)))
    return (ratio_score, summary.best_score)


def _is_task_acceptable(summary: SolverSummary, config: GenerationConfig) -> bool:
    if summary.total_candidates == 0:
        return False
    if summary.valid_candidates < config.min_valid_candidates:
        return False

    valid_ratio = summary.valid_candidates / summary.total_candidates
    if valid_ratio > config.max_valid_ratio:
        return False
    if summary.random_baseline_score > config.max_random_baseline_score:
        return False
    if summary.near_optimal_count > config.max_near_optimal_count:
        return False
    if summary.best_score < config.min_best_score:
        return False
    return True


def generate_validated_task(
    seed: int,
    difficulty: str = "medium",
    overrides: GenerationOverrides | None = None,
) -> tuple[CalendarTask, SolverSummary, GenerationConfig]:
    config = get_generation_config(difficulty, overrides)

    best_fallback: tuple[CalendarTask, SolverSummary] | None = None
    best_quality = (-1.0, -1.0)
    for attempt in range(config.max_generation_attempts):
        task_seed = seed + (attempt * 104_729)
        task = _generate_task(task_seed, config)
        summary = solve_task(task, near_optimal_delta=config.near_optimal_delta)

        if summary.valid_candidates > 0:
            quality = _task_quality_key(summary, config)
            if quality > best_quality:
                best_quality = quality
                best_fallback = (task, summary)

        if _is_task_acceptable(summary, config):
            return task, summary, config

    if best_fallback is None:
        raise RuntimeError(
            "Unable to generate a solvable calendar scheduling task with current settings"
        )
    return best_fallback[0], best_fallback[1], config


def render_task_brief(task: CalendarTask) -> str:
    duration_minutes = task.meeting_duration_slots * (60 // task.slots_per_hour)
    lines: list[str] = []
    lines.append("Schedule a cross-team meeting using the available tools.")
    lines.append("")
    lines.append("Scoring:")
    lines.append("- A submission gets score 0.0 if any hard constraint is violated.")
    lines.append("- Otherwise score = weighted average attendee utility (each utility in [0, 1]).")
    lines.append("- Weights are normalized to sum to 1 for this task.")
    lines.append("")
    lines.append(
        "Hard constraints include: required-attendee availability, hard local-time limits, and room availability."
    )
    lines.append(
        "Soft constraints include: early/late local-time penalties, day preference penalties, and back-to-back penalties."
    )
    lines.append("")
    lines.append(
        f"Window: {task.num_days} days, candidate starts from {task.search_start_hour:02d}:00 to {task.search_end_hour:02d}:00 UTC"
    )
    lines.append(f"Meeting duration: {duration_minutes} minutes (fixed)")
    lines.append(
        f"Score-check budget: {task.score_check_budget} calls to check_proposal (then the tool is blocked)"
    )
    lines.append("")
    lines.append("Rooms:")
    for room in task.rooms:
        lines.append(f"- {room.room_id}: {room.display_name}")

    lines.append("")
    lines.append("Attendees:")
    for attendee in task.attendees:
        role = "required" if attendee.required else "optional"
        lines.append(
            f"- {attendee.attendee_id}: {attendee.display_name} | {role} | UTC{attendee.timezone_offset_hours:+d} | weight={attendee.weight:.3f}"
        )
    lines.append("")
    lines.append(
        "Use submit_window(day_index, start_time_utc, duration_minutes, room_id) when ready."
    )
    lines.append("If you do not submit before turns run out, the reward is 0.0.")
    lines.append(
        "Day indices are integers (0-based). Example: check_proposal(day_index=1, start_time_utc='11:00', duration_minutes=90, room_id='room_1')."
    )
    lines.append("Tool outputs include remaining_turns.")
    return "\n".join(lines)


def build_example(
    task: CalendarTask,
    summary: SolverSummary,
    config: GenerationConfig,
) -> vf.Task:
    prompt = [{"role": "user", "content": render_task_brief(task)}]
    answer_payload = {
        "optimal_score": summary.best_score,
        "optimal_windows": [proposal.to_dict(task) for proposal in summary.best_proposals[:5]],
        "valid_candidates": summary.valid_candidates,
        "total_candidates": summary.total_candidates,
    }
    info = {
        "calendar_task": task.to_dict(),
        "oracle": summary.to_dict(task, top_k=5),
        "difficulty": config.difficulty,
    }
    return vf.Task(
        {
            "prompt": prompt,
            "answer": json.dumps(answer_payload, sort_keys=True),
            "info": info,
        }
    )


def format_blocks_for_day(
    blocks: tuple[tuple[int, int], ...],
    day_index: int,
    slots_per_hour: int,
) -> list[dict[str, str]]:
    day_slots = 24 * slots_per_hour
    day_start = day_index * day_slots
    day_end = day_start + day_slots
    formatted: list[dict[str, str]] = []
    for start, end in blocks:
        overlap_start = max(day_start, start)
        overlap_end = min(day_end, end)
        if overlap_start >= overlap_end:
            continue
        formatted.append(
            {
                "start": slot_to_time(overlap_start - day_start, slots_per_hour),
                "end": slot_to_time(overlap_end - day_start, slots_per_hour),
            }
        )
    return formatted


def timeline_segment(
    blocks: tuple[tuple[int, int], ...],
    day_index: int,
    search_start_hour: int,
    search_end_hour: int,
    slots_per_hour: int,
    highlight: tuple[int, int] | None = None,
) -> str:
    day_slots = 24 * slots_per_hour
    day_start = day_index * day_slots
    start_slot = search_start_hour * slots_per_hour
    end_slot = search_end_hour * slots_per_hour

    chars: list[str] = []
    for slot in range(start_slot, end_slot):
        absolute_slot = day_start + slot
        is_busy = _interval_overlaps_blocks(absolute_slot, absolute_slot + 1, blocks)
        if highlight is not None and highlight[0] <= slot < highlight[1]:
            chars.append("#" if is_busy else "=")
        else:
            chars.append("X" if is_busy else ".")
    return "".join(chars)

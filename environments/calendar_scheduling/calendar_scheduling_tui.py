import argparse
import json

from calendar_problem import (
    CalendarTask,
    GenerationOverrides,
    JsonObject,
    MeetingProposal,
    day_label,
    generate_validated_task,
    timeline_segment,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _parse_overrides(raw: str | None) -> GenerationOverrides | None:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--overrides must decode to a JSON object")
    return GenerationOverrides.model_validate(parsed)


def _styled_timeline(raw: str) -> Text:
    text = Text()
    for char in raw:
        if char == ".":
            text.append(".", style="grey50")
        elif char == "X":
            text.append("X", style="bold red")
        elif char == "=":
            text.append("=", style="bold green")
        elif char == "#":
            text.append("#", style="bold yellow")
        else:
            text.append(char)
    return text


def _highlight_range(
    task: CalendarTask,
    proposal: MeetingProposal | None,
    day_index: int,
) -> tuple[int, int] | None:
    if proposal is None or proposal.day_index != day_index:
        return None
    start = proposal.start_slot_in_day
    end = proposal.start_slot_in_day + proposal.duration_slots
    return (start, end)


def _render_overview(console: Console, task: CalendarTask, oracle: JsonObject) -> None:
    duration_minutes = task.meeting_duration_slots * (60 // task.slots_per_hour)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan", width=24)
    table.add_column(style="white")
    table.add_row("Task ID", task.task_id)
    table.add_row("Difficulty", task.difficulty)
    table.add_row("Seed", str(task.seed))
    table.add_row("Window", f"{task.num_days} days")
    table.add_row(
        "Candidate UTC hours",
        f"{task.search_start_hour:02d}:00 - {task.search_end_hour:02d}:00",
    )
    table.add_row("Meeting duration", f"{duration_minutes} minutes")
    table.add_row("Score-check budget", str(task.score_check_budget))
    table.add_row("Total candidates", str(oracle["total_candidates"]))
    table.add_row("Valid candidates", str(oracle["valid_candidates"]))
    table.add_row("Oracle best score", f"{oracle['best_score']:.4f}")
    table.add_row("Random baseline", f"{oracle['random_baseline_score']:.4f}")

    console.print(
        Panel(
            table,
            title="[bold white]Calendar Scheduling Problem[/bold white]",
            border_style="cyan",
        )
    )


def _render_attendees(console: Console, task: CalendarTask) -> None:
    table = Table(title="Attendees", border_style="blue")
    table.add_column("ID", style="bold")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("TZ")
    table.add_column("Weight")
    table.add_column("Preferred Day")
    table.add_column("Preferred Local")
    table.add_column("Hard Local")

    for attendee in task.attendees:
        role = "required" if attendee.required else "optional"
        hard_window = (
            "none"
            if attendee.hard_start_hour is None or attendee.hard_end_hour is None
            else f"{attendee.hard_start_hour:.1f}-{attendee.hard_end_hour:.1f}"
        )
        table.add_row(
            attendee.attendee_id,
            attendee.display_name,
            role,
            f"UTC{attendee.timezone_offset_hours:+d}",
            f"{attendee.weight:.3f}",
            day_label(attendee.preferred_day),
            f"{attendee.preferred_start_hour:.1f}-{attendee.preferred_end_hour:.1f}",
            hard_window,
        )

    console.print(table)


def _render_timeline_panel(
    console: Console,
    task: CalendarTask,
    proposal: MeetingProposal | None,
) -> None:
    day_heading = " | ".join(day_label(day_index) for day_index in range(task.num_days))
    rows = Table(
        title="Availability Timeline (X busy, . free, =/# best window overlay)",
        border_style="magenta",
    )
    rows.add_column("Lane", style="bold")
    rows.add_column("Timeline", overflow="fold")

    for attendee in task.attendees:
        parts = []
        for day_index in range(task.num_days):
            parts.append(
                timeline_segment(
                    blocks=attendee.busy_blocks,
                    day_index=day_index,
                    search_start_hour=task.search_start_hour,
                    search_end_hour=task.search_end_hour,
                    slots_per_hour=task.slots_per_hour,
                    highlight=_highlight_range(task, proposal, day_index),
                )
            )
        rows.add_row(
            attendee.attendee_id,
            _styled_timeline("|".join(parts)),
        )

    for room in task.rooms:
        parts = []
        for day_index in range(task.num_days):
            parts.append(
                timeline_segment(
                    blocks=room.unavailable_blocks,
                    day_index=day_index,
                    search_start_hour=task.search_start_hour,
                    search_end_hour=task.search_end_hour,
                    slots_per_hour=task.slots_per_hour,
                    highlight=_highlight_range(task, proposal, day_index),
                )
            )
        rows.add_row(
            f"{room.room_id} (room)",
            _styled_timeline("|".join(parts)),
        )

    console.print(Panel(day_heading, border_style="magenta", title="Day Labels"))
    console.print(rows)


def _render_constraints(console: Console, task: CalendarTask) -> None:
    block = Text()
    block.append("Hard constraints\n", style="bold yellow")
    block.append("- Required attendees must be able to attend\n")
    block.append("- Hard local-time bounds cannot be violated\n")
    block.append("- Chosen room must be available\n")
    block.append("- Duration must match task duration exactly\n\n")
    block.append("Soft utility penalties\n", style="bold green")
    block.append("- Early/late local-time penalties\n")
    block.append("- Day-preference distance penalties\n")
    block.append("- Back-to-back penalty near busy blocks\n")
    block.append("- Optional attendee absence penalty\n")
    console.print(Panel(block, title="Constraint Model", border_style="green"))


def _render_oracle(console: Console, task: CalendarTask, oracle: JsonObject) -> None:
    table = Table(title="Oracle Best Windows", border_style="cyan")
    table.add_column("Rank")
    table.add_column("Window")

    best_windows = oracle.get("best_proposals")
    if best_windows is None:
        best_windows = []
    if not isinstance(best_windows, list):
        raise ValueError("Oracle best_proposals must be a list")
    for index, window in enumerate(best_windows, start=1):
        if not isinstance(window, dict):
            raise ValueError("Oracle best proposal entries must be objects")
        window_payload = {str(key): value for key, value in window.items()}
        table.add_row(str(index), str(window_payload.get("window", "unknown")))

    if not best_windows:
        table.add_row("-", "No valid window found")

    console.print(table)
    console.print(
        Panel(
            f"Best score: {oracle['best_score']:.4f}\n"
            f"Valid windows: {oracle['valid_candidates']} / {oracle['total_candidates']}",
            title="Oracle Summary",
            border_style="cyan",
        )
    )


def render_problem(
    difficulty: str,
    seed: int,
    overrides: GenerationOverrides | None,
    show_oracle: bool,
) -> None:
    task, summary, _ = generate_validated_task(
        seed=seed,
        difficulty=difficulty,
        overrides=overrides,
    )
    oracle = summary.to_dict(task, top_k=5)
    best_proposal = summary.best_proposals[0] if summary.best_proposals else None

    console = Console()
    console.print("\n[bold cyan]Calendar Scheduling TUI[/bold cyan]\n")
    _render_overview(console, task, oracle)
    _render_attendees(console, task)
    _render_timeline_panel(console, task, best_proposal if show_oracle else None)
    _render_constraints(console, task)
    if show_oracle:
        _render_oracle(console, task, oracle)
    else:
        console.print(
            Panel(
                "Run with --show-oracle to reveal best windows and score.",
                border_style="grey50",
                title="Oracle Hidden",
            )
        )
    console.print(
        "\n[dim]Legend: X busy, . free, = highlighted best free slot, # highlighted busy slot[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a calendar scheduling task TUI")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--show-oracle",
        action="store_true",
        help="Show the oracle best windows and best score",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help="JSON object of fine-grained generation overrides",
    )
    args = parser.parse_args()

    overrides = _parse_overrides(args.overrides)
    render_problem(
        difficulty=args.difficulty,
        seed=args.seed,
        overrides=overrides,
        show_oracle=args.show_oracle,
    )


if __name__ == "__main__":
    main()

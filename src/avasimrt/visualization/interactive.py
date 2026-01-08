from __future__ import annotations

import logging

from avasimrt.result import AnchorReading, Sample, SimResult

from .plots import plot_amp_phase_for_reading, plot_mean_db_and_distance_over_time, show_anchor_over_time

logger = logging.getLogger(__name__)


def _get_anchor_reading_at_step(samples: list[Sample], anchor_id: str, step_index: int) -> AnchorReading:
    sample = samples[step_index]
    if not sample.readings:
        raise ValueError(f"No readings available at step {step_index}.")
    reading = next((ar for ar in sample.readings if ar.anchor_id == anchor_id), None)
    if reading is None:
        raise ValueError(f"No readings for anchor {anchor_id!r} at step {step_index}.")
    return reading


def handle_visualization_command(sim_result: SimResult, command: str) -> None:
    """Handle interactive visualization commands.

    Supported syntax:
      - "@"              : show overview (mean_db + distance over time)
      - "<id>@<step>"    : show detail for anchor <id> at step index <step>
      - "<id>@"          : anchor-over-time view
    """
    samples = sim_result.samples
    cmd = command.strip()
    if not cmd:
        return

    if cmd == "@":
        plot_mean_db_and_distance_over_time(samples, show=True, save_dir=None)
        return

    if "@" not in cmd:
        raise ValueError(f"Invalid command {cmd!r}. Expected '@', '<id>@<step>' or '<id>@'.")

    anchor_part, step_part = cmd.split("@", 1)
    anchor_id = anchor_part.strip()

    if step_part.strip() == "":
        show_anchor_over_time(samples, anchor_id=anchor_id)
        return

    try:
        step_index = int(step_part.strip())
    except ValueError as e:
        raise ValueError(f"Invalid step index {step_part.strip()!r} in command {cmd!r}; must be an integer.") from e

    if step_index < 0 or step_index >= len(samples):
        raise IndexError(f"Invalid step index {step_index}; must be in [0, {len(samples) - 1}]")

    reading = _get_anchor_reading_at_step(samples, anchor_id=anchor_id, step_index=step_index)
    plot_amp_phase_for_reading(reading=reading, graphs="both", prefix="", show=True, save_dir=None, amp_in_db=True)


def interactive_visualization_shell(sim_result: SimResult) -> None:
    """CLI loop for interactive visualization."""
    if not sim_result.samples:
        print("No results available for interactive visualization.")
        return

    print("\nInteractive visualization:")
    print("  @           -> overview (mean_db + distance over time)")
    print("  <id>@<step> -> anchor <id> at step index <step>")
    print("  <id>@       -> anchor-over-time view")
    print("  q / quit    -> exit\n")

    while True:
        try:
            cmd = input("vis> ").strip()
        except EOFError:
            print()
            break

        if cmd.lower() in {"q", "quit", "exit"}:
            break

        if not cmd:
            continue

        try:
            handle_visualization_command(sim_result, cmd)
        except Exception as e:
            logger.exception("Error while handling visualization command %r", cmd)
            print(f"Error: {e}")

from __future__ import annotations

from pathlib import Path

from sim.visualization import write_dashboard

DASHBOARD_HOURS = 72


def build_demo_action_frames(hours: int = DASHBOARD_HOURS) -> list[dict[str, dict[str, object]]]:
    action_frames: list[dict[str, dict[str, object]]] = [{} for _ in range(hours)]

    def add(hour: int, entity_id: str, action: dict[str, object]) -> None:
        if 0 <= hour < hours:
            action_frames[hour].setdefault(entity_id, {}).update(action)

    for hour in range(0, 10):
        add(hour, "brevik", {"load_vessel": "northern_pioneer"})
    add(10, "northern_pioneer", {"sail_to": "oygarden_terminal"})

    for hour in range(2, 14):
        add(hour, "celsio", {"load_vessel": "northern_pathfinder"})
    add(14, "northern_pathfinder", {"sail_to": "oygarden_terminal"})

    add(37, "oygarden_terminal", {"unload_vessel": "northern_pioneer"})
    for hour in range(37, 42):
        add(hour, "oygarden_pipeline", {"flow_tph": 300.0})
    add(38, "northern_pioneer", {"sail_to": "brevik"})

    add(45, "oygarden_terminal", {"unload_vessel": "northern_pathfinder"})
    for hour in range(45, 52):
        add(hour, "oygarden_pipeline", {"flow_tph": 300.0})
    add(46, "northern_pathfinder", {"sail_to": "celsio"})

    return action_frames


def main() -> None:
    output = write_dashboard(
        Path("docs") / "physical_layer_dashboard.html",
        hours=DASHBOARD_HOURS,
        action_frames=build_demo_action_frames(DASHBOARD_HOURS),
    )
    print(output.resolve())


if __name__ == "__main__":
    main()

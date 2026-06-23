from __future__ import annotations

from pathlib import Path

from sim.visualization import write_northern_lights_phase2_dashboard

DASHBOARD_HOURS = 24 * 10


def main() -> None:
    output = write_northern_lights_phase2_dashboard(
        Path("docs") / "physical_layer_phase2_dashboard.html",
        hours=DASHBOARD_HOURS,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()

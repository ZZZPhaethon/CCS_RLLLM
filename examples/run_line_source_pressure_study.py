from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from sim.line_source import (
    LineSourceParameters,
    bottomhole_pressure_bar,
    load_line_source_parameters,
    pressure_at_radius_bar,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "northern_lights_line_source_parameters.json"
CSV_PATH = ROOT / "data" / "northern_lights_line_source_pressure_results.csv"
REPORT_PATH = ROOT / "docs" / "northern_lights_line_source_pressure_study.md"


def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    base_parameters = load_line_source_parameters(DATA_PATH)
    active_wells = int(config["line_source_inputs"].get("active_wells", 1))
    elapsed_days = float(config["study_controls"]["elapsed_days"])
    radii_m = [float(value) for value in config["study_controls"]["observation_radii_m"]]
    rates_mtpa = [float(value) for value in config["study_controls"]["total_injection_rates_mtpa"]]

    rows = []
    for case in config["sensitivity_cases"]:
        parameters = replace(base_parameters, permeability_md=float(case["permeability_md"]))
        for total_rate_mtpa in rates_mtpa:
            per_well_rate_mtpa = total_rate_mtpa / active_wells
            row = _pressure_row(
                case_id=case["case_id"],
                parameters=parameters,
                total_rate_mtpa=total_rate_mtpa,
                per_well_rate_mtpa=per_well_rate_mtpa,
                elapsed_days=elapsed_days,
                radii_m=radii_m,
            )
            rows.append(row)

    _write_csv(rows)
    _write_report(config, rows, active_wells, elapsed_days, radii_m)
    print(REPORT_PATH)
    print(CSV_PATH)


def _pressure_row(
    *,
    case_id: str,
    parameters: LineSourceParameters,
    total_rate_mtpa: float,
    per_well_rate_mtpa: float,
    elapsed_days: float,
    radii_m: list[float],
) -> dict[str, float | str]:
    bottomhole = bottomhole_pressure_bar(parameters, per_well_rate_mtpa, elapsed_days=elapsed_days)
    row: dict[str, float | str] = {
        "case_id": case_id,
        "permeability_md": parameters.permeability_md,
        "total_rate_mtpa": total_rate_mtpa,
        "per_well_rate_mtpa": per_well_rate_mtpa,
        "elapsed_days": elapsed_days,
        "cumulative_injected_mt": total_rate_mtpa * elapsed_days / 365.25,
        "bottomhole_pressure_bar": bottomhole,
        "bottomhole_delta_bar": bottomhole - parameters.initial_pressure_bar,
    }
    for radius_m in radii_m:
        pressure = pressure_at_radius_bar(parameters, per_well_rate_mtpa, elapsed_days=elapsed_days, radius_m=radius_m)
        radius_label = _radius_label(radius_m)
        row[f"pressure_{radius_label}_bar"] = pressure
        row[f"delta_{radius_label}_bar"] = pressure - parameters.initial_pressure_bar
    return row


def _write_csv(rows: list[dict[str, float | str]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    config: dict,
    rows: list[dict[str, float | str]],
    active_wells: int,
    elapsed_days: float,
    radii_m: list[float],
) -> None:
    lines = [
        "# Northern Lights Line-Source Pressure Screening",
        "",
        "This is a standalone screening result for the infinite-acting line-source formula. The same formula is exposed through simulator well and reservoir snapshots.",
        "",
        "## Inputs",
        "",
        f"- Data file: `{DATA_PATH.relative_to(ROOT)}`",
        f"- Active wells used for rate split: {active_wells}",
        f"- Elapsed time: {elapsed_days:g} days",
        f"- Reservoir pressure means point pressure at radii: {', '.join(f'{radius:g} m' for radius in radii_m)}",
        f"- Initial pressure: {config['line_source_inputs']['initial_pressure_bar']:g} bar",
        f"- Effective thickness: {config['line_source_inputs']['thickness_m']:g} m",
        f"- Porosity: {config['line_source_inputs']['porosity_fraction']:g}",
        f"- CO2 viscosity assumption: {config['line_source_inputs']['viscosity_pa_s']:.2e} Pa s",
        f"- CO2 density assumption: {config['line_source_inputs']['co2_density_kg_m3']:g} kg/m3",
        "",
        "## Parameter Audit",
        "",
        "| Parameter | Current value | Source | Confidence | Note |",
        "|---|---:|---|---|---|",
    ]
    for row in config["line_source_input_audit"]:
        lines.append(
            "| {parameter} | {value} | {source} | {confidence} | {rationale} |".format(
                parameter=row["parameter"],
                value=_format_value(row["value"]),
                source=row["source"],
                confidence=row["confidence"],
                rationale=row["rationale"],
            )
        )

    lines.extend(
        [
            "",
            "Confidence convention: high = directly reported or directly derived from a project PDF; medium = sourced model/report range but selected representative value; low = engineering assumption pending replacement.",
            "",
        ]
    )
    lines.extend(
        [
        "## Results",
        "",
        "| Case | k (mD) | Rate (Mt/y) | Bottomhole pressure (bar) | dBHP (bar) | Reservoir pressure @100 m (bar) | dP @100 m (bar) | Reservoir pressure @1000 m (bar) | dP @1000 m (bar) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {case_id} | {permeability_md:.0f} | {total_rate_mtpa:.1f} | "
            "{bottomhole_pressure_bar:.2f} | {bottomhole_delta_bar:.2f} | "
            "{pressure_100m_bar:.2f} | {delta_100m_bar:.2f} | "
            "{pressure_1000m_bar:.2f} | {delta_1000m_bar:.2f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Pressure response is linear in injection rate for fixed time and fixed parameters.",
            "- Lower effective permeability produces much larger pressure buildup because the coefficient scales approximately with 1/(k h).",
            "- These are point-pressure estimates. They should not be read as whole-reservoir average pressure.",
            "- Reservoir pressure columns are line-source point pressures p(r,t) at 100 m and 1000 m from the injection well.",
            "- The baseline uses a moderate selected permeability; the high-k case is retained only as an optimistic sensitivity.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _radius_label(radius_m: float) -> str:
    if radius_m.is_integer():
        return f"{int(radius_m)}m"
    return f"{radius_m:g}m".replace(".", "p")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


if __name__ == "__main__":
    main()

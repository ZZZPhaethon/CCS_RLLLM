from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
MILLIDARCY_TO_M2 = 9.869233e-16
PA_PER_BAR = 100_000.0
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class LineSourceParameters:
    initial_pressure_bar: float
    permeability_md: float
    thickness_m: float
    porosity_fraction: float
    total_compressibility_1_pa: float
    viscosity_pa_s: float
    co2_density_kg_m3: float
    well_radius_m: float
    skin: float = 0.0


def annual_mt_to_kg_s(rate_mtpa: float) -> float:
    return rate_mtpa * 1_000_000_000.0 / SECONDS_PER_YEAR


def pressure_at_radius_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
    radius_m: float,
) -> float:
    pressure_change_bar = _pressure_change_bar(
        parameters,
        injection_rate_mtpa,
        elapsed_days=elapsed_days,
        radius_m=radius_m,
        skin=0.0,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def bottomhole_pressure_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
) -> float:
    pressure_change_bar = _pressure_change_bar(
        parameters,
        injection_rate_mtpa,
        elapsed_days=elapsed_days,
        radius_m=parameters.well_radius_m,
        skin=parameters.skin,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def variable_rate_pressure_at_radius_bar(
    parameters: LineSourceParameters,
    rate_history_mtpa: list[tuple[float, float]],
    *,
    elapsed_days: float,
    radius_m: float,
) -> float:
    pressure_change_bar = _variable_rate_pressure_change_bar(
        parameters,
        rate_history_mtpa,
        elapsed_days=elapsed_days,
        radius_m=radius_m,
        skin=0.0,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def variable_rate_bottomhole_pressure_bar(
    parameters: LineSourceParameters,
    rate_history_mtpa: list[tuple[float, float]],
    *,
    elapsed_days: float,
) -> float:
    pressure_change_bar = _variable_rate_pressure_change_bar(
        parameters,
        rate_history_mtpa,
        elapsed_days=elapsed_days,
        radius_m=parameters.well_radius_m,
        skin=parameters.skin,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def multiwell_bottomhole_pressures_bar(
    parameters: LineSourceParameters,
    injection_rates_mtpa_by_well: dict[str, float],
    *,
    elapsed_days: float,
    well_distances_m: dict[str, dict[str, float]],
) -> dict[str, float]:
    pressures: dict[str, float] = {}
    for well_id, injection_rate_mtpa in injection_rates_mtpa_by_well.items():
        pressure_bar = bottomhole_pressure_bar(
            parameters,
            injection_rate_mtpa,
            elapsed_days=elapsed_days,
        )
        for source_well_id, source_rate_mtpa in injection_rates_mtpa_by_well.items():
            if source_well_id == well_id or source_rate_mtpa == 0.0:
                continue
            distance_m = _well_distance_m(well_distances_m, well_id, source_well_id)
            pressure_bar += (
                pressure_at_radius_bar(
                    parameters,
                    source_rate_mtpa,
                    elapsed_days=elapsed_days,
                    radius_m=distance_m,
                )
                - parameters.initial_pressure_bar
            )
        pressures[well_id] = pressure_bar
    return pressures


def multiwell_variable_rate_bottomhole_pressures_bar(
    parameters: LineSourceParameters,
    injection_rate_history_mtpa_by_well: dict[str, list[tuple[float, float]]],
    *,
    elapsed_days: float,
    well_distances_m: dict[str, dict[str, float]],
) -> dict[str, float]:
    pressures: dict[str, float] = {}
    for well_id, rate_history in injection_rate_history_mtpa_by_well.items():
        pressure_bar = variable_rate_bottomhole_pressure_bar(
            parameters,
            rate_history,
            elapsed_days=elapsed_days,
        )
        for source_well_id, source_rate_history in injection_rate_history_mtpa_by_well.items():
            if source_well_id == well_id or not source_rate_history:
                continue
            distance_m = _well_distance_m(well_distances_m, well_id, source_well_id)
            pressure_bar += (
                variable_rate_pressure_at_radius_bar(
                    parameters,
                    source_rate_history,
                    elapsed_days=elapsed_days,
                    radius_m=distance_m,
                )
                - parameters.initial_pressure_bar
            )
        pressures[well_id] = pressure_bar
    return pressures


def load_line_source_parameters(path: str | Path) -> LineSourceParameters:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _parameters_from_mapping(payload["line_source_inputs"])


def _parameters_from_mapping(payload: dict[str, Any]) -> LineSourceParameters:
    fields = LineSourceParameters.__dataclass_fields__
    values = {key: payload[key] for key in fields if key in payload}
    return LineSourceParameters(**values)


def _well_distance_m(
    well_distances_m: dict[str, dict[str, float]],
    observer_well_id: str,
    source_well_id: str,
) -> float:
    try:
        distance_m = well_distances_m[observer_well_id][source_well_id]
    except KeyError as exc:
        raise ValueError(f"Missing distance between {observer_well_id} and {source_well_id}") from exc
    if distance_m <= 0.0:
        raise ValueError("well distance must be positive")
    return distance_m


def _pressure_change_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
    radius_m: float,
    skin: float,
) -> float:
    if elapsed_days <= 0.0:
        raise ValueError("elapsed_days must be positive")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if injection_rate_mtpa < 0.0:
        raise ValueError("injection_rate_mtpa must be non-negative")

    return _signed_pressure_change_bar(
        parameters,
        injection_rate_mtpa,
        elapsed_days=elapsed_days,
        radius_m=radius_m,
        skin=skin,
    )


def _variable_rate_pressure_change_bar(
    parameters: LineSourceParameters,
    rate_history_mtpa: list[tuple[float, float]],
    *,
    elapsed_days: float,
    radius_m: float,
    skin: float,
) -> float:
    if elapsed_days <= 0.0:
        raise ValueError("elapsed_days must be positive")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")

    pressure_change_bar = 0.0
    previous_rate_mtpa = 0.0
    previous_start_day = -math.inf
    for start_day, rate_mtpa in rate_history_mtpa:
        if start_day < 0.0:
            raise ValueError("rate history start time must be non-negative")
        if start_day < previous_start_day:
            raise ValueError("rate history must be sorted by start time")
        if rate_mtpa < 0.0:
            raise ValueError("rate history rates must be non-negative")
        previous_start_day = start_day
        if start_day >= elapsed_days:
            break
        delta_rate_mtpa = rate_mtpa - previous_rate_mtpa
        if delta_rate_mtpa != 0.0:
            pressure_change_bar += _signed_pressure_change_bar(
                parameters,
                delta_rate_mtpa,
                elapsed_days=elapsed_days - start_day,
                radius_m=radius_m,
                skin=skin,
            )
        previous_rate_mtpa = rate_mtpa
    return pressure_change_bar


def _signed_pressure_change_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
    radius_m: float,
    skin: float,
) -> float:
    q_m3_s = annual_mt_to_kg_s(injection_rate_mtpa) / parameters.co2_density_kg_m3
    permeability_m2 = parameters.permeability_md * MILLIDARCY_TO_M2
    elapsed_s = elapsed_days * 24.0 * 3600.0
    diffusivity_argument = (
        parameters.porosity_fraction
        * parameters.viscosity_pa_s
        * parameters.total_compressibility_1_pa
        * radius_m
        * radius_m
        / (4.0 * permeability_m2 * elapsed_s)
    )
    response = _exponential_integral_e1(diffusivity_argument) + 2.0 * skin
    pressure_change_pa = (
        q_m3_s
        * parameters.viscosity_pa_s
        * response
        / (4.0 * math.pi * permeability_m2 * parameters.thickness_m)
    )
    return pressure_change_pa / PA_PER_BAR


def _exponential_integral_e1(x: float) -> float:
    if x <= 0.0:
        raise ValueError("x must be positive")
    if x <= 1.0:
        total = -EULER_GAMMA - math.log(x)
        term_power = 1.0
        factorial = 1.0
        sign = 1.0
        for n in range(1, 200):
            term_power *= x
            factorial *= n
            term = sign * term_power / (n * factorial)
            total += term
            if abs(term) < 1e-15:
                return total
            sign *= -1.0
        return total

    tiny = 1e-300
    b = x + 1.0
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        a = -float(i * i)
        b += 2.0
        d_denominator = a * d + b
        if abs(d_denominator) < tiny:
            d_denominator = tiny
        d = 1.0 / d_denominator
        c = b + a / c
        if abs(c) < tiny:
            c = tiny
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x)

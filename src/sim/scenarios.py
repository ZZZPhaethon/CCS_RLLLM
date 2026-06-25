from __future__ import annotations

import csv
import json
from pathlib import Path

from .entities.emitter import Emitter
from .entities.manifold import SubseaManifold
from .entities.pipeline import Pipeline
from .entities.state import PhysicalState
from .entities.storage import InjectionWell, Reservoir
from .entities.terminal import Terminal
from .entities.vessel import Vessel
from .line_source import LineSourceParameters
from .network import PhysicalNetwork

ROOT = Path(__file__).resolve().parents[2]
NORTHERN_LIGHTS_PHASE1_DATA_PATH = ROOT / "data" / "northern_lights_phase1_demo.json"
NORTHERN_LIGHTS_PHASE1_PLUS_YARA_DATA_PATH = ROOT / "data" / "northern_lights_phase1_plus_yara_2026.json"
PHASE1_PLUS_YARA_HOURLY_CAPTURE_PROFILE_PATH = ROOT / "data" / "phase1plus_emitters_capture_rate_profile_hourly.csv"
NORTHERN_LIGHTS_PHASE2_DATA_PATH = ROOT / "data" / "northern_lights_phase2_scenario.json"


def _load_phase1_data() -> dict:
    with NORTHERN_LIGHTS_PHASE1_DATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_phase2_data() -> dict:
    with NORTHERN_LIGHTS_PHASE2_DATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_phase1_plus_yara_data() -> dict:
    with NORTHERN_LIGHTS_PHASE1_PLUS_YARA_DATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_hourly_capture_profiles(path: Path) -> dict[str, tuple[float, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    profile_columns = {
        column.removesuffix("_capture_tph"): column
        for column in rows[0]
        if column.endswith("_capture_tph") and column != "total_capture_tph"
    }
    return {
        entity_id: tuple(float(row[column]) for row in rows)
        for entity_id, column in profile_columns.items()
    }


def _coordinate(values: list[float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


PHASE1_DATA = _load_phase1_data()
PHASE1_ANNUAL_TARGET_EXPORT_TPY = float(PHASE1_DATA["phase1_annual_target_export_tpy"])
PHASE1_PIPELINE_ANNUAL_CAPACITY_TPY = float(PHASE1_DATA["pipeline_annual_capacity_tpy"])
PHASE1_PIPELINE_MAX_FLOW_TPH = PHASE1_PIPELINE_ANNUAL_CAPACITY_TPY / 8760.0
NATURGASSPARKEN = _coordinate(PHASE1_DATA["locations"]["naturgassparken"])
EOS_SUBSEA_TEMPLATE_LOCATION = _coordinate(PHASE1_DATA["locations"]["eos_subsea_template"])
C1_INJECTION_WELL_LOCATION = _coordinate(PHASE1_DATA["locations"]["c1_injection_well"])
OFFSHORE_PIPELINE_ROUTE = [
    _coordinate(point)
    for point in PHASE1_DATA["offshore_pipeline_route"]
]
AURORA_LINE_SOURCE_PARAMETERS = LineSourceParameters(
    **PHASE1_DATA["line_source_parameters"]
)
AURORA_LINE_SOURCE_PARAMETER_STATUS = dict(PHASE1_DATA["line_source_parameter_status"])


def build_northern_lights_phase1_demo() -> tuple[PhysicalNetwork, PhysicalState]:
    """Small Phase 1-style topology for smoke tests and downstream integration.

    Values are intentionally first-order, based on the project documents:
    1 h step, roughly 8,000 t ship cargo, about 800 t/h loading/unloading,
    one-ship terminal buffer, one pipeline, and two injection wells.
    """

    network = PhysicalNetwork(time_step_hours=float(PHASE1_DATA["time_step_hours"]))
    for emitter in PHASE1_DATA["emitters"]:
        annual_target_export_tpy = float(emitter.get("annual_target_export_tpy", PHASE1_ANNUAL_TARGET_EXPORT_TPY))
        network.add_entity(
            Emitter(
                emitter["entity_id"],
                nominal_capture_tph=annual_target_export_tpy / 8760.0,
                buffer_capacity_t=float(emitter["buffer_capacity_t"]),
                min_utilization=float(emitter["min_utilization"]),
                loading_rate_tph=float(emitter["loading_rate_tph"]),
                annual_target_export_tpy=annual_target_export_tpy,
                max_production_tph=float(emitter["max_production_tph"]),
                reference_name=emitter["reference_name"],
            )
        )
    for vessel in PHASE1_DATA["vessels"]:
        network.add_entity(
            Vessel(
                vessel["entity_id"],
                capacity_t=float(vessel["capacity_t"]),
                loading_rate_tph=float(vessel["loading_rate_tph"]),
                unloading_rate_tph=float(vessel["unloading_rate_tph"]),
                volume_capacity_m3=float(vessel["volume_capacity_m3"]),
                speed_knots=float(vessel["speed_knots"]),
            )
        )
    terminal = PHASE1_DATA["terminal"]
    network.add_entity(
        Terminal(
            terminal["entity_id"],
            storage_capacity_t=float(terminal["storage_capacity_t"]),
            berth_count=int(terminal["berth_count"]),
            site_name=terminal["site_name"],
        )
    )
    pipeline = PHASE1_DATA["pipeline"]
    network.add_entity(
        Pipeline(
            pipeline["entity_id"],
            max_flow_tph=PHASE1_PIPELINE_MAX_FLOW_TPH,
            ramp_tph=float(pipeline["ramp_tph"]),
            annual_capacity_tpy=PHASE1_PIPELINE_ANNUAL_CAPACITY_TPY,
            length_km=float(pipeline["length_km"]),
            route_color=pipeline["route_color"],
            route_coordinates=OFFSHORE_PIPELINE_ROUTE,
        )
    )
    manifold = PHASE1_DATA["manifold"]
    network.add_entity(
        SubseaManifold(
            manifold["entity_id"],
            max_flow_tph=PHASE1_PIPELINE_MAX_FLOW_TPH,
        )
    )
    for well in PHASE1_DATA["injection_wells"]:
        network.add_entity(
            InjectionWell(
                well["entity_id"],
                max_injection_tph=float(well["max_injection_tph"]),
            )
        )
    reservoir = PHASE1_DATA["reservoir"]
    network.add_entity(
        Reservoir(
            reservoir["entity_id"],
            storage_capacity_t=float(reservoir["storage_capacity_t"]),
            initial_pressure_bar=float(reservoir["initial_pressure_bar"]),
            pressure_at_capacity_bar=float(reservoir["pressure_at_capacity_bar"]),
            max_pressure_bar=float(reservoir["max_pressure_bar"]),
            depth_m=float(reservoir["depth_m"]),
            line_source_parameters=AURORA_LINE_SOURCE_PARAMETERS,
            line_source_observation_radii_m=tuple(
                float(radius_m)
                for radius_m in reservoir["line_source_observation_radii_m"]
            ),
            line_source_well_distances_m=reservoir["line_source_well_distances_m"],
            line_source_parameter_status=AURORA_LINE_SOURCE_PARAMETER_STATUS,
        )
    )

    network.connect("brevik", "northern_pioneer")
    network.connect("celsio", "northern_pioneer")
    network.connect("brevik", "northern_pathfinder")
    network.connect("celsio", "northern_pathfinder")
    network.connect("northern_pioneer", "oygarden_terminal")
    network.connect("northern_pathfinder", "oygarden_terminal")
    network.connect("oygarden_terminal", "oygarden_pipeline")
    network.connect("oygarden_pipeline", "aurora_subsea_manifold")
    network.connect("aurora_subsea_manifold", "aurora_well_a")
    network.connect("aurora_subsea_manifold", "aurora_well_c")
    network.connect("aurora_well_a", "aurora_reservoir")
    network.connect("aurora_well_c", "aurora_reservoir")

    state = PhysicalState(
        entity_inventory_t={
            "brevik": 0.0,
            "celsio": 0.0,
            "northern_pioneer": 0.0,
            "northern_pathfinder": 0.0,
            "oygarden_terminal": 0.0,
            "oygarden_pipeline": 0.0,
            "aurora_subsea_manifold": 0.0,
            "aurora_well_a": 0.0,
            "aurora_well_c": 0.0,
            "aurora_reservoir": 0.0,
        }
    )
    return network, state


def build_northern_lights_phase2_demo() -> tuple[PhysicalNetwork, PhysicalState]:
    """Phase 2-style topology from the public-data scenario JSON.

    The scenario keeps the current two public well locations for visualization and
    high-level operations checks; it does not instantiate placeholder Phase 2 wells
    whose coordinates are not public in the reviewed sources.
    """

    return _build_network_from_scenario_data(_load_phase2_data())


def build_northern_lights_phase1_plus_yara_demo() -> tuple[PhysicalNetwork, PhysicalState]:
    """Commercial ramp-up scenario with Brevik, Celsio, Yara, and four Phase 1 ships."""

    return _build_network_from_scenario_data(
        _load_phase1_plus_yara_data(),
        hourly_capture_profiles=_load_hourly_capture_profiles(PHASE1_PLUS_YARA_HOURLY_CAPTURE_PROFILE_PATH),
    )


def _build_network_from_scenario_data(
    data: dict,
    hourly_capture_profiles: dict[str, tuple[float, ...]] | None = None,
) -> tuple[PhysicalNetwork, PhysicalState]:
    network = PhysicalNetwork(time_step_hours=float(data["time_step_hours"]))
    hourly_capture_profiles = hourly_capture_profiles or {}

    for emitter in data["emitters"]:
        annual_target_export_tpy = float(emitter["annual_target_export_tpy"])
        network.add_entity(
            Emitter(
                emitter["entity_id"],
                nominal_capture_tph=float(emitter.get("nominal_capture_tph", annual_target_export_tpy / 8760.0)),
                buffer_capacity_t=float(emitter["buffer_capacity_t"]),
                min_utilization=float(emitter["min_utilization"]),
                loading_rate_tph=float(emitter["loading_rate_tph"]),
                annual_target_export_tpy=annual_target_export_tpy,
                max_production_tph=float(emitter["max_production_tph"]),
                reference_name=emitter["reference_name"],
                hourly_capture_profile_tph=hourly_capture_profiles.get(emitter["entity_id"]),
            )
        )

    for vessel in data["vessels"]:
        network.add_entity(
            Vessel(
                vessel["entity_id"],
                capacity_t=float(vessel["capacity_t"]),
                loading_rate_tph=float(vessel["loading_rate_tph"]),
                unloading_rate_tph=float(vessel["unloading_rate_tph"]),
                volume_capacity_m3=float(vessel["volume_capacity_m3"]),
                speed_knots=float(vessel["speed_knots"]),
            )
        )

    terminal = data["terminal"]
    network.add_entity(
        Terminal(
            terminal["entity_id"],
            storage_capacity_t=float(terminal["storage_capacity_t"]),
            berth_count=int(terminal["berth_count"]),
            site_name=terminal["site_name"],
        )
    )

    pipeline = data["pipeline"]
    network.add_entity(
        Pipeline(
            pipeline["entity_id"],
            max_flow_tph=float(pipeline["max_flow_tph"]),
            ramp_tph=float(pipeline["ramp_tph"]),
            annual_capacity_tpy=float(pipeline["annual_capacity_tpy"]),
            length_km=float(pipeline["length_km"]),
            route_color=pipeline["route_color"],
            route_coordinates=[_coordinate(point) for point in data["offshore_pipeline_route"]],
        )
    )

    manifold = data["manifold"]
    network.add_entity(
        SubseaManifold(
            manifold["entity_id"],
            max_flow_tph=float(manifold["max_flow_tph"]),
        )
    )

    for well in data["injection_wells"]:
        network.add_entity(
            InjectionWell(
                well["entity_id"],
                max_injection_tph=float(well["max_injection_tph"]),
            )
        )

    reservoir = data["reservoir"]
    network.add_entity(
        Reservoir(
            reservoir["entity_id"],
            storage_capacity_t=float(reservoir["storage_capacity_t"]),
            initial_pressure_bar=float(reservoir["initial_pressure_bar"]),
            pressure_at_capacity_bar=float(reservoir["pressure_at_capacity_bar"]),
            max_pressure_bar=float(reservoir["max_pressure_bar"]),
            depth_m=float(reservoir["depth_m"]),
            line_source_parameters=LineSourceParameters(**data["line_source_parameters"]),
            line_source_observation_radii_m=tuple(
                float(radius_m)
                for radius_m in reservoir["line_source_observation_radii_m"]
            ),
            line_source_well_distances_m=reservoir["line_source_well_distances_m"],
            line_source_parameter_status=dict(data["line_source_parameter_status"]),
        )
    )

    for source, target in data["connections"]:
        network.connect(source, target)

    state = PhysicalState(
        entity_inventory_t={
            entity_id: 0.0
            for entity_id in network.entities
        }
    )
    return network, state

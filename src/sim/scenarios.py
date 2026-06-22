from __future__ import annotations

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


def _load_phase1_data() -> dict:
    with NORTHERN_LIGHTS_PHASE1_DATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _coordinate(values: list[float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


PHASE1_DATA = _load_phase1_data()
PHASE1_ANNUAL_TARGET_EXPORT_TPY = float(PHASE1_DATA["phase1_annual_target_export_tpy"])
PHASE1_NOMINAL_CAPTURE_TPH = PHASE1_ANNUAL_TARGET_EXPORT_TPY / 8760.0
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
        network.add_entity(
            Emitter(
                emitter["entity_id"],
                nominal_capture_tph=PHASE1_NOMINAL_CAPTURE_TPH,
                buffer_capacity_t=float(emitter["buffer_capacity_t"]),
                min_utilization=float(emitter["min_utilization"]),
                loading_rate_tph=float(emitter["loading_rate_tph"]),
                annual_target_export_tpy=PHASE1_ANNUAL_TARGET_EXPORT_TPY,
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

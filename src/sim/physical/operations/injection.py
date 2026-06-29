from __future__ import annotations

from ..disturbances import well_max_injection_tph
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell, Reservoir


def inject_to_well(
    network,
    state: PhysicalState,
    flows: dict[tuple[str, str], float],
    source_id: str,
    well_id: str,
    amount_t: float,
) -> None:
    network._move(state, flows, source_id, well_id, amount_t)
    state.last_injection_flow_tph[well_id] = (
        state.last_injection_flow_tph.get(well_id, 0.0)
        + amount_t / network.time_step_hours
    )
    reservoir_id = network._single_downstream_of_type(well_id, Reservoir)
    if reservoir_id is not None:
        network._move(state, flows, well_id, reservoir_id, amount_t)


def well_remaining_capacity(network, well_id: str, state: PhysicalState) -> float:
    well = network.entities[well_id]
    assert isinstance(well, InjectionWell)
    effective_max_tph = well_max_injection_tph(state, well)
    if effective_max_tph <= 0.0:
        return 0.0
    well_capacity_t = effective_max_tph * network.time_step_hours
    reservoir_id = network._single_downstream_of_type(well_id, Reservoir)
    if reservoir_id is None:
        return well_capacity_t
    return min(well_capacity_t, _reservoir_remaining_capacity(network, reservoir_id, state))


def _reservoir_remaining_capacity(network, reservoir_id: str, state: PhysicalState) -> float:
    reservoir = network.entities[reservoir_id]
    assert isinstance(reservoir, Reservoir)
    current_t = state.entity_inventory_t.get(reservoir_id, 0.0)
    limit_t = min(reservoir.storage_capacity_t, reservoir.pressure_limited_capacity_t())
    return max(0.0, limit_t - current_t)

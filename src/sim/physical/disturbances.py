"""Runtime resolution of time-varying disturbances (the ``ξ_t`` channel).

The physical entities (:class:`Emitter`, :class:`InjectionWell`, ...) are frozen
dataclasses holding *nominal* parameters. Operational disturbances - weather
slowing vessels, capture-plant deratings, well maintenance, berth outages and
injectivity decline - are time-varying and therefore live in
:class:`PhysicalState` instead, written per timestep by a scenario generator or
an RL/evaluation harness.

Every consumer of a disturbance-affected parameter resolves it through these
helpers so there is a single place that defines the "override on state, else
fall back to the nominal entity value" rule.
"""

from __future__ import annotations

from .entities.emitter import Emitter
from .entities.state import PhysicalState
from .entities.storage import InjectionWell
from .entities.terminal import Terminal


def emitter_availability(state: PhysicalState, emitter: Emitter) -> float:
    """Capture availability factor in ``[0, 1]`` for an emitter."""
    value = state.emitter_availability.get(emitter.entity_id, emitter.availability)
    return max(0.0, min(1.0, value))


def well_is_available(state: PhysicalState, well: InjectionWell) -> bool:
    """Whether an injection well can accept flow this step."""
    return bool(state.well_available.get(well.entity_id, well.available))


def well_injectivity_factor(state: PhysicalState, well: InjectionWell) -> float:
    """Non-negative multiplier applied to a well's nominal max injection rate.

    Defaults to ``1.0`` (nominal). Values below ``1.0`` model injectivity
    decline or partial deratings; ``0.0`` means no injection capacity.
    """
    return max(0.0, float(state.injectivity_factor.get(well.entity_id, 1.0)))


def well_max_injection_tph(state: PhysicalState, well: InjectionWell) -> float:
    """Effective per-hour injection ceiling after availability and injectivity."""
    if not well_is_available(state, well):
        return 0.0
    return well.max_injection_tph * well_injectivity_factor(state, well)


def vessel_speed_factor(state: PhysicalState, vessel_id: str) -> float:
    """Non-negative multiplier on a vessel's nominal sailing speed.

    Defaults to ``1.0``. Values below ``1.0`` model weather-induced slowdowns.
    """
    return max(0.0, float(state.vessel_speed_factor.get(vessel_id, 1.0)))


def terminal_berth_count(state: PhysicalState, terminal: Terminal) -> int:
    """Number of usable berths after any outage override."""
    value = state.berth_count_override.get(terminal.entity_id, terminal.berth_count)
    return max(0, int(value))

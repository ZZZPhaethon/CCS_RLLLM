from __future__ import annotations

from ..entities.emitter import Emitter
from ..entities.state import PhysicalState, Violation


def apply_capture(
    network,
    state: PhysicalState,
    actions: dict[str, dict[str, object]],
    violations: list[Violation],
) -> float:
    generated_t = 0.0
    for emitter_id, emitter in network._entities_of_type(Emitter).items():
        utilization = actions.get(emitter_id, {}).get("utilization", emitter.default_utilization)
        utilization = max(emitter.min_utilization, min(1.0, utilization))
        requested_t = (
            emitter.nominal_capture_tph
            * utilization
            * emitter.availability
            * network.time_step_hours
        )
        current_t = state.entity_inventory_t.get(emitter_id, 0.0)
        free_t = max(0.0, emitter.buffer_capacity_t - current_t)
        actual_t = min(requested_t, free_t)
        state.entity_inventory_t[emitter_id] = current_t + actual_t
        generated_t += actual_t
        if actual_t < requested_t:
            violations.append(
                Violation(
                    "buffer_overflow",
                    emitter_id,
                    requested_t,
                    actual_t,
                    requested_t - actual_t,
                    "Emitter buffer capacity clipped captured CO2.",
                )
            )
    return generated_t

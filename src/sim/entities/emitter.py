from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Emitter:
    entity_id: str
    nominal_capture_tph: float
    buffer_capacity_t: float
    min_utilization: float = 0.0
    default_utilization: float = 1.0
    availability: float = 1.0
    loading_rate_tph: float = 800.0
    annual_target_export_tpy: float | None = None
    max_production_tph: float | None = None
    reference_name: str | None = None

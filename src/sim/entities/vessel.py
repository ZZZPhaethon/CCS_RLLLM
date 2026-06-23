from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vessel:
    entity_id: str
    capacity_t: float
    loading_rate_tph: float
    unloading_rate_tph: float
    volume_capacity_m3: float | None = None
    speed_knots: float | None = None

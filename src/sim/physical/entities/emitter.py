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
    hourly_capture_profile_tph: tuple[float, ...] | None = None

    def capture_rate_tph_at(self, interval_start_h: float) -> float:
        if not self.hourly_capture_profile_tph:
            return self.nominal_capture_tph
        hour_index = int(interval_start_h) % len(self.hourly_capture_profile_tph)
        return self.hourly_capture_profile_tph[hour_index]

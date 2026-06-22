from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Terminal:
    entity_id: str
    storage_capacity_t: float
    berth_count: int = 1
    site_name: str | None = None

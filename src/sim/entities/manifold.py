from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubseaManifold:
    entity_id: str
    max_flow_tph: float
    available: bool = True

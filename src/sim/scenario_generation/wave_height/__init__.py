"""Wave-height driven weather scenarios.

This package turns historical or forecast significant-wave-height fields into
the existing ``Scenario.vessel_speed_factor`` disturbance interface.
"""

from .netcdf import ClassicNetCDF, NetCDFVariable
from .routes import (
    RouteWaveConfig,
    aggregate_wave_heights,
    densify_route,
    route_wave_height_series,
)
from .scenario import WaveHeightScenarioGenerator

__all__ = [
    "ClassicNetCDF",
    "NetCDFVariable",
    "RouteWaveConfig",
    "WaveHeightScenarioGenerator",
    "aggregate_wave_heights",
    "densify_route",
    "route_wave_height_series",
]

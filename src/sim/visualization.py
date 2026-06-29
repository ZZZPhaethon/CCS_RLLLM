"""Backward-compatible import path for :mod:`sim.reporting.visualization`."""

from .reporting import visualization as _visualization

globals().update(
    {
        name: value
        for name, value in _visualization.__dict__.items()
        if name not in {"__builtins__", "__cached__", "__loader__", "__name__", "__package__", "__spec__"}
    }
)

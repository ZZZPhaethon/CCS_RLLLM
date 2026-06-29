"""Layered simulator and control stack for ship-based CCS logistics.

Primary implementation packages:

- :mod:`sim.physical` - physical entities, operations, scenarios, and simulator.
- :mod:`sim.rl` - RL environments, training adapters, and evaluation metrics.
- :mod:`sim.optimization` - MILP and rolling-horizon optimization baselines.
- :mod:`sim.policies` - hand-written dispatch policies.
- :mod:`sim.reporting` - visualization and dashboard generation.

Legacy modules such as :mod:`sim.env` and :mod:`sim.network` are kept as thin
compatibility import paths.
"""


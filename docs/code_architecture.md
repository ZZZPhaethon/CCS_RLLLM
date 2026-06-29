# Code Architecture

The `sim` package is split by responsibility so the research layers are easier
to navigate.

```text
src/sim/
  physical/       Physical CCS digital twin
    entities/     Emitters, vessels, terminal, pipeline, wells, reservoir, state
    operations/   Capture, loading, unloading, transport, injection, snapshots
    actions.py    Physical action dataclasses
    network.py    One-step mass-flow and constraint settlement
    simulator.py  Action resolution plus network stepping
    scenario.py   Exogenous disturbance generator
    scenarios.py  Northern Lights-style scenario builders
    economics.py  Step-level physical/economic accounting

  rl/             Reinforcement-learning layer
    env.py        Discrete hourly CCS control environment
    gym_env.py    Gymnasium adapter and action masks
    metrics.py    Common rollout scorecard
    train.py      PPO training entry point

  optimization/   Optimization baselines
    milp.py        Static perfect-information MILP benchmark
    rolling_milp.py Rolling-horizon MILP/MPC controller

  policies/       Non-RL controllers
    rule_based.py  Deterministic rule-based dispatcher

  reporting/      Visualization and dashboards
    visualization.py HTML/dashboard payload generation
```

Backward-compatible modules still exist at the old paths, for example
`sim.env`, `sim.network`, `sim.scenarios`, and `sim.visualization`. New code
should prefer the layered imports, such as `sim.physical.network.PhysicalNetwork`
or `sim.rl.env.CCSEnv`.

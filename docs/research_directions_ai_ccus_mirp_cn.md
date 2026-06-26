# AI + ship-based CCUS / MIRP 研究方向记录

记录日期: 2026-06-22

## 方向 A: Ship-based CCUS network operation

英文题目候选:

> A physics-constrained multi-agent reinforcement learning framework for real-time operation of ship-based CCUS networks under uncertain capture, transport, and injection dynamics.

中文定位:

> 面向船运 CCUS 网络的物理约束多智能体强化学习实时运营调度: 考虑捕集、运输和注入动态不确定性。

核心问题:

- 多个 emitter 持续产生/捕集 CO2, 但船运是 batch-wise 的。
- terminal 储罐容量有限, pipeline / injection wells 需要尽量连续、平滑地接收 CO2。
- 捕集量、船期、天气、泊位、终端库存、井注入能力都可能扰动。
- 需要实时协调 source, vessels, terminal, pipeline, wells, 同时保证物理约束可行。

可能的 agents:

| Agent | 决策内容 |
|---|---|
| Emitter agent | capture rate, source buffer, loading request, curtail/vent |
| Vessel agent | dispatch, routing, loading/unloading amount, waiting/sailing decision |
| Terminal agent | berth assignment, tank allocation, unloading rate, terminal-to-pipeline flow |
| Storage/injection agent | well allocation, injection rate, shutdown/restart |
| LLM/operator agent | high-level plan, violation explanation, emergency scenario reasoning |

状态空间:

- emitter inventory and capture forecast
- vessel location, cargo, ETA, status
- terminal tank inventory and berth status
- pipeline flow/pressure proxy
- well capacity, injectivity, bottomhole/reservoir pressure proxy
- weather, delay, price/carbon penalty, time

动作空间:

- vessel assignment and sailing destination
- loading/unloading amount or rate
- berth/tank choice
- terminal-to-pipeline flow
- well injection allocation
- curtail/vent emergency action

目标函数 / reward:

- maximize stored CO2
- minimize vessel waiting/demurrage
- minimize compression/transport/storage cost
- penalize terminal overflow/empty events
- penalize venting or uncaptured CO2
- penalize pipeline/injection fluctuation
- heavily penalize physical violations

核心创新点:

- RL/MARL 不直接拥有最终执行权, 而是提出 requested action。
- Physical layer 执行 action projection / safety shield, 将 requested action 投影到可行集合。
- 每次裁剪都返回 structured violation report, 如 `berth_required`, `flow_clipped`, `tank_capacity_clipped`, `injection_pressure_limited`。
- LLM 不直接控制连续物理流量, 而是用于高层协调、异常解释和应急方案生成。

风险:

- ship-based CCUS 运营调度本身文献较少, 容易被质疑实际价值和问题成熟度。
- 更适合作为 CCS 应用论文或 benchmark/framework 论文, 不一定适合作为第一篇主线。

## 方向 B: Dynamic Maritime Inventory Routing Problem (MIRP)

英文题目候选:

> Physics-constrained multi-agent reinforcement learning for dynamic maritime inventory routing under uncertain production and demand.

带 CCUS 应用版本:

> Physics-constrained multi-agent reinforcement learning for dynamic maritime inventory routing in ship-based CCUS supply chains.

中文定位:

> 面向不确定生产/需求的动态海运库存路径问题: 基于物理约束多智能体强化学习的实时调度方法。

为什么更稳:

- MIRP 是已知且成熟的问题, 在 LNG、原油、化学品、燃料补给和 CO2 船运中都有现实价值。
- 经典 MIRP 已经包含 production/source, vessel batch transport, terminal inventory, downstream demand/sink。
- 可以把 CCS 作为应用场景之一, 而不是声称发明一个全新问题。
- 论文叙事更容易成立: 已知问题 + 新方法 + 新物理约束场景。

统一物理结构:

```text
source / production
  -> vessel batch transport
  -> terminal inventory
  -> downstream continuous demand or sink
```

对应应用:

| 行业 | source | vessel batch | inventory | downstream |
|---|---|---|---|---|
| LNG | liquefaction/import terminal | LNG carrier | satellite terminal tank | coastal/inland gas demand |
| 原油 | VLCC parcels | ship unloading | refinery storage tanks | CDU continuous feed |
| CO2/CCUS | emitter/capture plant | LCO2 carrier | receiving terminal tank | pipeline/injection wells |

研究问题:

> 在动态、不确定 MIRP 中, 如何用 MARL 实时协调船舶、港口储罐和下游连续需求, 同时通过物理约束层保证库存、泊位、流量和安全约束不被违反?

推荐实验路径:

| Case | 规模 | 目的 |
|---|---|---|
| Toy MIRP | 2 sources, 2 vessels, 1 terminal, 2 sinks | 验证 MDP/MARL/action projection |
| Crude-like case | 1-3 vessels, 4-8 tanks, 1 continuous sink group | 对齐 Applied Energy 2019 原油调度结构 |
| LNG-like case | 1 supply port, multiple satellite terminals, uncertain demand | 对齐 Applied Energy / Energy LNG MIRP 文献 |
| CCUS case | emitters, vessels, terminal, pipeline/wells | 展示本文应用价值 |

Baselines:

- rule-based dispatch / FIFO
- rolling-horizon MILP / MPC
- single-agent PPO/SAC/TD3
- independent MARL
- CTDE MARL, e.g. MAPPO/MADDPG
- physics-constrained MARL
- optional: LLM planner + MARL + physical shield

建议主线:

1. 主问题选 MIRP。
2. 物理层使用当前 CCS simulator 的 action projection / violation reporting 思路。
3. 应用场景先做 LNG/crude-like, 再扩展到 ship-based CCUS。
4. 方法上突出 physics-constrained MARL, LLM 作为可选高层协调器。

## 关键参考文献

| 用途 | 文献 |
|---|---|
| LNG/MIRP 基础问题 | Applied Energy 2015, An MILP model for optimization of a small-scale LNG supply chain along a coastline, DOI: 10.1016/j.apenergy.2014.10.039 |
| LNG stochastic inventory routing | Energy 2017, A three-stage stochastic programming method for LNG supply system infrastructure development and inventory routing in demanding countries, DOI: 10.1016/j.energy.2017.05.090 |
| LNG fleet planning | Applied Energy 2024, LNG market liberalization and LNG transportation, DOI: 10.1016/j.apenergy.2024.122657 |
| 原油 batch + tank + continuous sink | Applied Energy 2019, Preventive crude oil scheduling under demand uncertainty, DOI: 10.1016/j.apenergy.2018.10.121 |
| CCUS agent-based transport | Applied Energy 2025, CCUS-Agent, DOI: 10.1016/j.apenergy.2024.124833 |
| AI + CCS/P2G 调度 | Applied Energy 2024, Dynamic optimization of IES with CCS-P2G, DOI: 10.1016/j.apenergy.2024.123390 |
| AI + PCCS/DACS 实时调度 | Applied Energy 2023, Automated DRL for multi-energy system with post-carbon and direct-air carbon capture, DOI: 10.1016/j.apenergy.2022.120633 |
| Safe RL 约束处理 | IEEE TSG 2020, Constrained EV Charging Scheduling Based on Safe Deep Reinforcement Learning, DOI: 10.1109/tsg.2019.2955437 |
| 运输网络 MARL | Transportation Research Part C 2020, Multi-vehicle routing problems with soft time windows: A MARL approach, DOI: 10.1016/j.trc.2020.102861 |
| 交通-能源耦合 MARL | Applied Energy 2026, Multi-agent heterogeneous graph RL for EV routing and charging, DOI: 10.1016/j.apenergy.2025.126958 |
| 工业调度 RL | IEEE TASE 2024, A Reinforcement Learning Based Large-Scale Refinery Production Scheduling Algorithm, DOI: 10.1109/tase.2023.3321612 |
| Crude scheduling safe RL | Computers & Chemical Engineering 2026, Safe RL for crude oil scheduling, DOI: 10.1016/j.compchemeng.2025.109480 |
| CO2 ship transport design | IJGGC 2020, Optimal design and cost of ship-based CO2 transport under uncertainties and fluctuations, DOI: 10.1016/j.ijggc.2020.103190 |
| Recent CO2 ship transport TEA | IJGGC 2025, Techno-economic analysis of large-scale CO2 ship transport with onboard BOG reliquefaction, DOI: 10.1016/j.ijggc.2025.104337 |

## 当前判断

更稳的第一主线是方向 B: Dynamic MIRP + physics-constrained MARL。

方向 A 可以作为方向 B 的重要应用场景, 或作为第二篇论文/benchmark paper 展开。这样既避免“无人研究所以价值不明”的风险, 又能保留 ship-based CCUS 的差异化和当前物理层代码资产。

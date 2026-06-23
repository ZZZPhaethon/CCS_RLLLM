# CCS_RLLLM

一个面向船运 CCS 链条的物理层仿真框架。当前版本重点实现 Layer V：把上层调度器、MPC/RL 或多 agent 系统给出的动作，投影到可执行的物理流量、库存、注入和压力状态上。

## 当前物理层包含什么

默认 demo 使用 Northern Lights Phase 1 风格的简化拓扑：

```text
Emitter -> Vessel -> Terminal -> Pipeline -> SubseaManifold -> InjectionWell -> Reservoir
```

主要实体包括：

- `Emitter`：排放端/捕集端，产生 CO2，并向船装载。
- `Vessel`：CO2 船，负责从排放端航行到接收终端。
- `Terminal`：接收终端，负责卸船和临时缓冲。
- `Pipeline`：从终端向海底侧外输。
- `SubseaManifold`：海底分配节点，把流量分给不同注入井。
- `InjectionWell`：注入井，控制可用性和注入能力。
- `Reservoir`：储层，记录库存、压力和 line-source 压力诊断。

物理层负责做容量裁剪、质量守恒检查和违规记录。例如船不在泊位时请求装卸，会产生 `berth_required`；请求流量超过管道、井或库存能力时，会产生 `flow_clipped`。

## 一步仿真怎么执行

每个仿真步接收一个 action frame。所有 action 先经过 `ActionResolver` 验证，然后进入物理层执行。

当前结算顺序是：

1. 排放端按捕集利用率产生 CO2。
2. Terminal/Pipeline/Storage 侧结算卸船、外输、井间分配和注入。
3. 排放端执行装船。
4. 更新注入历史、库存、压力和 snapshot。

注意：到达 `Terminal` 的 CO2 不会自动流入下游。需要同时给 `Pipeline` 的 `set_flow` 动作，才会从 terminal 外输到下游。

## Agent 动作接口

当前代码没有单独的 `Agent` 类；`agent_id` 是动作提交者标识。动作能力按物理实体类型定义：

| 类型 | 动作 |
|---|---|
| `Emitter` | `set_capture_utilization(utilization)`、`load_vessel(vessel_id)`、`hold()` |
| `Vessel` | `sail_to(destination_id)`、`hold()` |
| `Terminal` | `unload_vessel(vessel_id)`、`hold()` |
| `Pipeline` | `set_flow(flow_tph)`、`hold()` |
| `SubseaManifold` | `set_well_split(well_splits)`、`hold()` |
| `InjectionWell` | `set_available(available)`、`set_injection_limit(max_injection_tph)`、`hold()` |
| `Reservoir` | `hold()` |

示例：

```python
from sim.actions import ActionFrame, ActionProposal

frame = ActionFrame(
    time_h=0.0,
    proposals=[
        ActionProposal(
            agent_id="terminal_agent",
            entity_id="oygarden_terminal",
            verb="unload_vessel",
            params={"vessel_id": "northern_pioneer"},
        ),
        ActionProposal(
            agent_id="pipeline_agent",
            entity_id="oygarden_pipeline",
            verb="set_flow",
            params={"flow_tph": 200.0},
        ),
    ],
)
```

## 快速运行

PowerShell：

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
python examples\run_physical_layer_demo.py
python -m unittest discover -s tests
```

生成可交互 dashboard：

```powershell
$env:PYTHONPATH='E:\CCS_RLLLM\src'
python examples\build_physical_dashboard.py
```

输出文件：

```text
docs/physical_layer_dashboard.html
```

## 主要代码位置

- `src/sim/entities/`：物理实体定义。
- `src/sim/actions.py`：动作数据结构。
- `src/sim/action_resolver.py`：动作验证与翻译。
- `src/sim/network.py`：物理网络和一步结算逻辑。
- `src/sim/operations/`：捕集、装船、卸船、管道输送、注入等模块。
- `src/sim/scenarios.py`：Northern Lights Phase 1 demo 构建。
- `src/sim/visualization.py`：dashboard 数据和 HTML 生成。
- `tests/`：物理层、动作接口和可视化测试。

## 当前边界

当前仓库主要是物理层框架。上层调度器、MPC/RL 训练、LLM 协商协议和合同结算尚未作为完整模块实现。物理层的职责是接收外部动作，执行可行性投影，并返回可审计的状态、流量和违规信息。

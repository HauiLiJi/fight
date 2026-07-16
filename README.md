# 双机编队协同空战挑战

## 任务描述

本任务为双机编队协同空战。参赛者需要开发 Agent，根据己方两架 F-22 的状态和
可见敌方航迹，实时生成飞行控制与武器使用动作，控制编队与对方进行空中对抗。
Agent 需要完成态势判断、编队协同、机动决策和火力运用，并在保存己方作战能力的
同时争取击败对方编队。

## 项目结构

```text
src/air_combat_challenge/   可安装的 Python 包
configs/scenarios/          飞机部署与想定配置
scenarios/air_to_air/       F22 AFSIM 模型、武器和脚本资源
examples/rules/             基于 BaseAgent 的确定性规则示例
docs/schemas/               从 Pydantic 模型生成的 V1 JSON Schema
```

## 环境安装

```powershell
uv sync
```

## 比赛 Agent

Agent 实现统一接口：

```python
from air_combat_challenge.competition.agents import BaseAgent
from air_combat_challenge.competition.models import ActionBatchV1


class MyAgent(BaseAgent):
    def act(self, observation):
        return ActionBatchV1(
            actions=[]
        )
```

LLM 可以在 `act()` 内根据 `ObservationV1` 直接生成动作，也可以赛前生成完整
Agent 或规则文件。文件只在启动时加载，局内不进行热更新。

Agent 通过 JSON manifest 显式加载，不扫描目录。manifest 所在目录就是该 Agent 的
提交根目录，`entrypoint` 是相对于提交根目录的入口文件：

```json
{
  "api_version": "1.0",
  "topology": "team",
  "entrypoint": "agent.py",
  "step_timeout_s": 5.0
}
```

`topology` 可取 `team` 或 `per_aircraft`，分别表示一个 Agent 控制全队或每架飞机
独立运行一个 Agent。入口文件必须且只能直接定义一个 `BaseAgent` 子类；从其他
模块导入的 Agent 类不计入候选。建议每个提交目录只放置一个 Agent。

Agent 可以由多个 Python 文件组成，本地模块使用包内相对导入：

```text
my_agent/
├── agent.json
├── agent.py
├── planner.py
└── utils/
    ├── __init__.py
    └── geometry.py
```

```python
from .planner import Planner
from .utils.geometry import distance
```

加载器会递归编译提交目录内全部 `.py`，并根据规范化相对路径和文件内容计算统一
SHA-256。worker 启动时会重新计算并拒绝加载已变化的源码树；比赛期间不支持热更新。
第三方依赖必须已经安装在 baseline 环境中，框架不会读取依赖清单或自动安装包。
manifest、JSON、提示词、模型文件和其他非 Python 资源不包含在当前源码树哈希中。

验证示例规则 Agent：

```powershell
uv run air-combat validate-agent examples/rules/a2a_rule_agent.json
```

## 启动 AFSIM

### Warlock 可视化

Warlock 提供地图界面，可以观察飞机、导弹和平台状态，同时负责运行仿真和加载 gRPC
服务。通过鼠标操作启动：

1. 打开 AFSIM 安装目录，进入 `bin` 文件夹，双击 `warlock.exe`。
2. 在 Warlock 点击 `Brower`。
3. 选择项目中的 `scenarios/air_to_air/start_up.txt`，点击“打开”。
4. 点击 `Run`，等待想定加载完成，并保持 Warlock 窗口运行。飞机会在 Python 环境开始一局时创建。

当前 `grpc_config.txt` 将服务端口固定为 `19920`。`--episodes N` 会复用同一个 AFSIM
进程并按顺序重置、运行 N 局；当前 baseline 不支持同机多实例并行跑批。CLI 的
`--port` 只指定 Python 客户端连接的端口，不会修改 AFSIM 服务端口。如需并行运行，
必须为每个 AFSIM 实例生成独立的 gRPC 配置和端口。

保持 Warlock 窗口运行，在项目根目录打开 PowerShell 检查 gRPC 服务：

```powershell
uv run air-combat check-afsim
```

只有输出包含 `"ready": true`、`"state": "active"` 和 `"state_code": 4`
时再启动规则或比赛。

## 本地控制台

控制台提供浏览器中的场景配置、实时态势、飞机遥测和终局结果展示。它复用相同的
比赛校验、Agent 隔离和 JSONL 回放机制；AFSIM 仍需要按上文先启动并保持运行。

```powershell
uv sync
uv run air-combat serve
```

### 大模型赛后复盘

网页控制台会在每局结束后自动将该局的终局摘要、平台遥测摘要、事件、开火动作和锁定记录发送到配置的大模型，并显示中文复盘。系统提示词固定在 `src/air_combat_challenge/web/app.py` 的 `LLM_SYSTEM_PROMPT`，强调只根据回放数据作出可追溯判断。

在 `src/air_combat_challenge/web/app.py` 顶部填写 `LLM_API_KEY`，也可以在同处修改 `LLM_MODEL` 和 `LLM_API_BASE`。密钥只保留在服务端源码中，不会发送给浏览器；不要将含有真实密钥的文件提交到远程仓库。

```python
LLM_API_KEY = "你的密钥"
LLM_MODEL = "gpt-4.1-mini"
LLM_API_BASE = "https://api.openai.com/v1"
```

然后访问 `http://127.0.0.1:8787`。页面可以分别设置红蓝方的飞机数量、起始经纬度、
队内横向间隔（km）、初始高度和速度、Agent manifest、随机种子、步数、时间倍率及 AFSIM 地址。
“AFSIM 场景入口”、蓝方 Agent manifest 和红方 Agent manifest 通过“浏览”调起本机文件选择框；
场景需选择 `start_up.txt`，Agent 需选择 manifest JSON。默认场景为
`scenarios/air_to_air/start_up.txt`。当 `19920` 没有 AFSIM 服务且启用“未启动时自动加载场景”时，
控制台会使用 `mission.exe -es <start_up.txt>` 自动启动并加载所选场景；可通过 `AFSIM_HOME`
环境变量指定 AFSIM 安装目录。选择的路径可以位于项目外；Agent 的同目录会作为其源代码根目录。
已有外部 AFSIM 服务时，控制台会复用它，不会覆盖用户已打开的 Warlock 场景。

点击“检查 AFSIM”确认服务就绪后，再点击“启动仿真”；“结束仿真”会同时调用 AFSIM gRPC
的 `stop` 接口结束当前场景，并写入 `stopped_by_user` 回放记录。Warlock 窗口会保留，
但仿真不再推进。Web 控制台生成的场景配置保存在
`runs/web_scenarios/`，比赛回放和汇总仍保存在 `runs/`。

时间倍率支持 `0.2x`、`0.5x`、`1x`、`2x` 和 `3x`。AFSIM 场景不再使用 `realtime` 模式，
因此首次使用该版本或修改该文件后，需要在 Warlock 中重新加载 `start_up.txt` 并点击 `Run`。
网页中的态势图用于指挥与遥测；三维模型、导弹和传感器渲染请直接使用 Warlock 窗口查看。

每局会在 `runs/<episode_id>.jsonl` 保存逐步回放，在
`runs/<episode_id>.summary.json` 保存终局汇总。逐步回放包括：双方完整观测（每架战机的
经纬高、北东天速度、航向/俯仰/滚转、传感器状态、武器余量）、可见航迹、事件、Agent
提交动作、翻译后的 AFSIM 命令、动作执行回执、随机种子、场景哈希和 Agent 源码哈希。
对局结束后，网页会自动从 JSONL 加载全机三维轨迹，以及选定战机的高度和速度时间曲线。
实时态势图使用卫星影像瓦片；复盘使用可鼠标拖拽、缩放的 Cesium 三维地球，并支持按 `1x`、
`5x` 或 `20x` 播放战机轨迹动画。卫星影像与三维地球库通过公网 CDN 加载；离线时实时图会
回退到本地平面态势，三维复盘会提示网络不可用。

## 运行比赛

AFSIM 就绪后运行：

```powershell
uv run air-combat run `
  --blue-agent examples/rules/a2a_rule_agent.json `
  --red-agent examples/rules/a2a_rule_agent.json `
  --scenario configs/scenarios/aircraft_2v2.json `
  --step-delay 0.5 `
  --seed 1
```

`--blue-agent` 和 `--red-agent` 表示两个比赛席位，不要求对应两套重复代码。两边可以
指定同一个 manifest 进行自对弈，也可以分别指定规则 Agent 和 LLM Agent。即使使用
同一个 manifest，红蓝也会运行相互独立的 Agent 实例和 worker，并获得各自一侧的观测。

常用运行参数：

| 参数 | 含义 |
|---|---|
| `--blue-agent` | 蓝方 Agent manifest，必需 |
| `--red-agent` | 红方 Agent manifest，必需 |
| `--scenario` | 飞机部署与想定 JSON |
| `--episodes` | 连续运行局数，默认 1 |
| `--steps` | 每局最大决策步数，默认 2000 |
| `--seed` | 首局随机种子，后续局依次递增 |
| `--step-delay` | 每轮决策后的现实时间暂停秒数，默认 0 |
| `--output` | JSONL 回放和总结目录，默认 `runs/` |
| `--ip`、`--port` | AFSIM gRPC 服务地址 |
| `--global-view` | 调试用全局真值观测，正式比赛不要启用 |

当前底层环境每个决策步推进 1 秒仿真时间。`--step-delay` 只用于降低现实中的运行和
显示速度，不改变每步推进的仿真时长，也不会跳过 Agent 决策。例如
`--step-delay 1.0` 会在相邻决策之间额外等待约 1 秒。

运行结果保存在 `runs/`，包括逐步观测、Agent 动作、底层命令、动作报告、事件、
Agent 源码树哈希、场景 JSON 哈希和终局总结。

## 动作接口

Agent 只能返回以下安全动作：

| 动作 | 控制内容 |
|---|---|
| `set_flight` | 航向、高度、Mach 综合控制 |
| `set_heading` | 单独航向控制 |
| `set_altitude` | 单独高度控制 |
| `set_speed` | 单独速度控制，单位 m/s |
| `fly_path` | 结构化航路点飞行 |
| `fire` | 使用本机航迹攻击目标 |
| `co_fire` | 使用己方引导机航迹协同攻击 |

`create_platform`、`delete_platform` 和原始 `sendCommand` 不向参赛 Agent 开放。
每架飞机每步最多执行一个飞行动作和一个武器动作。越权、目标不可见、武器耗尽、
参数越界或动作冲突都会返回稳定的拒绝原因。

动作、观测和执行结果的机器可读定义位于 [`docs/schemas/`](docs/schemas/)。

当前 F22 模型仍挂载 `scenarios/air_to_air/processor/FighterState.txt`，其中的自动机动
和自动开火逻辑可能覆盖 Agent 命令。如需 Agent 独占底层控制，需要从
`scenarios/air_to_air/equipment/FighterSixDof.txt` 移除对应的 `include_once`、
`internal_link` 和 `processor fighter_state`，然后重启当前使用的 AFSIM 宿主进程。

## 观测与终局

正式模式只提供己方完整状态，以及己方 `MasterTrackList` 中存在的敌方航迹，
不会泄漏未探测敌机的全局真值。观测包含位置、速度、姿态、传感器和真实武器余量。

CLI 默认采用歼灭判胜，达到 `--steps` 限制时截断为平局；程序化构造
`CompetitionEnv` 时还可以设置最大仿真时间。Agent 超时、崩溃或返回无法解析的响应时，
对应控制器当前步不执行动作；网关会逐条拒绝越权或冲突动作，同批次中合法动作仍可
执行。上述情况都会记为违规，连续 3 次或单局累计 10 次违规后判负。worker 最多自动
重启一次。

worker 进程用于超时和崩溃隔离，不是恶意代码安全沙箱。正式比赛如接收不可信代码，
应由裁判环境在容器或虚拟机中运行整个参赛提交。

## A2A 规则 Agent

示例 `examples/rules/a2a_rule.py` 是一个 `per_aircraft` 的 `BaseAgent`，当前采用 50 km
战术开火阈值、真实武器余量和 30 秒发射间隔。该阈值不是武器的固定物理最大射程；
实际射程还取决于发射高度、速度和目标机动。它与 LLM Agent 使用相同的正式比赛入口：

```powershell
uv run air-combat run `
  --blue-agent examples/rules/a2a_rule_agent.json `
  --red-agent examples/rules/a2a_rule_agent.json `
  --scenario configs/scenarios/aircraft_2v2.json `
  --step-delay 0.5 `
  --seed 1
```

自定义确定性规则同样只需要继承 `BaseAgent`：

```python
from air_combat_challenge.competition.agents import BaseAgent
from air_combat_challenge.competition.models import ActionBatchV1


class CustomRuleAgent(BaseAgent):
    def act(self, observation):
        return ActionBatchV1(actions=[])
```

规则 Agent 接收受限的 `ObservationV1`，返回 `ActionBatchV1`，所有动作继续经过
`ActionGateway` 的权限、可见性、武器余量、参数和动作冲突校验。武器发射距离由规则
自行决定，当前网关不计算导弹动力射程。

## JSON Schema

`docs/schemas/` 是从 `src/air_combat_challenge/competition/models.py` 生成的协议快照，
比赛运行时不会读取这些 JSON 文件。重新生成 V1 Schema：

```powershell
uv run air-combat schema --output docs/schemas
```

# AFSIM F22 空对空模型

本目录是当前 F22 2v2 baseline 使用的最小 AFSIM 模型资源树。除 F22 及其直接、
间接依赖外，其他平台、武器和设备模型均已移除。目录内部的相对路径属于运行时约定，
迁移或重命名文件时必须同步检查所有引用。

```text
air_to_air/
  equipment/    F22 六自由度平台能力定义
  parts/        F22 使用的运动模型、雷达和武器定义
  platform/     F22 平台类型
  processor/    F22 任务处理器
  scripts/      gRPC 命令及态势、事件采集脚本
  start_up.txt  模型加载入口
```

`start_up.txt` 只加载 `platform/F22.txt` 和 Python 客户端需要的脚本。
`scripts/command.txt` 定义外部命令函数，`scripts/CollectData.txt` 通过 gRPC 服务
发布 `SituationData` 和 `events`。

默认 2v2 飞机部署单独保存在 `configs/scenarios/aircraft_2v2.json`。AFSIM 模型
启动后，Python 环境根据该配置创建具体平台。

重新引入其他平台或武器时，需要补齐对应的 `equipment/`、`parts/`、`processor/`
或 `platform/` 文件，并检查 `start_up.txt` 中的 `include_once` 加载顺序。

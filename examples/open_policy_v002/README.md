# Strategy 3 开放策略种子

这是用于 DSL 3.0 完整策略代码进化的初始冠军。默认 `policy.py` 委托给 Strategy 3
关闭 LLM 后的纯规则控制器，所以起点和 `strategy3_bt_seed` 动作等价。

与 DSL 2.0 不同，大模型后续可以重写整个 `Policy` 类，自行维护状态、分配目标、预测
威胁、控制双机编队，并生成比赛协议允许的所有飞行与武器动作。模型提交的是
`behavior_tree.json` 中的 `policy_source`；编译器经过源码校验后生成 `policy.py`。

`agent.py`、`agent.json` 和基线运行模块受到保护。文件、网络、外部进程、动态代码、
反射和读取其他 Agent 源码均被禁止，但战术算法本身没有固定行为树或参数表限制。

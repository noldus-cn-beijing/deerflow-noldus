# 范式 → 默认 EV19 模板降级表

## 何时使用

当 agent 在 ask_clarification 反问后用户答 "不知道" / "随便" / "你决定"，或者 LoopDetectionMiddleware 阻断了第二次反问时，使用此降级策略。

**重要**：填到 `set_experiment_paradigm(ev19_template=...)` 之前，先在用户面前确认一次：
> "您的实验我会按 EPM 标准模板（PlusMaze-AllZones）分析。这是 90%+ EPM 实验的默认配置。如果您的实验有特殊设置，分析后告诉我我会重做。"

不要默不作声地默认。

## 权威源

**范式 → 默认变体的映射以 Python 代码为唯一权威源**：
- `ethoinsight.ev19_facts.EV19_TEMPLATE_PARADIGM_MAP` — 每个 paradigm_key 对应的候选变体列表（第一个 = 默认值）
- `ethoinsight.ev19_facts.get_default_template_for_paradigm(key)` — 返回推荐默认变体
- `identify_ev19_template_tool` — agent 在反问失败时调此工具获取默认值

本文件是快速参考，与代码不一致时以代码为准。

## 决策流程

```
用户答 "不知道" / 反问被 LoopDetection 阻断
    ↓
1. 看用户文字 + 文件名能否推断 paradigm_key
    ↓
2. 调 identify_ev19_template_tool 获取该范式的默认 ev19_template
   (内部调用 get_default_template_for_paradigm)
    ↓
3. 告知用户："我会按 <默认模板> 分析。如有特殊设置，分析后告诉我我会重做。"
    ↓
4. 调 set_experiment_paradigm(paradigm=<推断>, ev19_template=<默认>, ...)
```

## 当前默认值快照（仅供参考，以 EV19_TEMPLATE_PARADIGM_MAP 为准）

| 范式 | 默认 ev19_template | 说明 |
|---|---|---|
| epm | PlusMaze-AllZones | 90%+ EPM 实验 |
| open_field | OpenFieldRectangle-AllZones | 矩形 arena 最常见 |
| zero_maze | ZeroMaze-AllZones | 环形迷宫标准 |
| light_dark_box | OpenFieldRectangle-Subdivided2x2 | 临时兜底，待行为学同事确认 |
| tail_suspension | NoTemplate | TST 不用 zone |
| forced_swim | PorsoltCylinder-AllZones | FST 标准 |

## 更新规则

行为学同事 PR 修改 `by-experiment/<范式>.md` 中的"适用模板"字段后，同步更新 `EV19_TEMPLATE_PARADIGM_MAP` 的候选顺序。本文件的快照表随之更新。

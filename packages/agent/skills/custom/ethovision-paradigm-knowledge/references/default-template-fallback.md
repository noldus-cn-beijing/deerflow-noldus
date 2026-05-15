# 范式 → 默认 EV19 模板降级表

## 何时使用

当 agent 在 ask_clarification 反问后用户答 "不知道" / "随便" / "你决定"，或者 LoopDetectionMiddleware 阻断了第二次反问时，按此表选默认变体进入分析。

**重要**：填到 `set_experiment_paradigm(ev19_template=...)` 之前，先在用户面前确认一次：
> "您的实验我会按 EPM 标准模板（PlusMaze-AllZones）分析。这是 90%+ EPM 实验的默认配置。如果您的实验有特殊设置，分析后告诉我我会重做。"

不要默不作声地默认。

## 范式 → 默认变体

| 学术范式 (paradigm key) | 默认 ev19_template | 选择理由 |
|---|---|---|
| `epm` | `PlusMaze-AllZones` | 90%+ EPM 实验用 AllZones（含开闭臂 + 头探出区） |
| `open_field` | `OpenFieldRectangle-AllZones` | 大多数 OFT 实验。圆形 arena 看 demodata 几何形状判断是否切到 `OpenFieldCircle-AllZones` |
| `zero_maze` | `ZeroMaze-AllZones` | 同上 |
| `light_dark_box` | `OpenFieldRectangle-Subdivided2x2` | **未来由行为学同事确认**。LDB 在 EV19 表里无独立大类，2x2 子分区可手工指明明暗箱区 |
| `tail_suspension` | `NoTemplate` | TST 不用 zone，仅活动度（不动时间） |
| `forced_swim` | `PorsoltCylinder-AllZones` | FST 标准（圆柱形容器） |
| `shoaling` | `OpenFieldCircle-NoZones-Fish` | 斑马鱼鱼群（多动物 2D） |
| `novel_object` | `OpenFieldCircle-NovObjZones` | NOR 实验，圆形 arena + 物体区 |
| `y_maze` | `Y-Maze-AllZones` | 三臂 + 中央交汇区 |
| `barnes_maze` | `BarnesMaze-20Holes` | 标准 20 孔配置 |
| `morris_water_maze` | `MWM-AllZones` | 平台 + 象限 + 走廊 + 边缘 |
| `sociability` | `Sociability-AllZones` | 三箱社交（社交区 + 对照区） |
| `radial_arm_maze` | `Radial-8-arm-AllZones` | 标准 8 臂 |

## 决策流程

```
用户答 "不知道" / 反问被 LoopDetection 阻断
    ↓
1. 看用户文字 + 文件名能否推断 paradigm_key（epm / open_field / ...）
    ↓
2. 查上表 → 默认 ev19_template
    ↓
3. 在与用户的下一条消息里告知："我会按 <默认模板> 分析。如有特殊设置，分析后告诉我我会重做。"
    ↓
4. 调 set_experiment_paradigm(paradigm=<推断>, ev19_template=<查表>, ...)
```

## 此表的更新

行为学同事 PR 中的 `by-experiment/<范式>.md` 一旦填写"适用模板（按推荐顺序）"，本表的相应行应同步更新。**保持单一事实源**：未来若数据飞轮启动 + agent 自学偏好，可考虑由 agent 自己改这个文件（启用 update_agent / Skill Evolution 后）。

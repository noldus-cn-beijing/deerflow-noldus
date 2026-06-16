# Skill ↔ Catalog 指标核对表（基于真实 DemoData，2026-06-16）

> **2026-06-16 行为学同事裁决已并入**（见第零节）。本表上半部分的"真实列推断"已被同事裁决覆盖处，以同事裁决为准。

## 零、行为学同事裁决（权威，覆盖下方推断）

来源：行为学同事（曲若衡）2026-06-16 回复。

| 指标 | 同事裁决 | skill 处置 |
|---|---|---|
| **rearing（后腿直立）** | **不支持**。真因=① EthoVision 精细行为分类器**信效度不佳**（销售/同事均认可不好）② 在焦虑/旷场范式里**非必须**，只在专门测精细行为的范式需要，那类实验不在支持范围。本质=速度不变+身长变短+投影面积变小，理论上能从 elongation 推但**没必要** | 标注"本系统不支持"，给出信效度+非必须的真实理由，不产出、报告不得编造 |
| **伸展注意 SAP / 身体伸长 elongation** | 实践中**很少使用**（"stretch 和 elongation 很少会用"） | 降级为"很少用、默认不算、报告不提，用户明确要求才算" |
| **head-dipping（低头探索）** | **只适用 EPM 和未来巴恩斯迷宫**——只有这两个有不带墙的边缘，鼠才能探头往下看。其他实验都带墙，不存在探头/低头，**既不需统计也无法计算** | EPM 保留（开臂无墙，走 `nose_over_edge_open_arms` 分析区列）；OFT/LDB 不提（带墙） |
| **Zero Maze 低头/探出** | **一般不测，优先用 FewZones 模板**做开放区时间/进入分析；**就算测，用鼻尖点在不在"探头区"分析区**来做，不是 stretch/elongation；stretch/elongation 很少用 | ZM skill 撤掉"stretch-attend 是核心洞察/最敏感指标"的拔高；低头改"一般不测优先 FewZones，测则用鼻尖点在探头区"；`hesitation_count`（catalog default 真实指标）保留为辅助 |
| **LDB peek-out / nose-in-zone** | LDB **不测 nose-in-zone**，以**身体中心点**为标准 | LDB 标注"封闭箱无开放边缘 + 以中心点判定、无鼻尖探出指标、不产出" |

**FST / TST 的 velocity 列**：同事说明——FST/TST **不需要测速度**，导出 sample 时**手动删了 velocity 列**，但**默认导出是带 velocity 的**。
→ 影响：FST/TST catalog 的 `activity_intensity` 图表（must_have）`requires_columns: velocity`。**默认导出有 velocity，依赖合理，catalog 不改**。但若用户上传删了 velocity 的导出，该图会缺列失败 → 属数据契约边界，code-executor 记 warning，非 bug。

**关键纠错记录**（防后续 agent 重蹈）：本次会话最初基于"真实数据列推断"得出"SAP/head-dip 普遍可算"，被同事裁决推翻——**可算性 ≠ 该算**。elongation 列虽存在，但 SAP 很少用；head-dipping 受**物理范式（有无开放边缘）**约束只 EPM 适用；rearing 即便能从 elongation 推也因信效度+非必须而不做。**指标取舍以行为学同事的实践判断为准，不以"数据列在不在"为准。**

---

## 一、6 范式真实列清单（parse_header 实测）

| 范式 | 文件 | 判读相关列 |
|---|---|---|
| **EPM** | 原始数据-Elevated Plus Maze XT190-Trial 1.xlsx | x/y_{center,nose,tail}、`elongation`、`heading`、distance_moved、velocity、`in_zone_open_arms_center`、`in_zone_closed_arms_center`、**`nose_over_edge_open_arms`**、entries_open/closed_arms |
| **OFT** | 原始数据-Open Field test XT190-Trial 1.xlsx | x/y_{center,nose,tail}、`elongation`、`heading`、distance_moved、velocity、`in_zone`、`distance_to_wall`、**`nose_over_wall_zone`** |
| **O-Maze** | 原始数据-oMaze-试验 1.xlsx | x/y_{center,nose,tail}、`elongation`、`heading`、distance_moved、velocity、`in_zone` |
| **FST** | 原始数据-Porsolt forced swim test XT190-Trial 1.xlsx | x/y_center、`elongation`、`mobility_continuous`、mobility_state_{highly_mobile,mobile,immobile} |
| **TST** | 原始数据-tstHelperDemoVideo-试验 1.xlsx | x/y_center、`elongation`、`activity(samplerate_1/5)`、`mobility_state` |
| **LDB** | 原始数据-ethoInsightDemo-ldb-试验 1.xlsx | x/y_center、`elongation`、`in_zone` |

注：以上是 demo 数据的列。真实客户数据列名可能因 Analysis profile 而异（如 OFT 的 `in_zone` 可能是用户自定义中心区列，走 HITL 列对齐），但**身体点 / elongation / heading / nose_over 这类因变量是否导出，取决于录制配置**，需以每份真实数据为准。

---

## 二、skill 边界指标 × 真实数据可算性

| skill 讲的指标 | 真实列证据 | 可算? | 处置 |
|---|---|---|---|
| **伸展注意 / SAP（伸展姿势）** | **6 范式全有 `elongation`** | ✅ 普遍可算 | **保留**（是 elongation 代理；脚本 `compute_body_elongation_stats` 已存在） |
| **低头探索 head-dipping** | EPM/OFT/O-Maze 有 `heading`；EPM 有 `nose_over_edge_open_arms`、OFT 有 `nose_over_wall_zone` | ✅ EPM/OFT/O-Maze 可算；FST/TST/LDB 无 heading 不可算 | **EPM/OFT 保留**；其余范式不提 |
| **EPM 开臂边缘探出** | EPM 专列 `nose_over_edge_open_arms` | ✅ 可算（有现成列！） | **EPM 可新增为真实指标**（待 catalog 登记决策） |
| **OFT 越壁探出** | OFT 专列 `nose_over_wall_zone` | ✅ 可算 | 同上 |
| **LDB 向外探出 peek-out / lean-out** | LDB 真实列**只有 `in_zone`**，无 nose_over、无 heading | ❌ **LDB 算不出** | **从 LDB skill 排除** |
| **后腿直立 rearing（无支撑/支撑）** | **无任何范式有 Z 轴/高度/rearing 列** | ❌ 全算不出 | **从所有 skill 排除**（需 3D 或人工评分） |

---

## 三、各 skill 待改清单（精确到行）

| 范式 skill | 当前措辞 | 真实数据判定 | 动作 |
|---|---|---|---|
| **epm.md** | "风险评估指标（低头向下探索、后腿直立、伸展注意）" | SAP✅ head-dip✅（有 nose_over_edge）rearing❌ | **改**：拆出 rearing 标"算不出"；SAP/低头探索保留（甚至可点名 `nose_over_edge_open_arms`） |
| **open_field.md** | "直立 rearing（无支撑/支撑）""倒 U 型曲线" | rearing❌ | **改**：rearing 整段移除/标"算不出"；可补 SAP（elongation 在 OFT 有）+ 越壁探出（nose_over_wall_zone） |
| **light_dark_box.md** | "向外探出暗箱 peek-out""风险评估行为（伸展姿势/鼻触/后腿直立）" | peek-out❌（无列）SAP✅（elongation 有）rearing❌ | **改**：删 peek-out + rearing + 鼻触；伸展姿势可保留（elongation 在 LDB 有） |
| **zero_maze.md / O-Maze** | stretch-attend / head-dipping / 犹疑次数 | O-Maze 真实列有 elongation + heading；`hesitation_count` 是代理 | **基本保留**；犹疑次数=hesitation_count（catalog default）一致 |
| **forced_swim.md / tail_suspension.md** | 只有 immobility 三件套 + 攀爬/游泳 | 无姿态指标问题 | **不动** |

---

## 四、catalog 侧（独立决策，本次可不动）

`elongation` / `nose_over_*` 在真实数据里有，对应脚本部分已存在（`compute_body_elongation_stats` 在 EPM/OFT catalog 的 optional），但：
- LDB / O-Maze 的 catalog **没登记** body_elongation_stats → 若要让 SAP 进 plan，需补 optional 条目
- `nose_over_edge_open_arms` / `nose_over_wall_zone` **没有对应 compute 脚本** → 若要算"探出率"需新写脚本

→ 这些是"补能力"的工作，与"修 skill 别诱导幻觉"正交，可分开决策。**本次只改 skill 与真实数据对齐**。

---

## 五、不变量

- skill 不得让 agent 以为 plan 里不存在的指标能产出数值（防下游幻觉）。
- catalog 是指标 SSOT，"能不能算"以真实数据列 + 脚本为准。
- **code-executor 的 `ethoinsight-code` skill 不受影响**（只读 plan 执行，不含范式判读）。

# Spec：EV19 列语义对齐 Sprint 2 — 结构聚合（N 列 → 1 概念）的坐实、固化与测试补齐

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `1c1033a7`
> 性质：🟡 中 · 后端 ethoinsight 库（聚合语义验证 + 测试固化 + 特殊规则声明，**非从零造聚合**）
> 阻塞解除：[Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) 已 CLOSED（2026-06-18）；行为学同事方法论已交付（PR #115 → `docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`）。**不再 blocked，可开工。**

---

## 〇、最重要的事实（先读，否则会重复造已有的轮子）

milestone 把 Sprint 2 标为「结构聚合：EPM 4 区 open_arm1/2 → 标准 2 区聚合」，听起来像要从零实现聚合。**但读码坐实：N 列→1 概念的聚合机制全链路其实已经存在并工作。**

| 层 | 现状 | 证据 |
|---|---|---|
| **resolve 层（多列收集）** | ✅ 已实现 | `catalog/resolve.py:_build_zone_aliases_overrides` Step 3-4（L688-744）按 concept 把**多个物理列分组**，`overrides[param_name] = cols`（列表，wrap_list=True 时）。两列 `open_arm1`/`open_arm2` 都 alias 成 `open` → `open_arm_zones=['open_arm1','open_arm2']` |
| **compute 层（聚合算法）** | ✅ 已实现且正确 | `metrics/epm.py`：`compute_open_arm_time_ratio` L101 `df[cols].max(axis=1)`（多臂**按帧 OR**＝同事说的"整合"）；`compute_total_entry_count` L173 注释"each group combines its columns with OR to avoid overcounting"（OR 防重复计数） |
| **catalog glob（多列匹配）** | ✅ 已支持 | `catalog/epm.yaml:9` `in_zone_open_arms_*` glob 天然匹配多列 |
| **多列聚合的直接测试** | ❌ **缺口** | grep `tests/` 无一处喂 `open_arm_zones=['a','b']`（>1 列）验证聚合数值正确——现有测试只覆盖单列/列对齐，**没锁住"两列 OR 聚合"这个 Sprint 2 核心行为** |
| **同事方法论的特殊规则** | ❌ **未固化** | LDB 隐藏区"忽略直接算暗区"、MWM"不合并"、Zero Maze"同 EPM"、累积分析区（EV 内已聚合）"别再叠加"——这些规则散在同事文档里，**代码/catalog 里没有显式声明或测试守护** |

**所以 Sprint 2 的真实任务 = 用同事方法论「坐实 + 固化 + 补测」已有的聚合机制，而不是实现它。** 这让本 sprint 风险骤降（不动承重墙），价值是把"恰好能跑"变成"被测试和声明锁死、不会回归、特殊范式不算错"。

---

## 一、同事方法论的权威规则（来自 PR #115 文档，逐范式聚合语义）

> SSOT：`docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`。实施时以该文件为准，本节是摘要。

| 范式 | 聚合规则 | 实现含义 |
|---|---|---|
| **EPM / Zero Maze** | 两个开放臂（A/B）→ **考虑聚合**（OR 合并）；两个封闭臂同理；中心区一般不导出 | 多列 `open_arm_zones` OR（已实现）；**累积分析区（EV 内已聚合的"开放臂all"）若与 A/B 同时输出，别再二次叠加**（去重/择一） |
| **LDB** | 明区 + 暗区；**隐藏分析区（入口）→ 忽略**（其坐标最终落入暗区，直接算暗区即可） | 隐藏区列**不参与聚合、不报缺列**——alias 成 `__ignore__` 或暗区 |
| **OFT** | 中心区 + 外周区；墙区/九宫格非标准 | 标准两区，v1.0 不做九宫格 |
| **FST / TST** | **不分区** | 无 zone 聚合，不应触发列对齐反问 |
| **MWM** | 平台/象限/thigmotaxis/whishaw 多画法 → **不合并**（不同分析角度+实验阶段，保留差异） | v0.1 不支持 MWM，但**聚合逻辑绝不能把不同画法 OR 到一起**——若未来支持，规则是"不聚合" |

**跨分析区群组的关键认知（同事原话）**：同一分析区群组内互斥（OR 安全）；**不同群组间可能堆叠**（如 open_arm_A/B + 累积 open_arm_all 三者都输出 → 不能三者 OR，会双重计数）。这是聚合的唯一真正陷阱。

---

## 二、本 Sprint 要做的（按风险从低到高）

### 任务 1（核心）：补多列聚合的直接回归测试

为已实现的 OR 聚合补"锁行为"的测试，防未来回归。`packages/ethoinsight/tests/`：

- **EPM 两列开放臂聚合**：构造 df 有 `open_arm_A`/`open_arm_B` 两列（不同帧各自==1），`compute_open_arm_time_ratio(df, open_arm_zones=['open_arm_A','open_arm_B'])` 断言 == 两列 OR 后的占比（手算对照）；同理 `compute_open_arm_time` / `compute_open_arm_entry_count`（验 OR 后跳变计数 ≠ 两列分别计数之和，坐实"防重复计数"）。
- **resolve 端到端**：`column_aliases = {'open_arm_A':'open','open_arm_B':'open'}` → `resolve_metrics` → 断言 plan 里 `open_arm_zones == ['open_arm_A','open_arm_B']`（两列都进，不丢）。
- **累积分析区去重（陷阱测试）**：df 同时有 `open_arm_A`/`open_arm_B`/`open_arm_all`（后者是 EV 内已聚合），三列都 alias 成 `open` → 断言**不会三列 OR 双重计数**（要么择 `open_arm_all`、要么只 OR A/B，**两者数值应一致**——这正是同事说的陷阱）。**若当前实现这里会双重计数，任务 3 修它。**

### 任务 2：固化特殊范式规则（LDB 忽略隐藏区 / FST·TST 不分区）

- **LDB 隐藏区**：确认 LDB catalog + 列对齐链路里，隐藏分析区列能被 alias 成 `__ignore__`（已有机制，`_apply_aliases` 移除）或归入暗区，**不报 `columns_missing`、不参与明/暗聚合**。补 LDB 测试：有隐藏区列时 plan 正常、暗区指标只用暗区列。
- **FST/TST 不分区**：确认这两范式的 catalog 无 zone metric、列对齐链路**不会对它们触发 zone 反问**（`_build_zone_aliases_overrides` 对无 zone_pattern 的范式返回 `{}`，已坐实 L685-686）。补一条 negative 测试锁住。

### 任务 3（仅当任务 1 的陷阱测试暴露 bug 才做）：修累积分析区双重计数

若任务 1 第三条测试证明"A/B + all 三列同 alias 会双重计数"，则在 `_build_zone_aliases_overrides` 或 compute 层加去重：**同一 concept 下若存在"累积列"（EV 内已聚合），优先用累积列、排除被它包含的分量列**。如何识别累积列：同事文档说 EV 的"整合分析区"命名（如 `open_arm` 无 A/B 后缀 vs `open_arm_A`）——但**别靠列名字面猜**（违反 Sprint 1 铁律），应通过列对齐 HITL 让用户标明哪列是"累积/整合区"。**这条若触发，范围扩大到需 schema 加"累积区"标记 + HITL 询问，单独评估是否本 sprint 做还是拆下一 sprint。**

### 任务 4：milestone + 文档状态更新

- `docs/milestone/column-semantics-alignment.md`：Sprint 2 状态从 `blocked` 改为实际状态（机制已存在 + 本 sprint 补测固化）；更新"下一 milestone"。
- 把同事方法论文档从 `docs/review-packages/2026-06-09-feedbacks/` **正式落位**到知识源（若 SOP 要求搬入 `ethovision-paradigm-knowledge` skill references，按 CLAUDE.md 第 1 条三件套）。

---

## 三、关键文件

- `packages/ethoinsight/ethoinsight/metrics/epm.py`（多列 OR 聚合 compute，**已实现，主要是加测试**；任务 3 触发才改）
- `packages/ethoinsight/ethoinsight/catalog/resolve.py:_build_zone_aliases_overrides`（L649-744，多列收集，**已实现**；任务 3 触发才改去重）
- `packages/ethoinsight/ethoinsight/catalog/{epm,ldb,oft,zero_maze,fst,tst}.yaml`（zone 概念声明，确认特殊规则）
- `packages/ethoinsight/tests/`（**主要工作量在这**：多列聚合 + 累积去重陷阱 + LDB 忽略 + FST/TST negative）
- `docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`（同事方法论 SSOT，实施时逐范式照此）
- `docs/milestone/column-semantics-alignment.md`（状态更新）

---

## 四、验证

1. `cd packages/ethoinsight && pytest tests/`（全绿；新测试覆盖多列聚合 + 陷阱 + 特殊规则）。
2. **聚合正确性断言**（常驻回归，守 memory `feedback_2026-06-16_io_boundary_asymmetry_and_aggregator_half_built` 的"mean 正确性常驻断言"纪律）：两列 OR 的占比/计数手算对照，不是"跑通即过"。
3. **真实数据 dogfood**：用一份真有两列开放臂的 EPM 数据（或构造），走完整 HITL 列对齐 → 确认 plan 的 `open_arm_zones` 含两列 → 指标值 == 手算 OR 聚合值。可用 `/noldus-insight-e2e`。
4. **特殊范式不误伤**：FST/TST 数据上传不触发 zone 反问；LDB 有隐藏区列时正常分析。
5. **后端铁律**：改 ethoinsight 库后 `cd packages/ethoinsight && pytest`；若动到 catalog 被 agent 消费的字段，额外裸导入两生产入口（CLAUDE.md「harness 模块顶层 import 闭环风险」）。

---

## 五、不做 / 边界

- **不从零造聚合**——OR 聚合机制已存在且正确，本 sprint 是坐实+固化+补测，不是重写。
- **不靠列名字面猜 zone 身份 / 累积区身份**（Sprint 1 铁律 + 同事铁律"虽然猜中也不要猜"）——累积区识别若需要，走 HITL 让用户标明。
- **不实现 MWM / 九宫格 / 墙区**（v1.0 范围外；但聚合逻辑要保证"MWM 多画法绝不 OR 合并"，即不误聚合）。
- **任务 3 是条件性的**——仅当任务 1 陷阱测试证明双重计数才做；若证明当前实现已正确（如用户数据通常不同时输出 A/B + all），则只留测试守护，不改代码。

---

*依据：读码坐实 N 列→1 概念聚合全链路已实现（`resolve.py:_build_zone_aliases_overrides` 多列收集 + `metrics/epm.py` OR 聚合 + `epm.yaml` glob 匹配多列）；缺口是多列聚合直接测试 + 同事特殊规则（LDB 忽略隐藏区 / 累积区去重陷阱 / MWM 不合并 / FST·TST 不分区）固化。Issue #98 已 CLOSED、同事方法论 PR #115 已交付（`docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`），阻塞解除。*

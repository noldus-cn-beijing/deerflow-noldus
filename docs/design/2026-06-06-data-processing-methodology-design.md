# 数据处理方法论：范式确定后，如何系统性地识别、对齐、聚合、验证用户数据

**日期**：2026-06-06
**状态**：设计草案，已经 agent 调研 review + grill 修订（2026-06-06）
**上游启发**：[karpathy/autoresearch](https://github.com/karpathy/autoresearch) — program.md 作为 agent 决策框架的方法论模式

> **2026-06-06 调研修订说明**：本文档经代码事实核实 + grill 后修订了三处：
> 1. **删除 `bash head -1 | tr ',' '\n'` 列名提取方案**（§3/§5）—— 经核实该命令对 EV19 文件三处全错（UTF-16 编码 / 分号分隔 / 第 0 行是 `标题行数：N` 元数据，列名在第 N-2 行）。列名由现有 tool（`identify_ev19_template` / `inspect_uploaded_file`，均走 `parse_header()`）提供。
> 2. **Layer 3 聚合语义降级为开放问题**（§4）—— 是否 sum 取决于未确认的行为学事实（zone 是否空间重叠）。**聚合语义的 SSOT 是 [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)，本文档只引用不内嵌**，待专家确认后落入 catalog/skill。
> 3. **补全 §6 catalog-driven 验证的三个缺口** + **修正 §7 的 train.py↔catalog 反向类比**。

---

## 一、问题陈述

EthoInsight 的 code-executor + data-analyst 流水线假设用户数据"看起来像 catalog 期望的样子"——列名匹配、分析区数量匹配。但真实数据是：

```
Demo data（干净）：                    Real data（混乱）：
┌─────────────────┐                   ┌──────────────────────┐
│ Open arm         │                   │ 试用时间              │ ← 固定列
│ Close arm        │                   │ X 中心               │ ← 固定列
│ 固定列...         │                   │ Open arm1            │ ← 用户自定义名
└─────────────────┘                   │ Open arm2            │ ← 同逻辑区的子区
                                      │ Close arm1           │
                                      │ Close arm2           │
                                      │ 中心区                │ ← 用户自定义名
                                      └──────────────────────┘
```

核心矛盾：**你无法穷举用户自定义的列名和分析区粒度，但每个范式的计算逻辑是确定的**。

Sprint 1（列语义对齐 1:1 名称映射）解决了"中心区 → center"的命名映射。但两个更深的问题没有解决：

1. **列名提取**：列名由现有 tool 直接提供——`identify_ev19_template` 内部已用 `parse_header()` 解析列结构判模板，`inspect_uploaded_file` 用同一 parser 返回完整列名 + 数据预览 + 每列几何证据。**无需新建提取 step**（原稿曾提议 `bash head -1`，已证伪删除，见 §3）。
2. **结构聚合**：多个物理子区对应一个逻辑区（Open arm1 + Open arm2 → Open arm），当前系统完全不知道如何处理。聚合语义取决于一个未确认的行为学事实，是本设计的核心开放问题（见 §4 Layer 3 + [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)）。

需要的不是按 case 修，而是一套**通用方法论**——就像 AutoResearch 的 `program.md` 教 agent"如何做实验"，我们需要一份文档教 agent"如何处理行为学数据"。

---

## 二、AutoResearch 启发：方法论优于补丁

AutoResearch 的力量不在于 `if isnan(loss): exit(1)` 这一行代码，而在于 `program.md` 建立了 agent 做实验的决策框架：

| AutoResearch program.md | EthoInsight 对应 |
|---|---|
| "只能改 train.py，不改 prepare.py" | "catalog 是 SSOT，不猜测指标定义和分析区" |
| "val_bpb 越低越好，NaN 立即失败" | "确定性校验（NaN/范围）走代码，判断性校验走 LLM" |
| "边际收益小的 hack 不如简单改进" | "优先用 catalog 声明的规则，fallback 到保守默认，不要臆测" |
| "实验结果记录到 results.tsv" | "每步决策落盘到 experiment-context.json，可追溯" |

**核心原则**：不是给 agent 写 100 条具体规则（"检测 center_entry_count ≥ 0"），而是教它一套可泛化的决策框架（"读 catalog → 检查 output_unit → 用代码做确定性校验 → 不确定的走 HITL"）。

---

## 三、工具分工：两个 tool 各司其职

列名提取**不需要第三个 step**——两个现有 tool 已覆盖。明确分工：

| Tool | 职责 | 何时调用 | 返回 |
|------|------|---------|------|
| `identify_ev19_template` | **模板识别**（需要领域知识匹配） | 分析开始第一步 | template_id, paradigm_key, 候选列表, evidence（内部已 parse 列名，但只返回前 20 列做参考） |
| `inspect_uploaded_file` | **数据探查 + 完整列名 + 几何证据** | identify 之后，带上已确定的 paradigm | 完整列名清单, 前 5 行数据预览, EV19 metadata, 每列 `column_assessment`（binary_zone 等几何证据，用于列对齐预填） |

**关键区别**：

- `identify_ev19_template` 内部用 `parse_header()` 解析列结构做模板匹配，返回聚焦在 template ID 和候选列表；`evidence.columns` **被硬截断为前 20 列**（`identify_ev19_template_tool.py:524`），仅供参考，**不可当作完整列名清单消费**。
- `inspect_uploaded_file` 用同一个 `parse_header()` 返回**完整列名（无截断）** + 数据预览 + `column_assessment`。它是"获取完整列名 + 列值几何证据"的**单一来源**。注意：`column_assessment` 依赖传入 paradigm 才能给出对齐证据，所以 inspect 应在 identify 确定 paradigm **之后**调。
- 两个 tool 各自独立 parse 同一文件、互不传递（架构现状，并列而非上下游）。重复 parse 的代价（一次 UTF-16 文件读取）远小于引入 tool 间数据依赖的复杂度，保持独立是正确的。

> ⚠️ **为什么不用 `bash head -1 | tr ',' '\n'`**（原稿方案，已删除）：EV19 导出文件是 **UTF-16 LE 编码、分号（`;`）分隔**，**第 0 行是 `标题行数：N` 元数据标记**（列名在第 `N-2` 行）。`head -1 | tr ','` 在编码、分隔符、行号三处全错，拿到的是错位的字节流，不是列名。代码依据：`packages/ethoinsight/ethoinsight/parse/_core.py`。若确需 headless/CLI 的零-LLM 列名提取，正确工具是 `python -m ethoinsight.parse.dump_headers`（已存在，处理 UTF-16 + 标题行数 + 分号 + 四格式），**永远不用 `head -1`**。

---

## 四、三层列处理框架

用户数据的所有列分为三层，每层的处理方式不同：

### Layer 1: EV19 Template 固定列（确定性识别，0 HITL）

每个 EV19 template 导出的数据包含一组**跨用户、跨实验不变的固定列**。这些列名不受用户自定义影响：

| 固定列 | 说明 | 状态 |
|--------|------|------|
| 试用时间 | Trial time | **需专家确认** |
| 录制时间 | Recording time | **需专家确认** |
| X 中心 | X center coordinate | **需专家确认** |
| Y 中心 | Y center coordinate | **需专家确认** |
| 区域 | Area | **需专家确认** |
| 面积变化 | Area change | **需专家确认** |
| 伸长 | Elongation | **需专家确认** |
| 移动距离 | Distance moved | **需专家确认** |
| 速度 | Velocity | **需专家确认** |
| Result 1 | Result 1 block | 部分 template 无此列 |

处理方式：从 `inspect_uploaded_file` 返回的完整列名清单中 → 白名单匹配固定列 → 匹配的直接标记，不进入反问流程。

> **[待专家确认]** 清单是否完整？不同 template 有无差异？中英文版 EV19 列名是否不同？

### Layer 2: 自定义分析区列 → 1:1 概念映射（Sprint 1 已完成）

Layer 1 过滤后剩余的分析区列，用户命名无法穷举。但每个范式只有固定的几个**逻辑分析区概念**（来自 catalog）：

| 范式 | 逻辑分析区概念 | 状态 |
|------|--------------|------|
| EPM | open_arms, closed_arms | **需专家确认** center 是否需要 |
| OFT | center, border | **需专家确认** corner 是否是独立区 |
| LDB | light, dark | **需专家确认** |
| Zero Maze | open, closed | **需专家确认** |
| FST | 无自定义分析区 | 已确认 |
| TST | 无自定义分析区 | 已确认 |

处理方式（Sprint 1 已实现）：agent 读取 catalog 合法概念菜单 → 根据列值分布（来自 `inspect_uploaded_file`，仅在需要时调用）预填最佳猜测 → HITL 合并反问用户确认 → 写入 `column_semantics` 到 `experiment-context.json`。

### Layer 3: 分析区结构聚合 N:1（Sprint 2 — 缺失，待设计）

用户可能将**一个逻辑分析区拆成多个物理子区**：

```
EPM 示例：
  标准 2 区：Open arm, Close arm
  真实 4 区：Open arm1, Open arm2, Close arm1, Close arm2
  聚合：    Open arm = Open arm1 ∪ Open arm2
           Close arm = Close arm1 ∪ Close arm2
```

检测方式：Layer 1+2 匹配后，用户分析区列数 > catalog 声明的逻辑区数。

Agent 决策流程：
1. **检测**：用户列数 > 逻辑区数 → 可能存在子区拆分
2. **模式匹配**：对用户列名做相似度聚类（共享前缀/关键词 → 可能属同一逻辑区）
3. **HITL 确认**：预填聚合方案，反问用户确认
4. **聚合执行**：⚠️ **聚合语义未定，是本设计的核心开放问题** —— 见下方
5. **产出**：聚合后的列映射表 + `aggregation_rules` → `experiment-context.json`

> ⚠️ **聚合语义是开放问题，SSOT 在 [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)，本文档不内嵌结论。**
>
> 原稿曾写死"时间/距离/进入次数 = sum"——这是**未经专家确认就替专家做的判断，已撤销**。是否该 sum 取决于一个工程侧无法从代码或数据格式推断的行为学事实：
>
> **EV19 的子区在空间上互斥还是可重叠/嵌套？**
> - **若互斥**（动物任一帧最多在一个子区）：帧级 OR 与 sum 数值等价，聚合纯机械。
> - **若可重叠/嵌套**（如大开放区内又画入口子区，动物同帧可同时在两区）：sum 会 **double-count**（时间重复计、相邻子区间移动被数成多次进入），必须**帧级 OR**（`df[cols].max(axis=1)`）合并后再算。
>
> **中性事实参考**：现有指标脚本对多 zone 列的合并用的是帧级 OR（`epm.py:96` / `_common.py:75` / `zero_maze.py:87`），`compute_total_entry_count` 的 docstring 明写 "combine with OR to avoid overcounting"。但这是历史实现，其正确性同样取决于上面这个问题的答案——故本文档**不预设 OR 正确**，等专家定论。
>
> 待 Issue #98 专家回答 zone 空间关系后，聚合规则落入 catalog/skill，本文档届时只引用。

---

## 五、完整数据处理流水线

```
用户上传数据
    ↓
Step 0: identify_ev19_template（tool）              ← 确定 template + paradigm（内部已 parse 列名，返回前 20 列参考）
    ↓
Step 1: inspect_uploaded_file（tool，带上 paradigm） ← 完整列名 + 5 行预览 + 每列几何证据（column_assessment）
    ↓
Step 2: Layer 1 — 白名单匹配固定列
    → 匹配的 → known_fixed_columns，不进入反问
    → 不匹配的 → 进入 Step 3
    ↓
Step 3: Layer 2 + Layer 3 — 分析区列处理
    → 读 catalog：该范式有几个逻辑分析区？
    → 子区检测：用户分析区列数 > 逻辑区数？
      - 否 → 仅做名称映射（Sprint 1）
      - 是 → 需要结构聚合（Sprint 2，聚合语义待 Issue #98 专家确认）
    → 预填推测（基于列名模式 + catalog 合法菜单 + inspect 的几何证据）
    → HITL 合并反问（所有问题一次问清）
    ↓
Step 4: 聚合执行（如需要，聚合方式 = 专家确认的语义）
    → 产出：column_semantics + aggregation_rules → experiment-context.json
    ↓
Step 5: code-executor 按常规流程计算指标
    → 聚合后的数据已匹配 catalog 期望
    ↓
Step 6: S2 validate.py 检查指标合理性（当前 suffix 匹配；catalog-driven 升级见 §6）
    ↓
Step 7: S3 data-analyst fast-fail + 统计审核
```

---

## 六、Catalog-Driven 指标验证（S2 进阶方案）

当前 S2 (`validate.py`，已实现于 `feature/s1-s4-implementation`) 用硬编码的 naming convention 推导验证规则。实际识别 6 个后缀：`_ratio`（[0,1]）/ `_pct`（[0,100]）/ `_count` / `_time` / `_latency` / `_distance`（后四者 ≥0），外加 name-agnostic 的 NaN/Inf 检查。这比 prefix 匹配好，但仍脆弱——经调研，**31 个实际 emit 的指标中只有约 14 个被 suffix 规则命中（~45%）**，`_stats` / `_filtered` / `_bins` / 无后缀的指标全部漏网。

**正确的做法**：从 catalog 读取 `output_unit`，推导验证规则：

```yaml
# catalog/epm.yaml（已有字段，output_unit 100% 覆盖、loader 强制必填）
metrics:
  open_arm_time_ratio:
    output_unit: ratio     → 自动应用 0–1 范围检查
  center_entry_count:
    output_unit: count     → 自动应用非负整数检查
  cumulative_distance:
    output_unit: cm        → 自动应用非负检查
```

`validate_metrics` 接受 catalog context，按 `output_unit` 推导规则，不再依赖 naming convention：

| output_unit | 验证规则 |
|-------------|---------|
| ratio | ∈ [0, 1] |
| count | ≥ 0, 整数 |
| seconds / cm / radians / mm_s2 | ≥ 0（物理单位，下限 0；上限见下方缺口 4） |
| pct | ∈ [0, 100]（当前 catalog 无此单位，预留） |

好处：新范式加一个新 metric 到 catalog，验证规则自动生效，不需要改 `validate.py`。

**落地前必须闭合的 4 个缺口**（调研发现，否则 catalog-driven PR 写不出来）：

1. **`mm_s2` 未在原 4 条规则内**：`acceleration_stats` 的 output_unit=`mm_s2`，已补入上表"物理单位 ≥0"一类。catalog 实际出现的 output_unit 共 6 种：`ratio` / `seconds` / `count` / `radians` / `cm` / `mm_s2`——验证器必须穷举它们，建议 loader 加 `output_unit` 合法值枚举校验，新单位未登记即报错而非静默放过。
2. **3 个孤儿指标**：`velocity_stats` / `thigmotaxis_index` / `distance_moved` 由 compute 脚本 emit，但**不是任何 catalog 的 metric id**。catalog-driven 验证查不到它们的 output_unit，必须决定：判 `CATALOG_UNKNOWN`（暴露孤儿，推荐）还是静默放过（退化成当前行为）。
3. **复合 `_stats` 指标是验证盲区**：`body_elongation_stats` / `turn_angle_stats` 等的"值"不是标量而是 `{mean, std, ...}` 字典，当前 `validate.py` 的 `isinstance(value,(int,float))` 直接跳过。output_unit=ratio 的 `body_elongation_stats` 永远不会被 [0,1] 校验。catalog-driven 同样需决定复合指标怎么验（拆字段验 mean？单列一类？）。
4. **物理单位的上限缺失**：`radians` 是 [0,2π] / [-π,π] / 仅非负？`cm` 距离有无 plausible 上限（追踪丢失会导致距离爆炸，≥0 检不出）？仅"≥0"太松，真正的脏数据逃得过。建议 catalog 可选声明 `plausible_max`，验证器据此加上限。

**当前状态**：✅ **已实施并合 `dev`（PR #102）**。`ethoinsight/validate_catalog.py` 实现了 catalog-driven 验证，上述 4 缺口全部闭合：
- 缺口 1（mm_s2）：规则表穷举 6 种 output_unit（ratio/seconds/count/radians/cm/mm_s2），未登记单位报 `unknown_output_unit`。
- 缺口 2（孤儿）：`validate_metrics_against_catalog` 入口对不在 catalog 的指标报 `catalog_unknown`（注：CLI 走 plan 路径不遇孤儿）。
- 缺口 3（复合 _stats）：`_validate_composite_stats` 拆字段——mean/median/min/max/p25/p75 套范围，std/sem/var 只验 ≥0，n/count 验整数。
- 缺口 4（物理上限）：规则表 `upper=None` 预留 `plausible_max`，等 catalog 加可选字段（见 §8 待实施）。

L-A（`validate.py`）已**收窄为只 NaN/Inf**，suffix 范围逻辑全部移除（迁到 L-B）。两层分工最终形态见 §8。

---

## 七、与 AutoResearch 的深度对应

类比在两处是真洞见，在两处是误导——下表已标注，避免把类比当论证。

| AutoResearch | EthoInsight | 对应质量 |
|---|---|---|
| `prepare.py`（不可变评估代码） | `ethoinsight/statistics.py`（确定性决策树）+ Golden Cases | ✅ **真等价**——都是 agent 不能改、定义对错的代码 |
| `train.py`（agent **唯一可改**文件，实验的动作空间） | **`experiment-context.json`**（agent 通过 set_experiment_paradigm 写入，schema 受控）。**不是 catalog** | ⚠️ **原表把 train.py 对应到 catalog，方向反了**：train.py 可变，catalog 是 agent **绝对只读**的 SSOT，二者在"可变性"这个最关键维度上正相反。agent 真正的可写动作空间是 context.json |
| catalog YAML | 无 AutoResearch 直接对应（AutoResearch 没有"只读真理源"层） | catalog 是 EthoInsight 独有的 SSOT 层，硬塞进 train.py 这一格只会模糊它 |
| `program.md`（人类迭代、**agent 运行时实时读**的指令） | SKILL.md + references（agent 主动 read_file） | ⚠️ **有致命非对称**：program.md 的力量全在"agent 真的会读它"。**本设计文档在 `docs/design/` 下，agent 运行时读不到**——它只是人看的设计稿。要兑现 program.md 的价值，方法论必须**下沉成 skill references**（见 §8 待办），否则类比落空 |
| `val_bpb`（不可变评估指标） | Golden Case 通过率 + Experiment Log 的 user_feedback 信号 | 待 S5 实施 |
| `results.tsv`（追加实验日志） | Experiment Log（S5） | 待实施 |
| `if isnan(loss): exit(1)`（代码级 fast-fail） | S2 validate.py（其 docstring 自注 "AutoResearch-inspired"）+ S3 fast-fail | ✅ **真等价，已落进代码** |
| git commit 管理实验状态 | experiment-context.json 版本化 + column_semantics 可追溯 | 已有基础 |

**小结**：类比在"不可变评估代码"（statistics.py）和"代码级 fast-fail"（validate.py）两处是真洞见、已落地。但 AutoResearch 是**单文件可改 + 文档被 agent 实时读**，EthoInsight 是 **SSOT 只读 + 设计文档目前没人读**——这是两套不同架构，类比提供灵感，不能当论证。§9 第 9 问"是否过度类比"的答案是：**部分过度，已在上表标注修正**。

---

## 八、实施状态与后续

### 已完成（已合 `dev`）

| Item | 内容 | 落地 |
|------|------|------|
| S1 | Subagent 级 LoopDetectionMiddleware（每次新实例，防 thread_id 污染） | PR #100 |
| S2（L-A） | `ethoinsight/validate.py` — 收窄为**只查 NaN/Inf**（name-agnostic 进程内安全网）+ `_cli.py:emit_result` 注入点 | PR #100→#102 |
| S2→S3 通路 | VALIDATION_ERROR → code-executor data_quality_warnings（`code=METRIC_VALIDATION`）→ data-analyst 可读 | PR #101 |
| S3 | data-analyst fast-fail 规则（n<3 → partial / all-metrics-failed → failed / gate failure → failed） | PR #100 |
| S4 | ethoinsight/ 写保护 red anchor test ×3 | PR #100 |
| **S2 catalog-driven（L-B）** | **`ethoinsight/validate_catalog.py`** — 按 `output_unit` 范围校验，**4 缺口全闭合**（mm_s2 枚举 / 孤儿判 catalog_unknown / 复合 _stats 拆字段 / 物理单位上限预留 `plausible_max`）。两层分工：L-A 进程内只 NaN/Inf，L-B 在 **code-executor 层** CLI `python -m ethoinsight.validate_catalog --plan`。**CLI 按 subject 逐条验证**（plan 按 subject 展开，同 metric_id 多条不互相覆盖），**直接用 plan 自带的 output_unit**（不二次 load_catalog）。guardrail 白名单已精准扩 `validate_catalog`。 | PR #102 |
| **方法论 skill 下沉** | `ethoinsight-column-confirmation/references/column-processing-methodology.md` — 三层框架 + 工具分工 + 禁 head -1 + 列名来源链；**不内嵌结构化知识**（指向 catalog/#98）、**Layer 3 聚合留白**指向 #98 | PR #102 |

> **catalog-driven 验证两层分工最终形态**（PR #102 经 review + P0/P1 修复）：
> - **L-A**（`validate.py`，进程内 `emit_result`）：只查 NaN/Inf，拿不到 paradigm，name-agnostic 安全网。
> - **L-B**（`validate_catalog.py`，code-executor 层）：两个入口——
>   - `validate_plan_results(plan)`（CLI 用）：吃 plan_metrics.json，用每条 metric 自带的 `output_unit`（resolve 从 catalog 透传），**不 load_catalog**；按 subject 逐条验证，违规标签带 `#<subject_index>`。
>   - `validate_metrics_against_catalog(results, paradigm)`（直接调函数）：load_catalog 查 output_unit，能检出 catalog 之外的孤儿（catalog_unknown）。CLI 路径不遇孤儿（plan 只含 catalog 内指标）。
>   - 新增缺口暴露：`plan_missing_output_unit` / `result_file_unreadable`。

### 待实施

| Item | 阻塞项 | 优先级 |
|------|--------|--------|
| S5 Experiment Log | 设计已就绪（经 Opus review），待实施 | v0.1 |
| Layer 1 固定列清单 | **等行为学专家确认**（[Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)） | Sprint 2 |
| Layer 2 概念列表完整性 | **等行为学专家确认**（Issue #98） | Sprint 2 |
| Layer 3 结构聚合 | **等行为学专家确认 zone 空间关系（互斥/重叠）→ 决定聚合语义**（Issue #98 决定性问题） | Sprint 2 |
| `references/column-structure-methodology.md` 的**聚合规则部分** | 专家确认聚合语义后补写（方法论骨架已随 `column-processing-methodology.md` 下沉，仅聚合规则留白） | Sprint 2 |
| 物理单位 `plausible_max` 上限 | 机制已在 `validate_catalog.py` 预留（rule 表 `upper=None`），等 catalog MetricEntry 增加可选字段 + 专家给合理上限 | v0.1 后 |
| ~~`bash head -1` 列名提取 step~~ | **已取消**——经核实对 EV19（UTF-16/分号/标题行数）不可用；列名由 identify/inspect 的 parser 提供 | — |

---

## 九、调研 Review 结论（2026-06-06 已完成）

以下 9 问已由 agent 调研 + grill 核实，结论标注如下。

### 代码事实（已核实）

1. **Tool 分工** — ✅ 核实：`identify_ev19_template` **不返回完整列名**（`columns[:20]` 硬截断，`identify_ev19_template_tool.py:524`）；`inspect_uploaded_file` 返回**完整列名 + 5 行预览 + column_assessment**（`:414`），用 `parse_header()`。`bash head -1` 与二者都不可比——它对 EV19 根本拿不到列名（见第 6 问）。
2. **Catalog output_unit** — ✅ 核实：**100% 覆盖**（35/35 metric 都有，loader 强制必填）。出现 6 种值：ratio/seconds/count/radians/cm/**mm_s2**（mm_s2 是原 4 规则漏的，§6 已补）。
3. **column_semantics 流程** — ✅ 核实：1:1 映射通路**完整且有测试**。`_apply_aliases`（`resolve.py:542`）单点重映射，`test_column_semantics.py:208` 用真实 OFT 数据（中心区→center）验证通过。无断点。
4. **validate.py 覆盖** — ✅ 核实：validate.py 存在于 `feature/s1-s4-implementation`（74 行）。suffix 匹配覆盖 **~14/31 实际 emit 指标（~45%）**，`_stats`/`_filtered`/`_bins`/无后缀全漏。佐证"suffix 脆弱、应转 catalog-driven"。另发现 3 个孤儿指标（§6 缺口 2）。

### 设计决策（已评估）

5. **三层框架通用性** — ✅ 成立：FST/TST 无自定义分析区，Layer 2/3 自然跳过（catalog 声明的逻辑区数为 0 → 子区检测不触发）。框架对无分析区范式正确降级。
6. **bash head -1 可用性** — ❌ **彻底证伪**：EV19 是 UTF-16 LE + 分号分隔 + 第 0 行是 `标题行数：N`（列名在第 N-2 行）。`head -1 | tr ','` 在编码/分隔符/行号三处全错，**TXT/CSV/XLSX/XLS 无一可用**（XLSX 还是二进制）。**已从 §3/§5/§8 全部删除**。正确轻量工具：`python -m ethoinsight.parse.dump_headers`。
7. **聚合正确性** — ⚠️ **悬而未决，依赖行为学事实**：sum 在子区空间重叠时 double-count；现有代码用帧级 OR（`epm.py:96`）。是否 sum/OR **取决于 EV19 zone 是否互斥**——已升级为 Issue #98 决定性问题，本文档不预设结论（§4）。
8. **catalog-driven 覆盖度** — ⚠️ 机制对、规则集不完整：4 个缺口（mm_s2 枚举 / 孤儿指标判 UNKNOWN / 复合 _stats 盲区 / 物理单位上限），已在 §6 列出，是开 PR 的前置。
9. **AutoResearch 类比** — ⚠️ **部分过度**：`prepare.py↔statistics.py`、`fast-fail↔validate.py` 真等价；但 `train.py↔catalog` **方向反了**（可变 vs 只读，真正对应是 context.json），`program.md↔本文档` 有"运行时不可读"非对称。已在 §7 修正。

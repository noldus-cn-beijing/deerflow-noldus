# EV19 列语义对齐（Column Semantics Alignment）

**状态**：Sprint 1 done（已合 dev）· Sprint 2 结构聚合机制**已在线**（#98 已 CLOSED、同事方法论已交付）· 剩余=坐实+固化+补测（[实施 spec](../superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md)，非新增能力、不阻塞）
**时间跨度**：2026-06-05 ~ 2026-06-06（Sprint 1）· 2026-06-18 #98 CLOSED（Sprint 2 解冻）
**dev HEAD**：`6e6a3c27`（PR #99 merge commit，Sprint 1）

## 做了什么

真实 OFT 数据（34 个 Xuhui XLSX）上传后被 agent 拒绝分析。根因不是模板识别，而是：用户在 EthoVision「按分析区输出」自定义的分析区列，列名 100% 由用户命名（`中心区`/`边缘区`/`Center`/`zone_A`…），归一化后既不在 COLUMN_MAP、也不匹配 catalog 的 `in_zone_center_*` glob → default metric 报 `columns_missing` → 整个 plan 中止。

Sprint 1 建了一条 **HITL 列语义对齐** 链路：inspect 标出未识别的自定义分析区列 + 客观证据（占时分布）→ lead 用 catalog 合法概念菜单**预填推测**让用户一键确认/纠正（绝不字面猜，守"预填反诘≠断言"）→ 决议落 `column_semantics`（SSOT）→ resolve 在 `columns` 入口单点重映射，把用户列翻译成 catalog 概念列 → metric 算出来。标准命名数据零额外开销（不触发反问）。

## 关键节点

| 日期 | 事件 | handoff |
|------|------|---------|
| 6/5 | 设计讨论 → v2 设计定稿（16 条决策 D1–D16），实测 34 文件钉死根因 | [design v2](../design/2026-06-05-column-semantics-hitl-design-v2.md) |
| 6/5 | Sprint 1 实施 spec（泛化 PR #89 单区机制为通用列别名表） | [Sprint 1 spec](../superpowers/specs/2026-06-05-column-semantics-alignment-sprint1-spec.md) |
| 6/5 | 开 Sprint 2 结构勘察 issue 给行为学专家 | [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) || 6/6 | code review 暴露 3 critical（集成断链 / 静默丢指标 / 假集成测试）+ 2 medium，全部修复 | — |
| 6/6 | merge dev（161 commit 落后，7 文件 18 冲突全解）→ PR #99 合入 dev | — |
| 6/18 | **Issue #98 CLOSED** — 同事逐范式聚合方法论交付（PR #115 → `docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`），Sprint 2 解冻 | [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) |

## 当前状态

- **完成项（Sprint 1 = 名字对齐，1:1）**：
  - `assess_column_confidence`（utils.py）：纯函数判定列 recognized/unrecognized，不靠 LLM
  - 通用列别名表 + 概念翻译（resolve.py `_apply_aliases` / `_zone_concept_map` / `_materialize_concept`）：LLM 只填概念关键词（`center`），代码翻译成匹配 catalog glob 的列名；`_apply_aliases` 两边过 normalize 鲁棒匹配（raw `中心区` ↔ 归一化 `center` 都命中）
  - `inspect_uploaded_file(paradigm=…)` 增强：返回 column_assessment + open_questions + 每列证据
  - `set_experiment_paradigm(column_semantics=…)`：写 SSOT + 写盘时单向投影 column_aliases
  - `prep_metric_plan` + catalog CLI 读 column_aliases 并传 resolve（metrics + charts 两路）
  - guardrail 加 column_semantics 未确认子检查；`columns_missing` hint 改为先走列对齐
  - 新 skill `ethoinsight-column-confirmation`（交互方法论，lead-only，三件套齐全 + 运行时实测 lead 可见，三层渐进披露）
  - 测试：ethoinsight 624 passed / agent-backend column_semantics 15 passed
  - 与 PR #89 的 `anonymous_zone_is`（裸 in_zone 单区）机制**并存**：别名重映射先跑、anonymous_zone 兜底后跑
- **Sprint 2（= 结构对齐，N:1）状态：机制已在线、剩余为坐实+固化+补测**：粒度/聚合差异（如 EPM 4 区 open_arm1/2 → 标准 2 区聚合）。读码坐实 N 列→1 概念的 OR 聚合机制**全链路早已实现并工作**（`resolve.py:_build_zone_aliases_overrides` 多列收集 + `metrics/epm.py` `df[cols].max(axis=1)` OR 聚合 + `epm.yaml` glob 匹配多列）。#98 已 CLOSED（2026-06-18）、同事方法论已交付（PR #115）。**不再 blocked**——剩余工作是补多列聚合回归测试 + 固化特殊范式规则（LDB 忽略隐藏区 / 累积区去重陷阱 / FST·TST 不分区）+ 更新文档，**非新增能力**，spec 见 [2026-06-26 结构聚合 spec](../superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md)。schema 已前向兼容（`resolves_to` 可扩展为多列）。
- **下一步**：派实施 Sprint 2 坐实 spec（低风险、不卡能力）；后续真实数据出现新结构 case 时增量加 `resolves_to`。**行为学分析方向当前唯一硬阻塞已转移到 Golden Cases（[Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90)，微调路线）**，与本 track 正交。

## 设计铁律（沉淀）

- **不字面猜 zone 身份**（同事铁律"虽然这次猜中了但不要猜"）：预填来自占时证据 + catalog 菜单，不来自列名字面。
- **预填反诘 ≠ 断言**：系统可预填最佳猜测让用户一键确认/纠正，但绝不直接采纳。
- **概念翻译下沉到代码**：机器列名后缀（`_point` 等）是 resolve 的契约，不靠 markdown 教 LLM 填；mapping skill 只承载纯语义层。
- **承重墙消费入口不动**：别名重映射锁在 `columns` 单点，下游 `_missing_columns` 等零改动。

## 相关 handoff

- [design v2](../design/2026-06-05-column-semantics-hitl-design-v2.md) — 完整设计 + 16 决策 + 6 场景工作流
- [Sprint 1 spec](../superpowers/specs/2026-06-05-column-semantics-alignment-sprint1-spec.md) — 实施清单（精确到 file:line）

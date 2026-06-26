# 阻塞清单 — 等行为学同事的范式方法论

> 更新：2026-06-26 ｜ 维护：技术同事
> 用途：把"等同事"这件笼统的事拆成**精确、可逐项交付**的待办，降低同事回合成本。
> 上游一动就解冻；在此之前，harness/基础设施层照常推进（见 [milestone README](README.md)）。

---

## 0. 一句话现状

v0.1 推进的行为学分析方向**只剩一条真实阻塞**，卡在 **行为学同事的 Golden Cases**（不是工程卡点）。另有一条**新 feature 的方法论依赖**（不卡 v0.1，但卡跨范式对比 feature）：

| # | 阻塞 | Issue | 卡住的能力 | 不卡的部分 |
|---|------|-------|-----------|-----------|
| ② | **Golden Cases**（微调 benchmark + 回归种子） | [#90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90) OPEN | SkillOpt 优化循环 + SFT 数据生成（微调路线） | v0.1 端到端分析**不卡**（识别+判读+聚合全已交付） |
| ③ | **跨范式对比方法论**（同批动物多范式结论怎么对比） | [#226](https://github.com/noldus-cn-beijing/noldus-insight/issues/226) OPEN | Experiment 跨范式 synthesizer 的判读规则（落成新 skill `ethoinsight-cross-paradigm`） | experiment 工程骨架（建表/import/并列展示）**不卡**，可先建 |

> ✅ **结构聚合（原阻塞 ①，Issue #98）已解除**（#98 于 2026-06-18 CLOSED）：同事方法论已交付（PR #115 → `docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`）。读码坐实 N 列→1 概念的 OR 聚合机制**全链路早已实现并工作**（`resolve.py:_build_zone_aliases_overrides` 多列收集 + `metrics/epm.py` `df[cols].max(axis=1)` OR 聚合 + `epm.yaml` glob 匹配多列）——milestone 此前标的 blocked **滞后于代码**。**功能性上 v0.1 六范式端到端分析现已可用**，复杂多分区数据也能聚合。剩余是"坐实+固化+补测"（不是新增能力），spec 见 `docs/superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md`，**不再 blocked、可随时开工**。详见本文 §1。
>
> ✅ **识别 + 判读领域知识也已交付**：v0.1 六范式（EPM/OFT/LDB/FST/Zero Maze/TST）的识别 + 判读知识同事已填实，落在 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/` 下对应的 `by-experiment/*.md` + `by-template/*.md`（占位注释已删、含必算指标/判读语言/参考文献）。未填的全是 v1.0 才支持的范式（鱼类/学习记忆迷宫/PhenoTyper/昆虫）。

---

## 1. 结构聚合（原阻塞 ①，Issue #98 — **已解除**）

> ✅ **#98 已 CLOSED（2026-06-18）、同事方法论已交付、聚合机制已在线。本节保留作历史与对接记录，不再是阻塞。**

### 1.1 问题本质（一句话）

用户数据的**分区粒度**和 catalog 最佳实践不同，需要**聚合/拆分**才等价。这不是"列名叫错了"（那是 Sprint 1 名字对齐，已解决），是"区的数量/结构不一样"。

典型 case（EPM）：
```
标准最佳实践按【2 区】算：open_arm, closed_arm
有的用户数据是【4 区】：    open_arm1, open_arm2, closed_arm1, closed_arm2
需聚合：open_arm = open_arm1 ∪ open_arm2，closed_arm = closed_arm1 ∪ closed_arm2
```
设计依据：`docs/design/2026-06-05-column-semantics-hitl-design-v2.md` §6.2（决策 D16）。

### 1.2 同事方法论已交付（PR #115）的逐范式聚合规则

| 范式 | 聚合规则 |
|------|---------|
| **EPM / Zero Maze** | 两开放臂 → OR 合并（已实现）；两封闭臂同理；累积分析区（EV 内已聚合的"all"）若与 A/B 同时输出，**别二次叠加**（唯一陷阱） |
| **LDB** | 明区 + 暗区；隐藏分析区（入口）→ 忽略（坐标落入暗区，直接算暗区） |
| **OFT** | 中心区 + 外周区；墙区/九宫格非标准（v1.0） |
| **FST / TST** | 不分区，无 zone 聚合，不应触发列对齐反问 |
| **MWM** | 多画法不合并（v0.1 不支持；未来支持时规则是"不聚合"） |

SSOT：`docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md`。

### 1.3 剩余工作 = 坐实+固化+补测（非新增能力，不阻塞）

OR 聚合机制已实现且正确，剩余只是补"锁行为"的回归测试 + 固化特殊范式规则（LDB 忽略隐藏区 / FST·TST 不分区 / 累积区去重陷阱）+ 更新文档状态。详见实施 spec `docs/superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md`。**风险低、随时可派、不卡分析能力**。

### 1.4 后续增量结构 case（开放式，不预先封闭）

每当真实 dogfood 数据出现新的非标准分区结构，技术同事增量加一条 `resolves_to` 多列映射（schema 已前向兼容，`resolves_to` 可为列表）。**这是持续迭代、不阻塞其他范式**。

---

## 2. 阻塞 ② — Golden Cases（Issue #90 — **当前唯一硬阻塞**）

### 2.1 问题本质（一句话）

SkillOpt 优化循环 + SFT 数据生成的前提是**每个 benchmark 任务有明确的对/错评分标准**。这个标准**只有行为学同事能定义**（synthetic fixture 只能当初始占位）。当前 golden case 数量 = **0**，这是微调路线的硬阻塞。

依据：`docs/plans/2026-06-04-skillopt-skill-optimization-plan.md` 阶段 1。

### 2.2 目标产出量

v0.1 六范式，**每个至少 2-3 个 golden case**，总计 **12-18 个**。覆盖：正常对照、显著差异、含离群个体、含混杂因素、样本量不足等场景。

### 2.3 框架已就绪（同事不用从零）

- 结构字典：[`golden-cases/SCHEMA.md`](../../golden-cases/SCHEMA.md)
- 模板：`golden-cases/TEMPLATE/`
- 校验脚本：`python3 scripts/validate_golden_case.py`
- 协作 SOP：[`docs/sop/golden-case-sop.md`](../sop/golden-case-sop.md)

### 2.4 分工（SCHEMA.md §5 标注流程）

| 步骤 | 谁 | 做什么 |
|------|----|----|
| Step 1 | 工程师 | 建目录 + 拷 raw data + 填 `metadata.yaml` + 填 `expected-analysis.yaml` 的**数值字段**（`expected_metrics`） |
| Step 2 | 工程师 | 跑一遍 agent，把当前输出贴给同事参考 |
| Step 3 | **行为学同事** | review `expected-analysis.yaml`，补齐**判断质量字段**（见 2.5）+ 写 `notes.md` 判读理由 |

### 2.5 同事要补的核心字段（FindingExpectation —— SCHEMA.md §2 最重要部分）

每个 golden case 的 `expected-analysis.yaml` 里，`expected_findings`（必填）每条要给：

| 字段 | 同事填什么 |
|------|-----------|
| `type` | 发现类型：`outlier_detection` / `counterfactual_analysis` / `confound_note` / `statistical_conclusion` / `data_quality_warning` / `phenotype_indication` |
| `claim` | 核心结论的措辞（如"组间差异不显著""效应量小"） |
| `reasoning` | **判断理由 1-3 句**（这是 SFT 最值钱的部分——专家怎么想的） |
| `severity` | 离群类发现的严重度：`low`/`moderate`/`high`/`critical` |
| `required_keywords` | agent 输出**必须包含**的关键词 |
| `forbidden_claims` / `should_not_contain` | agent **不得出现**的表述（捕幻觉/过度推断，如样本量 5 却报 `p<0.001`、不存在的 Subject 名、数据不支持的因果断言） |

### 2.6 一个 case 的最小交付

```
golden-cases/case-001-epm-control-vs-drug/
├── metadata.yaml            # 范式/物种/n/分组（工程师起草）
├── expected-analysis.yaml   # ★ 数值字段工程师填，findings/severity/reasoning 同事补
├── notes.md                 # 同事的判读全过程（双重用途：领域知识 + SFT 推理种子）
└── raw-data/                # EthoVision 导出 txt/xlsx（脱敏）
```

---

## 3. Golden Cases 解冻后会发生什么

- **② Golden Cases**：攒到每范式 ≥2 个 → SkillOpt 阶段 2/3 可启动（EnvAdapter + 优化循环）→ 产出 `best_skill.md` → 驱动 agent 生成高质量 SFT 轨迹 → 微调 Qwen3-30B。
- **① 结构聚合（已解冻）**：聚合机制已在线；每出现一个新的非标准分区结构 case → 技术同事增量加一条 `resolves_to` 多列映射（resolve.py 单点）→ 复杂分区数据可算。**增量、不阻塞、不卡能力**。

## 4. 同事交付后，技术同事要做的对接

- **② 收 golden case**：`validate_golden_case.py` 过 schema → 纳入 `golden-cases/` → 接入 SkillOpt dataloader（`dataloader.py` 扫 `case-*/`）。
- **① 收一条新结构 case（增量、非阻塞）**：在 `column_semantics` 别名表加多列 `resolves_to`，跑该范式 dogfood 验证聚合正确（不只是不 crash，是数值对）。
- 完成后回写本文件状态 + 更新 [milestone README](README.md) 对应 track。

---

## 相关文档

- [Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90) — Golden Cases（当前唯一硬阻塞）
- [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) — 结构聚合勘察（已 CLOSED 2026-06-18）
- [Sprint 2 实施 spec](../superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md) — 聚合机制坐实+固化+补测（非新增能力）
- [column-semantics design v2 §6.2](../design/2026-06-05-column-semantics-hitl-design-v2.md) — Sprint 2 结构对齐设计 + D16
- [column-semantics milestone](column-semantics-alignment.md) — Sprint 1/2 已交付内容
- [SkillOpt 实施计划](../plans/2026-06-04-skillopt-skill-optimization-plan.md) — 5 阶段微调路线
- [golden-cases/SCHEMA.md](../../golden-cases/SCHEMA.md) — 标注结构字典
- [golden-case SOP](../sop/golden-case-sop.md) — 协作流程

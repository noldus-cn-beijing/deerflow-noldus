# 阻塞清单 — 等行为学同事的范式方法论

> 更新：2026-06-10 ｜ 维护：技术同事
> 用途：把"等同事"这件笼统的事拆成**精确、可逐项交付**的待办，降低同事回合成本。
> 上游一动这两条就解冻；在此之前，harness/基础设施层照常推进（见 [milestone README](README.md)）。

---

## 0. 一句话现状

v0.1 推进**只有两条真实阻塞**，且都卡在同一个上游 —— **行为学同事的范式方法论**，不是工程卡点：

| # | 阻塞 | Issue | 卡住的能力 | 不卡的部分 |
|---|------|-------|-----------|-----------|
| ① | **结构聚合**（自定义分区粒度按范式聚合） | [#98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) | 复杂多分区数据（如 EPM 4 区）的指标计算 | 标准命名/标准分区数据**现在就能端到端跑**（Sprint 1 已合 dev） |
| ② | **Golden Cases**（微调 benchmark + 回归种子） | [#90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90) | SkillOpt 优化循环 + SFT 数据生成（微调路线） | v0.1 端到端分析**不卡**（识别+判读知识同事已交付） |

> ✅ **已交付（澄清，避免误以为全卡）**：v0.1 六范式（EPM/OFT/LDB/FST/Zero Maze/TST）的**识别 + 判读领域知识**同事已填实，落在 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/` 下对应的 `by-experiment/*.md` + `by-template/*.md`（占位注释已删、含必算指标/判读语言/参考文献）。未填的全是 v1.0 才支持的范式（鱼类/学习记忆迷宫/PhenoTyper/昆虫）。

---

## 1. 阻塞 ① — 结构聚合（Issue #98）

### 1.1 问题本质（一句话）

用户数据的**分区粒度**和 catalog 最佳实践不同，需要**聚合/拆分**才等价。这不是"列名叫错了"（那是 Sprint 1 名字对齐，已解决），是"区的数量/结构不一样"。

典型 case（EPM）：
```
标准最佳实践按【2 区】算：open_arm, closed_arm
有的用户数据是【4 区】：    open_arm1, open_arm2, closed_arm1, closed_arm2
需聚合：open_arm = open_arm1 ∪ open_arm2，closed_arm = closed_arm1 ∪ closed_arm2
```
设计依据：`docs/design/2026-06-05-column-semantics-hitl-design-v2.md` §6.2（决策 D16）。

### 1.2 为什么必须等同事（不能工程拍脑袋）

**聚合语义不一定都是 OR（并集）**：
- EPM open_arm1/2 → OR（占时取并集）✅ 这个已勘察清楚，`metrics/epm.py` 用 `df[cols].max(axis=1)` 实现（当前在 `epm.py:101`/`:153`，多列经 `_get_open_zone_cols` glob 抓取后取并集），`epm.yaml:9` 的 `in_zone_open_arms_*` glob 本就允许多列。
- 但其他范式/其他结构可能是：**加权平均**、**需区分臂身份的分别统计**、**先聚合再算比率 vs 先算比率再平均**（结果不同）。
- **这层语义只有行为学同事能拍**。错误聚合会静默产出错指标（比 crash 更危险）。

⚠️ EPM 这一例**绝不可当 Sprint 2 全貌**——它是 6 范式里 1 个范式的 1 种结构 case。

### 1.3 需要同事逐范式确认的清单（核心待办）

对 v0.1 六范式，每个范式回答：**真实用户数据里有哪些"非标准分区结构"，每种的正确聚合语义是什么？**

| 范式 | 标准分区 | 同事需确认 |
|------|---------|-----------|
| **EPM** | open_arm / closed_arm（2 区） | 4 区（open1/2+closed1/2）→ OR？是否需"哪条臂偏好"的分别统计？中心区是否单列 |
| **OFT** | center / border（或单列 in_zone） | 自定义环数（如 center/middle/periphery 3 区）→ 如何映射到 2 区？middle 归 center 还是 border |
| **LDB** | light / dark（2 区） | 是否有"过渡区/门口区"第三区？怎么归并 |
| **Zero Maze** | open / closed（无中心） | 4 象限（open1/2+closed1/2）→ 聚合语义同 EPM？ |
| **FST** | （游泳/不动，不分区，靠 pendulum 参数） | 主要是参数而非分区，确认是否有分区变体 |
| **TST** | （悬尾，pendulum 参数） | 同上 |

> **同事每确认一个范式的一种结构 case**，技术同事就增量落一条 `resolves_to` 多列映射（schema 已前向兼容，§6.2 决策：`resolves_to` 可为列表）。**开放式迭代，不预先封闭定义**（避免单 case 过拟合）。

### 1.4 同事回的最小信息单元（一条结构 case）

```
范式：EPM
非标准结构：4 区 open_arm1 / open_arm2 / closed_arm1 / closed_arm2
正确聚合：open_arm = open_arm1 ∪ open_arm2（占时并集）;
          closed_arm = closed_arm1 ∪ closed_arm2
是否需分别统计：否（标准 EPM 不区分左右臂）
特例/陷阱：若用户想看"臂偏好"则需保留分臂，但这是另一种分析意图
```

---

## 2. 阻塞 ② — Golden Cases（Issue #90）

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

## 3. 这两条解冻后会发生什么

- **① 结构聚合**：每确认一个范式结构 case → 技术同事加一条 `resolves_to` 多列映射（resolve.py `_apply_aliases` 单点）→ 复杂分区数据可算。**增量、不阻塞其他范式**。
- **② Golden Cases**：攒到每范式 ≥2 个 → SkillOpt 阶段 2/3 可启动（EnvAdapter + 优化循环）→ 产出 `best_skill.md` → 驱动 agent 生成高质量 SFT 轨迹 → 微调 Qwen3-30B。

## 4. 同事交付后，技术同事要做的对接

- **① 收一条结构 case**：在 `column_semantics` 别名表加多列 `resolves_to`，跑该范式 dogfood 验证聚合正确（不只是不 crash，是数值对）。
- **② 收 golden case**：`validate_golden_case.py` 过 schema → 纳入 `golden-cases/` → 接入 SkillOpt dataloader（`dataloader.py` 扫 `case-*/`）。
- 两者完成后回写本文件状态 + 更新 [milestone README](README.md) 对应 track。

---

## 相关文档

- [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) — 结构聚合勘察
- [Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90) — Golden Cases
- [column-semantics design v2 §6.2](../design/2026-06-05-column-semantics-hitl-design-v2.md) — Sprint 2 结构对齐设计 + D16
- [column-semantics milestone](column-semantics-alignment.md) — Sprint 1 已交付内容
- [SkillOpt 实施计划](../plans/2026-06-04-skillopt-skill-optimization-plan.md) — 5 阶段微调路线
- [golden-cases/SCHEMA.md](../../golden-cases/SCHEMA.md) — 标注结构字典
- [golden-case SOP](../sop/golden-case-sop.md) — 协作流程

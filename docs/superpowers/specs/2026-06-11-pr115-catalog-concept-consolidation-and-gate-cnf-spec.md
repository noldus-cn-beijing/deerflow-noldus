# Spec 总纲/索引 — PR#115 实施地基：catalog 概念声明收口 + 门控 CNF 表达力（Q1+Q3 重构）

> **本文档是总纲/索引**：承载「为什么这么做」（Fable 两轮裁决、地基验证、边界声明）。「怎么做」在 4 篇可执行 stage spec 里（见 §2 索引表），实施 agent 据 stage spec 执行、据本总纲建立全局观。
> 状态：**两决策门已闭合，4 篇 stage spec 已就绪，可开工。**
> 日期：2026-06-11
> 来源：行为学同事 PR#115（issue #98 列聚合方法论 + #63 调参指南）落地 → 问 Fable 两轮纯代码架构问题（第一轮三问 + 第二轮两决策门）→ 据代码逐条核实裁决 → 拆 4 篇 stage spec。
> 关联：[[feedback_fable_pr115_arch_verdicts_gate_cnf_ssot_generate]]、[[feedback_fable_pr115_stage_decisions_parambinding_optional_and_buildtime_gen]]、[[feedback_single_source_of_truth]]、issue #98 / #63。

---

## 0. 背景与 Fable 两轮裁决（沉淀，stage spec 不重复论证）

PR#115 是行为学同事把 issue #98（zone 聚合）/#63（调参）里"卡工程、等专家"的领域知识答复了。**PR 本身不含代码，实施 = 把知识落进代码/skill/catalog。**

实施前就纯软件架构取舍问了 Fable 两轮（裁决均已逐条对代码核实）：

### 第一轮 — 三个架构问题

- **Q1 门控**：catalog `requires_columns` 是 AND-over-patterns、无 OR；一个 chart 脚本已内化三级降级却被门控误毙 → **选 CNF：给 `requires_columns` 列表项支持嵌套列表 = 组内 OR、整体 AND（`[t, [P, Q]]`）**，非平行 `requires_any` key（会漏掉两分支都要的列）。声明层职责 = 如实表达脚本结构性缺失的那半个契约（plan 出现=产出承诺 / PlanSkipped=响亮带 reason 的不承诺）。
- **Q2 跨包测试**：SSOT↔派生文档一致性守卫归 **harness（消费者）侧**，不是 library（依赖边零新增 + 被依赖方须不知消费者就能测干净 + 派生副本一致性是 artifact 所有者的自断言）。
- **Q3 SSOT 漂移**：概念菜单双存（catalog vs skill md）→ **直接做"从 catalog 生成"，跳过守卫/混合**（守卫的"便宜"是幻觉：md 比 catalog 多 → 守卫今天就红 → 修绿=删知识或收进 catalog=生成的前置同一笔钱）；统一**内部模型**不强求统一 YAML 表面，判据是 `anonymous_zone_override` 能否无损规范化。

### 第二轮 — 落地浮出的两个决策门（已闭合）

- **决策门 1（统一模型该不该容纳"无注入点的概念"，如 OFT border）→ 容纳**。蕴含关系反向（可注入 ⊂ 可对齐）；对齐的本体职责是**认领/消解歧义**，注入只是部分概念的可选绑定。类型用 **`ResolvedZoneConcept(concept, binding: ParamBinding | None, source)`**（`binding` 整体可空，非裸 `param: str | None`，让非法状态不可表达）。OFT `border` = `binding=None` 一等概念。**不拆"可注入/仅识别"两个集合**（否则在统一模型内造小型双存）。与 Q1 同构（声明层如实映射脚本结构的镜像）。
- **决策门 2（概念菜单生成形态：运行期 CLI vs 构建期 md）→ 构建期**。变更节奏匹配（菜单随 catalog 随代码版本变=构建期节奏）；运行期 CLI 是 sandbox 白名单+guardrail+注册三层新失败面（援引 chart-maker 潜伏 bug）；不借机改 `read_file md` 披露契约（媒介撕裂+变更耦合违"根因未隔离别叠加"纪律）。"绝不双存"精确判据 = **人能手改且手改能静默存活**——生成+CI staleness 后 md 降为物化视图（同 uv.lock），达标。三硬性细节：①生成器拥有整个文件、不嵌哨兵块 ②staleness 红带可执行修复指令 ③进常规构建路径、CI 只兜底。

> **第二轮裁决取代了第一轮里的"待定"**：Q3 第一轮说"统一进 `zone_concept_params`"，第二轮精化为 `ResolvedZoneConcept.binding 可空`；Q3/Stage4 第一轮列"运行期 vs 构建期待选"，第二轮定为构建期。**本总纲与 4 篇 stage spec 均已按第二轮闭合形态对齐。**

---

## 1. 地基验证结论（事实依据）

### 1.1 两套机制的真实语义（已读 `schema.py` + `resolve.py` 全部消费点）

catalog 里"category → 分析区概念 → compute 参数"的知识今天有**两套并存机制**：

| 机制 | schema | 用它的 category | 语义 |
|------|--------|----------------|------|
| `zone_concept_params: dict[str, ZoneConceptParam]` | `ZoneConceptParam(param, wrap_list)`，**按概念名 key** | 仅 EPM | 多概念、**显式命名**：`open_arms→open_arm_zones`、`closed_arms→closed_arm_zones` |
| `anonymous_zone_override: AnonymousZoneOverride \| None` | `AnonymousZoneOverride(target_param, wrap_list)`，**无概念名** | OFT/LDB/ZM | 单概念、**概念名隐式**、resolve 期反推 |

### 1.2 决定性发现：`anonymous_zone_override` 承载**两个可分离的关注点**

读遍 `resolve.py` 所有消费点（`:224 :249 :289 :586-607 :647-674 :744-790 :1136-1139`）后确认，`anonymous_zone_override` 不是单一职责，它同时干两件**正交**的事：

**关注点 (1) — concept→param 映射（与 `zone_concept_params` 同质）**
`resolve.py:602-607` 在 `_build_zone_aliases_overrides`（`resolve.py:566`，本总纲早期草稿误写为 `_build_zone_param_overrides`，全仓 0 命中，已更正）里，对 `anonymous_zone_override`：用 `_derive_concept_from_zone_patterns(zone_patterns, target_param)`（`:692-699`，删 `target_param` 的 `_zone(s)` 后缀）**反推出概念名**，再当作 `(concept → param, wrap_list)` 注入 —— 与 `zone_concept_params` **走完全相同的下游路径**。
- **此推导是确定性的、可在加载期复现**。三个 category 实测：
  - OFT：`center_zone`→`center`，glob `in_zone_center_*`→`center` ✓
  - LDB：`light_zone`→`light`，glob `in_zone_light*`→`light` ✓
  - ZM：`open_zones`→`open`，glob `in_zone_open*`→`open` ✓

**关注点 (2) — `anonymous_zone_is` 统一输入键（`zone_concept_params` 无等价物）**
`resolve.py:760-790`（softgate `_detect_anonymous_zone`）+ `:1136-1139`（translate）：HITL 层可传一个**统一键** `anonymous_zone_is: <val>`，resolve 在缺命名区列、但数据里有裸 `in_zone` 列时，靠"本范式声明了 override"这个信号触发软门反问，并把 `anonymous_zone_is` 翻译成 `target_param`。EPM 的 `zone_concept_params` **没有**这条统一键路径。

### 1.3 「能否无损规范化」判据的答案：**分裂，部分可合**（已按决策门 1 精化形态）

| 关注点 | 能否规范化进统一内部模型 | 结论 |
|--------|--------------------------|------|
| (1) concept→param 映射 | **能，无损**：loader 加载期跑同一个 `_derive_concept_from_zone_patterns`，把结果写进 `ResolvedZoneConcept(concept, binding=ParamBinding(target_param, wrap_list))` | 生成器/对齐逻辑只读统一后的 `resolved_zone_concepts`，不再知道有两套表面语法 |
| (2) `anonymous_zone_is` 统一输入键 | **不能、也不该**：这是**运行期输入契约**，非概念菜单知识，与 Q3 的 SSOT-drift 正交 | `anonymous_zone_override` **保留**，但收窄职责为"只承载关注点 (2)" |

> **这正是 Fable "统一内部模型、不强求统一 YAML 表面" 的精确落地**（决策门 1 又把内部模型的形态精化为 `ResolvedZoneConcept.binding: ParamBinding | None`）：语义模型（concept→可选 binding）收敛到一个内部表示；`anonymous_zone_override` 不是它的语法糖（含 (2) 这个表达不了的运行期键），所以**两套 YAML 表面都保留**，但加载期把 (1) 规范化到统一内部模型。

### 1.4 catalog 真·不完整点（issue #98 答复暴露、生成的前置）

skill md 菜单 vs catalog 实际声明的概念，差异分两类（已对同事 issue#98 答复核实）：

| category | md 列出 | catalog 实声明 | 差异处置（已按决策门 1 定形态） |
|----------|---------|---------------|---------|
| EPM | open_arms / closed_arms / **center** | open_arms / closed_arms | `center` **删**（同事：EPM 不需 center） |
| OFT | center / **border** / **corner** | center（仅 anonymous override） | `corner` **删**（同事：罕见）；`border` **补为 `binding=None` 一等概念**（同事：真实"外周区"；metrics 无可注入 param，靠 regex 自动识别+三级降级） |
| LDB | light / dark | light（+ dark 隐式补集） | `dark` **补为 `binding=ParamBinding("dark_zone")`**（metrics 有独立 `dark_zone` 参数） |
| ZM | open / closed | open（+ closed 隐式补集） | `closed` **补为 `binding=ParamBinding("closed_zones")`**（metrics 有独立 `closed_zones` 参数） |

> **结论**：catalog 今天结构上**根本没把 `border`/`dark`/`closed` 这些"补集区"枚举为可对齐概念**。这是 Fable 说的"权威源不完整、副本部分成了权威"的精确定位。Stage 3 补这三个概念（按各自 metrics 形态定 binding），是 Stage 4 生成的前置。

---

## 2. 4 篇 stage spec 索引 + 依赖图（「怎么做」在这些里）

实施细节（逐文件改动、TDD、验收闸门、接口契约）全在 4 篇可独立执行的 stage spec：

| Stage | 文件 | 覆盖 | 一句话完成判据 |
|-------|------|------|---------------|
| **Stage 1** | [stage1-requires-columns-cnf](./2026-06-11-pr115-stage1-requires-columns-cnf-spec.md) | Q1：`requires_columns` 升 CNF（`[t,[P,Q]]`） | 纯 list-of-str 路径逐字节回归绿 + 接受嵌套 CNF + `missing` 含 list 项时下游不崩 + flatten 全消费者处理 + catalog yaml 零改动 |
| **Stage 2** | [stage2-zone-concept-normalization](./2026-06-11-pr115-stage2-zone-concept-normalization-spec.md) | Q3 前置：concept→param 统一内部模型（`ResolvedZoneConcept.binding: ParamBinding\|None`，加载期规范化） | `_build_zone_aliases_overrides` 对 EPM/OFT/LDB/ZM 输出 golden 逐字节相等（等价性=无损证明）+ `resolved_zone_concepts` 含 derive 出的 concept + 关注点(2) softgate/translate 零改动 |
| **Stage 3** | [stage3-complement-zone-concepts](./2026-06-11-pr115-stage3-complement-zone-concepts-spec.md) | Q3 前置：补 border/dark/closed 概念枚举（逐范式异质 binding） | LDB `dark`/ZM `closed` 走 `binding=ParamBinding(真 param)`、OFT `border` 走 `binding=None` + resolve 输出字节等价 + 无聚合语义泄漏 |
| **Stage 4** | [stage4-generate-concept-menu](./2026-06-11-pr115-stage4-generate-concept-menu-spec.md) | Q3 主体：构建期生成概念菜单（独立 `.generated.md`），消除双存 | 两份手写表删并改链接指向独立生成文件 + 重生成==已提交（staleness 绿，带修复指令）+ 菜单覆盖全 6 范式（含 FST/TST 空集）、含 dark/closed/border、不含 EPM center/OFT corner |

### 依赖图与执行顺序

```
轨道A（独立，可先合）:  Stage 1 ─────────────────────── 全程与 B 轨并行
                         (软缝：若先合, Stage2 loader 新增的 in_zone 迭代需 flatten, Stage2 自查)

轨道B（严格串行）:      Stage 2 ──→ Stage 3 ──→ Stage 4
                      binding 模型    补3概念       构建期生成菜单
                      (承重墙)    (注入同一 dict)   (遍历 dict 生成)
```

- **Stage 1 ∥ Stage 2/3/4 全程并行**（正交、行区不重叠）。可立即派单 agent 独立做、独立合。
- **Stage 2 → 3 → 4 不可并行**：Stage 3 的 red 测试在无 Stage 2 时会因 `AttributeError`（字段不存在）误红；Stage 4 读不到字段则不可启动。
- **两决策门已闭合**，B 轨无待拍板卡点。

---

## 3. 不在本系列（边界声明 — 防止下一 agent 重新捡起已排除的工作）

- **C 区（待 Golden Case）**：zone N:1 聚合语义（OR/sum）、Layer3 方法论 md 正文、整合区重复计的代码防守必要性 —— 同事 PR body 明示"#98 样例数据要和 Golden Case 一起跑"。**本系列只补概念枚举结构，不碰聚合。**
- **跨群组坑**：核实确认现有 HITL-ignore + `_apply_aliases` 删列已覆盖，**无需新工程逻辑**（详见 workflow 核验）。
- **B 区默认值领域判断**（#63 的 motor=1-2 / signal=2 / velocity 等默认值改动）：归同事/用户拍板，非代码架构问题，会破坏硬编码测试，**单开 PR**。
- **A4/A5 纯文档**（调参方法论 md / catalog description 中性化）：不依赖地基，**可并行单开 PR-文档**，A4 守零数字 SSOT。
- **OFT border 静默精度降级 TODO**（`_find_periphery_zone_column` 固定英文 regex 匹物理列名 → 用户对齐的中文列不中 → 静默降级）：干净终态=给 OFT 加真实 `periphery_zone` param + 注入列作第零优先级，届时 border 的 `binding` 从 None 换 `ParamBinding` 即可。**留 follow-up，本系列不做（一次只动一层）。**

---

## 4. 风险与铁律核对

1. **Stage 1 最易漏点 = flatten 回归**：`requires_columns` 升 CNF 后，所有 `pat.startswith(...)` 式消费者必须先 flatten，否则 list 上调 str 方法炸。**grep 全仓所有迭代 requires_columns 处**是 Stage 1 第一动作。
2. **Stage 2 等价性是无损规范化的可执行证明**：前后 resolve 输出字节不变才算"无损"成立，否则规范化引入了行为漂移。
3. **TDD 强制**：每个 stage red 先行。
4. **SSOT 单存**：Stage 4 生成后，概念菜单只剩 catalog 一份；A4 文档零参数数字。
5. **不跨范式复用**（CLAUDE.md 第 14 条 / [[feedback_no_cross_paradigm_reuse_accept_duplication]]）：Stage 3 补概念按范式各补（OFT border / LDB dark / ZM closed），不因"结构一样"合并；三者 binding 形态本就不同（None vs ParamBinding）。
6. **import 闭环**：本系列改动全在 `ethoinsight` 包内（catalog/resolve/loader/concept_menu），不碰 harness `subagents/agents/tools`。Stage 4 的 harness 侧 staleness 测试 import ethoinsight 是顺方向、安全。

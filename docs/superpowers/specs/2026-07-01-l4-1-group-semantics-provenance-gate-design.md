# 设计 spec：L4-1 组语义 provenance 结构门（治剂量幻觉越权）（2026-07-01）

> D0 UX audit findings **唯一 P0**（判读可信度）。用确定性 provenance 结构门，禁止 agent 把**未经用户本轮确认的组语义**当事实写进交付报告。**治越权，非消歧**——模糊本身诚实保留（中性名），不假装消除模糊（编造剂量）。
>
> 来源：`docs/reports/d0-audit/2026-06-30/findings.md` L4-1。与 #254（Bug② 列语义越权 provenance 门）**同病不同字段**——复用其 `_stamp_column_semantics_provenance` / `resolved.sources` 模式到 group_semantics。后端结构改动，不撞 C phase 前端。

---

## 根因（取证坐实）

- **现象**：用户只说 XX=对照、XY/YY/YZ=实验（基因型）。agent 在 `resolved.groups` 自作主张写 `XX=对照/XY=低剂量/YY=中剂量/YZ=高剂量`，data-analyst 据此产出「剂量-反应关系 / 中高剂量致焦虑 / 低剂量无效应」判读，写进交付研究员的 `report.html`（证据截图 `evidence/05-gallery-with-report.png`）。
- **真根因**：`resolved.groups` 是一个**混合字符串**，把合法的东西（组标签 + 分组结构 + 用户明说的"XX=对照"）和编造的东西（XY/YY/YZ=低/中/高剂量）混在一起，且**无 provenance 门**区分"用户确认的语义"vs"agent 自推的语义"。宪法 §2 已禁编造给药剂量，但那是 prompt 提醒，没拦住（HarnessX under-exploration：反复漏调不该靠加提醒 prompt，应上结构门）。
- **同族**：与 #254（列语义越权）同病不同字段。#254 建了 column_semantics 的 provenance 门，L4-1 把同一模式扩到 group_semantics。

## 核心立场：治越权，非消歧（边界诚实性）

语义模糊分两半：
- **① 越权**：agent 把未确认语义当事实交付 → **本 spec 治的就是这个**（确定性可验）。
- **② 消歧**：XY/YY/YZ 到底是不是剂量梯度 → **本 spec 不做**。那是研究员知识，agent 无权判断；正确姿势=不知就用中性名 + 把消歧权留给用户（用户回一句"XY=低剂量"，source 变 user_current_turn，门放行）。

守 `feedback_oft_single_zone_must_ask_not_guess`：工程手段不是"让 agent 更会猜"，而是"结构上禁止 agent 把猜的当真的"。**验收锚定 ①（可确定性验证），不验 ②（agent 本就做不到、不该验）。**

## 四个承重决策（brainstorm 锁定）

1. **修复定位**：结构门主治 + 宪法辅（宪法 §2 已禁编造剂量但 bug 仍发生 → 纯 prompt 不够，上结构门）。
2. **门拦什么**：**软门**——只降级未确认语义命名，**不阻断分析**（组间比较照跑；不强制反问、不 deny code-executor）。
3. **检测机制**：**拆字段**（`group_structure` 结构 + `group_semantics` 带 source），按 source 确定性判，**不用关键词黑名单**（黑名单必漏 + 误伤 + 靠正则解析自然语言，脆弱）。
4. **宪法辅层**：只补一句指向 `resolved.group_semantics`（SSOT），**不复制禁则清单**（避免与门重叠漂移）。

## 架构：结构门主治（拆字段 + provenance 降级）+ 宪法辅

`set_experiment_paradigm` 的 groups 从"混合字符串"拆成两显式部分，复用 #254 provenance 门：

```
set_experiment_paradigm(
  group_structure = {XX:[subj...], XY:[...], ...},        # 哪些 subject 属哪组——总合法，照写照跑
  group_semantics = {                                     # 组的解释性命名——带 source
     XX: {label:"对照组", source:"user_current_turn"},    # 用户明说 → confirmed=true 保留
     XY: {label:"低剂量",  source:"agent_inferred"},       # agent 自推 → 门降级中性名
     ...
  }
)
```

门（`experiment_context.py`，仿 `_stamp_column_semantics_provenance`）确定性行为：
- `group_structure` → 原样写 `resolved`，照跑分析（组间比较不受影响）。
- `group_semantics` 每项按 `source`：
  - `user_current_turn` → `confirmed=true`，语义标签保留。
  - `agent_inferred` / `prefilled_from_memory` → **降级中性名** `实验组 N`（N 按 structure 顺序稳定；用户明说的对照组保留），`confirmed=false`，记 `resolved.sources`。
- 下游拿到的 `resolved.group_semantics` **物理上不含未确认剂量语义** → 报告写不出剂量-反应。

**软门**：门在写盘那刻降级，不阻断 code-executor、不强制反问。

## 三个单元

### 单元 1：`set_experiment_paradigm` schema 拆字段
- **做什么**：groups 参数拆成 `group_structure`（label→subject 集）+ `group_semantics`（label→{label_text, source}）。向后兼容：老 `resolved_facts=[{key:"groups", value:"..."}]` 路径保留（legacy 不带 source → 不降级、行为不变，同 #254 `source=None` 处理）。
- **接口**：`set_experiment_paradigm(..., group_structure: dict | None, group_semantics: dict | None)`。
- **依赖**：现有 `_apply_resolved_facts` / `resolved` 写盘链。

### 单元 2：`_stamp_group_semantics_provenance(gs, source)` 纯函数
- **做什么**：给每个 group_semantics 项盖 `confirmed_source`，执行 provenance 不变式——非 `user_current_turn` 的语义标签**确定性降级中性名**（`实验组 N`，按 structure 顺序稳定编号；用户明说的对照组保留），`confirmed=false`。
- **接口**：输入 `gs: dict` + `source: str | None`；输出降级后 dict。**纯函数、无副作用 → 直接单测。**
- **依赖**：无。**L4-1 承重逻辑，测试重心。**

### 单元 3：宪法一句 + prompt 指引
- 宪法 §2 补一句：**"组语义以 `resolved.group_semantics` 为唯一来源；禁止超出它另编剂量/处理命名或剂量-反应叙事。"**（SSOT 指向，不复制禁则清单）。
- 分组/identify prompt 指引 agent 分开填 group_structure（结构）vs group_semantics（带 source）；user 本轮明说的才标 user_current_turn。
- **依赖**：单元 1 schema。

## 数据流

```
lead/grouping agent 识别分组
  → set_experiment_paradigm(group_structure=结构, group_semantics={带 source})  (单元1)
  → _stamp_group_semantics_provenance 按 source 降级未确认语义 (单元2)
      · user_current_turn → 保留标签 confirmed=true
      · agent_inferred/memory → 降级"实验组N" confirmed=false + 记 resolved.sources
  → 写 resolved.group_structure（照跑）+ resolved.group_semantics（已脱敏）
  → data-analyst 组间比较（用中性名，照做）
  → report-writer 按宪法只引 resolved.group_semantics (单元3)
      → 报告物理上无未确认剂量语义 → 写不出"剂量-反应"
```

**关键边界**：门在写盘那刻降级，下游全链拿到脱敏后的 resolved——不靠下游自觉、不靠 report-writer 记得禁则。

## 错误处理 / 边界

| 场景 | 处理 |
|---|---|
| 只有 group_structure 无 group_semantics | 照跑分析，组用标签本身（XX/XY…）当中性名，不编造语义 |
| group_semantics 有 source 但值缺失 | 该项降级中性名（缺失=未确认）|
| legacy caller（老 resolved_facts 字符串路径）| source=None → 不降级、行为不变（向后兼容）|
| 用户明说部分组、其余没说 | 明说的 user_current_turn 保留，没说的 agent_inferred 降级（逐项独立）|
| 中性名与已有标签冲突 | 按 structure 顺序稳定编号，同输入同输出 |

## 测试策略（守 TDD + 防 vacuous + 裸导入两入口）

**核心 = 单元 2 纯函数**（仿 #254 `test_column_semantics_provenance.py`）：
1. `agent_inferred` 的"低剂量" → 降级"实验组N"、`confirmed=false`、记 `resolved.sources`。
2. `prefilled_from_memory` → 同样降级（memory 预填不能替用户确认）。
3. `user_current_turn` 的"对照组" → 保留标签、`confirmed=true`（不误伤）。
4. `source=None`（legacy）→ 不动（向后兼容）。
5. 中性名编号稳定（按 structure 顺序）→ 同输入同输出。
6. **防 vacuous 探针（必须实跑观察，非声称）**：实施 agent 必须**真的**临时删掉/注释掉 `_stamp_group_semantics_provenance` 里的"降级那行"（把 `agent_inferred`→中性名的赋值去掉），**跑测试 1/2 观察它们变红**（贴出红的输出证据），**再恢复那行**跑绿。仅"写了探针测试"不算——要证明"被测行为不存在时该测试会红"。（守 `feedback_deterministic_html_image_product_needs_end_to_end_reparse_assertion` 同源：断言必须能在坏产物上失败。）

**端到端越权不变式（治越权的验收）——断言结构，不查 substring：**
7. "XY=低剂量(agent_inferred) + XX=对照(user_current_turn)"走完整写盘后，遍历 `resolved.group_semantics` 每一项，**逐项断言**：`source == "user_current_turn"` 的项 label 保留原值（XX="对照组"），**其余每一项** label **必须匹配中性名模式**（`实验组\d+` 或原始组标签，绝不含 agent 自推的语义文本）。**不要**只写 `"剂量" not in str(resolved)`——那是 substring 巧合命中的假绿（agent 换个词"高浓度"就漏）；要断言"每个未确认项都被换成了中性名"这个**结构不变式**。
8. `group_structure` 照写照跑 → 组间比较不受影响（软门不阻断）：断言 `resolved.group_structure` 原样保留 subject→group 映射，且分析路径（data-analyst 派遣）未被门 deny。

**裸导入两入口**（改 harness 核心，守铁律）：
9. `PYTHONPATH=. python -c "import app.gateway"` + `from deerflow.agents import make_lead_agent` 两者 0 退出。

## sync 友好性

- `experiment_context.py`（middleware）在 deerflow 子树内，但是 **Noldus 独有定制文件**（上游无 experiment_context）→ 改它是前向 feature 开发，非违 sync（#254 就改了它）。
- `output-constitution.md` 在 `skills/custom/` = Noldus 定制。
- 分组 prompt 若在 `lead_agent/prompt.py`（受保护）→ **surgical**：纯加指引、不删既有定制。

## 不做什么（YAGNI + 治越权非消歧，防实施 agent 跑偏）

- ❌ **不做消歧**：不试图让 agent 识别 XY 到底是不是剂量——那是用户知识，用中性名 + 留给用户。
- ❌ **不阻断分析**：软门，组间比较照跑（不强制反问、不 deny code-executor）。
- ❌ **不做关键词黑名单**：按 source 确定性判，不正则扫"剂量"词。
- ❌ **宪法不复制禁则清单**：只补一句指向 resolved.group_semantics。
- ❌ 不碰 column_semantics 门（#254 已做，正交）。

## 验收标准（诚实锚定"越权"可验，不验"消歧"）

1. 未确认组语义（agent_inferred/memory）确定性降级中性名、`confirmed=false`、记 source。
2. 用户本轮确认的语义保留、不误伤。
3. `resolved.group_semantics` 端到端**结构不变式**：每个未确认项 label 都是中性名（`实验组N`/原标签），只有 `source=user_current_turn` 的项保留语义值 → report 无从写出未确认的剂量-反应（断言结构，非查 substring）。
4. 分组结构照写、组间比较不受影响（软门不阻断）。
5. 纯函数全测 + 防 vacuous 探针 + 裸导入两入口绿。
6. 宪法一句指向 resolved.group_semantics；prompt 指引分开填 structure/semantics。

## 关联

- findings：`docs/reports/d0-audit/2026-06-30/findings.md` L4-1（唯一 P0）。
- 同族先例：#254（`a428d2fd` Bug② 列语义 provenance 门）——复用 `_stamp_column_semantics_provenance` / `resolved.sources` 模式 + `test_column_semantics_provenance.py` 测试模式。
- 现有代码：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py`（`_stamp_column_semantics_provenance` / `_apply_resolved_facts` / `resolved.sources`）、`packages/agent/skills/custom/ethoinsight/references/output-constitution.md` §2。
- 守 memory：`feedback_memory_prefill_must_not_be_counted_as_current_turn_confirmation`（memory 预填 confirmed=false）、`feedback_oft_single_zone_must_ask_not_guess`（不知就问别猜=不知用中性名别编）、`feedback_seal_missing_root_cause_is_react_no_toolcall_exit_gate_not_fallback`（结构门 > prompt 打地鼠）、`feedback_single_source_of_truth`（宪法指向 resolved 不复制禁则）、`feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`（确定性门=消除，验收问"现象还会不会发生"）。
- CLAUDE.md 第 9 条（判读只看组间差异，不脑补实验设计语义）。

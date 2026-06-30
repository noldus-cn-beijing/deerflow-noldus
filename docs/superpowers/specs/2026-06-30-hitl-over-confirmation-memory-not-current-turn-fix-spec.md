# 设计 spec：修复 HITL 越权——agent 拿 memory 历史偏好替用户确认本轮未回答的分组/列语义（结构门）（2026-06-30）

> 来源：2026-06-30 EPM 28 文件端到端 dogfood 实测发现的真 bug。诊断走 `diagnose` skill，
> **磁盘证据坐实**（失败 thread `8827351d-e2b1-4292-b69e-a3bde14b5fb0`）。本 spec 只设计修复，交别的 agent 实施。

---

## 一、现象（磁盘坐实，非推测）

反问含 3 项（① EV19 模板 A/B ② 分组映射 XX/XY/YY/YZ ③ open/closed 列语义）。
**用户只回了第 ① 项**（选 A. PlusMaze-AllZones）。agent 却把 ②③ 用 memory 历史偏好**直接确认**并落盘：

`experiment-context.json`（失败 thread 磁盘真相）：
```json
"column_semantics": { "columns": {
    "open":  {"resolves_to":"open_arms","meaning_zh":"开臂","confirmed": true},
    "closed":{"resolves_to":"closed_arms","meaning_zh":"闭臂","confirmed": true}
  }, "confirmed_at": "2026-06-30T08:42:27.556347+00:00" },
"resolved": {
    "groups": "XX=对照组, XY=低剂量, YY=中剂量, YZ=高剂量",
    "column_semantics": "open=开臂, closed=闭臂" }
```
`column_semantics.confirmed_at` 与 `paradigm_confirmed_at`（08:42:27）**同一瞬间**——即 agent 在用户只回模板那一刻，把分组+列语义一起盖了 `confirmed: true`。

虽然预填值碰巧对，但**agent 不能替用户确认用户本轮没回答的项**。违反用户铁律
`feedback_oft_single_zone_must_ask_not_guess`（不知道/没确认的要问，不能猜）。

## 二、根因（已定位）

`agents/lead_agent/prompt.py:587` 已有规则「**用户未明确选模板 → 重问**」，但它**只护模板字段一个方向**
（用户答了分组/其他但没答模板 → 重问模板）。**反方向无规则**：用户答了模板、没答分组/列语义时，
没有「不许用 memory 替确认未回答项」的约束 → agent 拿注入的 memory 历史偏好当本轮确认。

这是**结构层缺失**（不是 prompt 指引硬度不够）——按 CLAUDE.md HarnessX 三病理自检：
**结构缺失 → 上门**（不是再加一条 prompt reminder 打地鼠，守 under-exploration 警告）。

## 三、修复设计：确认来源结构门

**核心不变式**：写进 `experiment-context.json` 的 `column_semantics[*].confirmed=true` /
`resolved.groups` / `resolved.column_semantics`，其值**必须来自本轮用户输入**；memory 注入的
历史偏好**只能作为预填建议（confirmed=false / pending），不能直接标 confirmed**。

**落点**：`agents/middlewares/experiment_context.py`——它是写 column_semantics / resolved 的唯一入口
（`_normalize_column_semantics:196`、`_apply_resolved_facts:310`、`_derive_column_aliases:203`）。

**门的形态**（确定性校验，不是 prompt）：
1. **`confirmed=true` 需 provenance**：`set_experiment_paradigm` 落 column_semantics 时，每个
   `confirmed=true` 的列必须带 `confirmed_source`（枚举：`user_current_turn` | `prefilled_from_memory`）。
   只有 `user_current_turn` 才允许 `confirmed=true` 落盘；`prefilled_from_memory` 强制降级为
   `confirmed=false`（预填待确认）。
2. **下游门联动**：`prep_metric_plan` / code-executor 派发前的 guard（path_sequence 或新增）检查——
   若 column_semantics 有列 `confirmed=false`（即仍是 memory 预填未经用户确认），**deny 派发并
   ask_clarification 重问该列**，不放行分析。
3. **resolved.groups 同理**：`_apply_resolved_facts` 写入的 groups/column_semantics 标记来源；
   memory 来源的不算「本轮已确认」，下游门据此决定是否重问。

> 关键：memory 历史偏好的正确用法是**预填进反问选项让用户一键确认**（降低点击成本），
> 不是**替用户跳过确认**。门保证「预填 ≠ 已确认」。

## 四、验收（TDD + 防 vacuous）

1. **结构门单测**：构造「用户只回模板、memory 有分组/列语义偏好」的输入 → 断言落盘的
   column_semantics 列 `confirmed=false`（不是 true）、`confirmed_source=prefilled_from_memory`。
2. **下游门联动测**：column_semantics 有 `confirmed=false` 列时，派 code-executor 的 guard **deny**
   并触发 ask_clarification（断言 deny + 重问）。
3. **正向路径不回归**：用户本轮明确回答了分组/列语义 → `confirmed=true` + `confirmed_source=user_current_turn`
   正常落盘，分析放行。
4. **防 vacuous**：去掉「prefilled_from_memory 强制降级」那行 → 测 1 应变红（断言精准打在降级逻辑上）。
5. **裸导入两生产入口**（改了 experiment_context / guard / 派发链）：`import app.gateway` +
   `make_lead_agent` 0 退出（守 CLAUDE.md 导入环铁律）。
6. `make test` + `make lint` 绿。

## 五、不做什么

- ❌ 不靠加 prompt reminder 修（守 HarnessX：累加 reminder 亚阈值耦合致崩）——上结构门。
- ❌ 不禁用 memory 预填（预填进反问选项是好体验，只是不能当确认）。
- ❌ 不改 memory 注入链本身（memory 提供历史偏好没错，错在被当本轮确认）。

## 六、关联

- 守 memory：`feedback_oft_single_zone_must_ask_not_guess`、`feedback_suspect_columns_must_fold_into_first_clarification`、
  `reference_harnessx_report_and_etho_spec_application`（结构门 > prompt）。
- 现有：`prompt.py:587`（只护模板的半边规则）、`experiment_context.py`（confirmed/resolved 写入点）、
  `path_sequence_provider.py`（下游派发门，可挂联动）。
- 同范式：SealGateMiddleware / GateEnforcementMiddleware（确定性门拦行为，不靠 prompt）。
- 证据 thread：`8827351d-e2b1-4292-b69e-a3bde14b5fb0`（`.deer-flow/users/e6cdba0f.../threads/`）。

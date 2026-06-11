# Handoff — EV19 列语义对齐 Sprint 1 落地 + 文档/配置收尾

> 日期：2026-06-06 ｜ 分支：`dev`（与 origin/dev 完全同步，HEAD `4f3c26e2`）
> 这是给下一位 AI Agent 的交接，不是给用户的总结。

---

## 1. 当前任务目标

本会话从「用户上传真实 OFT 数据被 agent 拒绝」出发，完成了 **EV19 列语义对齐（Column Semantics Alignment）Sprint 1** 的全周期：设计讨论 → 实施 review → 修复 → 合并 dev → 文档/配置收尾。**Sprint 1 已 done 并合入 dev**。下一步是 Sprint 2（结构聚合，阻塞在外部专家）或处理可能同源的用户 bug。

---

## 2. 当前进展（本会话已完成）

- ✅ **设计 v2 定稿**：`docs/design/2026-06-05-column-semantics-hitl-design-v2.md`（16 条决策 D1–D16，实测 34 文件钉死根因）
- ✅ **Sprint 1 实施 spec**：`docs/superpowers/specs/2026-06-05-column-semantics-alignment-sprint1-spec.md`
- ✅ **Sprint 1 代码实现 + review 修复**：3 critical + 2 medium 全修（见 §4）
- ✅ **PR #99 已 MERGED 进 dev**（merge commit `6e6a3c27`，02:52 UTC）
- ✅ **本地 dev 偏离已 merge 调和**（曾领先 1 落后 4），3 份设计文档已推 origin
- ✅ **项目级文档更新**（commit `f71d7b8e`）：新建 milestone + README 索引 + CLAUDE.md
- ✅ **extensions_config 脆弱性修复**（commit `4f3c26e2`）：显式声明全部 9 个 ethoinsight skill
- ✅ **Sprint 2 勘察 issue 已开**：[#98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)（等行为学专家）

---

## 3. 关键上下文（架构 + 文件清单）

### 3.1 问题本质（必读，否则会误判）

真实 OFT 数据（`/home/wangqiuyang/DemoData/real_data/Raw data-OFT-Xuhui-34`，34 个 XLSX）被拒**不是模板识别问题**。根因：用户在 EthoVision「按分析区输出」自定义的分析区列，列名 100% 用户命名（`中心区`/`边缘区`/`边缘区到中心区`），归一化后既不在 COLUMN_MAP、也不匹配 catalog 的 `in_zone_center_*` glob → default metric 报 `columns_missing` → 整个 plan 中止。

### 3.2 Sprint 1 数据流（已实现）

```
inspect_uploaded_file(paradigm=…) → column_assessment + open_questions + 每列证据(占时)
  → lead 用 catalog 合法概念菜单【预填推测】反问("中心区→center 区,对吗?")
  → 用户确认/纠正 → set_experiment_paradigm(column_semantics={…}) 写 SSOT
       + 写盘时单向投影 column_aliases
  → prep_metric_plan 读 column_aliases 传 resolve
  → resolve._apply_aliases 在 columns 入口单点重映射(概念关键词→catalog 概念列)
  → metric 算出来。标准命名数据 open_questions=[] → 零额外开销不触发
```

### 3.3 改动文件（全在 dev 上）

| 文件 | 改动 |
|------|------|
| `packages/ethoinsight/ethoinsight/utils.py` | `assess_column_confidence()` + `_FIXED_COLUMNS` |
| `packages/ethoinsight/ethoinsight/catalog/resolve.py` | `_apply_aliases` / `_zone_concept_map` / `_materialize_concept` + `column_aliases` 参数（resolve_metrics + resolve_charts）+ `_UNSET` 哨兵 |
| `packages/ethoinsight/ethoinsight/catalog/cli.py` | `--column-aliases-file` 转发到两 mode |
| `.../tools/builtins/inspect_uploaded_file_tool.py` | `paradigm` 参数 + `_attach_column_assessment` 分层叠加到 dev 成熟版 |
| `.../tools/builtins/prep_metric_plan_tool.py` | 读 column_aliases 传 resolve；`columns_missing` hint 改为先走列对齐 |
| `.../agents/middlewares/experiment_context.py` | `column_semantics` 入参 + `_derive_column_aliases`（用 raw_name 现算 key，不信 LLM 填的 normalized） |
| `.../guardrails/ev19_template_provider.py` | column_semantics 未确认子检查 |
| `.../agents/lead_agent/prompt.py` | 自定义列对齐反问场景 + skill 指针 |
| `packages/agent/skills/custom/ethoinsight-column-confirmation/` | **新 skill**（SKILL.md 67 行 + 2 references，交互方法论，lead-only） |
| `packages/agent/extensions_config.json` | 9 个 ethoinsight skill 全显式声明 |
| `tests/test_column_semantics.py`（etho + backend 各一份） | 18 + 15 测试 |

---

## 4. 关键发现（重要结论 + 易踩坑）

1. **`normalize_column_name("中心区") == "center"`（不是 `中心区`！）**——会话早期我在旧 utils 上测得 `中心区`，但**合并 dev 后的归一化（含 zone-prefix detection）把 `中心区` 归一成 `center`**。这是 review 修复时差点酿成静默丢指标的真坑。`_apply_aliases` 因此改成**两边都过 normalize 鲁棒匹配**（raw key `中心区` ↔ 归一化列 `center` 都命中）。**下一位若改归一化逻辑，必须重测这条链路。**

2. **review 暴露的 3 critical（已修，是 Sprint 1 实施的真实缺陷）**：
   - **C-1 集成断链**：原实现写了 column_aliases 但 prep_metric_plan 从没读 → feature 形同未做。已接通（含 chart-maker 的 CLI 路径，否则 zone 图静默丢）。
   - **C-2 静默丢指标**：skill/test 建立在错误前提 `中心区→center` 字面映射上。治本=概念翻译下沉代码（LLM 只填 `center` 概念关键词，代码翻译成匹配 glob 的列名），mapping.md 瘦身到纯语义层。
   - **C-3 假集成测试**：原"集成测试"用虚构 center 列绕过断链。新增真·端到端 `TestPrepMetricPlanReadsColumnAliases`（真实 TXT 走完整 prep_metric_plan）。

3. **与 PR #89 anonymous_zone 并存**：dev 已有 `anonymous_zone_is`（裸 in_zone 单区，#89）。Sprint 1 是它的超集泛化。resolve.py 里**两机制并存**：`_apply_aliases`（链路A，命名自定义列）先跑、`_detect_anonymous_zone`（链路B，裸 in_zone）兜底后跑。

4. **「预填反诘 ≠ 断言」铁律（D13/D15）**：系统**可**预填最佳推测让用户一键确认/纠正，但绝不直接采纳；推测来自**占时证据 + catalog 菜单**，**不来自列名字面**（同事铁律"虽然这次猜中了但不要猜"）。用户的设计意图：遇匿名区"推测一个值问用户，用户接受或自己输入"——已对齐。

5. **skill 默认 enabled 机制**：`config/extensions_config.py:204-206`——未列入 extensions_config 的 custom skill 默认 enabled。这是 §2 那个脆弱性修复的依据（5 个 skill 本就工作，补全只是显式化）。

---

## 5. 未完成事项（按优先级）

| 优先级 | 事项 | 说明 |
|--------|------|------|
| **P1** | 查 **issue #94**（"上传原始数据却要求上传统计学数据"）是否与 Sprint 1 同源 | 听起来正是"agent 不认数据就拒绝"的故障类型。Sprint 1 已合 dev，**可能直接关掉它**——但要先核实是不是同源，别猜 |
| **P2** | **Sprint 2 结构对齐**（issue #98） | EPM 4 区 open_arm1/2 → 标准 2 区聚合等。**开放式、blocked 在行为学专家回 #98**。schema 已前向兼容（`resolves_to` 可扩展多列）。已发现 `metrics/epm.py:96` 有现成 OR 聚合，但只是 1 范式 1 case，**不可当全貌** |
| **P3** | **issue #97** 上传失败（19 txt / 103MB） | 独立的上传基础设施问题，与列语义无关（更像超限/超时） |
| **P3** | 清理 sprint1 worktree | `.claude/worktrees/column-semantics-sprint1`（PR 已 merge，可删） |

---

## 6. 建议接手路径

1. **先读**：`docs/milestone/column-semantics-alignment.md`（本 track 全貌）+ 本 handoff
2. **验证现状**：
   ```bash
   cd packages/ethoinsight && pytest tests/test_column_semantics.py -q   # 应 18 passed
   cd packages/agent/backend && PYTHONPATH=. python -m pytest tests/test_column_semantics.py -q  # 应 15 passed
   ```
3. **若做 P1（查 #94 同源）**：`gh issue view 94` 拿 thread ID → 去 `packages/agent/backend/.deer-flow/users/*/threads/<tid>/` 找 checkpoints，看是不是 `columns_missing` / 自定义列拒绝。**核实后再判断**，不要假设。
4. **若做 P2（Sprint 2）**：等 #98 有专家回复再启动；先读 design v2 §6.2 + Sprint 1 spec 的 Sprint 2 前向兼容段。

---

## 7. 风险与注意事项

- ⚠️ **不要假设 `中心区` 归一化成什么**——现状是 `center`，但这依赖 utils 的 zone-prefix 逻辑。改归一化前重测 `_apply_aliases` 链路（见 §4.1）。
- ⚠️ **不要把 mapping 知识写回 markdown**——概念→机器列名翻译已下沉到 `resolve.py` 代码（SSOT）。skill 只承载纯语义层。往 answer-mapping.md 加 JSON schema/后缀细节 = 倒退（反脑补 + SSOT 双违反）。
- ⚠️ **不要在确认聚合语义前自行实现 Sprint 2 聚合**——必须逐范式问行为学专家（#98 已承诺）。EPM 的 OR 聚合是巧合兜住的 1 个 case，别当通用方案。
- ⚠️ **改 resolve.py 必跑全量**（6 范式 + 测试套共享承重墙）：`cd packages/ethoinsight && pytest`（应 624 passed / 70 skipped）。
- ⚠️ **sprint1 worktree 还在**（`03c11c7a`），里面的改动都已进 dev，可安全 `git worktree remove`。

---

## 8. 下一位 Agent 的第一步建议

**先做 P1**：`gh issue view 94 --json body` 拿 thread ID → 解码 checkpoints 核实失败类型。若确认是"自定义分析区列被拒"（与 Sprint 1 同源），则：
1. 用真实数据 dogfood 验证 Sprint 1 是否真的修好了这个 case（端到端跑 agent，不只是单测）
2. 修好则在 #94 评论说明 + 关闭
3. 没完全修好则记录残留缺口

若 #94 非同源，转 P3 查 #97 上传失败，或等 #98 启动 Sprint 2。

## milestone 建议

本会话已让「EV19 列语义对齐」track 到达 checkpoint（Sprint 1 done），milestone 已创建：`docs/milestone/column-semantics-alignment.md`，README 索引已更新。无需额外动作。

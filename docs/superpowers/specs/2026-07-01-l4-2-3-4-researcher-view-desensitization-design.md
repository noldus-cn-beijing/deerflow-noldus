# 设计 spec：L4-2/3/4 研究员视角信息脱敏（2026-07-01）

> D0 UX audit findings 的 L4-2/L4-3/L4-4 合一：研究员视角不暴露内脏。三处各自独立、无共享逻辑，合一个 spec 因同属「信息脱敏」主题、可一次派。**做掉则 D0 findings 全部转成 spec。**
>
> 来源：`docs/reports/d0-audit/2026-06-30/findings.md` L4-2/L4-3/L4-4。跨 1 后端 prompt 删除 + 2 前端小修。

---

## 核心理念

研究员视角不暴露内脏（findings L4：不给研究员看 config id / 内部诊断 code / token 黑话）。

## 三个单元（各自独立、可独立测）

### 单元 1 — L4-2：删 config id 展示指令（后端 prompt）

- **做什么**：删 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:800` 那行：
  `- Analysis Config ID: When presenting analysis results, mention the analysis_config_id (read from experiment-context.json) so users can reference this specific analysis. Example: "本次分析标识: a1b2c3d4e5f67890"`
  lead 收尾不再写「本次分析标识: xxx」。
- **依据**：`analysis_config_id` **零消费**（前端 grep 空 + 后端 `app/` grep 空，本会话核实）——它被算出、写进 experiment-context.json，prompt 让 lead 展示「以便引用」，但没有任何路径能让用户按此 id 引用某次分析。删展示指令**零破坏**。后端 `compute_analysis_config_id` 仍算仍写盘（内部追踪/去重不动），只不再面向用户展示。
- **改动**：`prompt.py`（受保护文件）**surgical 纯删一行指令**，不碰其它定制。
- **注**：findings L4-2 另提的 `[gate_signals]`/内部字段（`errors_count`/`statistical_validity`/`sections_missing`/`written_count` 等）**已被 #195 的 `stripInternalMarkers`（`frontend/src/core/messages/utils.ts:641`）在 render 层 strip**——本单元只补 config id 这个 **prose 泄漏**（自然语言，strip 的 marker 匹配抓不到）。

### 单元 2 — L4-3：quality code 映射人读标签（前端）

- **做什么**：`packages/agent/frontend/src/components/workspace/messages/quality-warning-banner.tsx:103` 的 `[{ws.label} {w.code}]` → 把 `w.code`（`METRIC_VALIDATION` 等内部诊断码）经一个 `code→人读标签` 映射（如 `METRIC_VALIDATION`→「指标校验」）再渲染。
- **现状**：code 已只在**展开详情**（`CollapsibleContent`）里、标题已是自然语言（「N 条数据质量警告」）——但展开后每行仍露 raw `METRIC_VALIDATION`。本单元治「展开详情里仍露 raw code」。
- **关键（守 SSOT + 防新 code 泄漏）**：映射表处理**未知 code fallback**——map 没命中的 code **退化成通用「数据质量」标签，绝不露 raw code**。这样后端新增 code 而映射表滞后时，也不会漏黑话。
- **改动**：`quality-warning-banner.tsx` + 一个小 `code→标签` 映射表（放组件同目录或 i18n）。

### 单元 3 — L4-4：移除对话区 token 指示器（前端）

- **做什么**：删两个调用点的 `<TokenUsageIndicator>` + import：
  - `packages/agent/frontend/src/app/workspace/chats/[thread_id]/page.tsx:167`
  - `packages/agent/frontend/src/app/workspace/agents/[agent_name]/chats/[thread_id]/page.tsx:176`
- **依据**：token 数在对话头（挨着 Export/文件按钮），每个研究员每个 thread 都看得到，对行为学研究员是无意义技术黑话。前端**无现成 admin/debug mode-gate**——不为单个黑话指示器造一整套调试模式（YAGNI）。开发者看用量走后端 `make training-stats`。
- **孤儿清理**：删调用点后核 `token-usage-indicator.tsx` + `usage.ts`（`formatTokenCount`/`accumulateUsage`）是否变孤儿；**仅本改动造成的 orphan 才删**（守 surgical，不删预存 dead code）。若 `usage.ts` 的函数还有别处引用则保留。

## 测试策略（守 TDD + 防 vacuous）

- **单元 1（后端）**：prompt 是文本删除，无逻辑测。验证：① `grep "analysis_config_id" prompt.py` 证不再有展示指令行；② 裸导入两入口 0 退出（改受保护 prompt 文件的最低验证）：
  `PYTHONPATH=. python -c "import app.gateway"` + `from deerflow.agents import make_lead_agent`。
- **单元 2（前端 vitest）**：
  1. 已知 code（`METRIC_VALIDATION`）→ 断言渲染人读标签、**输出不含 raw "METRIC_VALIDATION"**。
  2. **未知 code**（造 `FOO_BAR`）→ 断言退化成通用「数据质量」标签、**不含 raw "FOO_BAR"**（防新 code 泄漏）。
  3. **防 vacuous 探针**：临时删掉映射（让所有 code 走原 raw 渲染）→ 测试 1 **必须变红**（观察红再恢复，证真在测映射，非恒真）。
- **单元 3（前端 vitest）**：对话头渲染断言**不含** TokenUsageIndicator / token 数文本；孤儿删除后 `pnpm check` 0（无未用 import/组件残留）。

## sync / 影响面 / 边界

- 单元 1：`prompt.py` 在 deerflow 子树内、受保护 → **surgical 纯删指令行**（前向脱敏，非违 sync；不删既有定制）。
- 单元 2/3：`frontend/` 全 deerflow 子树外，sync 无关；不碰 ai-elements/registry 结构、`ui/` 只保 API。

## 不做什么（YAGNI）

- ❌ 不碰已被 #195 处理的 `[gate_signals]`/`[intent]` render strip（那部分 L4-2 已治）。
- ❌ 不建 token 的 debug/admin mode-gate（方案 B 已排除，前端无此基建，不为单指示器造）。
- ❌ 不删预存 dead code（只清本改动造成的孤儿）。
- ❌ 不动后端 `compute_analysis_config_id` 计算/写盘（只删面向用户的展示指令）。

## 验收标准

1. lead 收尾消息不再含「本次分析标识」；`grep analysis_config_id prompt.py` 无展示指令；裸导入两入口绿。
2. quality banner 展开详情里 code 显人读标签；已知 + 未知 code 都不露 raw 黑话；防 vacuous 探针实跑观察红。
3. 对话头无 token 指示器；孤儿清理后 `pnpm check` 0。

## 关联

- findings：`docs/reports/d0-audit/2026-06-30/findings.md` L4-2（config id/内部字段）、L4-3（quality code）、L4-4（token 指示器）。做掉后 D0 findings 全部转 spec。
- 现有代码：`lead_agent/prompt.py:800`（config id 展示指令）、`compute_analysis_config_id`（experiment_context.py:73，保留）、`quality-warning-banner.tsx:103`、`token-usage-indicator.tsx` + `usage.ts`、两个 page.tsx 调用点、`utils.ts:641 stripInternalMarkers`（#195 已治 gate_signals）。
- 守 memory：`feedback_single_source_of_truth`（code→标签映射未知 fallback，不双存）、`feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`（ui/ 只保 API）、`feedback_deepseek 正面提示`（prompt 删指令用正面表述不留「不要」痕迹）、`feedback_perf_is_efficient_impl_not_visual_downgrade`（无关，纯脱敏）。CLAUDE.md 第 6 条（deepseek 正面提示——删指令即可，不加「禁止提 config id」反向激活）。

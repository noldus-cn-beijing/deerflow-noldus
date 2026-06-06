# Handoff — 2026-06-05 全链路 Dogfood 修复 + 列语义 HITL 设计

> 给下一个 AI Agent 的上下文总结

## 当前任务目标

本次会话完成了三类工作：
1. **E2E 图片 404 问题诊断与修复**
2. **CSRF token missing 修复**
3. **新对话空白页修复**
4. **EV19 列语义 HITL 确认 — 架构设计讨论**（未实施，仅有设计文档）

## 当前进展

### ✅ 已完成：图片占位符系统（PR #95，已合）
- Spec: `docs/superpowers/specs/2026-06-05-chart-image-placeholder-resolution-spec.md`
- 三层防御：`{{img:<basename>}}` 占位符 → seal_report_writer_handoff 解析 → _normalize_report_image_paths 兜底

### ✅ 已完成：E2E dogfood 修复（commit 8213ae27，dev 已 push）
- 7 类修复，11 files，commit: `fix: E2E dogfood 修复 + CSRF 补发 + 新对话空白页`
- 详见下方"关键改动"

### ✅ 已完成：CSRF token missing 根因诊断与修复
- 文件: `app/gateway/csrf_middleware.py`
- 根因: CSRF cookie 是 session cookie（无 Max-Age），浏览器重启后丢失，但 access_token 仍在
- 修复: GET 请求时若 CSRF cookie 缺失，自动补发新 cookie

### ✅ 已完成：新对话空白页修复
- 文件: `frontend/.../page.tsx`
- 根因: Next.js 复用 ChatPage 组件实例，`welcomeDismissed` 状态跨路由残留
- 修复: `useEffect` 监听 `isNewThread` 变化 → 重置 `welcomeDismissed`

### ✅ 已完成：EV19 英文版 detect_ethovision 修复
- 文件: `ethoinsight/parse/_core.py`
- 根因: `_detect_ethovision_xlsx` 只认中文 marker `"标题行数"`，英文版 `"Number of header lines:"` 不识别
- 修复: 中英文双 marker 检测

### 🔄 设计讨论中：EV19 列语义 HITL 确认系统

完整设计文档: `docs/design/2026-06-05-column-semantics-hitl-design-discussion.md`

核心设计：
- Gate 1.5 插入 `inspect_uploaded_file` 之后、`prep_metric_plan` 之前
- 三层职责：工具层（column_assessment 产出）、Prompt 层（触发时机 + 反问合并规则）、Skill 层（交互方法论 thin SKILL.md）
- 新增 `ethoinsight-column-confirmation` skill
- 复用 `experiment-context.json`（加 `column_semantics` 字段）、`set_experiment_paradigm`（加 `confirm_column_semantics` 模式）、`Ev19TemplateGuardrailProvider`（加子检查）
- L1/L2/L3 列分类模型：fixed / recognized / zone / unknown
- 关键原则：标准数据不阻塞（所有列 recognized → 跳过确认），不建新 tool/新文件/新 guardrail 类型

## 关键上下文

### 项目 infra
- 仓库: `/home/wangqiuyang/noldus-insight`
- 当前分支: `dev`（commit `8213ae27`，已 push）
- DeerFlow fork: `packages/agent/` 是上游 subtree
- ECS 部署: `39.105.231.16`，通过 1Panel + OpenResty 反代，`make deploy-tar` 部署

### 关键架构知识
- `experiment-context.json` 是实验配置的全局状态文件（`/mnt/user-data/workspace/`）
- `set_experiment_paradigm` 有多个模式：Gate 1（paradigm 确认）、Gate 2（quality acknowledge）、未来 Gate 1.5（column_semantics 确认）
- `Ev19TemplateGuardrailProvider` 拦截未确认模板的 `task(code-executor)`
- `ask_clarification` + `ClarificationMiddleware` 实现 Command(goto=END) 反问
- skill 三层注入：system prompt（静态）、read_file（渐进披露）、references（深度细节）
- 受保护文件: `prompt.py`, `experiment_context.py`, `subagents/builtins/__init__.py` 等 22 个

### 硬性原则（memory）
- SSOT: 列分类规则只在 ethoinsight 库，skill 不存结构化知识
- 工具教学只讲"何时调"不讲"返回什么"（反脑补）
- deepseek 正面指令（不写"禁止 X"）
- subagent 消费包内文件必走 first-party tool resolve
- seal 必达（不因非关键错误阻断封存）

## 关键发现

1. **图片 404 根因**：LLM 在 report.md 中发明描述性文件名（`epm_trajectory_control.png`）而非使用实际文件名（`plot_trajectory_s0.png`）。ECS 日志证实同一台机器上有 thread 用正确名 → 200 OK，用发明名 → 404。修复：`{{img:basename}}` 占位符 + seal 时解析。

2. **CSRF 404 根因**：CSRF cookie 无 Max-Age（session cookie），浏览器重启后丢失。`access_token` 仍在 → Auth 通过 → 但 POST 请求无 `X-CSRF-Token` → 403。修复：GET 请求自动补发 CSRF cookie。

3. **新对话空白页根因**：Next.js 复用 ChatPage 组件 → `welcomeDismissed=true` 残留 → 新对话不显示 Welcome → 空白。修复：`isNewThread` 时重置。

4. **EV19 英文版不识别**：`detect_ethovision` 只认中文 header marker。修复：中英文双 marker。

5. **设计方向**：列语义确认应该是主动的协作仪式（"我看到这些列，我的理解对吗？"），不是被动的纠错（"resolve 报错了问你"）。要最大化复用 deerflow infra。

## 未完成事项

### [P0] EV19 列语义 HITL 确认系统 — 待实施
- 设计文档: `docs/design/2026-06-05-column-semantics-hitl-design-discussion.md`
- 用户意图: 主动确认仪式 + thin skill 方法论 + Gate 1.5
- 实施顺序建议: `ethoinsight/utils.py`（L1 固定列常量 + assess_column_confidence）→ `inspect_uploaded_file` tool 增强 → skill 创建 → prompt 修改 → guardrail 增强
- **用户原话**: "应该让 lead agent 有一个 skill 的方法论，需要和用户最终确认好（HITL）各个表头数据的含义"

### [P2] 测试覆盖缺口（来自上次 handoff）
- 并行 bash + template guardrail 缺少 red 锚点测试

### [P3] `_extract_template_recommendations` 正则脆弱（来自上次 handoff）

## 改动文件清单（本次会话）

| 文件 | 改动类型 | 描述 |
|------|----------|------|
| `packages/ethoinsight/ethoinsight/parse/_core.py` | MODIFY | detect_ethovision 中英文双 marker |
| `packages/agent/backend/app/gateway/csrf_middleware.py` | MODIFY | GET 请求 CSRF 补发 |
| `packages/agent/frontend/src/app/workspace/chats/[thread_id]/page.tsx` | MODIFY | isNewThread 重置 welcomeDismissed |
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py` | MODIFY | 图片占位符解析（PR #95） |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` | MODIFY | prompt 三处更新（PR #95） |
| `packages/ethoinsight/ethoinsight/utils.py` | MODIFY | 列名归一化 + 中央区映射 |
| `packages/ethoinsight/tests/test_parse.py` | MODIFY | golden parse fixtures |
| `packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py` | MODIFY | 并行 bash 白名单 |
| `packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py` | MODIFY | 模板确认拦截 |
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py` | MODIFY | 写 candidates 到 workspace |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` | MODIFY | user_confirmed_template 参数 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | MODIFY | 去重 inspect + 模板确认规则 |
| `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` | MODIFY | 思考面板流式展开 |
| `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` | MODIFY | subagent 卡片默认展开 |
| `docs/design/2026-06-05-column-semantics-hitl-design-discussion.md` | NEW | 列语义 HITL 设计讨论文档 |

## 重要文档

| 文档 | 路径 |
|------|------|
| 列语义 HITL 设计讨论 | `docs/design/2026-06-05-column-semantics-hitl-design-discussion.md` |
| 图片占位符 spec | `docs/superpowers/specs/2026-06-05-chart-image-placeholder-resolution-spec.md` |
| Thinking streaming spec | `docs/superpowers/specs/2026-06-05-thinking-streaming-and-display-spec.md` |
| CLAUDE.md（项目） | `noldus-insight/CLAUDE.md` |
| CLAUDE.md（backend） | `packages/agent/backend/CLAUDE.md` |
| CLAUDE.md（frontend） | `packages/agent/frontend/CLAUDE.md` |

## 建议接手路径

1. **读设计文档**: `docs/design/2026-06-05-column-semantics-hitl-design-discussion.md` — 了解列语义确认的完整架构
2. **读用户原话**（memory 中有很多相关原则，MEMORY.md）— 了解 SSOT、反脑补、seal 必达等硬性约束
3. **熟悉 deerflow infra**: `CLAUDE.md` 中关于 `experiment-context.json`、`set_experiment_paradigm`、`Ev19TemplateGuardrailProvider`、skill 渐进披露的段落
4. **确认实施范围**: 用户可能已拿到另一个 agent 的分析结论，先读那个结论
5. **按顺序实施**: `ethoinsight/utils.py`（列分类函数）→ `inspect_uploaded_file` tool 增强 → skill 创建 → prompt → guardrail

## 风险与注意事项

- **受保护文件**: `prompt.py`、`experiment_context.py`、`ev19_template_provider.py` 是受保护文件，sync 上游时需 surgical merge
- **全量测试污染**: 有 4 个 pre-existing 测试失败（与改动无关），不要归因到自己
- **skill "三件一起做"**: 创建文件 + extensions_config 注册 + prompt 引用，漏任何一步 subagent 不可用
- **SSOT 铁律**: 列分类规则只在 ethoinsight 库，skill 不存结构化分类逻辑
- **Gateway hot-reload OFF**: 改后端后必须 `make stop && make dev`
- **ethoinsight editable install**: 改完立即生效无需重装

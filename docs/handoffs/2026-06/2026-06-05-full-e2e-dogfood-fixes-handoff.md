# Handoff — 2026-06-05 全链路 Dogfood 修复

> 给下一个 AI Agent 的上下文总结

## 当前任务目标

修复 EthoInsight 端到端分析流水线中的多个问题，使真实 EthoVision 数据（中文导出、多 zone 变体）能正确跑通完整 E2E。

## 已完成的改动（所有改动在工作区，未 commit）

### 1. 前端 Thinking 显示修复（1A + 2A）

**问题**：DeepSeek 模型产生 reasoning token 时，前端只显示 loading dots，思考面板默认折叠。用户长时间看不到 agent 在做什么。

**根因**：`message-group.tsx` 的 `showLastThinking` 默认 false（折叠），而最终答案的 `ReasoningPanel` 默认展开——两套组件默认态不对称。

**修复**：
- `frontend/.../message-group.tsx`：`showLastThinking` 流式期间默认展开，用户手动折叠后尊重操作
- `frontend/.../subtask-card.tsx`：subagent running 时卡片默认展开

### 2. 列名归一化补全

**问题**：真实 EthoVision 中文导出（`开放臂(开放臂1 / 中心点)`）不匹配 catalog 的 `in_zone_open_arms_*` 模式，`prep_metric_plan` 报 `columns_missing`。

**根因**：`utils.py` 的 `normalize_column_name` 规则不全——zone-前缀列（`开放臂(...)`）和中文 mobility 列（`活动状态`）未收敛到规范名。

**修复**：
- `packages/ethoinsight/ethoinsight/utils.py`：
  - `_slugify` subs 新增 `活动状态→mobility_state`、`活动→activity`
  - 新增 `_ZONE_PREFIX_TOKENS` + `_is_zone_prefix_column` + zone-prefix 检测
  - 补充 `中央区→center`、`中心区→center` 映射
- `packages/ethoinsight/tests/test_parse.py`：新增 `TestRealDataNormalization` 5 个 golden parse fixtures

**验证**：EPM real data `open_arm_time_ratio=0.6806` (实验组) vs `0.4047` (对照组)，3 次 dogfood 可复现。

### 3. 并行 Bash Guardrail

**问题**：code-executor 的 `bash -c "python -m ethoinsight.scripts.* & ... wait"` 被 guardrail 拦截。

**修复**（3 轮迭代）：
- `packages/agent/backend/.../guardrails/script_invocation_only_provider.py`：
  1. 新增 `_PARALLEL_BASH_PATTERN` + `_validate_parallel_bash_content` 白名单验证
  2. 新增 `_DANGEROUS_META` 拒绝 shell 元字符（`|><\`$`）
  3. `_PARALLEL_BASH_PATTERN` 支持 `cd` 前缀（`cd /mnt/... && bash -c`）
  4. `_PARALLEL_INNER_SPLITTER` 加 `\n` 分隔（修复 `wait\necho` 同 segment 问题）

### 4. 模板确认 Guardrail

**问题**：`identify_ev19_template` 返回 ambiguous 时，Agent 有时默认选"推荐"项不等用户确认。

**修复**：
- `packages/agent/backend/.../tools/builtins/identify_ev19_template_tool.py`：ambiguous/ok/unknown 时写 `template_candidates.json` 到 workspace
- `packages/agent/backend/.../guardrails/ev19_template_provider.py`：`set_experiment_paradigm` 时检查——若 ambiguous 且未传 `user_confirmed_template=True`，硬性拦截
- `packages/agent/backend/.../agents/middlewares/experiment_context.py`：`set_experiment_paradigm_tool` 新增 `user_confirmed_template` 参数

### 5. Prompt 优化

- `packages/agent/backend/.../agents/lead_agent/prompt.py`：
  - 明确 `identify_ev19_template` 已含列结构，不需要额外调 `inspect_uploaded_file`
  - 新增规则：用户未明确选模板时禁止默认选推荐项
  - 修复 L537 矛盾（原来同时说"用 inspect_uploaded_file"和"不需要调 inspect"）

## 关键发现

1. **数据归一化是确定性映射问题**，应在 `parse_header` 层解决，不应交给 LLM 运行时处理
2. **Prompt-only 不靠谱**——需要 guardrail 硬约束（模板确认是教科书案例）
3. **并行 bash 正确的 guardrail 模式**：白名单验证 > 黑名单拒绝。bash -c 内只允许脚本调用 + wait + echo
4. **TXT 和 XLSX 走同一个 `normalize_columns`**——改一处同时覆盖

## 未完成事项

1. **[P2] 优化：inspect_uploaded_file 仍有少量冗余调用**——每次 ~2-3s，可接受但可优化。方案：工具内部检查 `template_candidates.json` 已存在时返回 hint
2. **[P2] 测试覆盖缺口**：并行 bash + template guardrail 缺少 red 锚点测试
3. **[P3] `_extract_template_recommendations` 正则脆弱**——硬编码模板类名白名单，新增 EV19 类别需手动同步

## 改动文件清单

| 文件 | 改动类型 |
|------|----------|
| `packages/ethoinsight/ethoinsight/utils.py` | 列名归一化 + 中央区映射 |
| `packages/ethoinsight/tests/test_parse.py` | golden parse fixtures |
| `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` | 思考面板流式默认展开 |
| `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` | subagent 卡片默认展开 |
| `packages/agent/backend/.../guardrails/script_invocation_only_provider.py` | 并行 bash 白名单 |
| `packages/agent/backend/.../guardrails/ev19_template_provider.py` | 模板确认拦截 |
| `packages/agent/backend/.../tools/builtins/identify_ev19_template_tool.py` | 写 candidates 到 workspace |
| `packages/agent/backend/.../agents/middlewares/experiment_context.py` | `user_confirmed_template` 参数 |
| `packages/agent/backend/.../agents/lead_agent/prompt.py` | 去重 inspect + 模板确认规则 |
| `docs/handoffs/2026-06/2026-06-05-e2e-dogfood-thinking-display-issues.md` | 问题分析文档 |
| `docs/superpowers/specs/2026-06-05-thinking-streaming-and-display-spec.md` | 修复 spec |

## 建议接手路径

1. 读 `docs/handoffs/2026-06/2026-06-05-e2e-dogfood-thinking-display-issues.md` 了解问题全貌
2. 读 `docs/superpowers/specs/2026-06-05-thinking-streaming-and-display-spec.md` 了解 spec
3. 重启 `make dev` + 跑一次完整 E2E dogfood 验证并行 bash 是否生效
4. 如果用户要求：补 P2 测试覆盖
5. 如果用户要求：用 `perf/e2e-pipeline-speed-optimization` 分支做速度优化

## 风险与注意事项

- **受保护文件**：`experiment_context.py`、`prompt.py` 是受保护文件，sync 上游时需 surgical merge
- **并行 bash guardrail 有 3 个子修复**，必须全部到位才能生效
- **Gateway hot-reload OFF**（watchfiles 挂），改后端后必须 `make stop && make dev`
- **ethoinsight 是 editable install**，改完立即生效无需重装
- **全量测试有 4 个 pre-existing 污染失败**，不要归因于自己的改动

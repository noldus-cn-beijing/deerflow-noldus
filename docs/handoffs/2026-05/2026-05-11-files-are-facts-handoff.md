# files-are-facts 重构交接（2026-05-11）

## 实施完成情况
- spec: docs/superpowers/specs/2026-05-11-subagent-file-is-facts-design.md (v2.1)
- plan: docs/superpowers/plans/2026-05-11-files-are-facts.md
- 全部 14 个 task 完成（Task 1-5, 5a, 6, 7, 7a, 8-12；14 个独立 commit）
- 14 个 commit 在 dev 分支上（9d7e3171..1932fadb）

## 验收结果

### 静态验证
- production 代码（packages/harness/deerflow/）0 处 code_summary 引用 ✅
- skill 目录 0 处 code_summary 引用（Task 5a 清理 ethoinsight-planning） ✅
- 6 个范式胶水脚本模板全部含 [gate_signals] 输出代码（Task 7a） ✅
- handoff_schemas 含 GateSignals（3 个 handoff schema 有 optional gate_signals 字段） ✅
- 4 个文件含 GateSignals/gate_signals（handoff_schemas + code_executor + data_analyst + report_writer） ✅
- task_tool 含 HANDOFF_FILE_REGISTRY + {{handoff://}} 占位符解析 ✅
- SubagentExecutor 含 authorized_handoff_paths 参数 + 属性存储 ✅
- task_tool → SubagentExecutor 透传 authorized_handoff_paths ✅
- HandoffIsolationProvider 实现 + 接入 SubagentExecutor._create_agent 中间件链 ✅
- lead prompt 派遣 subagent 全部用占位符（新增加"派遣规约"章节） ✅
- lead 自己 read_file 仍用真实路径（特权角色，不改） ✅
- make test: 2194 passed（含 26 个新增测试），14 pre-existing failures 未变 ✅
- make lint: 13 pre-existing errors 未变，新增代码 0 lint errors ✅

### 新增测试
- test_gate_signals_schema.py — 6 tests（GateSignals Pydantic 模型）
- test_task_tool_handoff_placeholders.py — 6 tests（占位符解析 + registry）
- test_subagent_executor_authorized_paths.py — 3 tests（executor 参数接受）
- test_handoff_isolation_provider.py — 9 tests（provider 授权逻辑）
- test_subagent_handoff_isolation_integration.py — 2 tests（集成验证）

### dogfooding 实测
- 用例 A（正态流程 + knowledge-assistant 追问 + [gate_signals] 输出验证）：[待 dev 环境实测]
- 用例 B（critical 数据质量拦截 + lead 不读 handoff）：[待 dev 环境实测]
- 用例 C（subagent 未占位符授权时 read_file handoff 被 Guardrail 拦截）：[待 dev 环境实测]

## 架构变更摘要
1. **code_summary.json 中转层已删除** — lead 不再 write_file 到 /mnt/shared/
2. **{{handoff://}} 占位符** — 既指路径也授权，task_tool 解析为真实路径 + 采集 authorized_handoff_paths
3. **HandoffIsolationProvider** — GuardrailProvider 协议实现，subagent read_file handoff_*.json 需 lead 显式授权；读自己的 outbox 允许；lead 调用不受限
4. **[gate_signals] 块** — subagent 最终消息的结构化决策信号，lead 不读 handoff 也能做 Step 1.5 拦截
5. **胶水脚本确定性生成** — code-executor 的 [gate_signals] 由 Python 代码 print 输出（6 范式模板统一一致）

## 受保护文件改动记录
- `executor.py`: +authorized_handoff_paths 参数（__init__ 签名 + 属性）+ HandoffIsolationProvider 接入（_create_agent 中 append GuardrailMiddleware）
- `task_tool.py`: +HANDOFF_FILE_REGISTRY + _resolve_handoff_placeholders() + 透传 authorized_handoff_paths
- 上述改动均为追加性（新参数/新 import/新 middleware append），不改核心调度循环

## 后续路径
- {{shared://}} 占位符清理（follow-up）
- inbox/outbox 目录改造（v0.1 之后）
- LocalSandboxProvider 物理隔离（v0.1 之后）

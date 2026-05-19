# 细粒度 Tool 重构交接文档

> 日期：2026-04-17
> 上一份交接：`docs/handoffs/2026-04-17-granular-tool-refactor-handoff.md`

---

## 1. 当前任务目标

把 `run_paradigm_analysis`（monolithic tool）拆成 5 个细粒度 tool + skill 编排，解决 code-executor E2E 无限循环问题。

**根因**：`run_paradigm_analysis` 在 `ethoinsight/templates/tool.py:410` 实现但从未在 `config.yaml` 注册，code-executor 工具列表里没有它，导致 LLM 反复尝试脚本方式、在 UTF-16 编码上反复失败、SummarizationMiddleware 冻结上下文后无限重复。

## 2. 当前进展

### ✅ Step 1: tool.py 追加 5 个工具 wrapper
- 文件：`packages/ethoinsight/ethoinsight/templates/tool.py`（line 473 之后追加 ~400 行）
- 新增辅助函数：`_get_thread_data`, `_write_json`, `_read_json`, `_fail`, `_ok`
- 新增 5 个 @tool：`parse_trajectories_tool`, `compute_metrics_tool`, `run_statistics_tool`, `generate_charts_tool`, `assess_and_handoff_tool`
- 验证：7 个 @tool 装饰器（原 2 + 新 5），语法正确，全部可导入

### ✅ Step 2: config.yaml 注册 5 个 tool
- 文件：`packages/agent/config.yaml` tools 区块
- ethoinsight group 现有 6 个工具（原 1 + 新 5）

### ✅ Step 3: code_executor.py 重写
- 文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`
- tools 列表改为 5 个新工具 + `get_analysis_template`（fallback）+ sandbox 工具
- max_turns: 8 → 12，timeout_seconds: 600 → 900
- prompt 使用正面句式（GLM-5.1 约束）

### ✅ Step 4: ethoinsight-analysis skill 重写
- `SKILL.md` 重写为 v2.0（6 步编排指南）
- 新增 `references/tool-reference.md`、`error-recovery.md`、`quality-checks.md`
- 删除旧的 `references/data-quality-checks.md`
- 更新 `references/fallback-workflow.md`
- 保留 `references/run-paradigm-analysis-api.md`（旧 monolith API 参考）

### ✅ Step 5: 测试
- 新增 `packages/ethoinsight/tests/test_granular_tools.py`（7 passed, 3 skipped — skip 因为 demo-data 路径不在测试目录下）
- 新增 `packages/agent/backend/tests/test_ethoinsight_analysis_skill.py`（5 passed）
- 后端全量测试：**1551 passed**（原 1546 + 5 新增）
- ethoinsight 全量测试：**114 passed**

### ✅ Step 6: 文档更新
- `CLAUDE.md` 流水线章节已更新
- `docs/specs/paradigm-analysis-tools-spec.md` 重写为技术规格
- `docs/handoffs/2026-04-17-granular-tool-refactor-handoff.md` 已创建

### ⬜ Step 7: E2E 验证（未完成）
- 需要 `make stop && make dev` 启动服务
- 在 localhost:2026 用 demo-data（斑马鱼鱼群行为）测试
- 确认 code-executor 不再循环、依次调用 5 个工具

## 3. 关键上下文

### 工具注册流程
```
config.yaml tools: → get_available_tools() → resolve_variable() → subagent tools whitelist
```
subagent 的 `tools` 列表是精确名称匹配的白名单，不匹配的名称会静默跳过。

### 虚拟路径解析
- `/mnt/user-data/*` → `.deer-flow/threads/{thread_id}/user-data/*`
- 由 `sandbox/tools.py:396` 的 `replace_virtual_path()` 处理
- tool.py 中的 `_resolve_virtual_path()` 复用了相同逻辑

### Handoff 契约（不能破坏）
code-executor 写 `handoff_code_executor.json` → lead agent 压缩为 `/mnt/shared/code_summary.json` → data-analyst + report-writer 消费。
字段：`status`, `summary`, `output_files`, `metrics_summary`, `statistics`, `assessment`, `metadata`, `errors`。

### GLM-5.1 正面指令约束
绝对不要使用"禁止 X"、"不要 Y"句式，会反向激活。全部使用正面指令描述期望行为。

### 循环导入问题
DeerFlow 框架有 `deerflow.subagents.__init__` → `executor` → `agents` → `subagents` 循环。
测试时通过 `conftest.py` 注入 mock 绕过：`sys.modules["deerflow.subagents.executor"] = _executor_mock`。

## 4. 关键发现

1. **config.yaml 是唯一的工具注册入口** — 即使工具代码写好了，不注册就不会出现在 agent 的工具列表里
2. **monolithic tool 的设计缺陷** — 7 步塞进一个 tool call，LLM 看不到中间状态，出错只能整体重试
3. **ToolRuntime 是 pydantic dataclass** — 测试中不能用 MagicMock，必须构造真正的 ToolRuntime 实例
4. **demo-data 在 `demo-data/DemoData/斑马鱼鱼群行为/`** — 不是简单的 `demo-data/*.txt`，E2E 测试需要正确路径

## 5. 未完成事项（按优先级）

### P0: E2E 验证
```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make stop && make dev
# 等启动完成，前端 localhost:2026
# 上传 demo-data/DemoData/斑马鱼鱼群行为/ 下的轨迹文件
# 输入："帮我分析，Subject 1-2 是对照组，Subject 3-5 是实验组"
# 期望：lead agent → code-executor 依次调 5 个 tool → handoff → data-analyst → report
```

### P1: 提交代码
E2E 验证通过后分 3 个 commit：
1. `feat(ethoinsight): 拆 run_paradigm_analysis 为 5 个细粒度 tool`
2. `feat(agent): 更新 code_executor 和 ethoinsight-analysis skill 适配新工具`
3. `docs: 更新 CLAUDE.md、spec、交接文档`

### P2: Phase 0 剩余
- EPM + OFT 范式模板补全
- 429 重试策略改为 5s/15s/30s（`llm_error_handling_middleware.py`）
- 之前未提交的工作（test_client.py 修复、CLAUDE.md 创建、ethoinsight-planning skill）也应一并提交

## 6. 建议接手路径

1. 读本文件了解全貌
2. `cd packages/agent && make stop && make dev` 启动服务
3. 在 localhost:2026 做 E2E 测试（上传 demo-data 轨迹文件）
4. 确认 code-executor 不循环 → 提交代码
5. 继续 Phase 0 剩余任务

## 7. 风险与注意事项

- **不要删除 `run_paradigm_analysis_tool`** — 作为内部 fallback 和旧测试兼容
- **不要修改 `lead_agent/prompt.py`** — handoff 压缩逻辑与 data-analyst 契约不能破坏
- **noldus-kb MCP 当前禁用** — `extensions_config.json` 里 `"enabled": false`
- **循环导入** — 测试里必须有 conftest.py 的 executor mock，否则导入失败

## 8. 下一位 Agent 的第一步建议

```bash
# 1. 读交接文档
cat /home/qiuyangwang/noldus-insight/docs/handoffs/2026-04-17-granular-tool-refactor-handoff.md

# 2. 启动服务做 E2E
cd /home/qiuyangwang/noldus-insight/packages/agent
make stop && make dev

# 3. 如果 E2E 有问题，查看日志
tail -50 logs/langgraph.log
```

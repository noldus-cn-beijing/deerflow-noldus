# 2026-05-21 deerflow upstream sync handoff

## 背景

执行 `scripts/sync-deerflow.sh --auto-apply` 将 deerflow fork 从 `f0dd8cb` 同步到上游 `e19bec1`（150 commits）。脚本自动合入了 126 个"安全文件"+ 12 个新增文件，11 个受保护文件留待人工处理。

## 已完成的工作

### 手工 surgical merge（4 个文件，需保留）

| 文件 | 做了什么 | 原因 |
|------|---------|------|
| `sandbox/sandbox.py` | 添加 `download_file` 抽象方法 | 上游新增接口，本地缺失 |
| `sandbox/local/local_sandbox.py` | ① 添加 `download_file` 实现 ② `_resolve_path` 加路径穿越保护（`relative_to` 校验） | 上游 bug fix + 新功能 |
| `subagents/config.py` | 补回 Noldus 字段（`when_to_use`/`input_contract`/`output_contract`/`required_upstream_handoffs`）+ `format_subagent_capability` / `validate_subagent_handoff_refs` 函数 | 安全文件自动合入覆盖了 Noldus 定制版，恢复之 |
| `config/database_config.py` | 补回 `_resolve_sqlite_dir` 函数（`@lru_cache` + PWD 兜底，blockbuster 安全） | 上游简化版 `Path.resolve()` 会触发 blockbuster 告警，且测试直接 import 此函数 |

### 跳过的受保护文件（13 个中的 11 个出现在 diff 中，2 个无变更）

**已包含上游改进，无需 merge：**
- `llm_error_handling_middleware.py` — 本地已有总超时上限、timeout 关键字、circuit breaker 延迟加载等全部改进
- `config/paths.py` — 本地已有 `Path(__file__).resolve().parents[4]` 修复，且额外有 `SHARED_PATH_PREFIX` + `shared_dir()`
- `mcp/tools.py` — 本地已有 truncation 包装 + `_make_sync_tool_wrapper` 线程池执行器
- `sandbox/tools.py` — 本地已有 `_WriteFileArgs` Pydantic schema + `WRITE_FILE_MAX_CONTENT_CHARS` + 路径安全校验

**完整 Noldus 重写，上游改动不适用：**
- `prompt.py`（871 行 diff）— 全部中文 prompt、EthoInsight 调度规则、capability-exposure、orchestration_guide 等
- `agent.py`（453 行 diff）— 中间件链完全不同（Guardrail / Gate / ThinkTag / TrainingData / _LEAD_EXCLUDED_TOOLS）
- `executor.py`（752 行 diff）— HandoffIsolationProvider / ScriptInvocationOnlyProvider / max_turns 硬限制
- `task_tool.py`（213 行 diff）— `{{shared://}}` / `{{handoff://}}` 占位符解析、auto-inject。唯一上游 fix（`_find_usage_recorder` BaseCallbackManager 兼容）本地已有

**无关键上游修复：**
- `thread_data_middleware.py` — 上游改动是给 HumanMessage 注入 run_id/timestamp（功能新增，非 bug fix）

**无变更（不在本次 diff 中）：**
- `subagents/builtins/__init__.py`
- `agents/thread_state.py`

## 当前问题

### 126 个安全文件的自动合入引入了大规模 API breakage

上游在 `f0dd8cb..e19bec1` 之间做了大量 API 重构（Tier 4 persistence / runtime / event store / checkpointer 等），自动合入后：

- **Pre-sync**: 2 failed, 2695 passed
- **Post-sync**: 128+ failed/errors（104 failed + 24 errors，2569 passed）

已知的破坏性变更：
- `claude_provider.py` — 删除 `_strip_malformed_thinking_blocks` 函数
- `runtime/runs/manager.py` — `RunManager.get` 变为 async
- `persistence/` — 引擎初始化、事件存储 API 变更
- `runtime/checkpointer/` / `runtime/store/` / `runtime/stream_bridge/` — 接口重构
- `agents/middlewares/loop_detection_middleware.py` — 消息文本变更
- `tools/builtins/setup_agent_tool.py` — 行为变更
- `agents/` — factory/features/memory 多处 API 变更
- `tools/tools.py` / `tools/builtins/__init__.py` — tool 加载逻辑变更
- `skills/` — loader/parser/types/installer/validation API 变更
- `config/` — agents_config/app_config/checkpointer_config/extensions_config/memory_config 等全部变更

### 已尝试的回退

回退了 `models/claude_provider.py`、`runtime/runs/manager.py`、`persistence/`、`runtime/events/`、`runtime/checkpointer/`、`runtime/store/`、`runtime/stream_bridge/`、`tools/builtins/setup_agent_tool.py`、`agents/middlewares/loop_detection_middleware.py` 到 HEAD，但仍有 128+ 失败——说明更多文件需要回退。

## 下一步：回退策略

**目标**：只保留 4 个 surgical merge + 12 个新增文件，回退其余 126 个安全文件。

**原因**：这 126 个文件中大多数是 Tier 4 重构（persistence/runtime/checkpointer/skills 体系），引入的 API breakage 修复成本远高于从这些文件中能 cherry-pick 的 bug fix 价值。我们的 Noldus 代码基于旧 API 运行正常，没有必要为追上游而大规模适配。

### 执行步骤

```bash
cd /home/wangqiuyang/noldus-insight
LOCAL_HARNESS="packages/agent/backend/packages/harness/deerflow"

# 1. 回退全部改过的安全文件到 HEAD（保留 4 个 surgical merge 文件）
# 安全文件列表在 /tmp/safe-files.txt（126 个）
# 受保护文件列表在 /tmp/protected-changed.txt（11 个）

KEEP_FILES=(
    "sandbox/sandbox.py"
    "sandbox/local/local_sandbox.py"
    "subagents/config.py"
    "config/database_config.py"
)

while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    skip=0
    for kf in "${KEEP_FILES[@]}"; do
        [[ "$file" == "$kf" ]] && skip=1 && break
    done
    [[ $skip -eq 1 ]] && continue
    git checkout HEAD -- "${LOCAL_HARNESS}/${file}"
done < /tmp/safe-files.txt

# 2. 删除新增文件（这 12 个上游新文件可能引用新 API，先删掉再逐个评估）
git clean -fd "${LOCAL_HARNESS}/"

# 3. 验证
cd packages/agent/backend && make test
# 预期：回到 2 failed, 2695 passed（pre-sync 基线）
```

### 验证通过后

1. 保留这 4 个 surgical merge 文件不变
2. 12 个新增文件中，可以逐个评估后手动加回（它们引用新 API 的概率低，但需验证）：
   - `agents/middlewares/dynamic_context_middleware.py`
   - `agents/middlewares/tool_call_metadata.py`
   - `community/serper/__init__.py`
   - `community/serper/tools.py`
   - `config/loop_detection_config.py`
   - `models/mindie_provider.py`
   - `persistence/json_compat.py`
   - `runtime/converters.py`
   - `skills/manager.py`
   - `skills/tool_policy.py`
   - `tools/builtins/update_agent_tool.py`
   - `tools/sync.py`
3. `git add -A && git commit -m "sync: deerflow upstream surgical merge — download_file + symlink protection + Noldus config fix"`

## 关键路径

- **同步脚本**：`scripts/sync-deerflow.sh`
- **受保护文件列表**：脚本内 `PROTECTED_FILES` 数组（13 个）
- **安全文件列表**：`/tmp/safe-files.txt`（126 个）
- **Diff 报告**：`/tmp/deerflow-sync-report/`（11 个受保护文件的逐文件 diff）
- **上游 head**：`deerflow/main` @ `e19bec1`
- **同步基准**：`f0dd8cb`

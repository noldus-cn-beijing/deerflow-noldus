# DeerFlow 上游同步 SOP

> 适用于: noldus-insight 项目从 deerflow-noldus 拉取 DeerFlow 官方框架更新

## 同步链路

```
DeerFlow 官方仓库
      ↓ git sync (手动)
deerflow-noldus (你的 fork)
      ↓ scripts/sync-deerflow.sh
noldus-insight (主项目)
```

## 5-25 sync 关键教训（必读）

5-25 这次同步 (PR #33/#34/#35/#36/#38, 15 commit) 暴露了 4 个新教训。后续 sync 必须按这些规则走：

### 教训 1: head-to-head 必做，不能只看 +/- 统计

**反例**: PR-B 时根据 `dcc6f1e6 loop_detection` +228/-78 行统计判 "本地与上游差异太大，跳过"。但实际差异其中 ~80% 是上游新架构（_pending_warnings + wrap_model_call），剩下 20% 才是 Noldus 定制（中文 / tool freq 3/5 / task per-subagent_type bucket）。**正确做法：整文件接受上游 head + 重新打 Noldus 定制**，而不是跳过。

```bash
# 正确流程
diff <(git show deerflow/main:<上游路径>) <本地路径> | head -50  # 先看具体内容
grep -nE "Noldus marker pattern" <本地路径>                       # grep 验证实际 marker
# 才能判 "可整文件接受 + 重新打" / "surgical merge 几个点" / "完全跳过"
```

### 教训 2: 注册类文件必看 BUILTIN_TOOLS / `__all__` 集合

**反例**: 5-21 PR #23 翻车根因 — `tools/tools.py` 含 `BUILTIN_TOOLS` 注册了 ethoinsight 4 工具 (set_experiment_paradigm / identify_ev19_template / prep_metric_plan / 等)，上游脚本判 "本地未改" 把整文件覆盖，**洗掉了 4 个工具注册**。FST agent 立刻卡死。

5-25 已修复: `scripts/sync-deerflow.sh` 的 `PROTECTED_FILES` 现在包含全部注册类文件:
- `tools/tools.py` / `tools/builtins/__init__.py`
- `subagents/builtins/__init__.py`
- `agents/__init__.py` / `agents/factory.py` / `subagents/registry.py`

任何**集合字面量 / 聚合 import 块**只要本地添加过项就必须加入 PROTECTED_FILES。

### 教训 3: 飞轮 / 训练数据 schema 必须 surgical merge

**反例**: 5-25 PR-C 时 `persistence/feedback/sql.py` 默认 12 grep marker (ethoinsight / shared_path 等) 都没命中，差点整文件接受。**实际含 `verdict` 三分类 + `revised_text` + `message_id` 训练飞轮定制**（CLAUDE.md #7 提到的 SQLite 后端 schema）。

**新增 grep pattern**: `verdict | revised_text | message_id` 也要查。`scripts/sync-deerflow.sh` 已把 `persistence/feedback/sql.py` 加入 `PROTECTED_FILES`。

### 教训 4: in-graph `create_chat_model` 必须传 `attach_tracing=False`

5-25 PR #38 (df951542 Langfuse 完整集成) 引入的架构约定。**未来任何在 `lead_agent` 模块内或可达的中间件内新加 `create_chat_model(...)` 必须传 `attach_tracing=False`**，否则同一 LLM 调用会发出重复 span，且 `langfuse_*` keys 被剥导致 `session_id` / `user_id` 永远进不到 trace。

详见 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 顶部 module docstring **"INVARIANT — tracing callback placement"**。当前 4 个 in-graph site:
1. `make_lead_agent` 内 bootstrap agent
2. `make_lead_agent` 内 default lead agent
3. `_create_summarization_middleware` 内 2 处 (config.model_name / 默认分支)
4. `agents/middlewares/title_middleware.py` 的 `TitleMiddleware._build_title_kwargs`

### 教训 5: 全量跟随会洗掉 `backend/pyproject.toml` 里的 ethoinsight workspace 声明

`packages/agent/backend/pyproject.toml` **是上游也有的文件**，全量跟随 sync（subtree/merge 整文件覆盖）会把它退回上游版，**删掉我们加的 ethoinsight workspace 依赖 3 行**（dependencies 的 `"ethoinsight"`、`[tool.uv.workspace] members` 的 `"packages/ethoinsight"`、`[tool.uv.sources]` 的 `ethoinsight = { workspace = true }`）。历史：`fbc2255e`(4-07) 加，`fa3418ec`(6-02 全量跟随 sync) 删 → dogfood 时 lead agent 调 `identify_ev19_template`/`set_experiment_paradigm`/`prep_metric_plan` 全 `ModuleNotFoundError: No module named 'ethoinsight'`。

这是**第 2 次复发**（同 vector）。每次全量跟随 sync 后**必查**：
```bash
grep -q '"ethoinsight"' packages/agent/backend/pyproject.toml \
  && grep -q 'packages/ethoinsight' packages/agent/backend/pyproject.toml \
  || echo "⚠ ethoinsight workspace wiring LOST — 参照 fbc2255e 重新加 3 行后跑 uv sync"
```
CI 已有硬兜底（`backend-blocking-io-tests.yml` 的 "Assert ethoinsight is installed" 步骤）：洗掉 wiring 的 PR 合并前就变红。但 SOP 这条 grep 让 sync 当场就能发现，不用等 CI。注意 `sync-deerflow.sh` 的 `PROTECTED_FILES` **保护不了它**（那些路径都相对 `backend/packages/harness/deerflow/`，作用域外）。

## 前置条件

- [ ] noldus-insight 工作区干净（`git status` 无未提交改动）
- [ ] deerflow-noldus 已从 DeerFlow 官方同步最新代码

## Step 1: 同步 deerflow-noldus

先确保你的 fork 是最新的：

```bash
# 在 deerflow-noldus 本地仓库中
cd /path/to/deerflow-noldus
git fetch upstream          # upstream = DeerFlow 官方
git merge upstream/main     # 合入官方最新改动
git push origin main        # push 到你的 fork
```

## Step 2: 在 noldus-insight 中运行同步脚本

```bash
cd /home/wangqiuyang/noldus-insight

# 先 fetch 上游最新
git fetch deerflow

# 推荐: 先 dry-run 看影响范围
./scripts/sync-deerflow.sh --dry-run

# 交互模式（会逐步确认）
./scripts/sync-deerflow.sh

# 临时覆盖同步基准（极少用 — 当 .deerflow-sync-state 失效或重算时）
DEERFLOW_LAST_SYNC=<commit-sha> ./scripts/sync-deerflow.sh --dry-run

# ⚠️ 不要用 --auto-apply，除非已 dry-run 确认每个安全文件分类正确
./scripts/sync-deerflow.sh --auto-apply   # 危险，5-21 翻车根因
```

### 同步基准从哪来

脚本按以下优先级找上次同步到的上游 commit:

1. **`DEERFLOW_LAST_SYNC` 环境变量**（一次性覆盖）
2. **`.deerflow-sync-state` 文件**（持久化记录，**推荐机制**）
3. **git subtree squash commit message**（fallback — 但 5-21/5-25 都没做 squash，这条经常失效）
4. 都没有 → 全量对比所有上游文件（噪声大）

`.deerflow-sync-state` 是项目根目录下的纯文本文件，格式:

```yaml
last_sync_commit: f9b70713
last_sync_date: 2026-05-25
```

完成一次 sync 后 **必须更新此文件**（见 Step 6）。

## Step 3: 处理受保护文件

脚本会输出类似：

```
--- 受保护文件 (3) ---
以下文件包含你的定制改动，上游也有新改动，需要逐个判断:

  agents/lead_agent/prompt.py (diff: 420 行)
    上游改动:
      7643a46 fix(skill): make skill prompt cache refresh nonblocking (#1924)
    报告: /tmp/deerflow-sync-report/agents_lead_agent_prompt.py.diff

  subagents/executor.py (diff: 18 行)
    上游改动:
      f0dd8cb fix(subagents): add cooperative cancellation (#1873)
    报告: /tmp/deerflow-sync-report/subagents_executor.py.diff
```

对每个文件：

### 选项 A: 保留你的版本（上游改动不重要）

不做任何操作。

### 选项 B: 手动合入上游的部分改动（最常见）

```bash
# 查看上游具体改了什么
cat /tmp/deerflow-sync-report/subagents_executor.py.diff

# 根据 diff 手动编辑文件，只合入你需要的部分
vim packages/agent/backend/packages/harness/deerflow/subagents/executor.py
```

### 选项 C: 整文件接受上游 + 重新打 Noldus 定制（5-25 推荐）

当本地定制集中在几个明确位置（如 4 个常量 + 1 个分支）时：

```bash
# 1. 备份本地版
cp packages/agent/backend/packages/harness/deerflow/<file> /tmp/noldus_<file>.bak

# 2. 整文件接受上游 head
git show deerflow/main:backend/packages/harness/deerflow/<file> \
  > packages/agent/backend/packages/harness/deerflow/<file>

# 3. 对照 /tmp/noldus_<file>.bak 用 Edit 重新打 Noldus 定制
# 4. 必须 grep Noldus markers 确认全部回来

grep -nE "verdict|revised_text|ethoinsight|shared_path|/mnt/shared|extra_env|ArchivingSummarization|ThinkTag|TrainingData|GateEnforcement|HandoffIsolation|Ev19Template|set_experiment_paradigm|identify_ev19|prep_metric_plan" \
  packages/agent/backend/packages/harness/deerflow/<file>
```

这种方式优势是同步上游的所有 bug fix / 新方法 / 新参数，且接管文档维护责任。

### 选项 D: 完全接受上游（放弃定制 — 极少用）

```bash
git show deerflow/main:backend/packages/harness/deerflow/subagents/executor.py \
  > packages/agent/backend/packages/harness/deerflow/subagents/executor.py
```

## Step 4: 回归测试

```bash
cd packages/agent/backend
MAIN_VENV=/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
PYTHONPATH=$PWD/packages/harness:$PWD \
  $MAIN_VENV/bin/python -m pytest tests/ --tb=line -q
```

⚠️ 在 worktree 内跑测试**必须显式设 PYTHONPATH 和 DEER_FLOW_CONFIG_PATH**，否则会用主仓库代码（uv editable install 副作用）。

确保所有测试通过。如果失败，检查是否遗漏了某个受保护文件的定制。

## Step 5: Lint

```bash
RUFF=/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/ruff
$RUFF check packages/agent/backend/<file>
```

## Step 6: 更新 `.deerflow-sync-state` 并提交

```bash
cd /home/wangqiuyang/noldus-insight

# 1. 更新 sync state (推进基准)
UPSTREAM_HEAD=$(git rev-parse deerflow/main)
cat > .deerflow-sync-state <<EOF
last_sync_commit: ${UPSTREAM_HEAD}
last_sync_date: $(date +%Y-%m-%d)
EOF

# 2. 提交所有改动
git add -A
git commit -m "sync deerflow upstream to ${UPSTREAM_HEAD:0:7}: [简要描述]"
```

## 受保护文件清单

以下文件包含 Noldus 定制改动，每次同步时需要特别注意（与 `scripts/sync-deerflow.sh` 中 `PROTECTED_FILES` 数组保持同步）：

### 高侵入 (Noldus 核心业务逻辑)

| 文件 | 定制内容 | 注意事项 |
|------|---------|---------|
| `agents/lead_agent/prompt.py` | 中文调度规则、subagent 描述、Gate 反问机制 | **最大冲突面**，上游经常改这个文件 |
| `subagents/builtins/__init__.py` | 注册 5 个 Noldus subagent | 上游很少改，但**含集合字面量** |

### 中侵入 (通用增强 + Noldus 功能)

| 文件 | 定制内容 | 注意事项 |
|------|---------|---------|
| `llm_error_handling_middleware.py` | 超时类型识别 + 总超时上限 | 通用增强 |
| `mcp/tools.py` | MCP 结果截断 4096 字符 | 通用增强 |
| `sandbox/tools.py` | shared workspace 路径映射 | `{{shared://}}` 占位符依赖 |
| `local_sandbox.py` | venv PATH + DEERFLOW_PATH_* 环境变量 | Python 执行环境 |
| `agents/lead_agent/agent.py` | tool_groups 过滤 + 多个 Noldus 中间件链 + tracing callback root + `attach_tracing=False` 约定 | lead agent 工具隔离 |
| `task_tool.py` | `{{shared://}}` 占位符解析 | shared:// 功能依赖 |
| `executor.py` | recursion_limit 修复 + max_turns 硬限制 | bug fix |
| `config/paths.py` | `/mnt/shared` 路径 + `shared_dir()` | shared:// 功能依赖 |

### 低侵入 (1-3 行改动)

| 文件 | 定制内容 |
|------|---------|
| `sandbox.py` | `execute_command` extra_env 参数 |
| `thread_state.py` | `shared_path` 字段 |
| `thread_data_middleware.py` | `shared_path` 初始化 |

### 5-25 新增 (注册类 + 飞轮 + Loop detection 中文化 + Guardrail)

| 文件 | 定制内容 | 教训来源 |
|------|---------|---------|
| `tools/tools.py` | BUILTIN_TOOLS 注册 ethoinsight 4 工具 | 教训 2 (5-21 PR #23) |
| `tools/builtins/__init__.py` | `__all__` 包含 Noldus 工具 | 教训 2 |
| `agents/__init__.py` / `agents/factory.py` | agent 注册 | 教训 2 |
| `subagents/registry.py` | subagent 注册 | 教训 2 |
| `persistence/feedback/sql.py` | verdict 三分类 + revised_text + message_id 四元组主键 | 教训 3 (5-25 PR #36) |
| `agents/middlewares/loop_detection_middleware.py` | 中文 + ethoinsight 提示 + tool freq 3/5 阈值 + task per-subagent_type bucket | 5-25 PR #35 |
| `tools/builtins/setup_agent_tool.py` | Noldus 定制 | 5-06 上游脚本误标 |
| `guardrails/middleware.py` | `name` kwarg 解决 langchain unique-name 限制 | (各 guardrail provider 各自实例的命名) |

## 常见问题

### Q: 上游新增了文件怎么办？

脚本会自动识别上游新增的文件并提示你合入，这些文件不会冲突。

### Q: 上游删了某个文件怎么办？

脚本目前不处理删除。如果上游删除了某个文件，你需要手动判断是否跟进删除。

### Q: 如何更新受保护文件列表？

编辑 `scripts/sync-deerflow.sh` 中的 `PROTECTED_FILES` 数组。当你对新的上游文件做了定制修改时，把它加入列表。**记得同步更新本文档表格。**

### Q: 多久同步一次？

建议每 1-2 周同步一次。DeerFlow 官方更新频率较高（每周几个 commit），但大部分改动在你没碰过的文件里，自动合入即可。**但每次都必须 head-to-head 验证而不只是看 +/- 统计**（教训 1）。

### Q: 同步基准 (`.deerflow-sync-state`) 失效了怎么办？

如果 `.deerflow-sync-state` 文件丢失或记录的 commit 不在 `deerflow` remote 中，脚本会 fallback 到 subtree squash 提取，再 fallback 到全量对比。**全量对比噪声大但不会出错**，可以正常用，sync 完后写回 state 文件即可。

也可以一次性覆盖：

```bash
DEERFLOW_LAST_SYNC=<known-good-commit> ./scripts/sync-deerflow.sh --dry-run
```

## 前端 streaming 吞字临时修复（2026-05-09 起）

> **背景**：上游 `frontend/src/core/rehype/index.ts` 的 `rehypeSplitWordsIntoSpans` 在 streaming 时让已渲染英文词重 mount，造成"吞字"。我们在 `noldus-insight` 删除该插件链作为临时止血（commit `d4171eed`），同时给 bytedance/deer-flow 提了永久修复 PR（streamdown 1.4 → 2.5 + 官方 `animated` prop），分支 `fix-streaming-word-animation-remount` 在 `noldus-cn-beijing/deerflow-noldus` 上。详见 [docs/handoffs/2026-05-09-streaming-fade-in-fixed-handoff.md](../handoffs/2026-05-09-streaming-fade-in-fixed-handoff.md)。

**Sync 时如果碰到下列前端文件冲突，按以下决策树处理**：

```
packages/agent/frontend/src/core/rehype/index.ts
packages/agent/frontend/src/core/streamdown/plugins.ts
packages/agent/frontend/src/components/workspace/messages/markdown-content.tsx
packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx
packages/agent/frontend/src/components/workspace/messages/message-list.tsx
packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx
packages/agent/frontend/src/components/workspace/messages/message-group.tsx
packages/agent/frontend/src/styles/globals.css  (--animate-fade-in 段)
```

### 决策树

1. **上游有没有合并我们 streamdown 升级 PR？**
   ```bash
   git -C <deerflow-checkout> log deerflow/main --oneline -- frontend/package.json | head -5
   git -C <deerflow-checkout> show deerflow/main:frontend/package.json | grep streamdown
   ```
   - 看到 `streamdown: "^2.5.0"` 或更高 → **走 A**
   - 还是 `streamdown: "1.4.0"` → **走 B**

2. **A 路径（上游已 merge）**：
   - **接受上游版本**覆盖这 8 个文件（删 noldus-insight 的临时改动）
   - 在 noldus-insight 里 `git revert d4171eed` **或** 直接重新 checkout 这些文件到上游版本
   - 业务恢复 word fade-in 视觉，吞字已由上游根治
   - 验证：浏览器 streaming 一遍，看每个 word span 的 `data-sd-animate` 属性已存在（streamdown 2.x 标记）

3. **B 路径（上游还没 merge）**：
   - **跳过**这 8 个文件（保留 noldus-insight 临时修复）
   - sync 脚本提示冲突时人工选 "ours"
   - 注意：streamdown 仍是 1.4.0，没有 word fade，但没吞字

### 监控 PR 状态

我们提交的 PR 链接：[bytedance/deer-flow PR (待补编号)](https://github.com/bytedance/deer-flow/pulls)（搜索 "fix-streaming-word-animation-remount" 或 "streamdown's built-in animation"）。

merged 之后**优先做 sync**，避免维护双重逻辑。

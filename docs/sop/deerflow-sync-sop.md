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
cd /home/qiuyangwang/noldus-insight

# 先 fetch 上游最新
git fetch deerflow

# 方式一: 交互模式（推荐，会逐步确认）
./scripts/sync-deerflow.sh

# 方式二: 只看报告，不动文件
./scripts/sync-deerflow.sh --dry-run

# 方式三: 安全文件自动合入，只人工处理受保护文件
./scripts/sync-deerflow.sh --auto-apply
```

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

### 选项 C: 接受上游版本（放弃你的定制 — 谨慎使用）

```bash
git show deerflow/main:backend/packages/harness/deerflow/subagents/executor.py \
  > packages/agent/backend/packages/harness/deerflow/subagents/executor.py
```

## Step 4: 回归测试

```bash
cd packages/agent/backend && make test
```

确保所有测试通过。如果失败，检查是否遗漏了某个受保护文件的定制。

## Step 5: 提交

```bash
cd /home/qiuyangwang/noldus-insight
git add -A
git commit -m "sync deerflow upstream to [commit hash]: [简要描述]"
```

## 受保护文件清单

以下文件包含 Noldus 定制改动，每次同步时需要特别注意：

| 文件 | 定制内容 | 注意事项 |
|------|---------|---------|
| `prompt.py` | Noldus 调度规则、中文示例、subagent 描述 | **最大冲突面**，上游经常改这个文件 |
| `builtins/__init__.py` | 注册 4 个 Noldus subagent | 上游很少改 |
| `llm_error_handling_middleware.py` | 超时类型识别 + 总超时上限 | 通用增强 |
| `mcp/tools.py` | MCP 结果截断 4096 字符 | 通用增强 |
| `sandbox/tools.py` | shared workspace 路径映射 | shared:// 功能依赖 |
| `local_sandbox.py` | venv PATH + DEERFLOW_PATH_* 环境变量 | Python 执行环境 |
| `agent.py` | tool_groups 过滤 + ArchivingSummarization | lead agent 工具隔离 |
| `task_tool.py` | `{{shared://}}` 占位符解析 | shared:// 功能依赖 |
| `executor.py` | recursion_limit 修复 + max_turns 硬限制 | bug fix |
| `config/paths.py` | `/mnt/shared` 路径 + `shared_dir()` | shared:// 功能依赖 |
| `sandbox.py` | `execute_command` extra_env 参数 | sandbox 接口扩展 |
| `thread_state.py` | `shared_path` 字段 | 1 行改动 |
| `thread_data_middleware.py` | `shared_path` 初始化 | 1 行改动 |

## 常见问题

### Q: 上游新增了文件怎么办？
脚本会自动识别上游新增的文件并提示你合入，这些文件不会冲突。

### Q: 上游删了某个文件怎么办？
脚本目前不处理删除。如果上游删除了某个文件，你需要手动判断是否跟进删除。

### Q: 如何更新受保护文件列表？
编辑 `scripts/sync-deerflow.sh` 中的 `PROTECTED_FILES` 数组。当你对新的上游文件做了定制修改时，把它加入列表。

### Q: 多久同步一次？
建议每 1-2 周同步一次。DeerFlow 官方更新频率较高（每周几个 commit），但大部分改动在你没碰过的文件里，自动合入即可。

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

# 执行文档：Event loop is closed 终极修复 — sync 上游 memory 模块（含 user_id）

**日期**: 2026-04-29
**目标读者**: 接手执行的 Claude Code agent
**预计工时**: 60-90 分钟
**前置交接**: [2026-04-28-event-loop-fix-handoff.md](2026-04-28-event-loop-fix-handoff.md)

---

## 0. 一句话目标

E2E 测试又出现 `RuntimeError: Event loop is closed`。上次（4-28）的修复（清 langchain 的 lru_cache）只是缓解，**没修根**。上游 deerflow 已发布更彻底的修复（`5bfd23d5` 持久 loop → `82731aeb` 改用 sync `model.invoke()`）。**本次任务：完整 sync 上游 memory 模块（含 user_id 联动），把根因彻底切除**。

---

## 1. 背景与根因

### 1.1 上次修复（4-28）为什么不够

上次的 `_evict_provider_async_client_caches`（[updater.py:220-241](../../packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py#L220)）只在 memory 更新**之后**清 lru_cache。但 **memory 更新过程中已经在 connection pool 创建了绑定到 loop_mem 的 SSL 连接**——主 agent 下次复用时仍会触发 `Event loop is closed`。

实际错误堆栈（`packages/agent/logs/langgraph.log:124-195`）：

```
LLM call failed after 1 attempt(s): Event loop is closed
  File ".../langchain_anthropic/chat_models.py:1352, in _astream
  File ".../anthropic/_streaming.py:204, in __stream__
  File ".../httpx/_models.py:1063, in aiter_raw → aclose()
  File ".../httpcore/_async/connection_pool.py:420, in aclose
  File ".../anyio/_backends/_asyncio.py:1329, in aclose
  File "/usr/lib/python3.12/asyncio/selector_events.py:875, in close
  File "/usr/lib/python3.12/asyncio/base_events.py:541, in _check_closed
RuntimeError: Event loop is closed
```

时间线（同上日志）：
- `01:58:30` bg-loop-8 第一次 run
- `01:59:31` bg-loop-9 第二次 run
- `01:59:53` Memory updated successfully（Thread-6 daemon thread 用短命 loop）
- `02:00:17` **27 秒后** bg-loop-9 LLM stream 关闭时崩 — 复用了 memory update 创建的污染连接

### 1.2 上游最终方案（commit `82731aeb`）

> 完全用 sync `model.invoke()` 跑 memory 更新，不再创建任何 event loop。sync HTTP 用 langchain 的 `Client`（不是 `AsyncClient`），与主 agent 走的 `AsyncClient` 走**完全独立的 connection pool**，从架构层面消灭跨 loop 复用问题。

关键 commits：
- `5bfd23d5 fix(memory): replace short-lived asyncio.run() with persistent event loop` — 第一版（用持久 loop）
- `82731aeb update the code to address the review comments` — 终版（**改成 sync**，更优雅）
- `7289d2bb Merge branch 'main' into fix-2615` — 合到 main

---

## 2. 改动范围（精确清单）

### 2.1 必改文件（4 个 memory 文件 + 联动）

| 路径 | diff 行数 | 改动性质 |
|---|---|---|
| `packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py` | 356 | **核心**：sync model.invoke + user_id 参数 |
| `packages/agent/backend/packages/harness/deerflow/agents/memory/queue.py` | 76 | user_id 参数透传 |
| `packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py` | 141 | 缓存 key → (user_id, agent_name) tuple，调用 `get_paths().user_memory_file()` |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py` | 27 | 引入 `from deerflow.runtime.user_context import get_effective_user_id` |

### 2.2 联动新增文件（必须一并带入）

| 路径 | 性质 |
|---|---|
| `packages/agent/backend/packages/harness/deerflow/runtime/user_context.py` | **新增**（167 行）— 定义 `get_effective_user_id()` |
| `packages/agent/backend/packages/harness/deerflow/runtime/__init__.py` | **可能需要新增/更新** — 看上游有没有 export |

### 2.3 联动改动文件（受保护 — 部分合并）

| 路径 | 是否受保护 | 改动性质 |
|---|---|---|
| `packages/agent/backend/packages/harness/deerflow/config/paths.py` | ✅ 受保护 | 新增 `user_dir`/`user_memory_file`/`user_agent_memory_file` 等方法 + thread_dir 系列加 `user_id` 关键字参数 |

**关键判断**：`paths.py` 是 [scripts/sync-deerflow.sh:47](../../scripts/sync-deerflow.sh#L47) 的受保护文件，**不能直接 checkout 上游版本**。需要手动加新方法 + 给现有 thread_dir 系列加 `user_id` keyword-only 参数（默认 `None`）。

### 2.4 不改的相关文件

`agents/memory/message_processing.py` 和 `agents/memory/summarization_hook.py` —— diff 0 行，已与上游一致。

---

## 3. 执行步骤

### Step 0: 准备工作

```bash
cd /home/wangqiuyang/noldus-insight

# 0.1 检查 git 状态
git status -s
# 预期：可能有 packages/agent/temp/nginx.local.rendered.conf 改动（dev server 重渲染）

# 0.2 stash 干扰文件
git stash push packages/agent/temp/nginx.local.rendered.conf -m "temp nginx before memory sync" 2>/dev/null || true

# 0.3 确认 fetch 上游
git fetch deerflow 2>&1 | tail -3
git log --oneline deerflow/main -1
# 应该看到: 7289d2bb Merge branch 'main' into fix-2615 或更新

# 0.4 创建工作分支（保险）
git checkout -b fix/event-loop-sync-memory dev 2>/dev/null || git checkout fix/event-loop-sync-memory
```

### Step 1: 带入新文件 `runtime/user_context.py`

```bash
# 1.1 检查目录是否存在
ls packages/agent/backend/packages/harness/deerflow/runtime/ 2>&1
# 如果不存在，创建：
mkdir -p packages/agent/backend/packages/harness/deerflow/runtime/

# 1.2 直接从上游 checkout（这是新增文件，无冲突风险）
git show deerflow/main:backend/packages/harness/deerflow/runtime/user_context.py \
  > packages/agent/backend/packages/harness/deerflow/runtime/user_context.py

# 1.3 检查 runtime/__init__.py 状态
git show deerflow/main:backend/packages/harness/deerflow/runtime/__init__.py 2>/dev/null > /tmp/upstream_runtime_init.py
ls packages/agent/backend/packages/harness/deerflow/runtime/__init__.py 2>&1

# 如果本地不存在，复制上游版本
if [ ! -f packages/agent/backend/packages/harness/deerflow/runtime/__init__.py ]; then
  cp /tmp/upstream_runtime_init.py packages/agent/backend/packages/harness/deerflow/runtime/__init__.py
fi

# 1.4 验证 user_context.py 可导入（语法检查）
cd packages/agent/backend
.venv/bin/python -c "from deerflow.runtime.user_context import get_effective_user_id, AUTO, resolve_user_id; print('ok')"
cd ../../..
```

**验收**：`get_effective_user_id`、`AUTO`、`resolve_user_id` 三个符号都能导入。

### Step 2: 合并 `config/paths.py`（受保护文件，手动）

`paths.py` 是受保护文件，本地有 `shared_dir(thread_id)` 等定制方法（line 195），上游没有。上游加了 `user_dir`/`user_memory_file`/`user_agent_memory_file` 等方法且 thread_dir 系列加了 `user_id` keyword-only 参数。

**策略**：保留本地的 `shared_dir`，只把上游新增的方法加进来 + 给 thread_dir 系列**加可选 `user_id` 参数（默认 None）走旧逻辑**。

```bash
# 2.1 生成对比 diff
diff -u packages/agent/backend/packages/harness/deerflow/config/paths.py \
        <(git show deerflow/main:backend/packages/harness/deerflow/config/paths.py) \
        > /tmp/paths_diff.txt
wc -l /tmp/paths_diff.txt
```

**手动改动清单**（按上游 `git show deerflow/main:backend/packages/harness/deerflow/config/paths.py` 内容）：

#### 2.1 新增 `_validate_user_id`（在 `_validate_thread_id` 后面）

```python
def _validate_user_id(user_id: str) -> str:
    # 复制上游 line 26-32 内容
```

#### 2.2 在 `Paths` 类里新增以下方法（在 `agent_memory_file` 后面，`thread_dir` 前面）

```python
def user_dir(self, user_id: str) -> Path:
    # 上游 line 145-148

def user_memory_file(self, user_id: str) -> Path:
    # 上游 line 149-152

def user_agent_memory_file(self, user_id: str, agent_name: str) -> Path:
    # 上游 line 153-156
```

**注意**：本地已有的 `shared_dir(thread_id)` 方法（约 line 195）**保留不动**。

#### 2.3 thread_dir 系列加 `user_id` keyword-only 参数

`thread_dir` / `sandbox_work_dir` / `sandbox_uploads_dir` / `sandbox_outputs_dir` / `acp_workspace_dir` / `sandbox_user_data_dir` / `host_thread_dir` 等所有 line 139-227 的方法签名都要从：

```python
def thread_dir(self, thread_id: str) -> Path:
```

改成：

```python
def thread_dir(self, thread_id: str, *, user_id: str | None = None) -> Path:
```

**实现里**：保留本地原逻辑作为 `user_id is None` 分支，复制上游新逻辑作为 `user_id is not None` 分支。完整参考上游 `git show deerflow/main:backend/packages/harness/deerflow/config/paths.py` 的实现。

#### 2.4 验证

```bash
cd packages/agent/backend
.venv/bin/python -c "
from deerflow.config.paths import get_paths
p = get_paths()
# 旧 API（不带 user_id）必须仍然工作
print(p.thread_dir('test-thread'))
print(p.memory_file)
# 新 API
print(p.user_memory_file('test-user'))
print(p.user_agent_memory_file('test-user', 'test-agent'))
# 本地定制必须仍然工作
print(p.shared_dir('test-thread'))
"
cd ../../..
```

**验收**：6 个 print 都成功，无 AttributeError。

### Step 3: 合入 `memory/` 4 个文件

这些是**安全文件**（本地未改过），直接 checkout 上游版本即可。

```bash
# 3.1 备份本地版本（保险，万一回滚）
mkdir -p /tmp/memory-backup
cp packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py /tmp/memory-backup/
cp packages/agent/backend/packages/harness/deerflow/agents/memory/queue.py /tmp/memory-backup/
cp packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py /tmp/memory-backup/
cp packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py /tmp/memory-backup/

# 3.2 直接 checkout 上游版本
git show deerflow/main:backend/packages/harness/deerflow/agents/memory/updater.py \
  > packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py

git show deerflow/main:backend/packages/harness/deerflow/agents/memory/queue.py \
  > packages/agent/backend/packages/harness/deerflow/agents/memory/queue.py

git show deerflow/main:backend/packages/harness/deerflow/agents/memory/storage.py \
  > packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py

git show deerflow/main:backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py \
  > packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py

# 3.3 语法 + import 验证
cd packages/agent/backend
.venv/bin/python -c "
from deerflow.agents.memory.updater import MemoryUpdater, _SYNC_MEMORY_UPDATER_EXECUTOR
from deerflow.agents.memory.queue import get_memory_queue, ConversationContext
from deerflow.agents.memory.storage import get_memory_storage
from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
print('imports ok')
"
cd ../../..
```

**验收**：导入 ok，无 ImportError / AttributeError。

如果有 `_evict_provider_async_client_caches` 的引用残留导致测试失败，那是预期的（这函数已被上游删除）。**对应测试已经在 4-28 加进 `tests/test_memory_updater.py`**，需要手动清理（见 Step 4）。

### Step 4: 清理过时测试

上次会话（4-28）添加的 4 个测试在 [test_memory_updater.py](../../packages/agent/backend/tests/test_memory_updater.py)：

- `test_evict_provider_async_client_caches_clears_lru_caches`
- `test_evict_provider_async_client_caches_swallows_import_errors`
- `test_run_async_update_sync_evicts_caches_on_success`
- `test_run_async_update_sync_evicts_caches_on_failure`

这些是为了测旧的 `_evict_provider_async_client_caches`，**新方案下函数已不存在，必须删掉这些测试**。

```bash
# 4.1 找到这些测试在 test_memory_updater.py 里的位置
grep -n "test_evict_provider_async_client_caches\|test_run_async_update_sync_evicts_caches" \
  packages/agent/backend/tests/test_memory_updater.py

# 4.2 用 Edit 工具删除这些测试函数（包含其定义体）。
#     如果测试都连续放在文件末尾，可以一次性删除最后那段。
#     如果中间穿插，需要逐个删除。

# 4.3 跑测试确认这些测试已不存在且其他测试通过
cd packages/agent/backend
.venv/bin/pytest tests/test_memory_updater.py -v 2>&1 | tail -30
cd ../../..
```

**验收**：`tests/test_memory_updater.py` 通过，没有 collection error。

### Step 5: 跑全量测试，定位回归

```bash
cd packages/agent/backend
.venv/bin/pytest tests/ 2>&1 | tail -50 > /tmp/test_result.txt
cat /tmp/test_result.txt
cd ../../..
```

**预期基线**（来自 [2026-04-28-gate-enforcement-implementation-handoff.md](2026-04-28-gate-enforcement-implementation-handoff.md)）：
```
2 failed, 1725 passed, 14 skipped
```

两个 pre-existing 失败：
- `test_planning_skill_is_enabled_in_config`
- `test_usage_example_shows_ask_clarification_between_analyst_and_writer`

**新失败的可能原因排查**：

| 现象 | 可能原因 | 排查 |
|---|---|---|
| `ImportError: deerflow.runtime.user_context` | Step 1 的 user_context.py 没复制成功 | `ls packages/agent/backend/packages/harness/deerflow/runtime/` |
| `AttributeError: 'Paths' object has no attribute 'user_memory_file'` | Step 2 的 paths.py 没合好 | 重新过 Step 2 |
| memory storage 测试报 `(user_id, agent_name)` tuple 错 | storage.py 的 cache_key 改动需要测试也用新 key | 看测试代码是否直接 patch `_memory_cache` |
| `TypeError: thread_dir() got unexpected keyword argument 'user_id'` | Step 2.3 漏改某些 host_* 方法 | 把所有 `thread_dir`/`sandbox_*_dir`/`host_*` 方法的签名都加上 keyword |

**修完确认**：跑完 `make test`，结果应该 ≤ 2 failed（仅 pre-existing），其余全 pass。

### Step 6: E2E 烟测

```bash
cd packages/agent
make stop 2>/dev/null
make dev > /tmp/e2e.log 2>&1 &
DEV_PID=$!
sleep 30
# 检查启动日志
grep -E "Application startup complete|Starting queue with isolated loops|ImportError|Traceback" /tmp/e2e.log | head -20
```

**手动 E2E 步骤**（用户做）：
1. 浏览器打开 http://localhost:2026
2. 创建新 thread（manual mode）
3. 上传 `demo-data/shoaling/` 几个文件
4. 让 agent 跑分析直到结束
5. 全程**不应该出现** `Event loop is closed` warning
6. 后端 log 检查：`grep "Event loop is closed" packages/agent/logs/langgraph.log` —— 应无新增

### Step 7: Commit

```bash
git add packages/agent/backend/packages/harness/deerflow/runtime/user_context.py
git add packages/agent/backend/packages/harness/deerflow/runtime/__init__.py
git add packages/agent/backend/packages/harness/deerflow/config/paths.py
git add packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py
git add packages/agent/backend/packages/harness/deerflow/agents/memory/queue.py
git add packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py
git add packages/agent/backend/tests/test_memory_updater.py

git commit -m "fix(memory): sync upstream sync-HTTP memory update to eliminate cross-loop bug

切到上游 deerflow 终版方案：memory 更新改用 sync model.invoke()，不再创建任何
event loop。sync HTTP 走 langchain Client（独立连接池），与主 agent 的 AsyncClient
走完全分离的 pool，从架构层面消除跨 loop 复用问题。

替代 4-28 的 _evict_provider_async_client_caches 缓解方案（仅清缓存，仍残留连接）。

带入：
- runtime/user_context.py 新模块（get_effective_user_id 等）
- config/paths.py 新增 user_*_file 方法 + thread_dir 系列加 user_id keyword
- memory/{updater,queue,storage}.py 全量对齐上游
- middlewares/memory_middleware.py 在 enqueue 时捕获 user_id

参考: deerflow upstream commits 5bfd23d5 / 82731aeb / 7289d2bb (issue #2615)"
```

### Step 8: 还原 stash

```bash
git stash list | head -3
# 如果看到 "temp nginx before memory sync"，pop 它
git stash pop 2>/dev/null || true
```

---

## 4. 风险与回滚

### 4.1 回滚步骤

如果 E2E 烟测发现新 bug 而要回滚：

```bash
git reset --hard HEAD~1   # 回退本次 commit
# 或者切回 dev 分支
git checkout dev
git branch -D fix/event-loop-sync-memory  # 删工作分支
```

`/tmp/memory-backup/` 里有原始文件副本，可以手动 cp 回去恢复 sub-modular 状态。

### 4.2 已知风险

| 风险 | 应对 |
|---|---|
| **paths.py 手动合并出错**（最可能的故障点） | 改完后用 Step 2.4 的 print 测试逐个验证；导入异常先看 `import` 报错具体在哪行 |
| **user_id contextvar 没设置导致默认走 "default"** | 这是预期行为（fallback）。如果需要真正区分用户，要在 Gateway auth 里调 `set_current_user`，**本次不做**，留 follow-up |
| **storage cache key 改动可能命中本地测试** | 看 `tests/test_memory_storage*.py` 是否直接 patch `_memory_cache`；改成新的 tuple key |
| **上游可能假设 memory_config 有新字段** | `make test` 跑 memory_config 相关测试看；本地 `MemoryConfig` 与上游对比一下 |
| **`shared_dir(thread_id)` 是本地定制** | Step 2 必须保留，否则 sandbox 会找不到 `/mnt/shared` 路径 |

### 4.3 paths.py 手动合并失败的备用方案

如果 Step 2 改 paths.py 卡住，可以**降级**：

不带入 user_id 联动，只 sync `updater.py` 的 sync-HTTP 核心改动：

```bash
# 备用方案：只改 updater.py，跳过 user_id 联动
# 1. 不动 paths.py / queue.py / storage.py / memory_middleware.py
# 2. 只把 updater.py 改成 sync model.invoke()
# 3. 但要把 update_memory 签名里的 user_id 参数去掉（保持本地 caller 不破坏）
```

这个备用方案的代码改动等价于上游 5bfd23d5 + 82731aeb 但去掉 user_id —— 可以从上游 `82731aeb` 文件取出来后手工删 user_id 参数。

---

## 5. 验收标准

| 项 | 要求 |
|---|---|
| 单元测试 | `make test` 结果 ≤ 2 failed（pre-existing），原 1725 passed → 新增/减少几个可接受 |
| 静态检查 | `make lint` 通过 |
| E2E 烟测 | 完整跑一次 shoaling 分析，**全程无 `Event loop is closed`** |
| 启动日志 | 启动后看到 `Starting queue with isolated loops`（langgraph 框架）+ 没有 ImportError |
| Memory 流程 | 跑完 E2E 后查 `cat packages/agent/backend/.deer-flow/.ethoinsight/memory.json`，应该有正常更新 |

---

## 6. 上下文速查（给执行 agent 用）

### 6.1 关键源码位置

| 内容 | 路径 |
|---|---|
| 错误日志 | `packages/agent/logs/langgraph.log:124-195`（4-29 这次复发的堆栈） |
| 当前（旧）的 evict 实现 | `packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py:220-241` |
| 旧的 `_run_async_update_sync` | 同上 line 244-276 |
| ChatAnthropic `_async_client` cached_property | `packages/agent/backend/.venv/lib/python3.12/site-packages/langchain_anthropic/chat_models.py:1078` |
| `_get_default_async_httpx_client` lru_cache | `packages/agent/backend/.venv/lib/python3.12/site-packages/langchain_anthropic/_client_utils.py:67` |
| `_AsyncHttpxClientWrapper.__del__`（触发崩溃的 anti-pattern） | 同上 line 37-45 |
| 上游 sync 终版 updater.py | `git show deerflow/main:backend/packages/harness/deerflow/agents/memory/updater.py` |
| 上游 user_context.py | `git show deerflow/main:backend/packages/harness/deerflow/runtime/user_context.py` |
| 上游 paths.py | `git show deerflow/main:backend/packages/harness/deerflow/config/paths.py` |

### 6.2 关键 git refs

```bash
# 上游修复关键 commits
git show 5bfd23d5  # 第一版（持久 loop，已被废弃）
git show 82731aeb  # 终版（sync model.invoke），重要！
git show 7289d2bb  # merge 到 main

# fork remote
deerflow → git@github.com:noldus-cn-beijing/deerflow-noldus.git (fetch & push)
```

### 6.3 模型协议背景（避免误判）

本地 `config.yaml` 里的 `deepseek-v4-pro` 实际通过 `langchain_anthropic:ChatAnthropic` 类调用（走 NewAPI 转 Anthropic 协议），所以 `langchain_anthropic` 的 lru_cache 影响的就是这条线。改成 sync `model.invoke()` 后，sync 走 `_client`（不是 `_async_client`），完全独立的连接池。

### 6.4 项目规范提醒

- Python 3.12+，ruff line length 240
- 所有改动**必须有单测**
- commit message 用中文
- 不改 `noldus-kb` 的 `enabled: false` 状态（CLAUDE.md 第 2 条）

---

## 7. 完成后的交接

执行完后，**必须**写交接文档 `docs/handoffs/2026-04-29-event-loop-fix-v2-completed-handoff.md`，包含：

1. 实际改动文件清单（git diff --stat）
2. 测试结果（pass/fail 数对比基线）
3. E2E 烟测结果（用户跑完后填）
4. 是否需要更新上游 issue 草稿（[2026-04-28-deerflow-upstream-issue-draft.md](2026-04-28-deerflow-upstream-issue-draft.md)）说明已经被上游 fix
5. 遗留事项（比如 user_id contextvar 还没和 Gateway auth 联动）

---

## 8. 不在本次范围（明确排除）

- ❌ 上游同步的其他 65 个安全文件 + 9 个其他受保护文件 — 留 v0.1 后做
- ❌ Gateway auth 的 user_context 集成 — 当前 fallback 到 `DEFAULT_USER_ID = "default"` 即可，不影响修复
- ❌ Phase 0 P0 的 `set_experiment_paradigm` 和 `/mnt/shared` 问题 — 由其他 handoff 处理
- ❌ 上游 `2e05f380` per-user persistence 的完整带入 — 只引入 memory 相关的最小子集

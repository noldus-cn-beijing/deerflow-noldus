# DeerFlow 上游 Tier 2/3/4 执行文档

**日期**: 2026-05-07
**交接人**: Claude (本会话, opus-4-7-1m)
**接手对象**: 下一位 AI Agent
**任务状态**: 🟡 分析与分类完成,实施未开始
**前置依赖**: [2026-05-06-deerflow-upstream-sync-handoff.md](2026-05-06-deerflow-upstream-sync-handoff.md) 已完成 Tier 1 (8b19c667)

---

## 0. TL;DR

上次会话只做了 Tier 1。本次会话**重新分类了全部 105 个上游 commit** 并**重新评估了 Tier 4**,得出关键结论:

1. **noldus 已经吸收了 Tier 4 一半改动** — `runtime/user_context.py`、`agents/memory/storage.py`、`agents/memory/queue.py` 都已经是上游 head 版本(0 行差异)。**只缺 `persistence/`、`runtime/checkpointer/`、`runtime/events/` 这 3 个目录**。
2. **noldus 现有代码 0 处依赖**这 3 个缺失目录(harness + gateway 全 grep 过)。
3. **目标已经从「v0.1 单用户研究助手」校准为「现在就要上云」**,所以 Tier 4 的 backward-compat 部分应该**全拉**,better-auth 和前端登录可延后到 Phase 2。
4. **核心规则不变**:Noldus 价值在 **skill 内容**(`packages/agent/skills/custom/` 5 个 markdown skill 目录)+ **prompt** + **subagent 名字** + **关键 setting**(`extra_env`、`/mnt/shared`、`{{shared://}}`)。**skill 加载代码不是核心价值**,可随上游重构。

工作量过大,**分 3 轮做**。本文档定义**轮 1** 的精确执行步骤(本次会话剩余约 4-5 小时可完成),轮 2/3 标 backlog。

---

## 1. 总体策略与 3 轮分工

```
轮 1 (本文档,约 4-5h)
  ├── A. T1-sibling 批量合 (19 commit)
  ├── B. T2-keep surgical merge (9 commit)
  └── C. T3 ⭐⭐ 精选合并 (5 commit)

轮 2 (下次会话,约 5-7h)
  ├── E. Skill storage 重构 (1ad1420e)
  └── D. Tier 4 BC 持久化层 (11 commit, 主要新增 persistence/)

轮 3 (再下次会话,约 5-7h)
  ├── F. T3 体量大但有价值 (8 commit, 含 token usage)
  ├── G. 中间件链相关 surgical merge (6 commit)
  └── H. 复审 Skip 类(原本被低估的)+ Phase 2 better-auth 立项
```

---

## 2. 关键数据 (105 commit 完整分类)

**完整分类清单**: [data/all-105-commits-classified.txt](2026-05-07-tier234-walkthrough-data/all-105-commits-classified.txt)
**原始 commit list**: [data/raw-105-commits.txt](2026-05-07-tier234-walkthrough-data/raw-105-commits.txt)

格式: `<sha>|<tier>|<理由>|<commit 标题>`

| 分类 | 数量 | 说明 |
|---|---:|---|
| T1-DONE | 11 | 上次会话已合到 8b19c667 |
| **T1-sibling** | **19** | **轮 1 任务** — 直接合 |
| **T2-keep** | **9** | **轮 1 任务** — surgical merge |
| **T3** | **14** | 5 个 ⭐⭐ 在轮 1,9 个在轮 3 |
| T3-skip | 2 | 重审后实际属 Tier 4 BC,改 backlog |
| T4 | 20 | 重审后大部分可拉 BC 部分,见轮 2/3 |
| T5/T4 | 5 | merge release/2.0-rc 大杂烩,跳过 |
| Skip | 24 | 大部分轮 3 复审,有些可拉 |
| T5 | 1 | Serper(项目不用,跳) |

---

## 3. 关键事实(必读)

### 3.1 noldus subtree 的 Tier 4 现状

```bash
cd /home/wangqiuyang/noldus-insight

# 关键文件 noldus vs upstream 真实差异(行数)
diff <(git show deerflow/main:backend/packages/harness/deerflow/agents/memory/storage.py) \
     packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py | wc -l
# → 0 (noldus 已吸收)

diff <(git show deerflow/main:backend/packages/harness/deerflow/runtime/user_context.py) \
     packages/agent/backend/packages/harness/deerflow/runtime/user_context.py | wc -l
# → 0 (noldus 已吸收)

# noldus 缺什么
git ls-tree -d deerflow/main:backend/packages/harness/deerflow/ | awk '{print $4}' | while read d; do
    [[ ! -d "packages/agent/backend/packages/harness/deerflow/$d" ]] && echo "MISSING: $d"
done
# → MISSING: persistence

# runtime 子目录
git ls-tree -d deerflow/main:backend/packages/harness/deerflow/runtime/ | awk '{print $4}' | while read d; do
    [[ ! -d "packages/agent/backend/packages/harness/deerflow/runtime/$d" ]] && echo "MISSING: runtime/$d"
done
# → MISSING: runtime/checkpointer
# → MISSING: runtime/events
```

**含义**: 引入这 3 个目录是**纯新增**,noldus 现有代码 0 import,所以拉了不会破坏现状。

### 3.2 noldus 与 upstream 真实差异(关键文件)

| 文件 | diff 行数 | 含义 |
|---|---:|---|
| `agents/memory/storage.py` | 0 | 已吸收上游(含 user_id 参数) |
| `agents/memory/queue.py` | 0 | 已吸收 |
| `runtime/user_context.py` | 0 | 已吸收 |
| `config/paths.py` | 59 | 轻定制(`/mnt/shared` + `shared_dir()`) |
| `subagents/executor.py` | **455** | **重定制** — recursion_limit + max_turns + {{shared://}} 占位符 |
| `agents/lead_agent/agent.py` | **257** | **重定制** — Noldus 中间件链 |
| `runtime/runs/worker.py` | **230** | 中度定制 |
| `subagents/registry.py` | 103 | 中度定制 |
| `tools/builtins/task_tool.py` | 156 | 中度定制 |
| `skills/manager.py` | 160 | 中度定制(被 1ad1420e 整体替换) |
| `skills/installer.py` | 168 | 同上 |
| `skills/loader.py` | 104 | 同上 |
| `tools/skill_manage_tool.py` | 148 | 中度定制 |
| `agents/middlewares/thread_data_middleware.py` | 61 | 轻定制 |
| `uploads/manager.py` | 74 | 轻定制 |

`runtime/journal.py` 在 noldus **不存在**(上游 4e4e4f92 等改的就是这个文件)。

### 3.3 已合 Tier 1 (`8b19c667`) 包含的 commit

```
5b633449 c3170f22 e8675f26  # loop_detection_middleware
c91785dd                    # title_middleware
e8572b9d                    # jina_client
1f59e945                    # claude_provider
194bab46 616caa92 c99865f5  # models/factory
1b74d845 866d1ca4           # openai_codex_provider
```

### 3.4 sync remote 配置

```bash
cd /home/wangqiuyang/noldus-insight
git remote -v | grep deerflow
# deerflow → git@github.com:noldus-cn-beijing/deerflow-noldus.git (已对齐 upstream/main = 4ead2c6b)

git rev-parse deerflow/main         # 4ead2c6b
git log --oneline --all --grep="Squashed 'packages/agent/' changes from" -1
# 8a5e4043 Squashed 'packages/agent/' changes from 15640fb..f0dd8cb (上次 sync 起点)
```

**当前 sync 起点**: `f0dd8cb`(上游)→ noldus 上次 squash
**当前 sync 终点**: `4ead2c6b`(上游 head)
**范围内 commit**: 105 个影响 harness

---

## 4. 轮 1 详细执行步骤

### 4.0 总目标

完成 **A (T1-sibling 19 个)** + **B (T2-keep 9 个 surgical merge)** + **C (T3 ⭐⭐ 5 个)** = **共 33 个 commit**,跑通测试,分批 commit。

### 4.1 步骤 A:T1-sibling 19 个直接合

**性质**: 纯 bug fix + 安全修复 + 不动受保护文件 + 0 Tier 4 依赖。预期改动文件 noldus 端**未定制过**,可以直接 `git show <sha> -- <file> | git apply` 或 `git show deerflow/main:<path> > <local>`。

#### 19 个 commit 列表(按价值类别分组)

**A.1 安全修复 (5)**
- `e543bbf5` upload 拒绝 symlinked 目标 (#2623)
- `707ed328` skill 安装前扫描 archive (#2561)
- `f7dfb88a` aio-sandbox 日志 redact env (#2562)
- `80e210f5` upload 文档转换需 opt-in (#2332)
- `3b3e8e1b` bash 命令审计加强 (#1881)

**A.2 Bug fix (10)**
- `a664d2f5` checkpointer 创建 SQLite 父目录 (#2272)
- `0948c7a4` codex streamed output 保留 (#1928)
- `24fe5fbd` mcp get_cached RuntimeError 修复 (#2252)
- `29817c3b` memory timezone-aware UTC (#1992)
- `1df389b9` web_fetch readability 异步化 (#2157)
- `718dddde` sandbox file lock 内存泄漏修复 (#2096)
- `0b6fa8b9` sandbox 启动清理 orphan container (#1976)
- `ad6d934a` ClarificationMiddleware 字符串化 options (#1997)
- `f4c17c66` present_files thread id fallback (#2181)
- `e4f896e9` todo-middleware 防止未完成 todo 提早退出 (#2135)

**A.3 增强/收尾 (4)**
- `bb8b234d` codex token usage 收尾(跟 T1 已合的 866d1ca4)(#2689)
- `b1aabe88` DeerFlowClient AI 文本流式 token deltas (#1969)
- `f80ac961` harness 恢复 legacy skills path fallback (#2694)
- `eba3b9e1` log_level 从 config.yaml 统一 (#2601)

#### 执行方法

每个 commit 用以下流程:

```bash
cd /home/wangqiuyang/noldus-insight
SHA=<sha>

# 1. 列出该 commit 改动的 harness 文件
git show --name-only --format='' "$SHA" -- 'backend/packages/harness/deerflow/' | \
    sed 's|backend/packages/harness/deerflow/|packages/agent/backend/packages/harness/deerflow/|'

# 2. 对每个文件,先 diff 看 noldus 是否定制过
for upstream_path in $(git show --name-only --format='' "$SHA" -- 'backend/packages/harness/deerflow/'); do
    rel="${upstream_path#backend/packages/harness/deerflow/}"
    local_path="packages/agent/backend/packages/harness/deerflow/$rel"
    diff_lines=$(diff <(git show "deerflow/main:$upstream_path" 2>/dev/null) "$local_path" 2>/dev/null | wc -l)
    echo "  $diff_lines lines noldus_diff_vs_upstream  $rel"
done

# 3. 如果某文件 diff = 0(noldus 已是上游 head),不需要改它
# 4. 如果 diff > 0 但 noldus 改的是 prompt/subagent名/setting → 转 surgical merge
# 5. 否则直接 git show "deerflow/main:$upstream_path" > "$local_path"
```

**关键守则**:
- 一定要先 `diff` 看 noldus 是否定制过文件,**不要无脑覆盖**(即使 commit 是 T1-sibling,某些 commit 改的文件 noldus 端可能已经因为别的原因定制过)
- 测试文件可以直接拿上游版(noldus 不改测试)
- 把每个 commit 涉及的文件都过一遍

#### 测试

A 阶段完成后:
```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, 1714+ passed, 14 skipped`(2 个 pre-existing,见上次 handoff §4.5)。

#### Commit

```
sync deerflow upstream Tier 1-sibling: 安全修复 + bug fix 19 commit

- 安全: e543bbf5 707ed328 f7dfb88a 80e210f5 3b3e8e1b
- Bug fix: a664d2f5 0948c7a4 24fe5fbd 29817c3b 1df389b9 718dddde 0b6fa8b9
            ad6d934a f4c17c66 e4f896e9
- 增强: bb8b234d b1aabe88 f80ac961 eba3b9e1

详见 docs/handoffs/2026-05-07-deerflow-tier234-execution-plan.md §4.1
```

---

### 4.2 步骤 B:T2-keep 9 个 surgical merge

按从易到难顺序做。每个独立 commit。

#### B.1 `a62ca5dd` httpx.ReadError 加入 retriable (#2309) — ⭐ 最简单

**改的文件**: `agents/middlewares/llm_error_handling_middleware.py`(noldus 重度定制)

**上游做了什么**: 加 `"ReadError"` 和 `"RemoteProtocolError"` 到 retriable error class 列表。

**Surgical merge**:
```bash
git show a62ca5dd -- 'backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py' | grep '^+' | grep -E "ReadError|RemoteProtocolError"
```

把这两个错误名加入 noldus 现有的 retriable error class tuple,**保留 noldus 总超时 + 多种 timeout 关键字识别 + get_app_config() fallback**。

**验证**: `pytest tests/test_llm_error_handling_middleware.py -v`

#### B.2 `e5b14906` + `7dea1666` event loop conflict 修复 — ⭐⭐⭐ 最难

**改的文件**: `subagents/executor.py`(noldus 455 行重定制)

**上游做了什么**:
- `e5b14906`: SubagentExecutor.execute() 用持久 event loop 替代 `asyncio.run()` 短命循环
- `7dea1666`: avoid temporary event loops in async subagent execution

**Surgical merge 步骤**:
1. `git show e5b14906 -- '...executor.py' > /tmp/e5b14906.patch`
2. 找上游 patch 中实际的 event loop 修复代码块(关键代码片段大概 30-50 行)
3. 在 noldus 版的 `executor.py` 中找到对应位置
4. 手工 merge:**保留** noldus 的 `recursion_limit` + `max_turns` 硬限制 + `{{shared://}}` 占位符;**叠加** 上游的 event loop fix
5. 同样处理 `7dea1666`

**警告**: noldus 版 `executor.py` 是定制最重的文件之一,这次合并最容易出 bug。**建议**:
- 用 git stash 备份 noldus 版本
- 先看完两个 commit 的完整 diff 再下手
- 对比每个函数变更,**只挑 event loop 部分**
- 测试: `pytest tests/test_subagent_executor.py -v` (如果有)

#### B.3 `6bd88fe1` sandbox bash traversal 防御 (#2560) — ⭐⭐ 中等

**改的文件**: `sandbox/tools.py`(noldus 重度定制 — `{{shared://}}` 占位符)

**上游做了什么**: 引入 `_URL_*_PATTERN`、`_LOCAL_BASH_*` 常量,加强 host bash traversal 检测。

**Surgical merge**:
- 把上游的常量定义和检测函数 patch 进 noldus 版本
- **不要** 引入 `from deerflow.runtime.user_context import get_effective_user_id`(如果有)— 用 noldus 现有的 user_id 处理方式
- 保留 noldus 的 `{{shared://}}` 占位符解析逻辑

#### B.4 `af8c0cfb` view_image 限制 thread data 路径 (#2557) — ⭐⭐ 中等

**改的文件**: `sandbox/tools.py` + `tools/builtins/view_image_tool.py`

**上游做了什么**: 引入 `resolve_and_validate_user_data_path` 函数。

**Surgical merge**:
- 把 `resolve_and_validate_user_data_path` 函数加入 noldus `sandbox/tools.py`
- 调用方 `view_image_tool.py` 同步更新
- **注意** 上游可能传 `user_id` 参数,noldus 用默认值即可

#### B.5 `ca1b7d5f` ls_tool path masking 修复 — ⭐ 简单

**改的文件**: `sandbox/tools.py`

**上游做了什么**: 给 `ls_tool` 输出加 path masking。

**Surgical merge**: 把 masking 代码加到 noldus 的 `ls_tool` 实现中,保留 noldus 现有 `mask_path()` 逻辑。

#### B.6 `4d4ddb3d` LLM circuit breaker (#2095) — ⭐⭐ 中等

**改的文件**: `llm_error_handling_middleware.py` 等

**上游做了什么**: 引入 lightweight circuit breaker 防 rate limit。

**Surgical merge**: 加 circuit breaker 类和 wiring,与 B.1 的 retriable error 配合。**注意保留** noldus 的总超时机制。

#### B.7 `5ba1dacf` present_file → present_files 重命名 (#2393) — ⭐ 简单

**改的文件**: `lead_agent/prompt.py` 等

**上游做了什么**: 把工具名 `present_file` 改为 `present_files`。

**Surgical merge**: 在 noldus prompt.py 中查找 `present_file` 字符串,改为 `present_files`。如果 noldus prompt.py 已经用了新名字,跳过。

#### B.8 `2176b2bb` bootstrap agent names 校验 (#2274) — ⭐ 简单

**改的文件**: 可能是 `task_tool.py` 或 setup_agent 路径

**上游做了什么**: 写文件前校验 agent name 安全性。

**Surgical merge**: 加校验函数,在 setup_agent 入口调用。

#### Commit B 阶段

```
sync deerflow upstream Tier 2-keep: 受保护文件 surgical merge 9 commit

- llm_error_handling: a62ca5dd (ReadError/RemoteProtocolError) + 4d4ddb3d (circuit breaker)
- subagent executor event loop: e5b14906 + 7dea1666
- sandbox 安全: 6bd88fe1 (bash traversal) + af8c0cfb (view_image 路径) + ca1b7d5f (ls 掩码)
- present_files 重命名: 5ba1dacf
- bootstrap agent name 校验: 2176b2bb

保留 Noldus 全部定制(中间件链/recursion_limit/{{shared://}}/总超时/extra_env)
详见 docs/handoffs/2026-05-07-deerflow-tier234-execution-plan.md §4.2
```

---

### 4.3 步骤 C:T3 ⭐⭐ 精选 5 个

#### C.1 `f514e35a` clarification messages 幂等性 (#2350)

**价值**: ⭐⭐ Gate 反问机制相关,与 noldus 的 GateEnforcementMiddleware 协同
**改的文件**: `clarification_middleware.py` 或类似

#### C.2 `5db71cb6` middleware 修复 dangling tool-call history (#2035)

**价值**: ⭐⭐ 与 noldus loop 检测协同
**改的文件**: 可能 `loop_detection_middleware.py` 或别的 middleware
**注意**: noldus 已经合过 Tier 1 的 loop detection 改动 (5b633449 c3170f22 e8675f26),这个 commit 可能与之有交集

#### C.3 `ec8a8cae` gate deferred MCP tool execution (#2513)

**价值**: ⭐⭐ 与 noldus mcp 截断协同
**改的文件**: `mcp/tools.py` 等

#### C.4 `11f557a2` trace 加 run_name (#2492)

**价值**: ⭐ 简单但会改已合 Tier 1 的 `title_middleware.py`
**改的文件**: `agents/memory/updater.py` + `title_middleware.py` + `skills/security_scanner.py`(3 文件加 run_name 参数)
**警告**: title_middleware 已经合过 c91785dd(剥 think 标签),此次再改要 surgical

#### C.5 `f9ff3a69` 避免 rescue non-skill tool outputs in summarization (#2458)

**价值**: ⭐ 与 noldus ArchivingSummarizationMiddleware 可能冲突
**改的文件**: 可能 `summarization_middleware.py`

#### Commit C 阶段

```
sync deerflow upstream Tier 3 ⭐⭐ 精选: 5 个高价值改进

- clarification 幂等性: f514e35a
- dangling tool-call 修复: 5db71cb6
- gate deferred MCP: ec8a8cae
- trace run_name: 11f557a2
- summarization rescue 修复: f9ff3a69

详见 docs/handoffs/2026-05-07-deerflow-tier234-execution-plan.md §4.3
```

---

## 5. 轮 1 完成后的验证

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend

# 全量测试
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -10

# Lint
PYTHONPATH=. uv run ruff check packages/harness/deerflow/

# 启动应用做 smoke test
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev  # 看 LangGraph + Gateway 启动是否正常
```

期望:
- 测试: `2 failed, 1714+ passed, 14 skipped`(2 pre-existing 不变)
- Lint: 0 error
- 应用: 正常启动到 `localhost:2026`

---

## 6. 轮 2 backlog (下次会话)

### E. Skill storage 重构 (`1ad1420e`)

**关键洞察**: noldus 5 个 custom skill (`packages/agent/skills/custom/`) 是**纯 markdown 文件**(SKILL.md + references/templates 子目录),与加载代码完全解耦。所以 **skill 内容不变,加载代码可以全替换**。

**要做的事**:
1. 拉新 `skills/storage/` 目录(含 `SkillStorage` ABC + `LocalSkillStorage` 实现)
2. 改 `skills/__init__.py` 的 export
3. 删除旧 `skills/manager.py` `installer.py` `loader.py`
4. 更新调用方(约 5-8 处):
   - `agents/lead_agent/prompt.py:14` `from deerflow.skills import load_skills` → `get_or_new_skill_storage().load_skills()`
   - `app/gateway/routers/skills.py` 整个 router
   - 测试: `test_skills_installer.py`、`test_skills_custom_router.py`、`test_client_e2e.py:532`、`test_lead_agent_prompt.py:68/104/121`
5. **保留** `packages/agent/skills/custom/` 5 个目录原样
6. **保留** `extensions_config.example.json` 的 `skills: {}` 配置接口

**冲击文件 + diff**:
- `skills/manager.py` (160 lines noldus diff) — 删除
- `skills/installer.py` (168 lines) — 部分迁移到 `storage/local_skill_storage.py`
- `skills/loader.py` (104 lines) — 删除
- `tools/skill_manage_tool.py` (148 lines) — 调用方更新
- `agents/lead_agent/prompt.py` — 1 处 import + 1 处调用

**估算**: 4-6 小时(主要是测试更新)

### D. Tier 4 BC 持久化层 (11 commit)

**核心 commit**:
- `d8ecaf46` persistence scaffold #1930 — **新增 `persistence/` 目录** (alembic + DatabaseConfig)
- `56d5fa33` unified persistence #2134 — 加 event store + thread_meta + run repo
- `2e05f380` per-user filesystem isolation #2153 — 加 user_id 参数(默认 None,BC)
- `35ef8b7c` 默认 DB config (19 行)
- `16aedf45` `897dae54` `829e82a9` lint 修复
- `898f4e8a` memory cache corruption 修复
- `87609374` memory I/O `asyncio.to_thread`
- `35f141fc` checkpoint rollback on cancel(改 `runtime/runs/worker.py`,noldus 230 行定制,需 surgical)
- `ca3332f8` gateway ISO 8601 timestamps(改 persistence/thread_meta + runtime/runs/manager + utils/time.py)

**配置策略**: 在 `config.yaml` 加 `database: { backend: memory }` 默认值 — 不需要装数据库,行为和现在一样。上云时切 sqlite/postgres。

**估算**: 5-7 小时

---

## 7. 轮 3 backlog (再下次会话)

### F. T3 体量大但有价值 (8 commit)

按你的"激进"原则,原本被标 ❌ 的也要拉:

- `d02f762a` token usage display modes (398 行 token_usage_middleware + client) — 拉
- `105db009` 每 response token usage 显示 — 拉
- `31a3c9a3` client list_threads/get_thread — 拉
- `55bc09ac` mounted sandbox upload — 拉(虽然 noldus 用 LocalSandbox,但加上不影响)
- `fc94e90f` setup-agent 失败防数据丢失 — 拉
- `6dce26a5` skill parser/tool 重构 — 拉(注意与 E 的 skill storage 协调)
- `ac04f270` subagents config.yaml 模型 override — 拉
- `55474011` task_tool 继承 parent tool_groups — 拉

### G. 中间件链相关 surgical merge (6 commit)

这组都改 `lead_agent/agent.py`(noldus 257 行重定制)和 `subagents/executor.py`,需要细心:

- `8ba01dfd` thread app_config through lead/subagent — surgical
- `38714b6c` thread app_config through middleware factories — surgical
- `e82940c0` thread release config through lead — surgical
- `b9709934` read lead options from context — surgical (14 行)
- `b8bc4826` release config in gateway runtime(改 runtime/checkpointer 跟 D 一起) — surgical
- `83938cf3` subagent propagate user context across threaded execution — surgical(可与 E 的 user_id 体系协调)
- `487c1d93` subagent model override for tools/middleware — surgical
- `d78ed5c8` subagent skill allowlist 继承 — surgical(可能与 E 的 skill storage 重构冲突,后做)
- `30d619de` per-subagent skill loading (#2253) — surgical(配合 E)

### H. 复审 Skip 类(原本被低估)+ Phase 2 立项

- `189b8240` sandbox no_change_timeout 防 120s 提前终止 — 检查与 noldus extra_env 冲突,**应该拉**
- `c09c3345` 从 project root 解析 runtime paths — 检查与 noldus paths.py 冲突,**应该拉**
- `78633c69` agent_name 传入 ToolRuntime — 拉
- `8b61c94e` lead agent graph factory 兼容性 — 拉
- `dc50a7fd` LocalSandbox read_file/write_file 路径解析 — 拉
- `950821cb` local_backend 用 subprocess 替 os.system — 拉
- `02569136` sandbox 安全 + 多模态保留 — 拉
- `74081a85` Docker port bind loopback — 看 noldus LocalSandbox 是否需要,可能拉
- `39c5da94` local custom mount symlink — 与 noldus extra_env 冲突,**surgical**
- `f394c0d8` mcp 自定义 interceptor — 与 noldus 截断定制冲突,**surgical**
- `7e8e2937`(等)... 还有几个 sandbox/mcp 改进

**Phase 2 立项**(下下次会话):
- `94eee95f` 12 blocker auth release-validation
- `848ace98` first-boot setup wizard
- `da174dfd` gateway internal auth + CSRF
- `4e4e4f92` auth + journal logic 修复
- `98a5b34f` better-auth dep cleanup
- 配套前端登录 UI(在 `packages/agent/frontend/`)

预计 Phase 2 是 2-3 周冲刺,不在本次同步范围。

### 永久跳过(确认与目标无关)

- `844ad8e5` merge release/2.0-rc 进 main(无独立价值)
- `395c1435` `2bb1a2df` MindIE
- `eef0a6e2` Setup Wizard / doctor command
- `5350b2fb` Exa 搜索
- `44ab21fc` Serper 搜索
- `4ead2c6b` config 热重载 singleton reset(明显是 v2.0-rc 配套)
- `59c4a3f0` custom-agent self-update + user isolation(与 v2.0-rc 配套)
- `563383c6` agent prompt file-io guidance(改 prompt.py 受保护文件,价值低)
- `4ba3167f` flush memory before summarization(配套 BeforeSummarizationHook)
- `07fc25d2` memory updater async LLM
- `a7e7c6d6` custom-agent management API 默认禁用
- `680187dd` RemoteSandboxBackend list_running(noldus 用 LocalSandbox)
- `c0da2782` memory event-loop 修复(noldus 已合自版 82731aeb)

---

## 8. 重要约束(再次强调,所有轮都遵守)

### 8.1 Noldus 价值清单(永远保留)

按你 2026-05-07 的明确确认,Noldus 价值在以下 4 类:

1. **Skill 内容**: `packages/agent/skills/custom/` 5 个目录 (`compaction-recovery`、`ethoinsight`、`ethoinsight-code`、`ethoinsight-charts`、`ethoinsight-planning`) 的全部 markdown 文件
2. **流程/提示词**:
   - `agents/lead_agent/prompt.py` 中文调度规则 + Gate 反问 + subagent 描述 + EV19 模板路径
   - subagent 系统 prompt(在 `subagents/builtins/<name>/`)
   - tool description 中文化
3. **Subagent 名字与注册**:`subagents/builtins/__init__.py` 注册的 4 个 ethoinsight 子代理
4. **关键 setting**:
   - `sandbox/sandbox.py` 的 `extra_env` 参数
   - `sandbox/local/local_sandbox.py` 的 venv PATH + `DEERFLOW_PATH_*`
   - `sandbox/tools.py` 的 `{{shared://}}` 占位符
   - `config/paths.py` 的 `/mnt/shared` + `shared_dir()`
   - `agents/thread_state.py` / `thread_data_middleware.py` 的 `shared_path` 字段
   - `agents/middlewares/llm_error_handling_middleware.py` 的总超时 + 多种 timeout 关键字
   - `mcp/tools.py` 的 4096 字符截断
   - `subagents/executor.py` 的 `recursion_limit` + `max_turns` 硬限制
   - `agents/lead_agent/agent.py` 的中间件链(`ArchivingSummarizationMiddleware`、`ThinkTagMiddleware`、`TrainingDataMiddleware`、`GateEnforcementMiddleware`、`LoopDetectionMiddleware._loop_detection`)

### 8.2 不在保留范围内(可随上游)

- Skill **加载代码**(不是内容)
- 通用工具/中间件实现细节
- 数据持久化层(欢迎引入)
- 上云需要的基础设施

### 8.3 Tier 4 红线(上云前不要拉)

- `94eee95f` `848ace98` `da174dfd` `4e4e4f92` `98a5b34f` — better-auth 全套(后端 middleware + 前端 UI)
- 这部分的引入要**单独立项**,不混进同步工作

---

## 9. 测试约束

### 9.1 已知 pre-existing 失败(不要试图修)

```
tests/test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config
tests/test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer
```

经 git stash 实测,与同步工作无关。**轮 1/2/3 完成后,失败数应仍为 2**。

### 9.2 每个 commit 后跑测试

不要批量改完才测。**每个独立 commit 改完都跑一次**:

```bash
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
```

预期值: `2 failed, X passed, 14 skipped`,X 应该单调增长(因为我们也合上游测试)。

---

## 10. Commit 消息约定

按交接文档的 commit 拆分:
- A 阶段一个 commit(19 commit 合并 → 1 个 noldus commit)
- B 阶段一个 commit
- C 阶段一个 commit
- 轮 2 / 轮 3 同样按 phase 拆 commit

每个 commit 消息格式:
```
sync deerflow upstream <Tier>: <一句话总结> N commit

- 分组1: sha1 sha2 ...
- 分组2: sha3 sha4 ...

保留 Noldus 全部定制(<列出关键定制点>)
详见 docs/handoffs/2026-05-07-deerflow-tier234-execution-plan.md §X.Y
```

**不要 push** — 等用户决定。

---

## 11. 下一位 Agent 的第一步

1. **读这份文档**(你正在做)
2. **读上一份交接** [2026-05-06-deerflow-upstream-sync-handoff.md](2026-05-06-deerflow-upstream-sync-handoff.md) 了解 Tier 1 已合内容
3. **读 CLAUDE.md L123-L174** 上游同步核心规则
4. **跑 §3.4 验证 sync remote 配置正确**
5. **跑当前测试基线**:
   ```bash
   cd /home/wangqiuyang/noldus-insight/packages/agent/backend
   PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
   ```
   应看到 `2 failed, 1714+ passed`
6. **开始轮 1 步骤 A**(§4.1) — 19 个 T1-sibling 直接合
7. **每完成一个步骤 commit 一次**
8. **每完成一轮写 handoff,等用户 review**

---

## 12. 附录:数据文件

- [data/all-105-commits-classified.txt](2026-05-07-tier234-walkthrough-data/all-105-commits-classified.txt) — 全部 105 个 commit 的最终分类
- [data/raw-105-commits.txt](2026-05-07-tier234-walkthrough-data/raw-105-commits.txt) — 上游 commit 标题原始 list

格式:
```
<sha>|<tier>|<reason>|<title>
```

可用 `awk -F'|' '{print $2}' all-105-commits-classified.txt | sort | uniq -c` 看分布。

---

## 13. 风险与注意事项

### ⚠️ 受保护文件 surgical merge 是高风险区

`subagents/executor.py`、`agents/lead_agent/agent.py`、`sandbox/tools.py`、`llm_error_handling_middleware.py` 这几个文件 noldus 重度定制,合并出 bug 的概率最高。**逐函数对比**,**测试驱动**,**不批量**。

### ⚠️ 不要无脑 `git show <sha> > <file>` 覆盖

即使是 T1-sibling,也要先 `diff` 看 noldus 是否定制过该文件。"T1-sibling" 这个分类是按 commit 性质,不保证 commit 改的每个文件 noldus 端都未定制。

### ⚠️ skill storage 重构(轮 2 E)是 high-stakes

冲击 noldus prompt.py 的 `load_skills()` 调用、整个 gateway/routers/skills.py、3+ 测试文件。做之前先把 noldus 5 个 skill **复制一份到 /tmp 备份**,再开始。

### ⚠️ Tier 4 BC 部分(轮 2 D)新增 `persistence/` 目录

虽然现在 noldus 0 import,但拉了之后 `import deerflow.persistence` 至少要跑通(空操作)。要确认不会因为缺 alembic 配置导致启动失败。配 `database: { backend: memory }` 默认值。

### ⚠️ 测试基线偏移要立即排查

如果某个 commit 合完后失败数从 2 涨到 3+,**立即停下查根因**,不要继续往下做。

### ⚠️ 不要 force push,不要 push origin

所有 commit 留在本地 `dev` 分支,等用户决定是否 push。

---

## 14. 与上次交接文档的关系

本文档**取代**上次交接文档的 §5 P1/P2/P3 部分(Tier 2 手工合并清单 + Tier 3 candidates),其余部分(§1-§4 + §6-§9)仍然有效。

主要差异:
1. 上次只列了 Tier 2 的 4 个文件,本次列出**所有 105 个 commit 的精确分类**
2. 上次判 Tier 4 一律跳过,本次按"上云"目标**重新评估并大部分采纳**
3. 上次没识别 noldus 已部分吸收 Tier 4,本次确认**只缺 3 个目录,无依赖**
4. 上次工作量评估不准,本次**分 3 轮**,每轮 4-7 小时

---

## 15. 完成定义 (Done Criteria)

### 轮 1 完成
- [ ] A.1 + A.2 + A.3 (19 个 T1-sibling) 已实施 + commit
- [ ] B.1 ~ B.8 (9 个 T2-keep surgical merge) 已实施 + commit
- [ ] C.1 ~ C.5 (5 个 T3 ⭐⭐) 已实施 + commit
- [ ] 全测试套 = 2 failed (pre-existing) + 1714+ passed
- [ ] `make lint` 0 error
- [ ] `make dev` 启动成功,gateway + langgraph 正常
- [ ] 写新 handoff 文档,等用户 review

### 整体 33 commit 完成后
- 上游已采纳: 11 (T1-DONE) + 33 (轮 1) = **44 commit**
- 剩余: 105 - 44 = **61 commit** 在轮 2/3

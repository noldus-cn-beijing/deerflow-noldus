# DeerFlow 上游 Tier 2/3/4 同步 - 轮 2 设计完成交接

**日期**: 2026-05-07
**交接人**: Claude (本会话, opus-4-7-1m)
**接手对象**: 下一位 AI Agent (执行实施)
**任务状态**: 🟢 设计 + Plan 完成,等执行
**前置依赖**: [2026-05-07-deerflow-tier234-round1-completed-handoff.md](2026-05-07-deerflow-tier234-round1-completed-handoff.md)

---

## 0. TL;DR

本会话完成轮 2 的**设计 + 实施 plan**,**没有写代码**。用户决定:

1. **走上游 better-auth 路线**(原生 deerflow 自带的 auth 体系),弃用 noldus 自建 user-backend 计划
2. **本轮做 E + C.5 + D**(纯拉上游同步,10-15h)
3. **下一轮做 better-auth + 配套 wiring**(2-3 周冲刺,不在本轮)
4. 用户明说交给"能力比我差一点的 agent"执行,plan 必须高度可执行,无 placeholder

**产出 2 个文档**:
- 设计 spec: `docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md` (13KB)
- 实施 plan: `docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md` (72KB,38 个 task,每个 task 含 5-13 步)

**当前 git 状态**: dev 分支 HEAD 仍是 `2c0db62b` (轮 1 末尾),新 spec/plan **还未 commit**。

---

## 1. 关键决策(本会话定下,继续工作时不要质疑)

### 1.1 走上游 better-auth,弃用自建 user-backend

**用户原话**: "本着不想重复造轮子的最大规则" → 选 "走上游 better-auth"。

**调研结论**:上游 deerflow 已经手写了一整套 auth 体系(不是开源 better-auth.js 库,是上游自命名),包括:
- `94eee95f` 后端 auth 核心: JWT + bcrypt + SQLite UserRepository + Provider Factory + reset_admin CLI + 12 单测
- `848ace98` first-boot setup wizard + unified persistence (#1930 同 commit 内)
- `da174dfd` Gateway process-local internal auth + CSRF
- `98a5b34f` better-auth 前端依赖清理 (frontend/pnpm-lock.yaml + backend/uv.lock 改 2598 行)
- `4e4e4f92` 安全加固 (SHA-256 预哈希、错误信息脱敏、setup-status rate limit)

**对比 noldus 4-23 user-backend 计划**:noldus 自建路线本质是**重复造轮子**(自己写 LocalAuth/JWT/前端登录页/nginx 路由),上游已就绪整套且更完善。

**已弃用**:`docs/plans/2026-04-23-multi-user-deployment.md`(Plan F.2 会标记 deprecated)。

### 1.2 本轮范围:E + C.5 + D,不动 auth

| 阶段 | 内容 | 工作量 |
|---|---|---|
| **E** | Skill storage 重构 (`1ad1420e`) | 4-5h |
| **C.5** | Summarization skill rescue (`f9ff3a69`) | 1-2h |
| **D** | Tier 4 BC 持久化层 11 commit | 5-7h |
| **F** | 收尾(handoff + deprecate 文档) | 30min |

总计 **10-15h**, 产出 **3 + 2 = 5 个 commit** (3 个代码 commit + 2 个文档 commit)。

### 1.3 D 阶段默认 memory backend

不强制启用 SQLAlchemy/Postgres。`config.yaml` 加 `database: { backend: memory }`,运行时仍是内存模式,**noldus 现有行为完全不变**。轮 3 上 better-auth 时再切 sqlite/postgres。

### 1.4 v0.1 目标已升级

**用户原话**: "我们不应该只将目标停留在单用户研究助手这里。我们应该有一个长远的目标。"

CLAUDE.md L37 第 5 条 "v0.1 单用户研究助手" 的判断**已过时**,但本会话**没修 CLAUDE.md**(等执行完 plan 再统一更新)。

---

## 2. 关键调研结果(执行 plan 时重要)

### 2.1 noldus subtree 现状

```
HEAD = 2c0db62b (轮 1 handoff)
deerflow remote = git@github.com:noldus-cn-beijing/deerflow-noldus.git
deerflow/main 起点 = 4ead2c6b (上游 head)
```

工作区: 干净。但有 2 个未跟踪文件等 commit:
- `docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md`
- `docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md`

### 2.2 关键文件 noldus diff vs upstream(执行时必看)

| 文件 | noldus diff | 处理策略 |
|---|---:|---|
| `agents/lead_agent/prompt.py` | **1051 行** | E 阶段必须 surgical edit,只改 2 处 (L14 + L29) |
| `gateway/routers/threads.py` | **614 行** | D 阶段必须 surgical merge,只挑 timestamp 转换 |
| `subagents/executor.py` | 455 行 | E 阶段只改 4 行 (skills.loader → skills.storage),保留 recursion_limit/max_turns/{{shared://}} |
| `agents/lead_agent/agent.py` | 257 行 | C.5 阶段只加 1 个 kwarg (skills_container_path),保留中间件链 |
| `runtime/runs/worker.py` | 230 行 但 **0 noldus 定制标记** | D.8 阶段**可整覆盖**(实测 grep noldus/EthoInsight/shared_dir/`{{shared://}}` 全 0) |
| `config/paths.py` | 98 行 | D.3 阶段 surgical,保留 `/mnt/shared` + `shared_dir()` |
| `agents/memory/storage.py` | **0 行** | 已吸收上游,跳过 |
| `agents/memory/queue.py` | **0 行** | 已吸收上游,跳过 |
| `runtime/user_context.py` | **0 行** | 已吸收上游,跳过 |

### 2.3 D 阶段引入的目录(noldus 之前没有)

```
packages/agent/backend/packages/harness/deerflow/
├── persistence/                     ✨ 新 (21 个文件,含 SQLAlchemy ORM/migrations/feedback/user)
├── runtime/checkpointer/            ✨ 新 (从 agents/checkpointer/ mv,保留 shim)
├── runtime/events/store/             ✨ 新 (RunEventStore ABC + memory/db/jsonl)
├── runtime/runs/store/               ✨ 新 (RunStore ABC + MemoryRunStore)
├── runtime/journal.py                ✨ 新 (RunJournal: BaseCallbackHandler)
├── config/database_config.py         ✨ 新
├── config/run_events_config.py       ✨ 新
└── utils/time.py                     ✨ 新 (ISO 8601 helpers)
```

⚠️ **noldus 现有 `agents/checkpointer/` 不删,保留为 re-export shim**。多处旧 import (例 `app/gateway/deps.py:28` `langgraph.json:12` `tests/test_checkpointer*.py`) 才能继续工作。

### 2.4 E 阶段 skills/ 调用方完整清单

**生产代码**(执行时全部要改):
- `agents/lead_agent/prompt.py:14, :29` (surgical, prompt 受保护)
- `client.py:43, :754, :874, :901` (4 处)
- `config/skills_config.py:39`
- `subagents/executor.py:122, :124` (surgical, 受保护)
- `tools/skill_manage_tool.py:17` (整覆盖)
- `app/gateway/routers/skills.py:12, :13, :14, :105, :140, :150, :295, :317, :343` (整覆盖)

**测试代码**(10+ 文件):
- `conftest.py / test_client.py / test_client_e2e.py / test_lead_agent_prompt.py / test_lead_agent_skills.py / test_local_sandbox_provider_mounts.py / test_skill_manage_tool.py / test_skills_custom_router.py / test_skills_installer.py / test_skills_loader.py`

**新增**: `test_local_skill_storage_write.py` (162 行)

### 2.5 测试基线

```
当前: 2 failed, 1811 passed, 14 skipped
预期 (轮 2 完成): 2 failed, ≥1900 passed, 14 skipped (X 增加 80-160)
```

**Pre-existing failures (永远不要碰)**:
- `test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config`
- `test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer`

---

## 3. 已完成事项

- [x] 跟用户对齐路线(走上游 better-auth, 弃用自建 user-backend)
- [x] 调研上游 better-auth 5 个 commit 实际形态
- [x] 调研 D 阶段 11 个 commit 范围 + noldus diff
- [x] 调研 E 阶段所有 skills/ 调用方
- [x] 写设计 spec (`docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md`)
- [x] 写实施 plan (`docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md`,38 个 task)

---

## 4. 未完成事项 (执行 agent 的工作)

按 plan 顺序执行:

### 高优先级

1. **Phase 0: 基线确认** (Task 0, 5min) — 确认 HEAD 是 `2c0db62b`,基线 1811 passed
2. **Phase E: Skill storage 重构** (Task E.1 - E.7, 4-5h)
3. **Phase C.5: Summarization skill rescue** (Task C.5.1 - C.5.3, 1-2h)
4. **Phase D: Tier 4 BC 持久化层** (Task D.1 - D.10, 5-7h)
5. **Phase F: 收尾** (Task F.1 - F.2, 30min)

### 不要在轮 2 做 (留轮 3)

- 上游 better-auth 全套 (`94eee95f / 848ace98 / da174dfd / 98a5b34f / 4e4e4f92`)
- 配套 7 wiring (`8ba01dfd / 38714b6c / e82940c0 / b8bc4826 / 83938cf3 / 30d619de / 487c1d93`)
- 前端登录页中文化 + EthoInsight 品牌
- 写新 multi-user 部署 SOP

---

## 5. 建议接手路径

### 第一步: 读 plan 和 spec

```bash
cd /home/wangqiuyang/noldus-insight
cat docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md  # 13KB,5min 读完
cat docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md   # 72KB,30min 读完
```

### 第二步: 验证基线 (Plan Task 0)

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -1                        # 期望 HEAD = 2c0db62b
git status                                  # 期望干净 (已加未跟踪 spec/plan 文件可忽略)
git remote -v | grep deerflow               # 期望存在 deerflow remote
git fetch deerflow main                     # 拉最新

cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
# 期望 2 failed, 1811 passed, 14 skipped
```

如果基线 ≠ `2 failed, 1811 passed`,**立即停下问用户**。

### 第三步: 选择执行模式

**推荐**: 用 `superpowers:subagent-driven-development` skill,fresh subagent 每 task,主 agent 在 task 之间 review。中断点多但安全。

**备选**: 用 `superpowers:executing-plans` skill,inline 执行,适合连续工作场景。

### 第四步: 严格按 plan 顺序执行

不要跳 task,不要跳 step。Plan 中带 ⚠️ 标记的"立即停下问用户"触发条件出现时,**立即停下,不要试图自己解决**。

### 第五步: 每个 commit 之后跑全量测试

测试失败数从 2 涨到 3+,**立即停下**(plan §应急处置 §"测试失败数从 2 涨到 3+")。

---

## 6. 关键路径速记

```bash
NOLDUS=/home/wangqiuyang/noldus-insight
BACKEND=$NOLDUS/packages/agent/backend
HARNESS=$BACKEND/packages/harness/deerflow
GATEWAY=$BACKEND/app/gateway
TESTS=$BACKEND/tests
SKILLS_CUSTOM=$NOLDUS/packages/agent/skills/custom

# 测试基线命令
cd $BACKEND && PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5

# Lint
cd $BACKEND && PYTHONPATH=. uv run ruff check packages/harness/deerflow/

# 启动
cd $NOLDUS/packages/agent && make dev   # localhost:2026
cd $NOLDUS/packages/agent && make stop

# 查看上游 commit
cd $NOLDUS && git show <sha> --stat | head -50
cd $NOLDUS && git show <sha> -- '<path>' | head -50
cd $NOLDUS && git show deerflow/main:<path>            # 看上游某文件最新版

# 比较 noldus vs upstream
cd $NOLDUS && diff <(git show deerflow/main:<upstream-path>) <local-path>
```

---

## 7. Noldus 价值清单 (执行时永远要保留)

按用户 2026-05-07 明确确认:

1. **Skill 内容**: `packages/agent/skills/custom/` 5 个目录 (`compaction-recovery / ethoinsight / ethoinsight-code / ethoinsight-charts / ethoinsight-planning`) 的全部 markdown
2. **Prompt**: `agents/lead_agent/prompt.py` 中文调度规则 + Gate 反问 + subagent 描述 + EV19 模板路径
3. **Subagent 名字**: `subagents/builtins/__init__.py` 注册的 4 个 ethoinsight 子代理
4. **关键 setting**:
   - `sandbox/sandbox.py` `extra_env` 参数
   - `sandbox/local/local_sandbox.py` venv PATH + `DEERFLOW_PATH_*`
   - `sandbox/tools.py` `{{shared://}}` 占位符 + `mask_local_paths_in_output` + `SHARED_PATH_PREFIX`
   - `config/paths.py` `/mnt/shared` + `shared_dir()`
   - `agents/thread_state.py` / `thread_data_middleware.py` `shared_path` 字段
   - `agents/middlewares/llm_error_handling_middleware.py` 总超时 + circuit breaker
   - `mcp/tools.py` 4096 字符截断
   - `subagents/executor.py` `recursion_limit` + `max_turns` + `{{shared://}}`
   - `agents/lead_agent/agent.py` 中间件链 (Archiving/ThinkTag/TrainingData/GateEnforcement/LoopDetection)

---

## 8. 风险与注意事项

### ⚠️ 不要整文件覆盖受保护文件

CLAUDE.md L141-L155 列了禁止 import 的 Tier 4 模块,但本轮**故意打破** L141 那条边界(因为用户决定上 Tier 4 BC),所以:

- ✅ 允许 import `from deerflow.persistence import ...` (D.1 引入)
- ✅ 允许 import `from deerflow.runtime.checkpointer import ...` (D.2 引入)
- ✅ 允许 import `from deerflow.runtime.events import ...` (D.1 引入)
- ❌ **不要** import `from deerflow.runtime.journal import ...` 在 noldus 受保护文件中(留给轮 3 better-auth)
- ❌ **不要** import `from deerflow.utils.time import ...` 替换 noldus 现有时间处理(只在新文件用)
- ❌ **不要** 启用 SQLite/Postgres backend(默认 memory,轮 3 才切)

### ⚠️ Plan 中标"立即停下问用户"的地方,不要试图自己解决

包括但不限于:
- HEAD 不是 2c0db62b
- 工作区不干净
- 基线不是 2 failed, 1811 passed
- diff 太大不能简单整覆盖
- noldus 定制 grep check 数字 < 阈值
- 启动失败
- ImportError 排查不清楚根因

### ⚠️ 部分 step 故意保留"看 commit 后判定"

例如 Task E.3 step 1 / E.6 step 4 等。**这不是 placeholder**,而是因为某些 noldus 文件的具体定制需要执行时 grep 后判定。每处都给了明确的判定准则(diff 行数 + grep 关键字 + 数字阈值 + 备选行动)。**严格按判定准则执行,不要凭直觉**。

### ⚠️ D.1 拉 persistence/ 后 import 可能报错

如果 `from deerflow.persistence import ...` 报 SQLAlchemy 缺失,**不要立即 pip install**。先检查 `packages/agent/backend/pyproject.toml` 和 `packages/agent/backend/packages/harness/pyproject.toml` 是否需要加依赖。上游应该已加,如果 D.1 没拉到,执行 `cd $BACKEND && uv sync`。

### ⚠️ agents/checkpointer/ shim 可能让 mock path 失效

`tests/test_checkpointer.py` 等如果用 `patch("deerflow.agents.checkpointer.async_provider.xxx")` 形式 mock,在 D.2 把 `agents/checkpointer/` 改 shim 后,可能 module identity 变化导致 mock 失败。Plan 中 D.2 step 7 标了:**临时方案**改 shim 为 `import + 直接赋值` 而非 `from ... import *`。**改之前问用户**。

### ⚠️ 用户会同时做端到端测试

用户上次原话: "我自己手动进行端到端测试"。这意味着执行 agent **每次 commit 后** 都可能被用户要求停下让他测试。给用户预留缓冲时间,**不要 commit 后立即开下一阶段**,等用户确认。

---

## 9. 下一位 Agent 的第一步

### Step 1: 读这份交接 + spec + plan

```bash
cd /home/wangqiuyang/noldus-insight
cat docs/handoffs/2026-05-07-deerflow-tier234-round2-design-completed-handoff.md  # 本文件
cat docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md
cat docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md
```

### Step 2: Commit spec 和 plan(避免后续混淆)

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md \
        docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md \
        docs/handoffs/2026-05-07-deerflow-tier234-round2-design-completed-handoff.md

git commit -m "$(cat <<'EOF'
docs: 轮 2 deerflow 同步设计文档 + 实施 plan + 设计交接

- spec: docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md
- plan: docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md (38 task)
- handoff: docs/handoffs/2026-05-07-deerflow-tier234-round2-design-completed-handoff.md

决策:
- 走上游 better-auth (轮 3 做)
- 弃用 docs/plans/2026-04-23-multi-user-deployment.md
- 本轮做 E + C.5 + D BC 持久化层
- D 默认 memory backend, noldus 行为不变

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Step 3: 跑 Plan Task 0 验证基线

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, 1811 passed, 14 skipped`。

### Step 4: 决定执行模式后开始 Phase E

按 plan §Phase E Task E.1 开始。每完成一个 task 跑相关单测,每个 commit 后跑全量测试。

### Step 5: 如果用户同时在做端到端测试

每个 commit 后停下问用户: "已完成 Task X.Y,是否继续下一阶段?",给用户测试时间。

---

## 10. 不要做的事

- ❌ 不要 push origin (用户明说留 dev 等他决定)
- ❌ 不要 push deerflow remote
- ❌ 不要修 pre-existing 2 failures
- ❌ 不要在轮 2 引入 better-auth / first-boot wizard / 前端登录
- ❌ 不要启用 sqlite/postgres backend (默认 memory)
- ❌ 不要修 noldus 4-23 user-backend 计划文件之外的事(只是加 deprecated 标记)
- ❌ 不要修 CLAUDE.md(等轮 3 完成后统一更新)
- ❌ 不要试图"清理" noldus skills/ 目录(5 个 custom skill 是项目核心价值)

---

## 11. 完成定义

执行 plan 完成后,验证:

- [ ] 5 个本地 commit (3 代码 + 2 文档),HEAD 不是 2c0db62b
- [ ] `git log --oneline -7` 看到完整序列
- [ ] 测试基线: `2 failed, ≥1900 passed, 14 skipped`
- [ ] `make lint` 0 error
- [ ] `make dev` 启动成功
- [ ] noldus 价值清单全部 grep check 通过
- [ ] 写新 handoff `docs/handoffs/2026-05-07-deerflow-tier234-round2-completed-handoff.md`
- [ ] **不 push** — 等用户决定

---

## 12. 参考

- 设计 spec: [docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md](../superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md)
- 实施 plan: [docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md](../superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md)
- 轮 1 完成 handoff: [2026-05-07-deerflow-tier234-round1-completed-handoff.md](2026-05-07-deerflow-tier234-round1-completed-handoff.md)
- 轮 1 执行计划: [2026-05-07-deerflow-tier234-execution-plan.md](2026-05-07-deerflow-tier234-execution-plan.md)
- 105 commit 分类: [2026-05-07-tier234-walkthrough-data/all-105-commits-classified.txt](2026-05-07-tier234-walkthrough-data/all-105-commits-classified.txt)
- 弃用的 user-backend 计划: [../plans/2026-04-23-multi-user-deployment.md](../plans/2026-04-23-multi-user-deployment.md)
- noldus 上游同步规则: [../../CLAUDE.md](../../CLAUDE.md) L123-L174

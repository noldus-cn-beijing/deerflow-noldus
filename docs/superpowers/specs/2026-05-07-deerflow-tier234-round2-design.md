# DeerFlow 上游 Tier 2/3/4 同步 - 轮 2 设计文档

**日期**: 2026-05-07
**状态**: 已批准，待实施
**前置依赖**: [2026-05-07-deerflow-tier234-round1-completed-handoff.md](../../handoffs/2026-05-07-deerflow-tier234-round1-completed-handoff.md)

---

## 0. 目标

按"上游能用就用，已造好的就别再造"的原则，本轮完成 deerflow 上游的：

1. **E. Skill storage 重构**（`1ad1420e`，4-6h）— 替换 noldus 旧的 `manager.py / installer.py / loader.py` 三件套为新的 `storage/` 抽象。
2. **C.5. Summarization skill rescue**（`f9ff3a69`，1-2h）— 紧跟 E 之后做，依赖新的 skill storage 路径检测。
3. **D. Tier 4 BC 持久化层**（11 个 commit，5-7h）— 引入 `persistence/`、`runtime/checkpointer/`、`runtime/events/` 三个新目录，配 `database: { backend: memory }` 默认值，noldus 现状不破坏。

总计 **10-15h**，目标产出 **3 个 commit**（每阶段一个）。

**显式不做**：
- 上游 better-auth 系列（`94eee95f`、`848ace98`、`da174dfd`、`98a5b34f`、`4e4e4f92`）— 留给 **轮 3**（独立 2-3 周冲刺）
- 前端登录 UI / setup wizard / CSRF / rate limit — 跟 better-auth 一起做
- noldus 自建 user-backend 路线（`docs/plans/2026-04-23-multi-user-deployment.md`）— **弃用**，被上游 better-auth 取代
- D 之外的 Tier 4 wiring（`8ba01dfd`、`38714b6c`、`e82940c0`、`b8bc4826`、`83938cf3`、`30d619de`、`487c1d93`）— 配套 better-auth 一起做

---

## 1. 关键决策

### 1.1 走上游 better-auth，不再自建 user-backend

调研了 5 个 commit 的代码量后确认：上游 `94eee95f` 已经手写了完整 auth 体系（JWT + bcrypt + SQLite UserRepository + Provider Factory + reset_admin CLI + 12 单测），`848ace98` 加了 first-boot setup wizard，`da174dfd` 加了 process-local internal auth + CSRF，`98a5b34f` 引入前端 better-auth 依赖，`4e4e4f92` 做了安全加固（SHA-256 预哈希、错误信息脱敏、setup-status 的 rate limit）。

**结论**：noldus 4-23 的 user-backend 16-task 计划是重复造轮子，弃用。上游已就绪，留给轮 3 整体合入。

### 1.2 D 阶段保留默认 memory backend

虽然引入 `persistence/`、`runtime/checkpointer/`、`runtime/events/` 三个目录，但不强制启用 SQLAlchemy/Postgres。`config.yaml` 加 `database: { backend: memory }`，运行时仍是内存模式，**noldus 现有行为完全不变**。轮 3 上 better-auth 时再切 sqlite/postgres。

### 1.3 E 阶段保留 noldus 5 个 custom skill 内容

`packages/agent/skills/custom/` 5 个目录是 **markdown 内容**，与加载代码完全解耦。`1ad1420e` 重构的是加载代码（`manager.py / installer.py / loader.py` → `storage/skill_storage.py + storage/local_skill_storage.py`），不动 markdown。

---

## 2. 架构变化

### 2.1 E 阶段后的 skills/ 目录结构

```
packages/agent/backend/packages/harness/deerflow/skills/
├── __init__.py            # 改：export 接口变更（旧 load_skills → 新 get_or_new_skill_storage）
├── parser.py              # 微改（11 行）
├── security_scanner.py    # 微改（3 行，配合 trace run_name）
├── types.py               # 微改（16 行，加新字段）
├── validation.py          # 微改（6 行）
└── storage/               # 新增目录
    ├── __init__.py
    ├── skill_storage.py        # ABC + 254 行
    └── local_skill_storage.py  # 实现 + 198 行
```

**删除**:
- `manager.py`（161 行 deprecated）
- `installer.py`（86 行 deprecated）
- `loader.py`（105 行 deprecated）

**调用方更新**:
- `agents/lead_agent/prompt.py` — 1 处 import + 1 处调用
- `tools/skill_manage_tool.py` — 92 行调整
- `subagents/executor.py` — 4 行
- `client.py` — 18 行
- `app/gateway/routers/skills.py` — 整个 router（118 行）
- `config/skills_config.py` — 10 行
- 测试文件：`test_client.py`、`test_client_e2e.py`、`test_lead_agent_prompt.py`、`test_lead_agent_skills.py`、`test_local_sandbox_provider_mounts.py`、`test_skill_manage_tool.py`、`test_skills_custom_router.py`、`test_skills_installer.py`、`test_skills_loader.py`、`conftest.py`

### 2.2 C.5 阶段后的 summarization

```
agents/middlewares/summarization_middleware.py
├── 现有：DeerFlowSummarizationMiddleware
├── 现有：BeforeSummarizationHook 协议
├── 现有：memory_flush_hook、archive hook
└── 新增：skill rescue 逻辑（lift skill bundles before summarization）
```

新增 4 个 config 项到 `summarization_config.py`：
- `preserve_recent_skill_count`
- `preserve_recent_skill_tokens`
- `preserve_recent_skill_tokens_per_skill`
- `skill_file_read_tool_names`

`agents/lead_agent/agent.py` 工厂函数注入 `skills_container_path`（依赖 E 的新 skill storage 接口）。

### 2.3 D 阶段后的目录结构

```
packages/agent/backend/packages/harness/deerflow/
├── persistence/                # 新增
│   ├── __init__.py
│   ├── engine.py              # async engine lifecycle
│   ├── base.py                # DeclarativeBase + to_dict mixin
│   ├── alembic/               # alembic skeleton
│   ├── models/                # ORM 模型
│   │   ├── run.py             # RunRow + token 字段
│   │   ├── thread_meta.py     # ThreadMetaRow
│   │   └── run_event.py       # RunEventRow
│   └── repositories/          # 仓储模式
│       ├── run_repository.py
│       └── thread_meta_repository.py
├── runtime/
│   ├── checkpointer/          # 新增
│   │   ├── __init__.py
│   │   └── async_provider.py
│   ├── events/                # 新增
│   │   ├── __init__.py
│   │   ├── run_event_store.py # ABC + MemoryRunEventStore
│   │   └── db_run_event_store.py
│   └── runs/
│       ├── worker.py          # surgical merge（35f141fc rollback fix）
│       ├── store.py           # 新增 RunStore ABC
│       └── manager.py         # 微改
├── config/
│   ├── database_config.py     # 新增
│   └── run_events_config.py   # 新增
└── utils/
    └── time.py                # 新增（ISO 8601 helpers）
```

`config.yaml` 加：
```yaml
database:
  backend: memory  # memory | sqlite | postgres
  # url: postgresql+asyncpg://user:pass@host/db  # 上线时启用

run_events:
  store: memory  # memory | db | jsonl
```

---

## 3. D 阶段 11 commit 清单

按依赖顺序：

| # | SHA | 作用 | 风险 |
|---|---|---|---|
| D.1 | `d8ecaf46` | persistence scaffold + RunEventStore + ORM models + RunJournal | 低（纯新增） |
| D.2 | `56d5fa33` | unified persistence rebase cleanup | 低（rebase 整理） |
| D.3 | `2e05f380` | per-user filesystem isolation（user_id 参数默认 None，BC） | 中（动 storage.py 路径） |
| D.4 | `35ef8b7c` | 默认 database config | 低（19 行） |
| D.5 | `16aedf45` `897dae54` `829e82a9` | lint 修复 | 低 |
| D.6 | `898f4e8a` | memory cache corruption 修复 | 中（动 memory/queue.py） |
| D.7 | `87609374` | memory I/O 用 asyncio.to_thread | 中（动 memory/storage.py） |
| D.8 | `35f141fc` | checkpoint rollback on cancel | **高**（worker.py noldus 230 行重定制，surgical merge） |
| D.9 | `ca3332f8` | gateway ISO 8601 timestamps | 中（改 thread_meta + manager + utils/time.py） |
| D.10 | `69649d8a` | persistent part #2566 review fix 收尾 | 低 |
| D.11 | `17447fcc` | rollback restore checkpoint supersede newer checkpoints | 中 |

---

## 4. 测试策略

### 4.1 基线
- 当前: `2 failed, 1811 passed, 14 skipped`（来自轮 1）
- 目标: `2 failed, X passed, 14 skipped`（X >= 1811 + 上游测试增量）

### 4.2 每阶段后跑全量测试
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

### 4.3 关键回归测试
- E 阶段: `test_skills_custom_router.py`、`test_lead_agent_prompt.py`、`test_local_skill_storage_write.py`（新增）
- C.5 阶段: `test_summarization_middleware.py`、`test_lead_agent_model_resolution.py`
- D 阶段: 27 个 RunEventStore 单测（新增）+ noldus 现有 worker.py 测试（确保 surgical merge 不破坏）

### 4.4 已知 pre-existing failures（不要修）
- `test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config`
- `test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer`

---

## 5. 受保护文件清单（surgical merge 不能整覆盖）

按 noldus diff 行数从大到小排序：

| 文件 | noldus diff | 影响阶段 |
|---|---:|---|
| `subagents/executor.py` | 455 行 | E（4 行调整 import） |
| `agents/lead_agent/agent.py` | 257 行 | C.5（注入 skills_container_path） |
| `runtime/runs/worker.py` | 230 行 | D（35f141fc rollback fix） |
| `agents/middlewares/summarization_middleware.py` | ? | C.5（核心改动） |
| `agents/middlewares/llm_error_handling_middleware.py` | ? | 不在本轮（下轮 better-auth 配套） |
| `agents/lead_agent/prompt.py` | ? | E（1 处 import + 1 处调用） |

每个文件的 surgical merge 步骤：
1. `diff` 看上游版本
2. 提取上游"真正修复"或"接口变更"
3. 手工编辑 noldus 文件，保留全部 noldus 定制
4. 跑相关单测验证

---

## 6. Commit 拆分

```
# 后续 3 个 commit
sync deerflow upstream Tier 4 E: skill storage 重构 (1ad1420e)
sync deerflow upstream Tier 3 C.5: summarization skill rescue (f9ff3a69)
sync deerflow upstream Tier 4 D: BC 持久化层 11 commit
```

每个 commit 消息按 noldus 约定：
```
sync deerflow upstream <Tier>: <一句话总结>

- 改动内容 1
- 改动内容 2

保留 Noldus 全部定制：<列出关键定制点>
详见 docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md §X
```

不 push origin。

---

## 7. 风险与回滚

### 7.1 高风险点
1. **E 阶段调用方多**：14 个文件涉及，漏改一处会导致 import error。**对策**：先合 storage/ 新增文件，再删旧 manager/installer/loader，最后跑全量测试。
2. **D.8 worker.py 35f141fc**：noldus 230 行重定制，rollback 逻辑深度耦合 noldus `recursion_limit` + `max_turns` + `{{shared://}}`。**对策**：surgical merge 时保留 noldus 全部定制，只叠加 rollback 逻辑。先备份 noldus 版本到 `/tmp/worker_noldus_backup.py`。
3. **D.3 per-user FS isolation**：上游 storage.py 加 user_id 参数默认 None，noldus storage.py 已经吸收过上游（diff=0）。**对策**：拉新版本，确认 user_id=None 时行为与旧版一致。

### 7.2 回滚机制
每阶段独立 commit，出问题用 `git revert <commit>`。最坏情况 reset 到 `2c0db62b`（轮 1 末尾的 handoff commit）。

### 7.3 中途中断协议
如果某阶段卡住超过 2h 且测试不通过：
1. `git stash` 当前未提交改动
2. 写中断 handoff，记录卡点
3. 不 commit 半成品

---

## 8. 完成定义

- [ ] E 阶段：`storage/` 新目录就位，旧 manager/installer/loader 删除，14 文件调用方更新，`test_local_skill_storage_write.py` 新增 162 行测试通过，全量测试 `2 failed, X passed`
- [ ] C.5 阶段：summarization_middleware.py 加 skill rescue 逻辑，4 个新 config 项就位，`test_summarization_middleware.py` 新增 327 行测试通过
- [ ] D 阶段：`persistence/`、`runtime/checkpointer/`、`runtime/events/` 三个目录新增，`config.yaml` 加 `database: { backend: memory }`，27 个 RunEventStore 测试通过，worker.py surgical merge 后 noldus 现有定制全保留
- [ ] 全量测试通过：`2 failed, 1811+ passed, 14 skipped`
- [ ] `make lint` 0 error
- [ ] `make dev` 启动成功
- [ ] 写新 handoff `docs/handoffs/2026-05-07-deerflow-tier234-round2-completed-handoff.md`
- [ ] **不 push** — 等用户决定

---

## 9. 与轮 3 衔接

轮 2 完成后，下一会话做轮 3 时：
- 持久化层 ✅ 已就位（D 阶段）— better-auth 的 SQLite UserRepository 直接复用 persistence engine
- skill storage ✅ 已就位（E 阶段）— better-auth 不需要再动 skill 路径
- summarization rescue ✅ 已就位（C.5）— 不影响 auth

轮 3 实施清单：
1. 拉 `94eee95f` 后端 auth 核心模块
2. 拉 `848ace98` first-boot setup wizard
3. 拉 `da174dfd` Gateway internal auth + CSRF
4. 拉 `98a5b34f` better-auth 前端依赖
5. 拉 `4e4e4f92` 安全加固
6. 拉配套 7 个 wiring commit（`8ba01dfd` 等）
7. 前端登录页适配中文 + EthoInsight 品牌
8. 写 multi-user 部署 SOP（取代弃用的 4-23 计划）

---

## 10. 参考

- 轮 1 完成交接: `docs/handoffs/2026-05-07-deerflow-tier234-round1-completed-handoff.md`
- 执行计划文档: `docs/handoffs/2026-05-07-deerflow-tier234-execution-plan.md`
- 105 commit 分类: `docs/handoffs/2026-05-07-tier234-walkthrough-data/all-105-commits-classified.txt`
- noldus 上游同步规则: `CLAUDE.md` L123-L174
- 弃用的 user-backend 计划: `docs/plans/2026-04-23-multi-user-deployment.md`

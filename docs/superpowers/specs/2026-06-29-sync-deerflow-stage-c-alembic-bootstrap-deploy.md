# Sync Spec — Stage C：采纳上游 alembic 迁移系统 + bootstrap，改造部署链（#3706 `debb0fd1`）

> 状态：待执行。本 spec 是 [2026-06-29-sync-deerflow-11415875-to-b3c312b7.md](2026-06-29-sync-deerflow-11415875-to-b3c312b7.md) 里 Stage C 的展开（父 spec 的 Stage A+B 是常规 sync，可独立先做）。
> **前置确认（用户已拍板）**：① 项目**未进生产**，本地/dev DB**可随便删** → 不需要 reconcile 历史数据，可 wipe & 重 bootstrap；② 顺带把部署链 `make deploy-tar` 的迁移步一并改成与上游 bootstrap 对齐。
> 上游：`deerflow/main` @ `b3c312b7`；本地 harness `packages/agent/backend/packages/harness/deerflow/`，上游路径 `backend/packages/harness/deerflow/`。

## Context（为什么做 / 解决什么）

上游 #3706（`debb0fd1`）把 persistence 从「裸 `Base.metadata.create_all`」升级成**正式 alembic 迁移系统 + 启动时 `bootstrap_schema` 状态机**（空库→create_all+stamp head；旧库→create_all+stamp baseline+upgrade head；已托管→upgrade head）。这是 infra 底座升级，我们该跟。

**但它和我们已有的一套 alembic 正面冲突**（父 spec 已坐实）：
- 我们自有 4 revision 一条线：`20260512_feedback_verdict_revised`(root, down=None) → `20260601_feedback_paradigm` → `20260622_run_token_usage_by_model` → `20260626_thread_cascade_fk`(我们 HEAD)；本地 `deerflow.db` 的 `alembic_version` 现停在 `20260626_1700`。
- 上游另一条线：`0001_baseline`(root, down=None) → `0002_runs_token_usage`(我们 `20260622` 的重复——都给 `runs` 加 `token_usage_by_model`)。
- 两个 root（down_revision 都=None）→ 直接共存 = alembic **multiple-heads**，`upgrade head` 歧义。

**因为 DB 可随便删，最干净的解法 = 折叠成「上游 baseline 单链 + 我们的 Noldus 增量作为其后一个 revision」**，彻底消除双 root，不做历史数据迁移。

**🔴 必保的 Noldus 定制（不能被 wholesale 采纳洗掉）**：
1. `persistence/feedback/model.py`（PROTECTED）：`rating` **nullable**（verdict-only 路径不写 rating）+ 三个 Noldus 列 `verdict`(三分类)/`revised_text`(SFT 种子)/`paradigm`。**上游 `0001_baseline` 的 feedback 表 `rating NOT NULL` 且无这三列** → 直接采上游 baseline 会与我们 ORM 漂移、丢业务语义。
2. `persistence/engine.py`（PROTECTED）：`init_engine_from_config` 顶部 idempotency guard `if _engine is not None: return`（防 `get_local_provider` 每请求重跑 `os.makedirs` 触发 langgraph blockbuster 500）。上游版**无**此 guard。

**预期结果**：单一 alembic 链（root=上游 `0001_baseline`）、启动时自动 bootstrap、`feedback` 表保留 Noldus 三列 + rating nullable、engine guard 保留；部署链 `deploy-via-tar.sh` 的迁移步与新 bootstrap 对齐（不再硬编码我们旧链的 `HEAD_REV`）。

---

## 方案（折叠成上游单链 + Noldus 增量 revision）

### C1. 引入上游 alembic 机器（NEW 文件，全量合）
`git show deerflow/main:<上游路径> > <本地路径>`：
- `persistence/bootstrap.py`（状态机入口 `async bootstrap_schema(engine, *, backend)`）
- `persistence/migrations/_env_filters.py`（autogenerate 的 LangGraph 表排除过滤）
- `persistence/migrations/_helpers.py`（`safe_add_column` 等幂等 helper）
- `persistence/migrations/script.py.mako`（autogenerate 模板）
- `persistence/migrations/versions/0001_baseline.py`（链根）
- `persistence/migrations/versions/0002_runs_token_usage.py`（runs.token_usage_by_model）

### C2. SAFE 文件全量覆盖
- `persistence/run/model.py`（上游 RunRow ORM）
- `persistence/migrations/env.py`（上游加了 `include_object` 过滤 + alembic spawned engine 的 SQLite busy_timeout）——我们自有 env.py 无 Noldus 定制，全量覆盖即可。

### C3. 删除我们旧的 4 个 revision
删 `persistence/migrations/versions/` 下：
- `20260512_1200_feedback_verdict_revised.py`
- `20260601_1500_feedback_paradigm.py`
- `20260622_1700_run_token_usage_by_model.py`（功能 = 上游 `0002`，被取代）
- `20260626_1700_thread_cascade_fk.py`（runs/run_events→threads_meta ON DELETE CASCADE；**核对上游 0001_baseline 是否已含此 FK cascade**，下方 C5 验证项）

> 因 DB 随便删，无需保留这些做历史迁移路径。

### C4. 新增一个 Noldus 增量 revision（承上游链）
新建 `persistence/migrations/versions/0003_noldus_feedback_ext.py`（或按上游命名风格），`down_revision = "0002_runs_token_usage"`，做：
- `feedback.rating` 改 **nullable**（上游 baseline 是 NOT NULL）。
- `feedback` 加列 `verdict`(String(16) null)、`revised_text`(Text null)、`paradigm`(String(64) null)。
- 用 C1 引入的 `_helpers.safe_add_column` 保证幂等。
- 若 C5 核出上游 0001_baseline **缺** thread_cascade FK，本 revision 一并补（或单开一个 revision），保留我们 `20260626` 的 cascade 语义。

这样链变成：`0001_baseline → 0002_runs_token_usage → 0003_noldus_feedback_ext`(新 HEAD)，**单 root、无 multiple-heads**，且 Noldus feedback 语义 = 一个干净的"上游之上的增量"。

### C5. 验证上游 0001_baseline 对我们表的覆盖（写 revision 前必做）
上游 `0001_baseline` 建表清单已确认含：`users / threads_meta / runs / run_events / feedback / channel_connections / channel_conversations / channel_credentials / channel_oauth_states`（与本地 `deerflow.db` 实际 10 张表中 9 张吻合，第 10 是 `alembic_version`）。**还需逐项核**：
- `runs` 表上游 baseline 是否已含 `token_usage_by_model`？（若 0001 已含，则 0002 对全新库是 no-op；对我们无害）
- `runs`/`run_events` → `threads_meta` 的 **ON DELETE CASCADE FK** 上游 0001/0002 是否已建？没有 → C4 补。
- `channel_*` 表列是否与我们 ORM 一致（我们没改过 channel ORM，应吻合）。

### C6. engine.py surgical（保 guard + 采 bootstrap）
对 `persistence/engine.py`（PROTECTED）：
- **保留** idempotency guard（`if _engine is not None: return`，函数顶部）原样。
- **替换** 现有 naive `Base.metadata.create_all` + "Production should use Alembic" 那段（约 153-179 行）为上游版的 `from deerflow.persistence.bootstrap import bootstrap_schema` + `await bootstrap_schema(_engine, backend=backend)`（含 postgres "does not exist" 自建库重试分支）。
- busy_timeout：上游新版 WAL pragma 块里有 `PRAGMA busy_timeout=30000`；我们当前版没有 → 跟随上游加上（与 alembic spawned engine 的 busy_timeout 对齐，跨进程 bootstrap 需要）。
- head-to-head：`diff -u <(git show deerflow/main:backend/packages/harness/deerflow/persistence/engine.py) packages/agent/backend/packages/harness/deerflow/persistence/engine.py`，逐行确认只保了 guard、其余跟上游。

### C7. 部署链改造（用户要求顺带改 `make deploy-tar`）
当前 `packages/agent/scripts/deploy-via-tar.sh`（136-140、197-219 行）：rsync `run-db-migrations.sh` 到远端 → 在 gateway 镜像里 `docker compose run gateway /app/scripts/run-db-migrations.sh`，**先于** `compose up`。`run-db-migrations.sh:155` 硬编码 `HEAD_REV="20260626_1700"`（我们旧链头）+ legacy auto-stamp 逻辑。

采纳 bootstrap 后两条路线（**spec 推荐 C7-b**）：

- **C7-a（最简，删手动步）**：既然 `engine.py` 启动即 `bootstrap_schema`（空库→create_all+stamp head；已托管→upgrade head，幂等），**删掉 deploy-via-tar.sh 的整个手动迁移步**（136-140 rsync + 197-219 run）+ 删 `run-db-migrations.sh`。Gateway 一起来 bootstrap 自动把 schema 带到 head。**风险**：迁移失败从「deploy 前显式 abort」变成「Gateway 启动时报错」，可观测性略降。
- **C7-b（推荐，保留显式步但改成 alembic upgrade head，去掉硬编码）**：保留「服务起来前先迁移」这层（失败即 abort、不让新代码撞未迁移 schema——守 memory 部署铁律），但把 `run-db-migrations.sh` **简化**为「`alembic upgrade head`」一行式（不再 legacy auto-stamp、不再硬编码 `HEAD_REV`——因为新链有正式 baseline，全新库走 bootstrap、已有库走 upgrade head 都由 alembic/bootstrap 处理）。HEAD_REV 硬编码彻底删除（它本就是「旧库无 alembic_version 时 stamp 哪个 rev」的补丁，新体系不需要）。

> 两条都要顺带核 `docker-compose.yaml` 的 gateway 镜像是否 bake 了 alembic + migration 文件（deploy-via-tar.sh 注释说 bake 了 alembic+迁移文件但没 bake scripts/）。守 [[feedback_deploy_compose_per_service_image_tag]]：deploy 完验镜像。

### C8. 本地 wipe & 重 bootstrap（验证用，用户已允许删）
```bash
# 备份（保险，虽说可删）后清空本地 3 个 db
mv packages/agent/backend/.deer-flow/data/deerflow.db{,.bak-$(date +%s)} 2>/dev/null || true
# users.db / checkpoints.db 同理按需
```
然后起 `make dev`，让 `bootstrap_schema` 从空库建全套 + stamp 到新 HEAD（`0003_noldus_feedback_ext`）。
> 注：799 个 training-data JSONL 在 `.deer-flow/training-data/`，**不在 DB 里**，wipe DB 不影响它们；feedback 表 7 行可弃。

---

## 验证（每步必过）

### 单测 / 迁移正确性
```bash
cd packages/agent/backend
make test                                                    # 全量回归
PYTHONPATH=. .venv/bin/python -c "import app.gateway"         # 裸导入入口 1（守闭环铁律）
PYTHONPATH=. .venv/bin/python -c "from deerflow.agents import make_lead_agent"  # 入口 2
```
- 新增 `tests/test_persistence_bootstrap.py`（若上游没带等价测试）：在**空库**和**旧库（无 alembic_version、已有 feedback 表但缺 Noldus 列）**两种状态跑 `bootstrap_schema`，断言终态 feedback 表**含 verdict/revised_text/paradigm 且 rating nullable**。守 [[feedback_processpoolexecutor_test_runner_injection_and_ssot_parity]] 式真产物断言——不只断言"跑通"，要断言列真在。
- 守 [[feedback_dependency_upgrade_review_must_uv_sync_real_venv_not_main_pth]]：若在 worktree 做，alembic 跑的是 worktree 的 env.py/versions，确认 PYTHONPATH/cwd 指向 worktree 源。

### 迁移链自检（防 multiple-heads）
```bash
cd packages/agent/backend/packages/harness/deerflow/persistence/migrations
alembic -c alembic.ini heads        # 必须只有 1 个 head（= 0003_noldus_feedback_ext）
alembic -c alembic.ini history       # 必须一条线：0001→0002→0003
```

### 端到端 wipe→bootstrap→读写
- C8 wipe 后 `make dev`，前端提交一次 feedback（带 verdict + revised_text）→ 查 `deerflow.db` feedback 行有这三列值（守「代码有修复≠现象消除」，验真写得进）。

### 部署链（C7 改完）
- 在本地用 gateway 镜像 dry-run：`docker compose run --rm gateway alembic -c <ini> upgrade head` 不报错、`alembic heads` 单头。
- 守 [[feedback_deploy_alembic_migration_for_added_columns]]：确认部署链真跑了迁移（不是 create_all 漏列）。

---

## 收尾
1. 父 spec 的 `.deerflow-sync-state` 注释从「persistence/alembic SKIP」改为「已采纳 #3706 + Noldus feedback 增量 0003 + 部署链改 C7-b」。
2. commit（精确路径 `git add`，**绝不 `-A`/`.`**——避开历史 untracked `docs/reports/`、`reports/report for june/`、`scripts/repro/` + 无关的 `golden-cases/*` 未提交改动）。改动文件：`persistence/{bootstrap.py,engine.py,run/model.py}` + `persistence/migrations/{_env_filters.py,_helpers.py,script.py.mako,env.py,versions/0001_*,versions/0002_*,versions/0003_*}` + 删 4 个旧 revision + `scripts/run-db-migrations.sh`(改/删) + `scripts/deploy-via-tar.sh` + 新测试。
3. commit message 中文，说明「折叠双 alembic 链为上游单链 + Noldus feedback 增量 + 部署链对齐 bootstrap」。
4. 进 dev（非 main）。push 与否问用户。

## 不做
- 不保留旧 4 revision 做历史迁移（DB 可删，折叠成上游单链更干净）。
- 不 wholesale 采纳上游 feedback baseline 而丢 Noldus 三列（C4 增量补回）。
- 不洗掉 engine.py idempotency guard（C6 surgical 保）。
- 不在父 spec 的 Stage A/B 里混入本 Stage C（独立 commit，便于回滚）。

## 待用户确认
1. **部署链取 C7-a（删手动迁移步，全靠启动 bootstrap）还是 C7-b（保留显式 `alembic upgrade head` 步、删硬编码 HEAD_REV）**？我推荐 C7-b（保「服务起来前迁移失败即 abort」的可观测性）。
2. 新 Noldus 增量 revision 命名：跟上游 `000N_` 风格（`0003_noldus_feedback_ext`）还是我们旧的日期风格（`20260629_...`）？建议跟上游风格保持单一体系一致。
3. 这是要我**直接实施**，还是先评审本 spec？（实施会动 PROTECTED `engine.py` + 删 revision + 改部署脚本 + wipe 本地 DB，建议父 spec Stage A+B 先落，再做本 Stage C。）

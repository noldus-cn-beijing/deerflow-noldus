# 交接：本地 dev 启动 500 根因（DB 缺列）→ DB 迁移自动化（A+B 未提交）+ EPM dogfood chart-maker 两 bug 修复（PR#172 已合）

> **会话日期**：2026-06-22
> **来源 thread**：`339512dd-96dc-4a26-a712-71dee5c74c27`（EPM dogfood，28×control7/treatment21）
> **本会话性质**：① 修本地启动 500（DB 缺列）+ 把 DB 迁移做成部署链自动化（**A+B 代码未提交，最重要的活待办**）② 诊断并修复 dogfood 暴露的 chart-maker 两个真 bug（**PR#172 已合 dev**）③ 写两份后续 spec（随 #172 进 dev）。

---

## 一、最重要的待办：DB 迁移自动化（A+B）**代码已写好但未提交**

### 背景（已彻底查清并本地验证）
本地 `make dev` 起来后 `/workspace/chats/new` HTTP 500，gateway log 底行：
```
sqlite3.OperationalError: no such column: runs.token_usage_by_model
```
**与任何业务 spec 无关**。根因：DeerFlow sync e418d729（PR#166）给 `runs` 表加了列 `token_usage_by_model`，但本地 DB `packages/agent/backend/.deer-flow/data/deerflow.db` **从没跑过那个迁移**（且从无 `alembic_version` 表——一直靠 `create_all` 建表，`create_all` 绝不 ALTER 已存在表）。**prod 部署链同样不跑 alembic，下次 `make deploy-tar` 会让现网 500。**

本地已修复（alembic stamp `20260601_1500` + upgrade head，DB 已到 `20260622_1700`，164 runs/54 threads 完好，已验证那条 ORM 查询 `session.get(RunRow,...)` 成功）。备份在 `.deer-flow/data/deerflow.db.bak-before-token-col-migration`。

### A+B 改动（✅ 写好 ✅ 本地全验过 ❌ 未提交未 PR）
工作树里**未提交**的三处（在主仓 `/home/wangqiuyang/noldus-insight`）：
1. **新增** `packages/agent/scripts/run-db-migrations.sh`（untracked，10759 字节，已 chmod +x）—— 幂等迁移脚本，**自动 stamp 无 alembic_version 的老库**（按"已存在哪些迁移加的列"探测 baseline→stamp→upgrade head）。已实测三分支：已 head 空跑 / 老库（无 version 表、停在 20260512_1200）自动 stamp+升到 head / 重跑幂等。
2. **改** `packages/agent/scripts/deploy-via-tar.sh` —— 远程脚本 `docker load` 后、`docker compose up` 前加一步 `docker compose run --rm --entrypoint sh gateway` 跑迁移（镜像自带 alembic+迁移文件、bind-mount 脚本、对宿主持久库迁移；失败 `exit 1` 中止部署留旧容器）；并加 ship `run-db-migrations.sh` 到 ECS `scripts/` + mkdir。
3. **改** `docs/sop/deploy-via-tar-sop.md` §3.5 —— 重写：自动迁移说明 + 手动兜底（`docker cp`/`run --rm`，修正原 SOP 直接 `upgrade head` 会在老库撞 `duplicate column` 的隐患）。

**两个 bash 脚本都过 `bash -n`。** 接手第一步建议：在主仓**开分支**把这三处 commit + 开 PR 到 dev。commit message 参考：`feat(deploy): DB 迁移自动化——部署链跑 alembic + 自动 stamp 老库`。

⚠️ **prod 首次部署前**：prod 库大概率也无 `alembic_version`，脚本的 auto-stamp 正为此设计，但建议先 `DRY_RUN=1` 或先看 ECS 上 `runs`/`feedback` 现有列确认 baseline。SOP §3.5 已写手动步骤。

### 关键踩坑（已记 memory `feedback_local_dev_db_also_needs_manual_alembic_migration_after_sync`）
- alembic `env.py` **只读 alembic.ini 的 `sqlalchemy.url`，忽略 `-x sqlalchemy.url=`**。
- `alembic.ini` 的 URL 是相对路径 `./data/deerflow.db`，须 **cd 到 `backend/.deer-flow/`** 跑 + `PYTHONPATH=<backend>/packages/harness` + `PWD=$(pwd)`。
- 真实库由 config.yaml `database.sqlite_dir`（默认 `.deer-flow/data`）决定；`checkpoints.db`/`users.db` 不含 runs 表，别迁错。
- `docker compose exec` **不支持 `-v` 挂卷**（要 `docker cp` 或 `run --rm`）。

---

## 二、已完成并合并：EPM dogfood chart-maker 两 bug（✅ PR#172 已合 origin/dev `45824364`）

> **本地 `dev` 是 stale 的**（HEAD 还在 #169 `b59d78a5`）。接手先 `git checkout dev && git pull` 拉到含 #172 的 `45824364`。

### #1 — chart-maker 重跑丢 `--parameters-json`（真 bug，治本）
- **现象**：`plot_open_arm_time_ratio_bar_s0/s1` 前两批生成失败、靠第三次单独重试救活（浪费 ~50s）。按 mtime+SandboxAudit 坐实 batch2 命令无 `--parameters-json`。
- **复现**：脚本不带该参数 → `compute_open_arm_time_ratio` 返回 None → 退出码 1 → 不出图。
- **根因**：`ethoinsight-chart-maker/SKILL.md` 第 5 步 args 描述漏列 `--parameters-json`，诱导 LLM 重跑重构命令丢参（plan 的 `entry.args` 本就含它，逐字拼就对）。
- **修法**：args 描述补该参数 + 加正面铁律「整体逐项拼接、不增不减不重构」+「重跑从 plan 的 args 重取」。新增 `backend/tests/test_chart_maker_skill.py`（5 契约测试锁文字）。
- memory：`feedback_chart_maker_drops_parameters_json_on_rerun_skill_args_omits_it`。

### #3 — 代表性子集不按组均衡（真 bug，治本）
- **现象**：dogfood 前 7 个 subject 全 control，budget=8 取 idx 0/1 → 画的全是 control，treatment 一张个体图都没有。
- **根因**：`select_charts_by_priority` 纯按 `subject_index` 排序。
- **修法**：`packages/ethoinsight/ethoinsight/catalog/resolve.py` 的 `select_charts_by_priority` 加 `group_of`（subject_index→组名）参数按组轮转；无分组退回旧行为（向后兼容）。两调用方 `prep_chart_plan_tool.py` + `cli.py` 接上组映射。

### 测试（都跑 worktree 源）
ethoinsight 全量 **961 passed/0 failed**；chart budget 16 passed（含 dogfood 复现）；backend chart 相关 56 passed（含守环 `test_gateway_import_no_cycle`）；裸导入 `app.gateway`/`make_lead_agent` OK。

---

## 三、随 #172 进 dev 的两份 spec（**待实施，未实施**）

| spec | 内容 | 建议 |
|---|---|---|
| `docs/superpowers/specs/2026-06-22-chart-budget-ask-user-not-auto-throttle-spec.md` | **#2**：chart 全画 vs 代表性子集**由 lead 反问**（subagent 不能 HITL），chart-maker 不再自设 budget。只在「意图未明确+subject 多」时问，和现有「要不要出图」合并成一个多选。 | **值得实施**。直接解决用户痛点「上传 28 个却只画几张」。改动清单/测试 spec 里都有，可让新 agent 照做。与 #3 配套（#3 管均衡，#2 管该不该限流）。 |
| `docs/superpowers/specs/2026-06-22-data-analyst-residual-thinking-cost-followup-spec.md` | **#4**：data-analyst 判读残留耗时（06-18 修复后仍 ~3min+2 次 seal-gate 催促）。 | **先别急着改代码**。spec 把「取证」设为强制第一步（当前训练数据不录 subagent thinking，得换源拿到）。**很可能结论是不用改**（若那 ~2.5min 是合理判读推理）。盲改 data-analyst 判读 prompt 风险高（直接进给导师的报告）。 |

---

## 四、用户原始问题的完整答案（已查清，供参考）

dogfood thread `339512dd` 用户问的几件事，已全部坐实：
1. **「为什么还有图 missing/生成失败」** → 就是 #1（两张 bar 图前两批失败靠重试救活）。另有 `rose` 图在 `plan_charts.json.skipped[]` 标 `columns.missing`——那是**正确跳过**（数据没导出 `Direction` 列，rose 是 optional），无关。
2. **「两个 thread（339512dd vs fb3ed752）图差别很大」** → `fb3ed752` 是上周（6-18）跑的，候选集只有 box/bar/trajectory 三种；`339512dd` 今天跑的 catalog 多了 heatmap+zone_entry_distribution。**同数据不同版本代码 → 不同图集，正确演进非 bug**。
3. **「report-writer 慢是不是读大文件」** → **不成立**。report-writer prompt 已只让它读精简 handoff（code_executor 18.6KB + data_analyst 3.8KB + metadata 1.6KB），实测没读 147K 的 plan_metrics。真慢因是 **data-analyst ~3min（thinking 重 + 2 次 seal 催）** + **lead 派遣间隙 ~50s×3（deepseek 延迟）** + chart-maker 重跑 ~50s（#1 修了就消除）。report-writer 本身 ~1.5min 是 deepseek 长文生成的合理成本。
4. **「s0/s1 是什么、按什么画」** → s0/s1=subject_index（0-based，按上传顺序）。s0=Trial1(control)、s1=Trial2(control)。画哪些图由 catalog（epm.yaml charts 段，6 种）定；画多少由 chart_budget（chart-maker 自设的 8，#2 spec 要改成 lead 反问）。

---

## 五、未完成事项（按优先级）

### 🔴 P0（最紧，prod 风险）
1. **提交 A+B（DB 迁移自动化）**——主仓工作树三处未提交（见 §一）。开分支 commit + PR 到 dev。下次 `make deploy-tar` 前**必须**让这个进去，否则现网 500。

### 🟡 P1
2. **实施 #2 spec**（chart 全画 vs 子集 lead 反问）——dev 已含 spec，可让 agent 照做。
3. **本地 `git checkout dev && git pull`** 拉到 `45824364`（含 #172），本地 dev 现 stale。

### ⚪ 待证
4. **#4 spec**：先取证 data-analyst 残留 thinking，再决定改不改。不阻塞。

---

## 六、风险与注意事项

1. **本地 `dev` stale**（HEAD #169，远程已 #172 `45824364`）——接手先 pull，别在旧 dev 上开分支。
2. **工作树有非本会话的未提交改动**：`prompt.py`、`identify_ev19_template_tool.py`、`uv.lock`、`docs/problems/2026-06-17-*` 等是**之前会话/其他人**的在制品，**不是 A+B 的一部分**。提交 A+B 时**只 `git add` 那三个 DB 迁移文件**（`run-db-migrations.sh` + `deploy-via-tar.sh` + `deploy-via-tar-sop.md`），别把别人的在制品一起 commit。
3. **别盲改 #4**（data-analyst 判读 prompt）——判读质量直接进报告，且 06-18 已修过一刀，残留可能是合理成本。先取证。
4. **DB 迁移老库必须先 stamp 再 upgrade**——直接 `upgrade head` 会从最早 revision 重放撞 `create_all` 已建列报 duplicate column。脚本已自动化这步。
5. **cwd 陷阱**：Bash 工具每次新调用 cwd 重置回 `/home/wangqiuyang`（非仓库根）。所有命令用绝对路径或开头 `cd /home/wangqiuyang/noldus-insight &&`。

---

## 七、下一位 Agent 的第一步建议

1. **读本 handoff** + memory `feedback_local_dev_db_also_needs_manual_alembic_migration_after_sync`（DB 迁移踩坑）+ `feedback_chart_maker_drops_parameters_json_on_rerun_skill_args_omits_it`（#1 根因）。
2. **拉新 dev**：`cd /home/wangqiuyang/noldus-insight && git checkout dev && git pull`（到 `45824364`）。
3. **提交 A+B**（P0）：开分支，**只**加 `packages/agent/scripts/run-db-migrations.sh` + `packages/agent/scripts/deploy-via-tar.sh` + `docs/sop/deploy-via-tar-sop.md`（别带其他在制品），commit + PR。
4. 若要继续画图体验：**实施 #2 spec**。

## milestone 建议
- 「EPM dogfood 图表流水线打磨」track：#1（丢参，PR#172 已合）+ #3（组均衡，PR#172 已合）已 checkpoint；#2（lead 反问 budget）待实施。
- 「部署/运维基础设施」track：DB 迁移自动化（A+B）待提交——这是补上「部署链不跑 alembic」的结构性缺口。

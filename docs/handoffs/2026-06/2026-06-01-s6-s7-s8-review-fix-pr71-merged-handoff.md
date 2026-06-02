# 2026-06-01 会话交接 — S6/S7/S8 review 修复 + PR #71 合 dev + S8 定位锁定 + deploy 迁移 SOP

> **本 handoff 用途**:交接 2026-06-01 一次 review 会话。任务是 review 用户 push 的 `worktree-sprint-6-7-memory-assumptions`(S6/S7/S8)。**结论**:三个 sprint 测试全绿但生产路径全静默失效,已修 5 缺陷 + 补迁移 + 加 deploy SOP,**PR #71 已合 dev**。
>
> **dev HEAD(写本文时)**:`065b4180`(Merge PR #71)。本地 = origin/dev 同步。
> **务必先 `git fetch && git log --oneline origin/dev -8` 看 dev 是否又前进**(多 agent 并行)。

---

## 0. 本会话产出(全部已合 dev,PR #71)

dev 上的 4 个相关 commit(`4b1ea0a4` 之后):

| commit | 内容 |
|---|---|
| `c3644546` | feat(S6+S7) 原始实现(用户/并行 agent 写) |
| `77264915` | feat(S8) 原始实现 |
| `f9c12cdf` | **fix(S6/S7/S8) 本会话修的 5 个生产缺陷 + 真实回归测试** |
| `742c4676` | docs(sop) deploy SOP §3.5 数据库迁移注意事项 |

**测试基线**:worktree 全量 **3807 passed / 84 skipped / 0 failed**(补 config.yaml 后跑;原 commit 自报 3283/3 是 worktree 缺 config.yaml 所致)。ruff 改动文件全绿。

---

## 1. 核心发现:三个 sprint「测试全绿但生产路径全死」

原 2 commit 的 47 个测试全绿,但 review(读源码 + 实跑真函数)发现**三个 sprint 的生产路径全部静默失效**——共因是 **mock / 测试契约误锁盖住了 bug**(典型 [[feedback_pr_merge_must_run_full_suite_on_shared_logic]] 的升级版:不是没跑全量,而是 mock 把签名错误藏了)。已修(`f9c12cdf`):

| # | 严重度 | 缺陷 | 修复 |
|---|---|---|---|
| 1 | 🔴 | **S6**:`create_memory_fact(source=)` 传不存在的 kwarg(真签名无 source,硬编码 "manual")→ TypeError 被 try/except 吞 → fact **永不写入**。21 个测试全 mock 该函数,反而断言传了 source | 去掉 `source=`,lineage 折进 content;**新增调真函数(仅隔离文件 IO)的回归测试**,签名不匹配将 fail |
| 2 | 🟠 | **S8**:加了 `paradigm` 列但**无 alembic 迁移** → create_all 不给现网已存在 feedback 表补列 → 写该表报 OperationalError | 新建 `20260601_1500_feedback_paradigm.py`(batch_alter_table,down_revision=20260512_1200),**实跑 upgrade 验证加列成功** |
| 3 | 🔴 | **S8**:prompt 注入两处错 import(`deerflow.agents.thread_state._current_thread_data` 不存在 + `deerflow.persistence.database` 应为 `.engine`)→ ImportError 被吞 → `<prior_corrections>` **永远为空**。2 个 prompt 测试断言返回 "" 反而锁定了坏行为 | 改由 `make_lead_agent` 在**已解析 workspace 处**取 paradigm/user_id 传进 `apply_prompt_template`(prompt-build 时无 thread ContextVar,原 ContextVar 设计不可行);修 import;**新增真 happy-path 测试渲染 `<prior_corrections>`** |
| 4 | 🟠 | **S8**:router `_read_paradigm_from_context` 手拼 `base/threads/<tid>` 漏多用户 `users/<uid>/` 段(本仓库是多用户,CLAUDE.md §13) | 改用 `paths.sandbox_work_dir(thread_id, user_id=)` + user-less 兜底 |
| 5 | 🟡 | **S7**:`present_assumptions` 全默认(无 override/warning/audit)仍渲染面板,违反 docstring + brief §7;死代码 `gates/gate_str`/`parameter_audit_section` | 仅非空 override 计入 has_content;删死码;修 2 个**名实不符**的测试(名字说返回空、断言却查 `<details>`) |

**修复手法的共性教训**:测试 mock 掉被测函数本身 → 签名/import 错误被完全掩盖。**改进**:S6/S8 都新增了"调真函数、仅隔离 IO/DB"的回归测试,这才是能抓住此类 bug 的测试。

---

## 2. S8 定位锁定(用户质疑后确认,重要)

用户问"S8 真的是 agent 需要的吗" → 确认 **S8 = 微调到位前的临时桥,不是核心能力,合了别再扩展**:
- 与训练飞轮微调**强重叠**,brief 原话"微调到位后收益减半";prompt 注入是"伪学习"(真学习是参数级);v0.1 feedback 数据稀疏多数会话不触发;且引入"必须手动跑迁移否则反馈提交 500"的运维成本。
- **结论**:合了放着,**不做**跨范式 prior_corrections / 加权重等扩展;微调链路起来后该段 prompt 注入可下线。
- **S6(记性)/S7(可追溯)是结构性能力**,微调不取代,正常保留。
- 已落 memory:[[2026-06-01-sprint-s6-s7-s8-completed]] + [[feedback_deploy_alembic_migration_for_added_columns]]。

---

## 3. S8 对现有项目的影响面(已逐文件核实)

**绝大部分向后兼容 + 容错降级**:`upsert(paradigm=None)` 老调用兼容;新 router 独立路径追加;prompt 注入无范式时 0 成本短路、有范式时 try/except 吞异常 → **agent 永远正常启动**;迁移 `add_column(nullable=True)` 不动老数据,env.py 已 scope 不碰 LangGraph checkpointer。

**唯一真实风险点**:`submit_feedback` 里 `upsert(paradigm=...)` **没额外 try 保护**。现网漏跑迁移 → `feedback` 表缺 `paradigm` → **专家反馈提交 500**(仅此一个功能;agent 分析/S6/S7/报告都不写 feedback 表,不受影响)。仅影响**已有 feedback 表的现网 ECS**;本地 dev/全新部署不受影响。**跑迁移后风险归零。**

---

## 4. 未完成 / 待办(按优先级)

1. **🔴 部署带 S8 的版本到现网前,必须先跑数据库迁移**(否则反馈通道 500)。完整命令见 `docs/sop/deploy-via-tar-sop.md §3.5`。**关键坑**:不能直接 `alembic upgrade head` 也不能靠 `-x sqlalchemy.url=`(env.py 用 `get_main_option` 取 url,不读 -x/环境变量,会迁错的空库)。必须内联 Python 调 `command.upgrade` + `set_main_option` 指向真实库 `${DEER_FLOW_HOME}/data/deerflow.db`(容器内 `/app/backend/.deer-flow/data/deerflow.db`)。详见 [[feedback_deploy_alembic_migration_for_added_columns]]。
2. **🟡 dogfood 验证「实现完 ≠ 生产能跑」**(本会话即活证)。建议端到端跑:
   - S6:跑完一次 EPM → 看 `memory.json` 真出现 `category=experiment_summary` fact(含精确数字);第二会话同范式 → system prompt `<memory>` 段含上次。
   - S7:有 critical warning / override 的分析末尾,lead 主动调 `present_assumptions` 渲染折叠卡;简单分析(全默认)不渲染。
   - S8:跑两次同范式(第一次专家给 needs_fix + 修正),第二次 lead prompt 含 `<prior_corrections>` 段(前提:DB 已迁移)。
3. **🟡 S3 FST mobility 判据**:工程结构已就位,卡行为学同事 **issue #63**(velocity 物种判据)。#63 答复后在 catalog `fst.yaml/tst.yaml` 加 mobility `parameters`(换数字不改结构)。**不要自己编阈值数字**(越权写领域知识)。
4. **🟡 S4 调参指南工程通路**:data_analyst workflow 加"grep `## 参数调整指南` 段"通路,**内容留空待同事**(SSOT 在 review-packages,见 [[feedback_ssot_lives_in_review_packages]])。
5. **🟢 善后**:worktree `sprint-6-7-memory-assumptions`(`742c4676`,已合 dev)可清理 —— `git cherry -v dev <分支>` 确认全 `-`(已合)后 `git worktree remove` + 删分支。**不要 `git log dev..` 判合并状态(被 squash/merge 骗)**。

---

## 5. 风险与注意事项

1. **roadmap v2 的 10 个 sprint 已全部落地**(S0–S8)。下一步重心是 **dogfood 收口 + 等 #63 同事**,不是"加新 sprint"。
2. **愿景层(实验本体 SSOT / 设计层决策智能 / 基元化)是 v1.0,不进 v0.1 sprint**([[feedback_version_boundary_v01_insight_v10_experiment_harness]])。v0.1(9月)= 只消费 EV19 raw data 的 insight 分析 harness。
3. **S8 别再扩展**(见 §2)。
4. **改 prompt.py / agent.py / persistence / 共享 helper 合并前必跑全量 `make test`**,且要警惕 mock 掩盖签名/import 错误——**对新加的"调真依赖"路径,至少一个测试要调真函数**([[feedback_pr_merge_must_run_full_suite_on_shared_logic]] 升级版)。
5. **worktree 测试陷阱**:worktree 无 gitignored config.yaml → 构造 lead-agent / 跑全量会 FileNotFoundError 或少收集测试。解法:`cp 主仓/packages/agent/config.yaml 进 worktree backend`(gitignored,不会误 commit;跑完删掉)。
6. **prompt.py / builtins/__init__.py 是受保护文件**:本次是纯新增(一行 import、一行 `__all__`、一条 prompt bullet、prior_corrections 段),没动中文调度/注册逻辑;下次 deerflow sync 仍按 surgical-merge 处理([[feedback_sync_protected_files_registry_loss]])。

---

## 6. 下一位 Agent 的第一步建议

1. `git -C /home/wangqiuyang/noldus-insight fetch && git log --oneline origin/dev -8` — 看 dev 最新(PR #71 已合,HEAD 应 `065b4180` 或更新)。
2. 跑一次全量确认 dev 真绿(不轻信 commit message):
   `cd packages/agent/backend && source .venv/bin/activate && python -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
3. **若要部署**:先读 `docs/sop/deploy-via-tar-sop.md §3.5`,部署后**必跑迁移**(否则反馈 500)。
4. **若要推进 v0.1**:做 §4.2 的 dogfood(最高价值,验证 S5–S8 生产真能跑),或起草给 #63 同事的 catch-up 解锁 S3。
5. **读这些 memory 再动手**:`project_2026-06-01_sprint_s6_s7_s8_completed`(本批最新真相)、`feedback_deploy_alembic_migration_for_added_columns`(迁移坑)、`feedback_version_boundary_v01_insight_v10_experiment_harness`(别飘到愿景层)。

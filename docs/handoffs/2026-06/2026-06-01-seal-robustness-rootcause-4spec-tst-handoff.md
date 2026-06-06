# 2026-06-01/02 会话交接 — seal handoff 系统性根因定位 + 4 spec 产出 + TST 上线 + 现网迁移待跑

> **本 handoff 用途**:交接一次跨 6-01→6-02 的长会话。主线是**端到端 dogfood 中暴露的 seal handoff 故障的系统性根因定位**(用 git + langgraph.log 双证据钉死),并产出 4 份 spec、合入 3 个修复 PR、提 1 个同事 issue、删库重建现网。**最重要的待办:`all-subagent-seal-robustness-design.md` spec 待实施(系统性根治全部 subagent seal)+ 现网迁移待跑。**
>
> **dev HEAD(写本文时)**:`f678e8c1`(PR #75)。**务必先 `git fetch && git log --oneline origin/dev -6` 看 dev 是否又前进**(多 agent 并行)。

---

## 0. 当前任务目标

把 EthoInsight 端到端流水线(lead → code-executor → data-analyst → chart-maker → report-writer)在真实 FST/TST 数据上**跑通且 seal 不出问题**。本会话发现:**两个 subagent(code-executor / data-analyst)的 seal handoff 都失败**,定位为 seal 从 Sprint 0 起变严苛、prompt 没同步教 + Sprint 3 step 2.8 卡死,后续兜底(5.7/5.8)是创可贴没拆病根。

---

## 1. 已完成(本会话)

### 1.1 合入 dev 的 3 个修复 PR
| PR | commit | 内容 | 状态 |
|---|---|---|---|
| #73 | `a45c1b98` | TST(悬尾)范式端到端支持,v0.1 5→6 范式 | ✅ 合 dev,我已 review 通过(单测 18 passed) |
| #74 | `91ed188e` | data-analyst pendulum 卡死修复(step 2.8 降级) | ✅ 合 dev,**但降级出口写窄了**(见 §4) |
| #75 | `217cdc1f` | report 图表网页 404 修复(前端 img rewrite + report_writer prompt) | ✅ 合 dev,我已 review 通过(normalize 5 路径实跑验证 + 前端 typecheck) |

### 1.2 seal handoff 故障 — 系统性根因定位(git + langgraph.log 双证据)
**两个故障,两个机制,都源于"seal 从 Sprint 0 变严苛 + prompt 没教":**

| | code-executor | data-analyst |
|---|---|---|
| 现象 | seal **被调用**,schema 校验失败(5→2→过) | seal **没被调用**(forgot to call),seal-resume 补轮 2 次没救回 |
| 直接触发 | `data_quality_warnings`: `metric=None` / `code='insufficient_sample'`(下划线)/ `evidence`=字符串 | step 2.8 参数审计陷入"拿 per_subject 输出指标比对需原始分布的参数"死循环,烧光 max_turns=12 |
| 引入 commit | **Sprint 0 `aed27729`**(seal schema 化加 DOT/dict/str 强校验)+ Sprint 1 `00741752` | **Sprint 3 `f5aa5b05`**(step 2.8) |

**证据出处**:`packages/agent/logs/langgraph.log` thread `9af3ba6d-61c0-479a-926e-0f5aaabb2448`,10:05/10:06 两次 `CodeExecutorHandoff ValidationError`(5→2 errors),10:13/10:18 两次 data-analyst seal-resume 没救回。**全局盘点:4 个 subagent prompt 全部 0 处教 schema 字段格式。**

### 1.3 产出 4 份 spec(均在 `docs/superpowers/specs/`,**全部 untracked `??`,未 commit**)
| spec | 状态 |
|---|---|
| `2026-06-01-tst-paradigm-e2e-support-design.md` | 已实施(PR #73) |
| `2026-06-01-report-chart-404-fix-design.md` | 已实施(PR #75) |
| `2026-06-01-data-analyst-pendulum-audit-fix-design.md` | 已实施(PR #74),被下条 spec 阶段1补全 |
| **`2026-06-01-all-subagent-seal-robustness-design.md`** | 🆕 **待实施**(本会话最终产物,取代已删的旧 seal-warnings spec) |

### 1.4 现网删库重建(用户确认测试数据无价值)
- ECS `root@39.105.231.16`:gateway+langgraph 容器**已停**,`/opt/ethoinsight/deer-flow-home/data/` 下 `deerflow.db`+`-wal`+`-shm` **已删**,data 目录空
- **未跑 `make deploy-tar`** → 容器还停着,现网当前不可用,等用户本地 deploy

### 1.5 提交同事 issue #72
[#72 TST 上线需行为学同事确认 3 项](https://github.com/noldus-cn-beijing/noldus-insight/issues/72):Q1 模板归类(NoTemplate vs PorsoltCylinder)/ Q2 NoTemplate.md 填空 / Q3 TST golden-case。关联 #63。

---

## 2. 关键发现(决定后续方向)

1. **seal 机制本身没错**(用户原话:"seal 绝对不是问题,问题是我们没做好")。是 Sprint 0 起 seal 加严校验 + Sprint 3 加 step 2.8 前置步骤,prompt 没同步教。
2. **治本分两类**:code-executor/chart-maker/report-writer = prompt 教格式即治本;**data-analyst ≠ 教 seal**,它卡在 seal 之前,必须**补数据通路**(让参数审计要比对的 velocity/periodicity 原始分布真正进 handoff)。
3. **路径 B(用户锁定)= 不回退 Sprint 3,把它做对**:`metrics/_pendulum.py:140-144` 逐帧 `periodicity`/`activity` 是**现成中间产物**,只是没写进 handoff。让 code-executor 输出其分布统计量(p10/p90/median)→ data-analyst step 2.8 有真数据可比。
4. **用户要求"所有 subagent seal 都没问题"** → spec 覆盖 4 个 subagent + 三层(prompt 教格式 / 工具归一化 / 补数据通路)。

---

## 3. 未完成 / 待办(按优先级)

1. **🔴 实施 `all-subagent-seal-robustness-design.md`**(最高价值,解锁卡住的 dogfood)。spec 分两阶段:
   - **阶段 1(止血,纯 prompt + 工具)**:4 prompt 教 seal 字段格式 + `_seal_handoff` 容错归一化 + data-analyst step 2.8 补"signal_distribution 缺失即整类 info 跳过"出口(补 `91ed188e` 漏的 case)。**此阶段后 dogfood 应跑通**。
   - **阶段 2(路径 B,单独 PR)**:code-executor 产 periodicity/velocity 分布进 handoff + data-analyst 用真分布审计 + handoff_schemas 加字段。
   - 建议:**先只做阶段 1**(止血优先),阶段 2 单独排。
2. **🔴 现网迁移/恢复**:用户本地跑 `make deploy-tar` → 新镜像在空 data 目录 `create_all` 建带 `paradigm` 列的新表(绕开 alembic 迁移)。deploy 后验证 4 容器健康 + feedback 表有 paradigm 列。详见 `docs/sop/deploy-via-tar-sop.md §3.5`。
3. **🟡 commit 4 份 spec**:它们现在 untracked,其他 worktree/agent 读不到。需 `git add docs/superpowers/specs/2026-06-01-*.md && commit`(或至少 commit 待实施的那份)。
4. **🟡 TST dogfood**:PR #73 静态改动 + 单测过,但 spec 核心验收项"真实悬尾数据端到端 dogfood"**没做**。需等 seal 根治后,用 `/home/wangqiuyang/DemoData/newdemodata/悬尾/` 真实数据跑通。
5. **🟢 issue #72 同事回复后**:Q1 定 TST 标准模板(工程先按 NoTemplate 跑通)。

---

## 4. 风险与注意事项

1. **`91ed188e`(PR #74)的降级出口写窄了**:只覆盖 `n<2` 和 `文档缺失`,**漏了"判据存在但 per_subject 无原始分布"这个真死因** → 所以 6-01 dogfood data-analyst 仍卡死(langgraph.log 10:13/10:18 实证)。新 spec 阶段1 必须补这个 case,否则继续卡。
2. **不回退 Sprint 3 / 不删 step 2.8 / 不删 seal-resume 兜底**(用户明确)。兜底是纵深防御最后一环,保留。
3. **不放宽 schema 契约**(下游 data-analyst/前端依赖),只"教 + 容错 + 补数据"。
4. **工程不编 pendulum/velocity 领域阈值数值**(issue #63 归同事)。路径 B 补的是"数据分布"(统计事实),p90×3 是统计背离判据,**不是领域阈值**。
5. **受保护文件**:4 个 subagent prompt(`subagents/builtins/*.py`)+ `seal_handoff_tools.py` 均受保护,surgical 改 + 改后全量 `make test`([[feedback_pr_merge_must_run_full_suite_on_shared_logic]]:改共享 helper `_seal_handoff` 必 grep 4 个调用方 + 跑全量)。
6. **改 data_analyst.py workflow 必 grep 编号唯一**(5.7 seal bug 教训):`grep -nE "^\s*[0-9]+\.|^\s*[0-9]+\.[0-9]+" data_analyst.py` 确认 `1/2/2.5/2.6/2.7/2.8/3/4` 不重复。
7. **prompt 教学是 schema 的"副本"**,字段约束权威源在 `handoff_schemas.py` validator。prompt 注释标注"约束权威源见 schema",schema 改了 prompt 要同步([[feedback_single_source_of_truth]])。
8. **TST 模板矛盾(Q1)工程不裁决**:`tail_suspension.md` 说 PorsoltCylinder-NoZones,`ev19_facts.py:62` 映射 NoTemplate,真实数据像 NoTemplate。已在文档标 `⚠️ 待同事裁决`,工程先按 NoTemplate 跑。

---

## 5. 下一位 Agent 的第一步建议

1. `git -C /home/wangqiuyang/noldus-insight fetch && git log --oneline origin/dev -6` — 确认 dev HEAD(应 `f678e8c1` 或更新)。
2. **读 `docs/superpowers/specs/2026-06-01-all-subagent-seal-robustness-design.md`**(本会话最终产物,完整根因 + 三层 + 两阶段方案)。
3. **若要实施 seal 根治**:先 commit 这份 spec(§3.3),再开 worktree 做**阶段 1**(纯 prompt + 工具,解锁 dogfood)。复现验收用 langgraph.log thread `9af3ba6d` 同款 FST n=1 数据。
4. **若要恢复现网**:用户本地跑 `make deploy-tar`,deploy 后我可验证(见 §3.2)。
5. **读这些 memory 再动手**:`feedback_pr_merge_must_run_full_suite_on_shared_logic`(改共享 helper 必跑全量)、`feedback_single_source_of_truth`(prompt 不双存 schema 约束)、`feedback_version_boundary_v01_insight_v10_experiment_harness`(别飘愿景层)、`feedback_ssot_lives_in_review_packages`(不编领域阈值)。

---

## 6. 关键文件/命令速查

- **seal 工具**:`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`(`_seal_handoff` helper `:193`,4 个 seal 工具 `:230/284/331/364`)
- **schema(契约 SSOT)**:`subagents/handoff_schemas.py`(DataQualityWarning `:80` + validator `:113`,ParameterAuditFinding `:129`,4 个 Handoff)
- **4 subagent prompt**:`subagents/builtins/{code_executor,data_analyst,chart_maker,report_writer}.py`
- **pendulum 中间产物**:`packages/ethoinsight/ethoinsight/metrics/_pendulum.py:140-144`(逐帧 periodicity/activity)
- **真实数据**:FST `原始数据-Porsolt forced swim test XT190-Trial 1.xlsx`(每组 n=1,Drug vs Saline);TST `/home/wangqiuyang/DemoData/newdemodata/悬尾/`
- **故障 log**:`packages/agent/logs/langgraph.log` thread `9af3ba6d`
- **测试**:`cd packages/agent/backend && source .venv/bin/activate && make test`;ethoinsight `cd packages/ethoinsight && pytest tests/`
- **现网**:`ssh root@39.105.231.16`,`cd /opt/ethoinsight/docker && docker compose -p deer-flow ps`

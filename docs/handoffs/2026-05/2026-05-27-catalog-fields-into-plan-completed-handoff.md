# 2026-05-27 catalog 判读字段下沉到 plan_metrics.json — 交接文档

## TL;DR

- **PR #54 已合 dev**:把 catalog YAML 的 5 个判读 / 展示字段透传到 `plan_metrics.json`,subagent 不再尝试 read 包内 catalog YAML 文件
- **8 commit + 1 commit spec/plan 归档**:全部合并到 dev,worktree 已清理
- **本地 make dev 端到端验证通过**: data-analyst thinking 不再"猜路径",直接读 plan_metrics.json,正确消费 `direction_for_anxiety` / `statistical_default` 等字段
- **后续待办很少**: 仅 1 个可选清理动作(删远端已合分支),3 份历史 handoff 未 commit(不在本次范围)

## 当前任务目标(已完成)

修复 ECS 生产 FST 端到端请求中 data-analyst subagent "~10 步反复猜 catalog YAML 路径"的故障。

**真根因(精确版)**: `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py:700` 的 `validate_local_tool_path` 是 sandbox 白名单闸门,只放行 5 类前缀(`/mnt/user-data/*` / `/mnt/skills/*` / `/mnt/acp-workspace/*` / `/mnt/shared/*` / `config.yaml.sandbox.mounts`)。`ethoinsight-metric-catalog/SKILL.md` 教 subagent 用 `import ethoinsight.catalog as c; c.__file__` 拿包内物理路径,该路径**不在白名单**,read_file 直接 PermissionError。dev / prod 行为一致(都走同一 LocalSandboxProvider + tools.py),不是 dev/prod 差异。

**最终方案(方案 D)**: 把 catalog YAML 这条 subagent 消费路径**彻底移除**;5 个判读 / 展示字段(`unit_zh` / `one_liner` / `output_unit` / `direction_for_anxiety` / `statistical_default`)由 `_metric_to_plan` 透传 + `plan_metrics_to_dict` 序列化 → 派遣前 `prep_metric_plan_tool` 写到 `/mnt/user-data/workspace/plan_metrics.json` → subagent 走白名单允许的虚拟路径 read 一次拿全。

## 当前进展

### ✅ Phase 1: 取证(完成)

- 读 5 份背景文档 + spec
- 用 `validate_local_tool_path` 实测复现拒绝(site-packages 路径 / 源码绝对路径 / 容器内路径 均 PermissionError;`/mnt/skills/*` 和 `/mnt/user-data/*` 通过)
- 通过 LLM thinking 转录确认 LLM 在 subagent 内**没有 bash 工具**(`data_analyst.py:158-160` `disallowed_tools` 含 `bash`),所以 SKILL.md 教的 `python -c "..."` 方法走不通

### ✅ Phase 2: brainstorm(完成)

A/B/C/D 4 选项 trade-off 分析:
- A. docker-compose volume mount:违反 `feedback_dev_prod_behavior_alignment`,需两份配置同步
- B. dump CLI:data-analyst 没有 bash,走不通
- C. plan_metrics.json schema 扩展(原 spec 提议):不够彻底,两份消费契约同时存在
- **D. catalog YAML 从 subagent 消费路径彻底移除**(本方案):
  - 字段下沉到 plan_metrics.json
  - 删 SKILL.md / data_analyst / report_writer 中 "read catalog YAML" 引导
  - 加 lint 防回滚

### ✅ Phase 3: spec + plan(完成,已 commit dev)

- spec: `docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md`
- plan: `docs/superpowers/plans/2026-05-27-catalog-fields-into-plan.md`
- dev commit: `dbbd91c0 docs(catalog): 加 spec + plan 文档`

### ✅ Phase 4: 实施(完成,PR #54 已合 dev)

10-task TDD 实施计划,subagent-driven-development 模式(sonnet model)派遣 8 个 implementer task(Task 9 用户手工 dev 验证,Task 10 PR 准备由我做):

| Task | commit | 内容 |
|------|--------|------|
| 1 | `0ad5b554` | PlanMetric dataclass 扩 5 字段(默认值向前兼容) |
| 2 | `0644d27f` | `_metric_to_plan` 透传 + 5 范式 parametrize 回归 |
| 3 | `ae86c81d` | `plan_metrics_to_dict` / `plan_to_dict` 输出 5 字段 |
| 4 | `89bdae99` | `prep_metric_plan_tool` e2e 测试(写出 JSON 含 5 字段) |
| 5 | `3e297bc7` | SKILL.md 重写(删 `__file__` 招数 + 完整字段字典) |
| 6 | `c681f0db` | data_analyst.py / report_writer.py prompt 改读 plan_metrics.json |
| 7 | `bf2f1460` | `test_subagent_prompts_no_catalog_yaml.py` lint 防回滚 |
| 8 | `c08ffa1c` | catalog `__init__.py` docstring + 2026-05-13 spec Addendum |

每 task 走"implementer DONE → spec compliance reviewer ✅ → code quality reviewer ✅"三阶段。

### ✅ Phase 5: 用户手工 dev 端到端验证(完成)

worktree 内 make dev,跑 FST 双 subject 端到端。data-analyst thinking 实测:
- ✅ 不再出现 "The catalog YAML file wasn't found at that path"
- ✅ 不再出现 "Let me try to find it via the python import approach"
- ✅ thinking 一开始就说 "读取 plan_metrics.json(用于指标元数据)"
- ✅ 正确消费 `direction_for_anxiety` 字段做方向判读("higher immobility = more depressive-like" / "shorter latency = faster giving up")
- ✅ 正确识别 `direction_for_anxiety: null` 字段(如 immobility_bout_count)不下硬判读

### ✅ Phase 6: 清理(完成)

- `git worktree remove` + `git branch -d fix/catalog-fields-into-plan` 本地清理
- spec/plan commit 到 dev(`dbbd91c0`)
- memory 更新:
  - **新增** `feedback_subagent_consumption_via_first_party_tool.md` — SSOT 契约修正模板(可复用到任何"subagent 消费包内文件"场景)
  - **删除** `feedback_worktree_uv_editable_install_pitfall.md`(用户判定过时;`uv run --project` 已自动规避)

## 关键发现

### 发现 1: dev/prod 在 sandbox 白名单层面**完全等价**

之前会话曾假设 catalog 读不到是 dev/prod 差异(`make dev` vs `make up`),实际证伪:`config.yaml:87` `sandbox.use: deerflow.sandbox.local:LocalSandboxProvider`,docker-compose.yaml 直接挂载同一份 config.yaml 进容器。dev / prod 走同一 LocalSandboxProvider + 同一份 tools.py + 同一 white list。LLM 在本地 dev 也撞同样的墙。

**反模式警告**: 看到"prod 出问题"先别假设 dev/prod 差异,先 grep 配置确认是否真的双轨。

### 发现 2: SSOT 派生 + tool 消费的正确模式

catalog YAML 仍是 single source of truth,但**消费形态**从"subagent 直接 read YAML 文件"改成"lead 调 first-party tool 在 sandbox 外 resolve → 派生到 /mnt/user-data/workspace/*.json → subagent read JSON"。这个模式可复用到任何"住在 Python 包内的结构化数据 → subagent 消费"场景。详见新加的 [[feedback-subagent-consumption-via-first-party-tool]] memory。

### 发现 3: subagent thinking 是最快的"prompt 改动起效"诊断

不要靠 logs / 工具 trace 判断 prompt 改动是否生效,**直接看 thinking 第一句**。如果改 prompt 让 LLM "直接读 plan_metrics.json",thinking 第一段应该说"读取 plan_metrics.json";如果还说"尝试 import ethoinsight.catalog",改动没生效或 prompt 缓存没刷新。

### 发现 4: subagent 没有 bash 工具是"教 import + __file__"反模式的硬约束

`data_analyst.py:158-160` `disallowed_tools=[..., "bash", ...]`,即便 SKILL.md 教 LLM 跑 `python -c "..."` 拿路径,LLM 也跑不了。改 SKILL.md 时如果设想用 bash 取数据,**先查 subagent disallowed_tools 是不是含 bash**。

## 未完成事项(按优先级)

### 优先级 P3(可选,5 分钟)

**清理远端 `origin/fix/catalog-fields-into-plan` 分支**

PR #54 已合,远端分支可删:

```bash
cd /home/wangqiuyang/noldus-insight
git push origin --delete fix/catalog-fields-into-plan
```

也可不删 — GitHub PR 详情页保留 commit 历史,远端分支可有可无。用户上次问到,被 /handoff 流程打断没明确回答。

### 优先级 P3(可选,5 分钟)

**3 份历史 handoff 仍 untracked 在 dev**

`git status` 仍显示这三份:
- `docs/handoffs/2026-05/2026-05-26-channel-todos-bug-diagnosis-handoff.md`(上次会话根因假设有误的 handoff)
- `docs/handoffs/2026-05/2026-05-26-session-final-handoff.md`
- `docs/handoffs/2026-05/2026-05-27-channel-todos-bug-resolved-handoff.md`(本次任务的输入 handoff)

不在本次任务范围,留给后续会话(或下个 agent)按上下文判断是否 commit。建议如果 commit,加 redirect 标注到 `2026-05-26-channel-todos-bug-diagnosis-handoff.md` 顶部,说明 hashseed 假设证伪。

### 优先级 P3(将来当 catalog 字段扩展时,SOP 提醒)

**catalog YAML 加新判读字段时的三件套**

(参考新 memory `feedback_subagent_consumption_via_first_party_tool.md`)

1. 在 `MetricEntry` schema(`packages/ethoinsight/ethoinsight/catalog/schema.py:32-41`)加字段
2. 在 `_metric_to_plan`(`resolve.py:427`)透传到 `PlanMetric` 构造
3. 在 `plan_metrics_to_dict` 和 `plan_to_dict`(`resolve.py:698, 745`)dict 字面量加 key

**LLM 看到 plan_metrics.json 多 key 会自然消费,大概率不需要改 prompt** — 这是本次设计的关键优势(用户原本担心"字段会频繁扩"的负担)。但如果新字段语义不直观,需要在 `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md` 的"metric 字段字典 reference"表里加一行。

`tests/test_plan_metrics_interpretation_fields.py` 的 `_EXPECTED_NEW_FIELDS` 集合也需同步更新(防 silent regression)。

## 建议接手路径

如果是新会话接手本任务的后续(几乎不需要,任务已闭环),按这个顺序:

1. **read** 本 handoff(就是你现在读的这份)
2. **read** spec: `docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md`(35 KB,设计 + 替代方案 + 风险)
3. **read** plan: `docs/superpowers/plans/2026-05-27-catalog-fields-into-plan.md`(46 KB,10-task TDD 详细步骤,有完整代码)
4. **跑 sanity test**:
   ```bash
   cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
   uv run pytest tests/test_plan_metrics_interpretation_fields.py -v
   ```
   Expected: 15 passed(3 个 parametrize 函数 × 5 范式)

## 风险与注意事项

### 不建议的方向

- **不要走方案 A**(加 catalog volume mount):违反 `feedback_dev_prod_behavior_alignment`(本地 config.yaml 和 docker-compose.yaml 双轨)
- **不要走方案 B**(加 dump CLI):data-analyst `disallowed_tools` 含 bash,CLI 走不通
- **不要再恢复 SKILL.md 的 `import + __file__` 引导**: `tests/test_subagent_prompts_no_catalog_yaml.py` 是 lint 兜底,会 fail
- **不要把 PlanMetric 改成 `frozen=True`** dataclass:resolve 阶段需要 mutable,frozen 会让透传逻辑炸

### 容易混淆的点

- `plan_metrics_to_dict` (line 745) 和 `plan_to_dict` (line 698,backward-compat wrapper) **两个序列化器都要改**,否则 `plan_to_dict` 输出和 `plan_metrics_to_dict` 输出不对齐。Task 3 已修。
- `_MINIMAL_COLUMNS` 测试 fixture(`tests/test_plan_metrics_interpretation_fields.py`)的列名要和各范式 catalog YAML `requires_columns` 真实 pattern 匹配(可能用 fnmatch glob),改 catalog YAML `requires_columns` 时这个 fixture 可能需要同步。Task 2 implementer 已遇到并补齐(oft / zero_maze 加 `distance_moved`)。
- worktree 内 backend pytest **如果 worktree 自己没 .venv,会用主仓库 .venv 导致 import 主仓库代码**(已删 memory `feedback_worktree_uv_editable_install_pitfall`)。规避:`uv run --project <worktree_backend> pytest ...` 会自动建 worktree 独立 venv。

### LLM 易踩的陷阱

- **不要把 catalog YAML 当"次级 SSOT"** — 在本次架构下,plan_metrics.json 才是 subagent 看到的"事实",它的字段值由 catalog 派生但**消费契约只有一份**。catalog 仍是源头,但 subagent 不直接看。

## 下一位 Agent 的第一步建议

**你大概率不需要做任何事**。本任务已完整闭环:
- 代码 + 测试已合 dev(PR #54)
- spec + plan 已 commit dev(`dbbd91c0`)
- worktree 已清理
- memory 已更新
- 端到端 dev 验证通过

如果用户启动新会话要求"继续 catalog 任务",**先 read 本 handoff 确认任务状态,大概率告诉用户'本任务已闭环'即可**。

如果用户提到"远端分支还在"或"未 commit 的 handoff",参考"未完成事项"段的 P3 待办。

如果用户提到 catalog 的**新需求**(例如加新指标 / 加新判读字段 / 加新范式),按 P3 "三件套" SOP 执行。

## 仓库状态快照

- 当前分支: `dev`
- HEAD: `dbbd91c0 docs(catalog): 加 spec + plan 文档`
- 本地 = origin/dev(已 push)
- worktree list:
  ```
  /home/wangqiuyang/noldus-insight                                           dbbd91c0 [dev]
  /home/wangqiuyang/noldus-insight/.claude/worktrees/paradigm-key-alignment  ce1376af [paradigm-key-alignment]
  /home/wangqiuyang/noldus-insight/.claude/worktrees/pr3-lead-robustness     a3e9e677 [worktree-pr3-lead-robustness]
  /home/wangqiuyang/noldus-insight/.claude/worktrees/retire-shoaling         b9ae1bc2 [fix/retire-shoaling-paradigm]
  /home/wangqiuyang/noldus-insight/.claude/worktrees/sync-cleanup            94faeed0 [chore/sync-protected-files-and-sop]
  ```
  (这 4 个 worktree 是历史遗留,不属于本任务)
- 未 commit 改动: 3 份历史 handoff(见 P3 待办)
- 远端分支: `origin/fix/catalog-fields-into-plan` 仍存在(PR merge 后未自动删,见 P3 待办)

## milestone 建议

本任务让 EthoInsight 的 **catalog 消费契约**从"双轨(派遣前 resolve + 派遣后 read YAML)"固化为"单轨(派遣前 resolve 是唯一路径)",是架构清理类的里程碑。建议在 `docs/milestone/` 找 catalog / SSOT 相关 track,加一行 checkpoint 摘要:

> **2026-05-27**: catalog 消费契约固化 — subagent 不再直接读 catalog YAML,所有判读 / 展示字段统一走 `plan_metrics.json`。PR #54 合 dev。dev 端到端实测 data-analyst thinking 不再猜路径,直接消费 catalog 元数据。

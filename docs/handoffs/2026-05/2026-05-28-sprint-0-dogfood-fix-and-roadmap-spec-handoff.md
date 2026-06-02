# 2026-05-28 Sprint 0 dogfood 修复 + 路线图 spec 全套就绪 — handoff

**会话主体**：评估 SOTA agent v2 路线图 → 全套 spec → Sprint 0 实施验证 → 4 个 dogfood bug 修复 → spec 库与代码同步
**触发分支**：dev
**核心 commit**：`db791480` (本次会话产出，已 push)；上游 `ff81bef2` (Sprint 0 主体 PR #60，已合)
**测试基线**：ethoinsight 463 passed + agent backend 3098 passed
**dogfood 验证**：thread `af5bc3a2-47d9-479f-a464-6224e7aa857c` FST 端到端全程跑通

---

## 1. 本次会话做了什么

### 第一阶段：路线图评估 + spec 全套

入会场景：用户给了 [`docs/plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md`](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) (SOTA agent v2 路线图 10 sprint / 14-17 周) 让评估是否过设计、是否重复造轮子、是否贴合实际。

**评估流程**：
1. 复核 v2 路线图 6 处 "代码审计发现" → 5/6 准确, 1 处部分准确 (ChartMakerHandoff prompt 契约已存在)
2. 找出 4 处必修正: 路径错误 / Sprint 5.5 LangChain API 误解 / Sprint 2a 工作量低估 / Sprint 8 前置工作没说
3. 修完之后用 grill-me skill 跟用户过了 10 轮设计决策, 锁定:
   - Q1 ChartMakerHandoff 用现有 prompt 字段
   - Q2 atomic write + 4 个 seal_*_handoff first-party tool (E 方案)
   - Q3 warning code 4 一级分类 SAMPLE / MOTOR / SIGNAL / METHOD (点分隔)
   - Q4 catalog 参数 C 混合 SSOT (shared_parameters in _common.yaml + 范式独有)
   - Q5 早绑定 + overrides JSON 文件 (early binding)
   - Q6 analysis_config_id = deterministic sha256 + 进 experiment-context.json
   - Q7 auto/manual 双轨 (β 字段双用) + acknowledge_quality 复用
   - Q8 lineage manifest 进 .lineage/ + fail-closed + audit log
   - Q9 Sprint 6 砍掉新顶层结构, 复用 deerflow facts (0.5 周)
   - Q10 Sprint 7 只做 tool 不做 GateProvider (0.5 周)
4. 把 grill 决定全部回写 roadmap v2 (16.5 周, 净省 2 周)

**产出 5 份 spec**:

| Spec | 行数 | 状态 |
|---|---|---|
| Sprint 0 (handoff schema + seal tools) | 1014 | ✅ 已实施 + dogfood |
| Sprint 1 (data_quality warnings 结构化) | 377 | 🟡 已交给新 agent 实施 |
| Sprint 2a (catalog 参数下沉) | 706 | 待执行 |
| Sprint 2b (参数管线 5 跳封闭) | 569 | 待执行 |
| roadmap v2 | 559 | living document |

### 第二阶段：Sprint 0 实施

Sprint 0 由独立 agent 执行 (PR #60, commit aed27729) 合 dev:
- handoff_schemas.py 加 6 字段 (DataQualityWarning code/evidence/blocks_downstream, MetricStat parameters_used, CodeExecutorHandoff paradigm/ev19_template, 4 handoff analysis_config_id)
- 4 个 seal_*_handoff first-party tool + _seal_handoff helper (Pydantic 校验 + atomic write + .lineage/manifest.json)
- 三档 strict mode (OFF/WARN/FAIL_CLOSED) + 紧急降级文件 `/tmp/disable_strict_handoff`
- script_invocation_only_provider 拦截 write_file 写 handoff_*.json
- 4 个 subagent prompt 改用 seal tool
- 22 files / +1246/-31 / 55 新增测试 cases / 全量 3066 passed

### 第三阶段：dogfood 暴露 4 个真问题 → 4 个 fix

第一次 dogfood (thread 485a899d) 卡 6+ 分钟, 第二次 (af5bc3a2) 跑通。修了:

**Bug #1 — MetricStat.parameters_used 不允许 None**
- 现象: code-executor seal 时填 `{"threshold_mm_s": None}` 被 Pydantic 拒
- 根因: Sprint 0 spec 类型定义 `dict[str, float|int|str]` 太严, LLM 不知道值时填 None 是合理行为
- 修复: 类型改 `dict[str, float|int|str|None]` + 注释说明"None = not applicable"
- 文件: `handoff_schemas.py:67`

**Bug #2 — DataQualityWarning.code 错误消息歧义**
- 现象: LLM 写 `SAMPLE_UNDERPOWERED` (下划线) 被拒 3 次才悟到要点分隔
- 根因: 错误消息只说 "must start with one of {...}.*"模糊, LLM 没看出"点分隔"
- 修复: 错误消息明示 "literal DOT '.' as separator, NOT underscore", 含具体示例
- 文件: `handoff_schemas.py:_validate_code_namespace`
- dogfood 实证: 第二次 dogfood code-executor 仍试了 1 次错才写对, 进步明显但治本需要 docstring 加示例

**Bug #3 — groups 链路完整闭环 (最大改动)**
- 现象: code-executor 拿不到分组信息, 幻觉 `inspect_drug_column` / `dump_subject_metadata` 等不存在脚本死循环
- 真根因 2 层:
  1. lead 没把用户答案 ("第一个是实验组") 翻译成 groups dict 传 prep_metric_plan
  2. lead 没有探查 xlsx 结构的工具 (派 general-purpose subagent 不存在, knowledge-assistant 读不了 xlsx)
- 修复路径:
  - **prep_metric_plan_tool.py 加 `groups: dict | None` 参数** → 写 workspace/groups.json + 注入 plan_metrics.json.inputs.groups_file
  - **新建 `inspect_uploaded_file` first-party tool**: lead 探查上传文件的 EV19 metadata header (Treatment / Dose / Animal ID), 自动提取分组信息, 多数情况无需反问用户
  - **新建 `ethoinsight-grouping` skill** (SKILL.md + 3 references): 单一权威分组链路, lead + code-executor 共用契约
  - code_executor.py prompt 加 critical_rule: 不存在脚本立即停手, 分组从 plan.inputs.groups_file 读
  - quality-gates.md 加分组落地引用
- **关键发现**: EV19 文件头部 (UTF-16 编码 txt + xlsx 都有) 直接含 `Treatment="Drug"` / `Treatment="Saline"` 元数据字段, lead 调一次 inspect_uploaded_file 就拿到所有分组信息, 完全不需要反问用户

**Bug #4 — artifacts 504 Gateway Time-out**
- 现象: chart-maker 完成后 4 个图片 GET 504, 刷新页面才能加载
- 根因: `Response(content=actual_path.read_bytes(), ...)` 同步 IO 阻塞 async event loop; SSE stream 并发时 gateway 单 worker 互锁 7 分钟
- 修复: 改 `FileResponse(path=...)` 走 starlette anyio threadpool
- 文件: `app/gateway/routers/artifacts.py:202`
- dogfood 实证: 第二次 dogfood 图片**直接显示**, 无需刷新

### 第四阶段：commit + push

- 18 文件 / +4150-12 (含 5 份 spec 文档 + 8 个代码改动 + 4 个新 skill 文件)
- commit db791480 "fix: Sprint 0 dogfood 暴露的 4 个 bug + groups 链路 + 路线图 spec"
- 已 push origin/dev
- **显式排除**: `.env.wecom` (含 WebHook 密钥) + 其他无关 handoff/plan 文件留在工作区

---

## 2. 当前状态

### 代码

| 模块 | 状态 |
|---|---|
| Sprint 0 (handoff schema + seal tools) | ✅ 已合 dev + dogfood 验证 |
| 4 个 dogfood bug fix | ✅ 已合 dev (db791480) |
| ethoinsight-grouping skill | ✅ enabled in extensions_config |
| inspect_uploaded_file tool | ✅ 注册到 BUILTIN_TOOLS |
| 全量测试 | ✅ 3098 passed, 0 退化 |

### Spec 库

| Spec | 状态 | 备注 |
|---|---|---|
| roadmap v2 | living | grill 10 决定已回写 |
| Sprint 0 | 实施完成 | spec.parameters_used 类型已与代码对齐 (允许 None) |
| Sprint 1 | 🟡 **新 agent 正在实施** | 用户在本次会话末尾启动 |
| Sprint 2a | 待执行 | 依赖 Sprint 1 合 dev |
| Sprint 2b | 待执行 | 依赖 Sprint 2a 合 dev |
| Sprint 3-8 | 未写 spec | 等 Sprint 2b 完工再说 |

### 待清理 (工作区残留, 没动)

- `.env.wecom` — 敏感凭证, **永不 commit**
- `docs/handoffs/2026-05/2026-05-27-p0-metrics-bugfix-and-ev19-audit-handoff.md` — 之前会话残留
- `docs/handoffs/2026-05/2026-05-28-b2-b10-implementation-complete-handoff.md` — 同上
- `docs/superpowers/plans/2026-05-27-p2-p3-ev19-completion-plan.md` — 同上

下次清理时判断是否单独 commit / .gitignore / 删除。

---

## 3. Sprint 1 实施 agent 要注意的事

新 agent 拿到 [`docs/superpowers/specs/2026-05-28-sprint-1-data-quality-structured-design.md`](../../superpowers/specs/2026-05-28-sprint-1-data-quality-structured-design.md) 执行 Sprint 1。要点:

### 前置已就位

- `DataQualityWarning` schema 已有 code/evidence/blocks_downstream 三新字段 (Sprint 0)
- 4 一级分类 (SAMPLE/MOTOR/SIGNAL/METHOD) 已在 field_validator 锁定
- LEGACY 过渡兜底**已在 `seal_handoff_tools.py:187`**, Sprint 1 完工时务必清理 (按 spec §2.6)
- DataAnalystHandoff schema 已有 (Sprint 0), Sprint 1 只加 `quality_warnings` 字段 + GateSignals 加 `quality_warnings_critical_count`

### 重点 task

1. dispatcher.py 9 处 warning 按 spec §2.1 映射表升级 (code + evidence + blocks_downstream)
   - SAMPLE.TOO_SMALL / SAMPLE.UNDERPOWERED / MOTOR.LOW_ENTRIES / MOTOR.LOW_DISTANCE / SIGNAL.LOW_TRANSITION_COUNT / METHOD.*
2. data-analyst subagent prompt 加步骤 1.5: 读 quality_warnings 分级写入 method_warnings
3. seal_data_analyst_handoff 加 quality_warnings 参数 (Sprint 0 没加)
4. GateSignals.quality_warnings_critical_count
5. lead prompt 加播报模板 (critical_count > 0 时)
6. 前端按 blocks_downstream + severity 二维渲染 (红/橙/黄/蓝)
7. **清理 LEGACY 兜底** (spec §2.6 5 步, 完工后 `git grep "LEGACY.UNCATEGORIZED" packages/agent/backend/` 必须返回空)

### 验收 checklist (spec §6)

最关键 3 项:
- [ ] `git grep "LEGACY.UNCATEGORIZED" packages/agent/backend/` 返回空
- [ ] `git grep "LEGACY" packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` 返回空
- [ ] dogfood 跑 4 范式 critical case (n<3 / velocity all low) → 前端红字标注 + handoff warning 含具体数字

### 跟今天修复的衔接

- DataQualityWarning 类型 (`dict[str, ...|None]`) 现在已经允许 None (Bug #1 修复) — Sprint 1 dispatcher.py evidence dict 不会用到 None, 但跨 sprint 一致性要保持
- code 错误消息现在更明确 — Sprint 1 dispatcher 端写 code 应该一次正确, 不会试错
- groups 链路 (Bug #3) 与 Sprint 1 无直接关系, 但 Sprint 1 dogfood 时会用到 ethoinsight-grouping skill 自动分组的能力

---

## 4. 关键文件地图

### Sprint 0 + dogfood 修复实际文件

| 类别 | 文件 |
|---|---|
| Handoff schema | `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` |
| Seal tools | `packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py` |
| Inspect tool (新) | `packages/agent/backend/packages/harness/deerflow/tools/builtins/inspect_uploaded_file_tool.py` |
| Prep metric plan (改) | `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py` |
| Tool 注册 | `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py` + `tools/tools.py` |
| Strict mode | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` |
| Guardrail | `packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py` |
| Artifacts 504 fix | `packages/agent/backend/app/gateway/routers/artifacts.py` |
| Subagent prompts | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/{code_executor,data_analyst,chart_maker,report_writer}.py` |

### Skill 文件

| Skill | 文件 |
|---|---|
| ethoinsight-grouping (新) | `packages/agent/skills/custom/ethoinsight-grouping/` (SKILL.md + 3 references) |
| ethoinsight-lead-interaction | `packages/agent/skills/custom/ethoinsight-lead-interaction/references/quality-gates.md` (加分组引用) |
| extensions_config | `packages/agent/extensions_config.json` (enable ethoinsight-grouping) |

### Spec / 文档

| 类别 | 路径 |
|---|---|
| 路线图 | `docs/plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md` |
| Sprint 0 spec | `docs/superpowers/specs/2026-05-28-sprint-0-handoff-schema-foundation-design.md` |
| Sprint 1 spec | `docs/superpowers/specs/2026-05-28-sprint-1-data-quality-structured-design.md` (🟡 正在执行) |
| Sprint 2a spec | `docs/superpowers/specs/2026-05-28-sprint-2a-catalog-parameters-design.md` |
| Sprint 2b spec | `docs/superpowers/specs/2026-05-28-sprint-2b-parameter-pipeline-design.md` |

---

## 5. 教训记录 (待写入 MEMORY.md 候选)

会话中出现的几个值得记下来的教训:

1. **EV19 文件头自带分组元数据** — Treatment / Dose / Animal ID 字段直接在 txt 头部 (UTF-16) 和 xlsx sheet 头部 (中文 metadata 区), 多数 dogfood 场景不需要反问用户。lead 应该先 `inspect_uploaded_file`, 看到 Treatment 字段就直接构造 groups dict
2. **gateway 同步 read_bytes 会阻塞 async event loop** — `Response(content=actual_path.read_bytes())` 是反模式, 用 `FileResponse(path=...)` 走 anyio threadpool
3. **Pydantic 错误消息要明示语法元素** — `{'SAMPLE',...}.*` 不够明显, 必须写 "literal DOT '.', not underscore" + 具体示例 LLM 才能一次写对
4. **subagent_type 是固定枚举 (5 个)** — lead 试 `task(subagent_type='general-purpose')` 会报错, deerflow Noldus fork 只有 chart-maker/code-executor/data-analyst/report-writer/knowledge-assistant
5. **LangChain @tool docstring 解析中文冒号会当 arg 名** — 中文"格式:"或"key: value"在 Args 块下方会触发 `Arg ... not found in function signature`, 写 docstring 时避免冒号格式描述, 用 "key=path, value=group_name" 平铺

---

## 6. milestone 建议

本次会话让 SOTA agent v2 路线图 track 到达 **Sprint 0 完成 + 4 个 dogfood bug 闭环 + Sprint 1 执行启动** checkpoint。

应创建/更新的 milestone:
- `sota-agent-v2`: 更新当前状态为 "Sprint 0 dogfood pass + Sprint 1 in flight"
- `groups-pipeline`: 新建 milestone, 记录 ethoinsight-grouping skill + inspect_uploaded_file tool 的设计与落地
- `dogfood-fst-thread-af5bc3a2`: 记录第二次 FST dogfood 全程跑通的完整 trajectory 作为参考案例 (4 个 bug 都不复现, lead 调 inspect → set_experiment_paradigm → prep_metric_plan(groups=...) → 派 code-executor → ask_clarification quality gate → set_experiment_paradigm(acknowledge_quality=True) → data-analyst → set_viz_choice → chart-maker → report-writer)

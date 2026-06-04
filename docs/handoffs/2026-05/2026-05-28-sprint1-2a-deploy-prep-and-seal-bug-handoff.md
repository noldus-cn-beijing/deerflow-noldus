# 2026-05-28 Sprint 1 dogfood + Banner 修复 + Sprint 2a + 三份 spec + seal bug 真根因 — handoff

**会话主体**：Sprint 1 复核 + Banner 数据通道修复 + Sprint 2a 复核 + 三份下游 spec (3/4/4.5) + serve.sh 启动提速 + data-analyst seal bug 真根因 + Sprint 5.7 spec
**触发分支**：dev
**核心 commit**：`00741752` (Sprint 1) → `da924145` (Sprint 2a) → `b593d66c` (Sprint 3/4/4.5 spec) → `e3e6b205` (serve.sh) → `2df10880` (data-analyst fix) → `bbff6a33` (Sprint 5.7 spec)
**测试基线**：ethoinsight **504 passed** (Sprint 1 +14 / Sprint 2a +27) / agent backend **3111 passed** (Sprint 1 +14) / frontend tsc 0 错
**dogfood 验证**：EPM n=1 thread 端到端跑通,Banner 红字渲染正常,data-analyst 漏 seal bug 已定位 + prompt 修复 + Sprint 5.7 兜底 spec 待实施

---

## 1. 本次会话做了什么

### 第一阶段：Sprint 1 复核 + Banner 数据通道修复

**Sprint 1 实施完成核验** — 用户用 Sonnet (xhigh effort) 让我复核 Sprint 1 agent 的产出。逐项过 spec §6 14 项验收: dispatcher 8 处升级 + DataAnalystHandoff schema + seal tool + prompt 步骤 1.5 + lead prompt 播报模板 + frontend banner + LEGACY 清理 — **13 项 PASS,1 项硬缺陷**:

🔴 **硬缺陷**:spec §2.5 写"frontend 从 SSE stream 中读 quality_warnings",但实际 data-analyst seal tool 只写 handoff_data_analyst.json 文件,**后端没有任何代码把 quality_warnings 注入 message additional_kwargs**。前端 `QualityWarningBanner` 组件齐、二维渲染逻辑正确,**但实际 dogfood 时永远不会触发** — 因为没人写这个字段。

**修复方案 B (我之前提议的 3 种里选的)**:写 `QualityWarningBroadcastMiddleware` 钩子 lead 的 after_model,检查最近的 ToolMessage 是不是 task(data-analyst) 的结果,如果是则读 handoff_data_analyst.json 拿 quality_warnings,**注入到 lead 即将产生的下一条 AIMessage additional_kwargs**。

实现 + 测试:
- 新增 `agents/middlewares/quality_warning_broadcast_middleware.py` (含 `_has_recent_data_analyst_toolmessage` / `_load_quality_warnings` helper)
- 注册到 `lead_agent/agent.py` middleware chain (紧跟 GateEnforcement,SafetyFinishReason 之前)
- 顺手 fix `handoff_schemas.py:124` 删 LEGACY 提示文字时残留的尾随空格
- 14 个单元测试 case 全绿 (注入路径 / 跳过路径 / chart-maker 穿插 / 异步入口)
- ethoinsight 477 / agent backend 3111 / frontend tsc 0 错

**Sprint 1 commit** `00741752`: 19 files / +1238/-51, 含 Banner 修复 + Sprint 5 双重提示设计约束追加到 roadmap v2 + 补 commit Sprint 0 milestone 文件。

### 第二阶段：用户手动 EPM n=1 dogfood — Sprint 1 端到端验证通过

用户上传 EPM xlsx,跑完完整 E2E_FULL_ASKVIZ 流程: code-executor → quality acknowledge → data-analyst → chart-maker → report-writer。

**Banner 验证现象**:最后一条 lead 播报消息下方出现红字"1 条数据质量警告 (含 1 条阻断级) 展开详情" — 这就是 `QualityWarningBanner` 组件渲染。完整数据通路打通确认。

三层 quality_warnings 数字对账正确: code-executor critical_count=1 → data-analyst quality_warnings_critical_count=1 → Banner 显示 1 条 critical+blocks。

### 第三阶段：用户告诉我 2a/2b 并行 + 写下游 spec

用户问:Sprint 2a/2b 能否并行? Sprint 3-8 spec 何时写? 我分析了 spec 依赖关系 (2b 末步要删 _common.py 常量,与 2a 强耦合) 后建议:

- **2a/2b 严格串行** (用户选了这个,保守)
- **现在写 Sprint 3+4+4.5 三份 spec**(用户选了这个;Sprint 5/5.5/6/7/8 后续再写)

写完三份 spec (b593d66c):
- Sprint 3 (1 周):data-analyst 参数审计 — `ParameterAuditFinding` schema + 5 元枚举 mismatch_kind
- Sprint 4 (1 周):paradigm md 加"参数调整指南"段 + CI 哨兵防 SSOT 漂移
- Sprint 4.5 (0.5 周):analysis_config_id = sha256(canonical_json(defaults+overrides))[:16] + experiment-context.json 字段扩展

合计 1011 行 / 3 files。

### 第四阶段：是否能发版讨论 + serve.sh 启动提速

用户问"今天能不能发版"。我看了 dev 落后 main **52+ commit** (5/19~5/28 累积),建议**能发**且 strongly recommend deploy。讨论了路径:

- **A. 直接从 dev `make deploy-tar`**:快,但 dev 还在涨;Sprint 2a agent 在改代码,有时序冲突风险
- **B. 走 dev → main → deploy**:正式版路径
- **推荐**:打 tag 锚定快照后 deploy

讨论中用户跑 `make dev`,发现 Gateway 启动等了 100+ 秒。诊断:
- `uvicorn --reload` 的 watchfiles 缺 `*.py` 显式 include → fallback 扫 sys.path 所有 .py
- `.venv` 含 **42538 个 py 文件**没被 exclude → watchfiles 建初始监听索引拖死

修复 `serve.sh:153`:
- 加 `--reload-include=*.py`
- 加 `--reload-exclude=.venv/** node_modules/** tests/** .git/**`

预期 ~100s → ~10s。commit `e3e6b205` 已 push。**只影响 dev 模式本地启动,不影响 make deploy-tar 生产部署**。

### 第五阶段：Sprint 2a 实施完成 + 复核 PASS

用户告诉我 Sprint 2a agent 改动完成 (11 改动 + 2 新测试)。我跑全套复核:

- **schema.py**:ParamSpec / SharedParameters / ParadigmParameters 三个 frozen dataclass + MetricEntry.parameters/parameters_ref + Catalog.paradigm_parameters + PlanMetric.parameters_in_use + CommonCatalog 从 loader.py 移入 schema.py (架构改进)
- **loader.py**:6 个新 helper (_parse_param_spec/_parse_param_block/_parse_metric_entry/_parse_catalog/load_common_catalog/validate_catalog_consistency/load_all_catalogs)
- **_common.yaml**:13 个 shared_parameters (2 velocity + 1 sample_size + 10 pendulum) — agent 主动检测出 sample_size 6 范式同值 + pendulum FST/TST 同值,**全部提升到 shared**(符合 spec §3 SSOT 原则)
- **6 个 paradigm yaml**:epm/ldb/zero_maze 各保留独有 1 参数;oft/fst/tst 显式 paradigm_parameters: {}
- **dispatcher.py**:5 处阈值改用 `_get_shared_param` + `_get_paradigm_param` (lru_cache + lazy import + fallback warn)
- **27 新测试全过 + 504 全量 passed + agent backend 3111 不退化**

**唯一 PARTIAL 发现**:parameters_ref 没在 immobility metric 上加 (Sprint 2b 启动时需补)。守住了所有 Sprint 6 不在范围项 (_common.py / _pendulum.py / scripts/ / resolve.py 都没动)。

用户在 commit + push Sprint 2a 改动后通知我 (`da924145`).

### 第六阶段：dogfood 后发现 data-analyst seal 漏调 bug — 找到真根因

用户跑第二次 EPM n=1 dogfood,前端流程跑到 chart-maker 后突然停下,lead 输出 "data-analyst 的 handoff 文件似乎没有被正确封存。让我先读取 code-executor 的结果,掌握数据后再重新派遣"。lead 自救式 re-dispatch 后第二次成功了。

**用户问**:为什么 seal handoff 失败? Sprint 1 测试通过为啥 dogfood 触发?

**我的诊断过程**:
1. 解析 training-data jsonl row 14 (data-analyst 第二次派遣 output)
2. 见到关键证据:
   ```
   ## 进度时间线
   1/4: read_file × 3
   2/4: read_file
   3/4: read_file
   4/4: "现在我已掌握所有必要信息,来进行综合分析"
   ## 最终结果
   ...
   现在封存分析结果:                    ← stop here, no tool_call
   ```
3. 读 `data_analyst.py:60-94` workflow,发现编号冲突:

   ```
   1. 必读输出宪法
   2. read handoff_code_executor.json     ← 第一个 step 2
   2.5 读 quality warnings
   2.6 按范式 read 判读文档
   2. 一次性完成核心分析推理              ← 第二个 step 2(冲突!)
   3. 封存 handoff
   4. 最终 AIMessage
   ```

**真根因**:Sprint 1 实施时插入 step 2.5/2.6 后,**没把原 step 2 改成 step 2.7**。LLM 走到第二个 step 2 时把"一次性完成单轮思考"理解为"在 thinking 里输出全部结论就算 done",产 content"现在封存分析结果:"但**没 emit seal_data_analyst_handoff tool_call**。executor 见无新 tool_call → terminate as COMPLETED → handoff 文件没产。

**不是 Sprint 2a regression**(2a 不涉及 data_analyst.py)。是 Sprint 1 遗留 prompt bug。Sprint 1 dogfood 当时 LLM 运气好 emit 了 tool_call,这次运气差就暴露。

**prompt 修复** (commit `2df10880`):
- 第二个 step 2 → step 2.7 (消除编号冲突)
- 段首加 positive 指令 "下一步 step 3 必须真的调 seal_data_analyst_handoff tool, 不能只在 thinking 里写'封存'"
- 28 个 test 全过

### 第七阶段：根治讨论 + Sprint 5.7 spec

用户问"光这么改 prompt 能行吗?"

**我的回答**:prompt 修复只是减少触发概率,**不能根治** — LLM 随机性 + 长链 reasoning 注意力衰减意味着同类 bug 会在其他 subagent / 其他 prompt 改动后复现。**真正根治 = harness 层硬约束**。

提出 4 种深度方案 A/B/C/D,推荐 **方案 C: executor 末端 handoff 存在性检查**。

用户认同:"我们的流程不够工程化导致的这个,需要做存在性检查"。

我写了 **Sprint 5.7 spec** (`bbff6a33`):
- `executor.py:723` 加 `_validate_handoff_emitted` helper
- white-list 4 个 ethoinsight subagent (code-executor / data-analyst / chart-maker / report-writer)
- 不存在 handoff 文件 → 标 task FAILED + 明确诊断 message ("terminated without emitting ...")
- lead prompt 加自动 retry 规则识别 "terminated without emitting" 关键字
- 与 Sprint 5.5 lineage 封印正交 (5.7 管 emission,5.5 管 integrity)
- 10 单元 + 5 集成 + 1 dogfood 模拟测试 case
- 0.5 周

---

## 2. 当前状态

### 代码 (dev HEAD = bbff6a33)

| 模块 | 状态 |
|---|---|
| Sprint 1 (data_quality_warnings 结构化 + Banner 通道) | ✅ 已合 dev + dogfood 验证 |
| QualityWarningBroadcastMiddleware | ✅ 已合 + 14 测试全绿 |
| Sprint 2a (catalog 参数下沉) | ✅ 已合 dev + 27 测试全绿 + 504 ethoinsight 回归 |
| data-analyst seal prompt 修复 | ✅ 已合 + 28 测试全绿 |
| serve.sh dev 启动提速 | ✅ 已合 (待用户重启 Gateway 验证 ~10s) |
| 全量测试 | ✅ ethoinsight 504 / agent backend 3111 / frontend tsc 0 错 |

### Spec 库

| Spec | 状态 | 备注 |
|---|---|---|
| roadmap v2 | living | 加了 Sprint 5 双重提示约束 |
| Sprint 0 | 实施完成 | |
| Sprint 1 | 实施完成 + Banner 通道补完 | |
| Sprint 2a | 实施完成 | |
| Sprint 2b | 待执行 | 严格串行 (等 2a 合 dev 后启动,现在 2a 已合) |
| Sprint 3 | 待执行 | 依赖 Sprint 2b |
| Sprint 4 | 待执行 | 依赖 Sprint 2a + 行为学同事 review |
| Sprint 4.5 | 待执行 | 依赖 Sprint 2b |
| Sprint 5 | 未写 spec | roadmap 概要 + 双重提示约束 |
| Sprint 5.5 | 未写 spec | roadmap 概要 |
| Sprint 5.7 (新增) | 待执行 | **独立于 5/5.5,可任何时点启动** |
| Sprint 6/7/8 | 未写 spec | |

### dogfood 验证状态

- EPM n=1 thread `68f6da40-6c62-494e-922c-8e3c82c156b5` 端到端跑通 (从 inspect → set_paradigm → prep_metric_plan → code-executor → data-analyst (第二次重试成功) → chart-maker → report-writer)
- Banner 红字 "1 条数据质量警告 (含 1 条阻断级)" 在 lead 播报 AIMessage 下方渲染成功
- data-analyst seal 漏调 bug 已定位 + 真根因修复 (prompt 编号)

### 工作区残留 (没动)

- `.env.wecom` — 敏感凭证,**永不 commit**
- `docs/handoffs/2026-05/2026-05-27-p0-metrics-bugfix-and-ev19-audit-handoff.md`
- `docs/handoffs/2026-05/2026-05-28-b2-b10-implementation-complete-handoff.md`
- `docs/handoffs/2026-05/2026-05-28-sprint-0-dogfood-fix-and-roadmap-spec-handoff.md`
- `docs/superpowers/plans/2026-05-27-p2-p3-ev19-completion-plan.md`

下次清理时判断是否单独 commit / .gitignore / 删除。

---

## 3. 下一会话(新 agent)的执行清单

### A. 立即可做 (用户回来后第一件事)

**deploy 决策**:用户原计划今天发版 demo,但精力散在 Sprint 2a 复核 + bug 诊断 + spec 写作上,**今天没 deploy**。下次会话可考虑:

1. 打 release tag (`release-2026-05-28-sprint1-2a`) 锚定快照
2. `cd packages/agent && make deploy-tar` 推 ECS
3. dogfood:上传 EPM xlsx 验证 banner 红字 + 整条 E2E

预期改进 (相对 5/19 当前 main):
- ✅ data_quality_warnings 结构化 + banner (Sprint 1)
- ✅ catalog 参数 SSOT (Sprint 2a)
- ✅ EV19 自动识别分组 (5-28 dogfood fix)
- ✅ artifacts 504 修复 (5-28 dogfood fix)
- ✅ 企微 channel 文件上传修复 (deb0e350)
- ✅ DeerFlow sync 5/25 + 4 项 surgical merge
- ✅ FST E2E 7 处修复 + chart-maker 5 根因修复

### B. Sprint 5.7 实施 (独立 agent 可启动)

`docs/superpowers/specs/2026-05-28-sprint-5.7-handoff-emission-validator-design.md` 已就位。0.5 周。

关键挂载点:`executor.py:723` 改 `try_set_terminal(COMPLETED)` 为 validate-first 路径。

**dogfood 验收方法 (spec §5)**:临时 revert `2df10880` 让 data_analyst.py 编号冲突复现 → 跑 EPM n=1 → 验证 task 标 FAILED + lead 自动 retry → 第二次大概率成功。

### C. Sprint 2b 实施 (独立 agent 可启动)

Sprint 2a 已合 dev (`da924145`),Sprint 2b 严格串行启动条件满足。

Spec:`docs/superpowers/specs/2026-05-28-sprint-2b-parameter-pipeline-design.md`

**Sprint 2b agent 启动备注**:Sprint 2a 没在 immobility metric 上挂 `parameters_ref` (spec §2.3 写"按实际加,先 grep")。2b 改 `compute_immobility_*.py` CLI 参数时,需要顺手补 `parameters_ref: [velocity_threshold, velocity_min_duration]` 到 FST/TST 的 6 个 immobility metric (catalog/fst.yaml + tst.yaml 各 3 个)。

### D. 重启 Gateway 验证 serve.sh 提速

用户重启时验证 100s → 10s 是否落地。如果还是慢,可能 lifespan 内 init (langgraph_runtime / SQLite / channel_service) 才是瓶颈,不是 watchfiles。

### E. 是否独立 commit 3 个老 handoff 文件 + 1 个老 plan

用户暂未决定。本会话 commit 都避开了这 4 个 untracked 文件。

---

## 4. 教训记录 (待写入 MEMORY.md 候选)

### LLM 工具调用的工程化兜底

**核心教训**:依赖 LLM"记得调 tool"是反工程化。所有产生关键产物的 subagent 终结路径必须有 harness 层强约束验证契约履行。

具体实例:
- Sprint 1 实施时 data_analyst.py 留下"两个 step 2"编号 bug
- Sprint 1 dogfood 当时 LLM 运气好 emit 了 seal tool_call,bug 没暴露
- Sprint 2a dogfood (5-28) LLM 运气差,bug 暴露 + task 静默 COMPLETED 但 handoff 不存在
- lead 自救式 re-dispatch 救了场,但不该依赖运气
- **根治 = Sprint 5.7 executor 末端 handoff 存在性检查**

### Sprint 实施 spec 编号冲突的检测哨兵

agent 实施 spec 改 prompt workflow 时,要 grep `^\d+\.` 验证编号唯一性。Sprint 1 spec §2.3 "在步骤 2.5 (quality warnings) 和 2.6 (按范式 read 判读文档) 之间插入步骤" 没明示原 step 2 该升 2.7,agent 没自动注意。

### Sprint 1 spec §2.5 与实现的契约缺口

spec 写"frontend 从 SSE stream 中读 data_quality_warnings"略含糊,实际实现需要 middleware 把 handoff_data_analyst.json 转发到 message kwargs。**下次写涉及前端/后端数据流的 spec 时,明确写出 transport 机制**,不要假设"自然流过去"。

### uvicorn --reload 显式 include 必须含 *.py

这是 watchfiles 隐藏行为:`--reload-include` 列表一旦给出,**只**匹配 include 里的 pattern,不会自动包含 *.py。Sprint 1 之前 serve.sh 漏写 *.py,导致 watchfiles 走 fallback 全量扫描 sys.path 的 42538 个文件,启动 100+s。

---

## 5. 关键文件地图

### 本次会话核心产出

| 类别 | 文件 |
|---|---|
| Quality Warning Banner middleware | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/quality_warning_broadcast_middleware.py` |
| Banner middleware 测试 | `packages/agent/backend/tests/test_quality_warning_broadcast_middleware.py` |
| Frontend Banner 组件 | `packages/agent/frontend/src/components/workspace/messages/quality-warning-banner.tsx` |
| Sprint 3 spec | `docs/superpowers/specs/2026-05-28-sprint-3-data-analyst-parameter-audit-design.md` |
| Sprint 4 spec | `docs/superpowers/specs/2026-05-28-sprint-4-tuning-guide-design.md` |
| Sprint 4.5 spec | `docs/superpowers/specs/2026-05-28-sprint-4.5-analysis-config-id-design.md` |
| Sprint 5.7 spec (新增,根治 seal 漏调) | `docs/superpowers/specs/2026-05-28-sprint-5.7-handoff-emission-validator-design.md` |
| data-analyst seal 漏调修复 | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py:95` |
| serve.sh 提速 | `packages/agent/scripts/serve.sh:153` |
| Sprint 2a catalog SSOT | `packages/ethoinsight/ethoinsight/catalog/{schema.py,loader.py,_common.yaml,*.yaml}` + `metrics/dispatcher.py` |

### dogfood reference

| 资源 | 路径 |
|---|---|
| EPM n=1 dogfood thread training-data | `packages/agent/backend/.deer-flow/training-data/auto-collected/68f6da40-6c62-494e-922c-8e3c82c156b5.jsonl` |
| data-analyst 漏 seal 证据 | 同上 row 14 |
| data-analyst 成功 seal (re-dispatch) | 同上 row 15 |

---

## 6. milestone 建议

本次会话让 SOTA agent v2 路线图 track 推进到:
- **Sprint 1 完整 dogfood-pass** (Banner 端到端验证)
- **Sprint 2a 完成 + 合 dev**
- **5 份累积 spec** (3/4/4.5/5.7 + Sprint 0/1/2a/2b 已存在)
- **data-analyst seal 漏调 bug 真根因 + prompt 修复 + 工程化根治 spec**

应创建/更新的 milestone:
- `sota-agent-v2`: 状态更新为 "Sprint 0+1+2a 完成 + 4 份新 spec 排队 + Sprint 5.7 兜底 spec"
- `quality-warning-banner`: 新建 milestone,记录 banner 通道的设计 + spec gap + middleware 修复 + dogfood 验证 + 与 Sprint 5 双重提示约束的关系

---

## 7. 给下一 agent 的友情提示

1. **dev HEAD 是 `bbff6a33`**,所有改动都已 push origin
2. **测试基线**:ethoinsight 504 / agent backend 3111 / frontend tsc 0 错。下次实施任何 sprint **必须保证基线不退化**
3. **`make deploy-tar` 还没跑** — 用户原计划今天 deploy 但精力被会话内其他事消耗。下次会话第一件事可询问用户是否仍要 deploy
4. **Sprint 2b 启动备注**:Sprint 2a 留了 immobility metric parameters_ref 没挂的小坑,2b agent 启动时顺手补 (FST/TST 的 6 个 compute_immobility_* metric)
5. **Sprint 5.7 spec 已就位**,可独立实施;它是 LLM 漏调 seal tool 这一类系统性 bug 的工程化根治方案,不依赖 5/5.5 推进
6. **不要重蹈 Sprint 1 spec 的覆辙**:写涉及 frontend/backend 数据流的 spec 时,明确写出 transport 机制(不要假设 SSE 自然流过去)
7. **prompt 编号检查**:agent 实施 spec 改 subagent prompt workflow 时,grep `^\d+\.` 验证编号唯一性

---

## 后续进展（2026-05-29 追加）

- **Sprint 5.7 实施文档已产出**：[`docs/superpowers/plans/2026-05-29-sprint-5.7-handoff-emission-validator-impl.md`](../../superpowers/plans/2026-05-29-sprint-5.7-handoff-emission-validator-impl.md)。基于 spec，但对 dev HEAD `bbff6a33` 真实代码逐项核验后**修正了原 spec 3 处会让实施 agent 踩空的点**：
  - **C1**：spec §2.3 让自造 `_resolve_workspace_path`，但 `ThreadDataState` 是 TypedDict（运行时即 dict）且 deerflow 已有现成 `resolve_workspace_from_state`（experiment_context.py:79）→ 改为 inline `self.thread_data.get("workspace_path")` + isinstance 守护，复用 > 自造
  - **C2**：`config.name` 用连字符（`data-analyst`），但 `HANDOFF_FILE_REGISTRY` 的 key 用下划线（`data_analyst`）→ 白名单字典 key 必须用连字符，不能直接拿 registry key 比
  - **C3**：723 行 COMPLETED 在 `try` 块内（725 行 except→FAILED）→ validation helper 内部必须 try/except fail-open，否则 IO 抖动会把正常 task 误判 FAILED；额外加了 spec 没有的鲁棒性测试 case
- **已交独立 agent 实施**（2026-05-29）。
- 文档同步更新：milestone README + sota-agent-v2-sprint-0.md（状态推进到 Sprint 0/1/2a 合 dev + 5.7 实施中，dev HEAD bbff6a33）+ roadmap v2（正式登记 Sprint 5.7 章节 + 依赖图）。

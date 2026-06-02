# 2026-06-01 剩余 Sprint 实施 Brief（交新 agent 执行）

> **本文用途**：把 roadmap v2 里"待实施"的 sprint，按**经代码核实的真实缺口**重写成可直接执行的实施说明。新 agent 拿这份文件即可开工，不需要再去 roadmap 里猜"改动是设计意图还是已验事实"。
>
> **权威来源**：roadmap v2 = `docs/plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md`（设计意图 + grill 复审结论）。本 brief = 2026-06-01 用 grep 核实后的真实代码状态 + 精确缺口。**两者冲突时以本 brief 的代码核实为准**（roadmap 的"已实施清单"有 stale）。
>
> **dev HEAD**：`29539c85`（2026-06-01 会话起点）。本会话先修了 3 个 red 测试（见 §0）。

---

## 0. 前置：基线已修复（本会话已做，新 agent 无需重做）

**问题**：2026-05-29 handoff 写"3729 passed / 0 failed"，但 2026-06-01 全量跑出 **3 failed**：
- `test_executor_handoff_emission.py::...::test_completed_subagent_with_handoff_passes`
- `test_seal_resume.py::...::test_resume_not_triggered_when_handoff_present`
- `test_seal_resume.py::...::test_resume_recovers_writes_handoff`

**根因**：PR #66（Sprint 5.5 handoff 内容非空校验）合入后，`_validate_handoff_emitted` 不再只查文件存在，还查核心字段非空（`_HANDOFF_CONTENT_CHECKS`）。但这 3 个旧测试的 fixture 用空 `"{}"` 占位 handoff，被新校验正确判 FAILED。**产品逻辑无错，错在 PR #66 漏更新旧测试 fixture**（合并时只跑了新测试没跑全量）。

**已修**：3 处 fixture 从 `"{}"` 改为满足契约的最小合法内容（`{"key_findings": [...]}` 等），并在两个测试文件加了 `_MINIMAL_VALID_HANDOFF` 映射。纯测试修复、零产品风险。

**教训沉淀**：PR 合并前必须跑**全量** `make test`，不能只跑新增测试——尤其当 PR 改的是被多处测试共享的判定逻辑（如 handoff validator）。

---

## 1. 经代码核实的真实 Sprint 状态（2026-06-01 grep 实测）

| Sprint | roadmap 自报 | **代码核实** | 本 brief 处理 |
|---|---|---|---|
| S0 handoff schema + seal tools | ✅ | ✅ 真实 | 完成，不动 |
| S1 data_quality 结构化 | ✅ | ✅ 真实（含 QualityWarningBroadcastMiddleware） | 完成，不动 |
| S2a 参数下沉 catalog | ✅ | ✅ ParamSpec + shared_parameters | 完成，不动 |
| S2b 参数管线 5 跳 | ✅ | ✅ resolve.py 含 overrides 合并逻辑 + parameters_in_use | 完成，不动 |
| **S3 参数审计** | ❌ 待实施 | ✅ **实际已实施**（data_analyst workflow + ParameterAuditFinding schema） | 见 §2（仅缺 FST mobility 判据，卡 #63 同事） |
| **S4 调参指南** | ❌ 待实施 | ❌ by-experiment md 无"参数调整指南"段 | 见 §3（工程通路可做，内容卡同事） |
| **S4.5 analysis_config_id** | ❌ 待实施 | 🟡 **半成品**：schema 字段有，helper + overrides 读取无 | 见 §4（**最高优先级，不卡人**） |
| **S5 数据质量门** | ❌ 待实施 | ❌ data_quality_provider.py 不存在；workflow_mode 地基在 | 见 §5 |
| S5.5 / S5.7 / S5.8 | ✅ | ✅ 真实 | 完成，不动 |
| **S6 跨会话 memory** | ❌ 待实施 | ❌ seal tool 无 add_fact | 见 §6（P2/v0.2） |
| **S7 假设面板** | ❌ 待实施 | ❌ present_assumptions.py 不存在 | 见 §7（P2/v0.2） |
| **S8 feedback 回流** | ❌ 待实施 | ❌（按 roadmap） | 见 §8（最低优先级） |

**推进顺序建议**（依赖 + 不卡外部优先）：
1. **S4.5**（半成品收尾，0.5 周，不卡人，是 S6/S7 前置）← 先做
2. **S5**（数据质量门，~1 周，依赖 S1✅ 已满足）
3. **S4 工程通路 + S3 FST 判据**（卡 #63 同事，先搭通路）
4. **S6 / S7**（P2/v0.2，公测后）
5. **S8**（最低优先级，微调到位后收益减半）

---

## 2. Sprint 3：参数审计（已实施，仅补 FST mobility 判据）🟡 卡 #63

**真实状态**：`data_analyst.py` workflow 已含参数适配性检查段（grep `parameter_audit` 命中 8 处），`handoff_schemas.py` 有 `ParameterAuditFinding` 类 + `parameter_audit_findings` 字段 + `parameter_audit_findings_count/critical_count` gate_signals。**结构完整。**

**唯一缺口**：2026-05-29 FST dogfood 实测 `parameter_audit_findings_count: 0`，但 data-analyst 正确识别 FST 不动时间 0.56s 异常低（疑 EV XT Mobility detection 未校准）——**参数审计没触发 FST 的 mobility threshold**。这是真实覆盖缺口，不是 bug。

**为何卡**：补判据需要行为学同事 issue #63（velocity 物种判据）的答复——"FST 大鼠 mobility 阈值应该是多少、什么算异常低"是行为学判断，不是工程能拍板的（[[feedback_ssot_lives_in_review_packages]]）。

**新 agent 动作**：
- **#63 答复前**：不动。不要自己编 mobility 阈值数字（那是越权写领域知识 + 伪造证据）。
- **#63 答复后**：在 catalog 的 fst.yaml / tst.yaml 加 mobility 相关 `parameters`（换数字，不重做结构），data_analyst 的审计段会自动比对。

---

## 3. Sprint 4：调参指南进 by-experiment md（工程通路可做，内容卡同事）🔴 SSOT 归属

**真实状态**：`by-experiment/*.md` 没有"## 参数调整指南"段。

**SSOT 归属铁律**（grill 复审 + [[feedback_ssot_lives_in_review_packages]] / [[feedback_ssot_skill_deployment_distinction]]）：
- `by-experiment/*.md` 的内容 SSOT = **行为学同事，走 review-packages**，不是工程团队直接编辑。
- "调到多少、为什么"是行为学判断。工程直接写 → 会被 review-packages 同步覆盖 + 越权写领域知识。

**新 agent 可做的（工程通路，不碰内容）**：
- 在 `data_analyst.py` workflow 加一步："read paradigm md 时 grep `## 参数调整指南` 段"——通路搭好，段不存在时优雅跳过。
- **不要自己写指南内容**。内容留空待同事 PR。

**不动**：`ev19-dependent-variables.md`（它是公式 SSOT，不混应用指南）。

---

## 4. Sprint 4.5：analysis_config_id 收尾（半成品，最高优先级，不卡人）🟢

**这是本批次最该先做的——半成品，且卡着 S6/S7 下游，不依赖任何外部。**

**已做（核实）**：
- 所有 4 个 handoff 类（CodeExecutor/DataAnalyst/ChartMaker/ReportWriter）已有 `analysis_config_id: str` 字段（`handoff_schemas.py`）。
- `resolve.py` 已支持 `overrides: dict` 参数（S2b 做的，含 default+ref+override 三层合并逻辑，line ~754/800）。

**缺口（精确）**：
1. **`experiment_context.py` 无 `parameter_overrides` 字段、无 `compute_analysis_config_id` helper**（grep 两者均无命中）。`set_experiment_paradigm_tool`（line 174）签名只有 `acknowledge_quality`，没有 `parameter_overrides`。
2. **`prep_metric_plan_tool.py` 调 `resolve_metrics()`（line 234-241）没传 `overrides=`**——所以即使有 override 也进不到 plan。
3. **resolve.py 的 `--overrides-file` CLI 入口**：`resolve_metrics()` 函数支持 overrides，但 `__main__`（line 974）的 CLI 是否暴露 `--overrides-file` 需核实（grep `add_argument` 未命中，可能 CLI 走的不是 argparse；prep_metric_plan 是直接 import 调用 `resolve_metrics()`，不走 CLI，所以 CLI 入口非必需——**优先走函数调用路径**）。

**实施步骤**：

### 4.1 experiment_context.py
- `experiment-context.json` schema 顶层加：
  - `analysis_config_id: str`（每次新参数自动计算）
  - `parameter_overrides: dict[str, float|int|str]`（默认 `{}`）
- 新增 helper `compute_analysis_config_id(catalog_default: dict, overrides: dict) -> str`：
  - `normalize`（sort keys）→ `json.dumps`（canonical）→ `sha256` → 取前 16 hex 字符
  - **deterministic**：同 input 必同 id（验收 #2 靠这个）
- `set_experiment_paradigm_tool` 加可选参数 `parameter_overrides: dict | None = None`：
  - 设置/修改 overrides 时，调 `compute_analysis_config_id` 生成新 id 写入 context。
  - **正面指令写法**（CLAUDE.md §6 deepseek 偏好）：参数 docstring 描述"传入用户确认的参数覆盖"，不要写"不要乱传"。

### 4.2 prep_metric_plan_tool.py
- Step 5（line 233 前）读 `experiment-context.json` 的 `parameter_overrides`。
- `resolve_metrics(...)` 调用（line 234）加 `overrides=parameter_overrides`。
- 把 overrides 写到 `/mnt/user-data/workspace/overrides_<config_id>.json`（lineage 留痕，可选但建议）。

### 4.3 seal_handoff_tools.py
- 4 个 seal tool 内部，自动从 `experiment-context.json` 读 `analysis_config_id` 注入 handoff（subagent 不手动传）。**复用现有读 context 的逻辑**（seal tool 已读 context 取 paradigm/ev19_template，加一个字段即可）。

### 4.4 lead_agent/prompt.py
- 报告末尾模板加 `{analysis_config_id}`（"本次分析标识：xxxx-xxxx"）。
- （S7 联动）假设面板顶部展示——S7 做时再接。

**验收**：
1. 两次不同参数跑同一份数据 → 报告末尾 `analysis_config_id` 不同，可反查参数差异。
2. 同一 overrides 两次跑 → 完全相同的 id（deterministic）。
3. overrides 为空（首次跑）→ id 仍生成（基于纯 catalog default 的 hash）。
4. **TDD**：`compute_analysis_config_id` 的纯函数单测（同 input 同 id、不同 input 不同 id、空 overrides 也有 id）。

---

## 5. Sprint 5：GuardrailProvider 数据质量门（~1 周，依赖 S1✅ 已满足）🟢

**真实状态**：`data_quality_provider.py` 不存在。但 **workflow_mode 双轨地基已就位**（`agent.py:429/512` 读 `workflow_mode`，`gate_enforcement_middleware.py` 已是"仅 manual 启用"的模板）。S1 的 `blocks_downstream` 字段已在 DataQualityWarning。

**克隆模板**：`Ev19TemplateGuardrailProvider`（`guardrails/ev19_template_provider.py`）是完美模板——ContextVar bridge（`_ev19_workspace`，line 31）、`evaluate(request) -> GuardrailDecision`（line 62）、含明确指令的 deny 消息都在。**本质是克隆 + 改读路径 + 改判定 + 改 deny 文案。**

> ⚠️ grill 复审提醒：roadmap 说"1 周富余"是乐观（5.8 也以为简单、核验冒出 3 坑）。**实施前先核验依赖就位**：`blocks_downstream`（S1✅）/ `workflow_mode`（✅）/ `gate2_quality_acknowledged`（需确认 `set_experiment_paradigm(acknowledge_quality=True)` 是否真写了这个 key）。估期可能 >1 周。

**Auto/Manual 双轨**（grill 锁定）：
| 模式 | 行为 | 实现 |
|---|---|---|
| `auto`（默认）| 不阻断，UI 红字（S1 已做） | **不挂** provider |
| `manual`（飞轮专家）| guardrail 拦下游 + 必须 ack | **挂** provider（仿 gate_enforcement 在 manual 启用）|

**实施**：
- 新建 `guardrails/data_quality_provider.py`（克隆 ev19 模板）：
  - 拦 `tool_name == "task"` 且 `subagent_type ∈ {data-analyst, chart-maker, report-writer}`（knowledge-assistant 免拦）。
  - 读 handoff 中 `severity=critical AND blocks_downstream=true` 的 warning。
  - 查 `experiment-context.json` 是否含 `gate2_quality_acknowledged` → 已确认放行。
  - 未确认 → **deny + 含明确指令**（[[feedback_deny_messages_must_direct]]）："请先 ask_clarification 告知用户以下 N 条质量问题：[code+message]，等用户确认后调 set_experiment_paradigm(acknowledge_quality=True) 再继续"。
  - **deny 消息直接带结构化 warning payload**（code/evidence/blocks_downstream），让前端用同一个 QualityWarningBanner 渲染——避免 EPM n=1 dogfood 的"双重提示"问题（粗粒度中断 + banner 二次展示）。
- `guardrails/__init__.py` 注册。
- `agent.py` **仅 `workflow_mode=manual` 时**挂（仿 gate_enforcement，line ~430）。
- `lead_agent/prompt.py` manual 模式调度边界加对应规则。

**验收**：manual 模式下，critical+blocks_downstream warning 出现 → 派 data-analyst 被拦 → ask_clarification → acknowledge → 放行；auto 模式不挂、不拦。

---

## 6. Sprint 6：跨会话范式 memory（P2/v0.2，0.5 周）🔵

**真实状态**：seal tool 无 `add_fact` / `experiment_summary`（grep 未命中）。

**关键原则**（grill 锁定，已吸取"LLM 抽取数字漂移"教训）：**experiment_summary 走确定性写入**，不走 LLM 抽取通道。复用 deerflow 现有 facts 通道（per-user 隔离 / facts 结构 / system prompt 注入 / eviction 全就位），不建新顶层结构。

**实施**：
- `agents/memory/storage.py` — 若无 `add_fact()` 公共方法则加（load → append → save helper）。
- `tools/builtins/seal_handoff_tools.py` 的 `seal_report_writer_handoff` 内部，atomic write + manifest 之后，**额外调** `memory_storage.add_fact(user_id, fact)`：
  - content 形如 `"{paradigm} analysis on {date}: n_per_group={n}, {metric1}={v1}; effect_size={es}; analysis_config_id={config_id}"`（数字来自 handoff 结构化字段，不经 LLM）
  - `category="experiment_summary"`、`confidence=1.0`、`source=f"thread:{tid}, config_id:{cid}"`
- **不动**：MemoryMiddleware / updater / config / prompt（现有路径全不变）。

> ⚠️ 协调：改 `seal_report_writer_handoff` 内部——与 S5.8 seal-resume 改**同一 seal 路径**，需协调顺序（先确认 5.8 的 seal 逻辑稳定）。
> ⚠️ PRD 把"跨范式关联"划出 MVP——确认 v0.1 是否真要做（这是主人哲学质变能力，非公测标准）。建议留 v0.2。

**验收**：完成一次 EPM → memory.json 出现 `category=experiment_summary` fact（含精确数字）；第二会话 EPM 时 system prompt `<memory>` 段含上次 fact；跑 200 次后仍 ≤100 facts（eviction），confidence=1.0 不被挤掉。

---

## 7. Sprint 7：假设暴露面板（P2/v0.2，0.5 周，轻量不强制）🔵

**真实状态**：`present_assumptions.py` 不存在。

**已自我纠偏**（grill 复审"最健康"）：**不做** GateProvider 强制每次渲染（用户看冗余卡片几次就无视）。改**可主动调用的聚合工具** + lead prompt 建议性指引。

**依赖**：S4.5（analysis_config_id ✅ 本批次做）/ S2b（parameters_in_use ✅）/ S3（parameter_audit ✅）。**本批次做完 S4.5 后依赖即满足。**

**实施**：
- 新建 `tools/builtins/present_assumptions.py`（first-party tool）：聚合查询
  - plan_metrics.json → in-use parameters（S2b）
  - handoff_data_analyst.json → parameter_audit / quality_warnings（S1/S3）
  - experiment-context.json → gate_completed / parameter_overrides / analysis_config_id（S4.5）
  - 渲染成 markdown 段（前端 collapsed 卡片）。
- `lead_agent/prompt.py` final delivery 段加**建议性**指引（非强制）："含 critical warning / 参数 override / 假设面板可能有价值时，主动调 present_assumptions"。
- 前端新增折叠卡片组件（默认 collapsed，标题"分析假设摘要 (config_id=...)"）。

**不做**：AssumptionPanelGateProvider 强制 / 每次 final delivery 强制渲染。

**验收**：含 critical warning 或 override 的分析末尾，lead 主动调并渲染折叠卡；简单分析（无 warning + 无 override + 全 default）lead 不调。

---

## 8. Sprint 8：feedback verdict 回流到 prompt（最低优先级，2 周）🔵

**自标"微调到位后收益减半"——最后做。**

**前置（grep 实测）**：`feedback` 表 schema 无 `paradigm` 字段，"查同 paradigm 上次 verdict ≠ correct"无法直查。
- **方案 A（推荐）**：feedback 表加 `paradigm: str | None`（schema migration + 提交时从 experiment-context.json 读 paradigm 落盘）。历史数据 paradigm 留 null。

**实施**：
- `app/persistence/feedback_repository.py` — schema 加 `paradigm` + migration。
- `app/gateway/routers/feedback.py` — 提交时读 paradigm 一并写；加查询 `GET /api/feedback/prior_corrections?paradigm=epm&user_id=X&limit=3`。
- `lead_agent/agent.py` — `make_lead_agent` 构 prompt 前 fetch prior_corrections。
- `lead_agent/prompt.py` — 模板加 `{prior_corrections}` 插槽。

---

## 9. 横切纪律（每个 sprint 都适用）

1. **改 `data_analyst.py` workflow 必 grep 编号唯一性**（`^\d+\.`）——S5.7 的 seal bug 就是"两个 step 2"编号冲突引起，复发代价高。S3/S5/S6 都改它。
2. **每个 sprint 收尾跑全量 `make test`**（不只新测试）——§0 教训。
3. **TDD 强制**（CLAUDE.md）：每个改动带单测。
4. **deny 必须含明确指令**（[[feedback_deny_messages_must_direct]]）：信息完备不够，必须"请改调 X 因为 Y 然后做 Z"。
5. **SSOT 唯一**：参数默认值只在 catalog YAML；调参权衡只在 paradigm md（同事写）；不双存。
6. **复用 deerflow 现成机制**优先于自造（memory facts / GuardrailProvider 模板 / workflow_mode）。
7. **deepseek 正面指令**（CLAUDE.md §6）：prompt/docstring 用"想要的行为"，不用"禁止 X"。

---

## 10. 参考锚点（本 brief 核实过的关键文件:行）

| 用途 | 路径:锚点 |
|---|---|
| handoff 内容校验判据 | `subagents/executor.py:104-134`（_HANDOFF_CONTENT_CHECKS）|
| S4.5 set_experiment_paradigm | `agents/middlewares/experiment_context.py:174`（缺 parameter_overrides）|
| S4.5 resolve overrides 已支持 | `catalog/resolve.py:754,800`（overrides 三层合并）|
| S4.5 prep 调 resolve 未传 override | `tools/builtins/prep_metric_plan_tool.py:234-241`|
| S5 克隆模板 | `guardrails/ev19_template_provider.py:31,62`（ContextVar + evaluate）|
| S5 manual 双轨地基 | `agents/lead_agent/agent.py:429,512` + `gate_enforcement_middleware.py`|
| S3 已实施 | `subagents/builtins/data_analyst.py`（parameter_audit ×8）+ `handoff_schemas.py:217-227,443`|

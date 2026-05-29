# SOTA Agent 构建路线图 v2：从"橡皮泥"到"可复现 AI Native Agent"

## 背景

### 项目现状

EthoInsight 是一个行为学 AI 分析助手，基于 DeerFlow 框架构建。用户上传 EthoVision XT 轨迹数据，agent 自动完成统计分析、图表生成、报告撰写。

近期完成了大量基础设施工作：P0 bug 修复（3 处）、P1（pendulum detect、activity_intensity 修复、velocity fallback）、P2（图表增强）、P3（基础指标补全 + catalog + CLI）、EV19 公式参考接入、Noldus 48 算法对照、chart 置信度分级。当前 ethoinsight 439 passed，agent backend 3043 passed。

### 从哪里来的

这个路线图源于一场关于尼采哲学的讨论：

> 凭什么 Noldus 的定义是对的？Noldus 只是另一个解释者，不是上帝。他们的 Activity 公式、Mobility State 编码、immobility 阈值——只是**视角**（perspectivism），不是客观真理。

当前 agent 像一个"奴隶"：机械执行 Noldus 公式、被动响应请求、不质疑输入、不跨实验关联。向"主人"的转变是：agent 知道定义从哪来，也知道的局限，能帮用户选择或创造更适合的分析视角。

**核心纠正**（Opus 审查）："主人"不是 agent 擅自调参，而是**暴露每一项假设并请用户决策**。参数审计只警告，不改参——agent 多走半步等于伪造证据。

### 当前 agent 的 5 个局限

1. **不质疑输入** — 数据有异常不警告。但 dispatcher.py 已经生成了 9 处 `data_quality_warnings`，只是没有浮出水面。
2. **不拒绝 + 替代** — 不支持的功能说"不支持"，不提供替代方案。用户要 n=3 的箱线图，不会建议用小提琴图+散点。
3. **不发现用户没问的问题** — 不会指出潜在混淆因素（"treatment 组 total distance 也降了 40%，可能不是特异性焦虑"）。
4. **不跨实验关联** — 上次会话和这次完全独立。
5. **不质疑自己的工具** — immobility 阈值硬编码 30mm/s，不适应用户数据。

### 细调后应具备的 5 个新能力

1. **输入质疑** — 识别数据质量问题，计算前主动提问或自动处理
2. **方法适配** — 根据实际数据特征判断预设方法是否合适（n 太少跳过 Shapiro-Wilk，方差不齐用 Welch）
3. **假设生成** — 看到数字后产生因果假设并建议验证路径
4. **跨会话记忆** — "你上次 EPM 也焦虑，OFT 现在也焦虑——汇聚置信度更高"
5. **拒绝+替代** — "箱线图对 n=3 很差，建议小提琴图+散点叠加"

### 新增核心设计目标（v2 新增）

经过架构讨论后，v2 路线图新增三个硬性设计目标：

6. **可复现性两层结构** — 确定性层（解析→指标计算→统计检验）必须 bit-identical；概率性层（解读→报告）可以同义不同字。两层之间的接口必须封闭、可校验。
7. **参数完整 lineage** — 从 raw data → 解析参数 → 指标参数 → 统计决策路径 → 图表参数 → 最终 artifact，每一步的参数选择都可追溯。
8. **AI Native 架构边界** — LLM 参与控制流决策（策略、解读、质疑），不参与计算（指标、统计、图表）。AI 做 PI 的认知功能，不做 RA 的计算功能。

### 为何基于 DeerFlow

DeerFlow 已有 5 个机制可以支撑这些能力，不需要新架构：

| DeerFlow 机制 | 当前使用 | SOTA 用途 |
|--------------|---------|----------|
| MemoryMiddleware | 只存 user facts | 存实验摘要，跨会话引用 |
| Skills 渐进披露 | 只用于 EV19 模板 | 注入方法论指导（统计选择、图表选择） |
| ClarificationMiddleware | 只用于范式反问 | 反问参数选择、图表选择 |
| GuardrailMiddleware（含 9 个 provider） | 没用 | 数据质量门禁 |
| TrainingDataMiddleware | 已录制但被动 | 每次决策都是 SFT 种子 |

---

## 代码审计发现（v2 新增）

在制定修正路线图之前，对现有代码做了逐文件审计。以下发现直接决定了 Sprint 的拆分和优先级调整：

### 发现 1：catalog 是"半 SSOT"

`catalog/schema.py` 的 `MetricEntry` 只有 display 相关字段（`id / script / requires_columns / output_unit / display_name_zh / unit_zh / one_liner / direction_for_anxiety / statistical_default`），**完全没有 `parameters` 字段**。同时 `metrics/_common.py:219` 硬编码 `_VELOCITY_THRESHOLD_MM_S = 30.0`，`metrics/_pendulum.py:22` 硬编码 `PERIODICITY_THRESHOLD = 0.55`。

参数从 catalog → PlanMetric → CLI args → 执行 → handoff → data-analyst prompt 一共**五跳**，每一跳都要新代码打洞。原路线图说"2-3 周"是严重低估。

### 发现 2：ChartMakerHandoff schema 完全缺失

`handoff_registry.py:16` 注册了 `chart_maker: handoff_chart_maker.json`，但 `handoff_schemas.py` 的 `__all__` 没有 `ChartMakerHandoff` 类。CodeExecutorHandoff / DataAnalystHandoff / ReportWriterHandoff 三个已就位，chart 这一环是断的。图表参数（colormap、bin 数、置信区间画法）完全不在 lineage 里。

### 发现 3：handoff validation 是 soft 的

`agents/middlewares/experiment_context.py:101-107` — schema violation 只写进 `_schema_violations` 字段，**返回的还是原 dict**。lead 仍然按错误结构推理。"封闭接口"目前是个建议，不是强制契约。

### 发现 4：data_quality_warnings 缺 code 和 evidence 字段

`dispatcher.py:169-300` 的 9 处 warning 结构是 `{severity, metric, message}` — message 是一句中文字符串。`handoff_schemas.py:69-76` 的 `DataQualityWarning` 也只有 `severity / metric / message`。Sprint 1 不是"让现有 warning 浮上水面"，而是先要加上 `code` 和 `evidence` 结构。

### 发现 5：Sprint 5（DataQualityGuardrail）被高估

`Ev19TemplateGuardrailProvider` 已有完整的拦截模板：ContextVar bridge 注入 workspace、evaluate 读文件、含明确指令的 deny 消息。`IntentPostStepAskGateProvider` 已实证 deny 模板。DataQualityGuardrailProvider 本质是克隆——改读路径、改判定逻辑、改 deny 文案。1 周富余。

### 发现 6：Sprint 6 的"LLM 抽取实验摘要"是反模式

`MemoryMiddleware.updater` 走通用 LLM 提取通道（`queue.add → debounce → LLM prompt`），让 LLM 从 markdown 里抽取 `n_per_group / effect_size` 这种精确数字会导致**数字漂移**。必须改成结构化注入：report-writer handoff 加 `ExperimentSummary` 字段 → middleware 直接 `storage.save()`，不经过 LLM。

---

## 2026-05-29 grill 复审（实施进展 + Sprint 5.8 grill 教训驱动的二次修订）

截至 2026-05-29，Sprint 0 / 1 / 2a / 2b / 5.7 已实施（5.7 + 5.8 是新增的 seal 可靠性 sprint）。在为 Sprint 5.8 跑 `/grill-with-docs` 时，暴露出**roadmap 制定方式本身的 4 个系统性问题**，它们不止影响 5.8，而是命中多个未实施 sprint。本段集中记录复审结论，下面各 sprint 标题带 `← grill 复审` 的即受影响项。

### 4 条可泛化教训（来自 grill 5.8 + 3 个 deepseek 真机探针）

1. **roadmap 的"改动清单"基于未核验的代码假设** → 5.8 核验后发现原 spec 3 处会让实施 agent 踩空 + 1 个负优化（max_turns）。**推论**：每个 sprint 真正实施前都需像 5.8 那样核验行号/字段是否真就位，roadmap 的"改动"是设计意图不是已验事实。
2. **roadmap 凭直觉选了"看起来对"的技术方案** → forced tool_choice 被探针证伪是陷阱（dashscope thinking 拒绝 / 关 thinking 产空 args），见 [ADR-0001](../adr/0001-seal-resume-not-forced-tool-choice.md)。**推论**：涉及模型 API 行为的方案必须探针验证，不能假设。
3. **roadmap 没把"模型无关"当硬约束** → 线上 deepseek-v4-pro（dashscope），未来换 Qwen3（vLLM/Fireworks），thinking/tool_choice/reasoning 行为各异。**推论**：任何依赖模型特定行为的设计都会在切模型时爆，防御应放 harness 层。
4. **roadmap 假设了错误的失败机制** → "data-analyst 漏调因撞 max_turns" 被查 dogfood jsonl 证伪（成功仅用 3-4/12 轮）。**推论**：失败归因要有数据，不能拍脑袋。

### 逐 sprint 复审结论

| Sprint | 复审裁决 | 依据（对应教训/原则） |
|---|---|---|
| **3 参数审计** | 🟡 设计对，实施加护栏 | 改 `data_analyst.py` workflow——**这是 5.7 seal bug 的同一文件**（加 step 必须 grep `^\d+\.` 验编号唯一，防 seal bug 复发）；且 3/5/5.8/6 改同一文件同片区域，是**热点**，需协调实施顺序；硬依赖 2b 的 `parameters_used`（已实施 ✅） |
| **4 调参指南** | 🔴 撞 single-source | `by-experiment/*.md` 的 SSOT 归属是**行为学同事（走 review-packages）**，不是工程团队直接编辑（[[feedback_ssot_lives_in_review_packages]] / [[feedback_ssot_skill_deployment_distinction]]）。"调参指南"内容（调到多少、为什么）是行为学判断，应由同事写。**Sprint 4 工程部分只做"data_analyst grep 该段"的通路，内容留空待同事填**，否则会被 review-packages 同步覆盖 + 越权写领域知识 |
| **4.5 analysis_config_id** | 🟢 健康（小问） | deterministic hash 判断对。小问：v0.1 用户改参数频率若极低，则"改参数重跑对比 config_id"用例少、价值低——但成本仅 0.5 周且是"标识"非"防篡改"，可留 |
| **5 数据质量门** | 🟡 估期乐观 | "克隆 Ev19TemplateGuardrailProvider，1 周富余"是 5.8 同款乐观（5.8 也以为简单，核验冒出 3 坑）。依赖未核验：`blocks_downstream`（Sprint 1）/ `workflow_mode=manual`（deerflow 现成？）/ `gate2_quality_acknowledged`（新加）。**估期可能 >1 周，实施前先核验依赖就位** |
| **5.5 lineage 封印** | 🔴 **降级**（最大改动） | 原设计是 handoff 内容 **sha256 封印 + fail-closed**（防篡改，2 周）。但：①威胁不存在——单产品/确定性脚本/自有 subagent 链，没有"恶意篡改 handoff"的威胁模型；②"写一半被读"的 race 已被 Sprint 0 的 atomic write（tmp+rename）覆盖；③用户/demo 无感（违反公测标准导向）；④与 grill 5.8 立案的 [[project_2026-05-29_handoff_content_validation_pending]]（handoff 内容**非空**校验）重叠——后者是真需求（空内容 handoff 静默产垃圾，实际会发生），前者是假需求。**裁决：5.5 从"sha256 hash 封印（2周）"替换为"handoff 内容非空/schema 完整性校验（0.5周）"**，hash 封印推 v0.2 或砍。复用 5.7/5.8 的 validator。 |
| **6 跨会话 memory** | 🟢 健康（2 个注意） | 设计对（复用 deerflow facts 通道、确定性写入不走 LLM 抽取——已吸取"LLM 抽取数字漂移"教训）。注意：①改 `seal_report_writer_handoff` 内部——与 5.8 seal-resume 改**同一 seal 路径**，需协调顺序；②PRD §8 把"跨范式关联"**划出 MVP**，确认 v0.1 是否要做（这是"主人哲学"质变能力，非公测标准） |
| **7 假设面板** | 🟢 最健康 | 已自我纠偏（拒绝 GateProvider 强制、改可主动调用工具）——正是 grill 会强调的"不过度工程"。仅需标依赖（2b ✅/3/5.5） |

### 横切主题（影响多个 sprint）

- **`data_analyst.py` 是改动热点**：Sprint 3 / 5 / 5.8 / 6 都改它的 workflow 或 handoff 路径。实施顺序需串行协调，且每次加 workflow step 必须 grep 编号唯一性（防 5.7 seal bug 复发）。
- **依赖时序**：很多 sprint 依赖尚未实施项的产出（5 依赖 1、5.5 依赖 0+4.5+2b、7 依赖 2b+3+5.5）。**写"代码核验版"spec 前需确认依赖已落地**，否则核验的是不存在的代码状态。当前已实施：0/1/2a/2b/5.7。
- **战略提醒（未决）**：5.5（hash）/6（跨会话）/7（假设面板）属"主人哲学上层建筑"，PRD 公测标准是"6 范式跑得准、不翻车、端到端"。golden-cases 仍为空、OFT/LDB/Zero Maze 无端到端 dogfood——若 v0.1 资源紧张，这几个应让位于"范式端到端验证 + golden-case"。本复审不擅自重排优先级，仅标记。

### 复审后 spec 编写策略

- **5.5 spec 按"内容非空校验"写**（不写 hash 封印）
- **Sprint 4 spec 明确"内容由同事写、工程只做通路"**
- 3/5/6/7 spec 写"设计骨架 + 标依赖/热点/实施前核验点"，真正实施时再升级为代码核验版（如 5.8）
- Sprint 8（feedback 回流）优先级最低（自标"微调到位后收益减半"），最后写

---

## 修正后路线图

### Sprint 0（2 周）：handoff 全面 schema 化 + seal_*_handoff tool 集 ← ★ 新增，最高优先级

**动机**：handoff 是一切下游的承接器。参数 lineage 没地方落、分析配置版本没地方挂、封印机制没东西可 hash——都依赖 handoff schema 先定型。当前 3/4 个 subagent 有 handoff schema（缺 chart_maker），validation 是 soft 的不 fail-closed，且**所有 subagent 都用 LLM 写 JSON 字符串**（漂移面大）。

**核心架构决策（grill 锁定）**：
1. **所有 subagent 改用 first-party tool seal handoff**（不再用 write_file 字符串写 JSON）—— LLM 只填结构化参数，Python 内部 Pydantic 校验 + atomic write + manifest 落盘
2. **atomic write**：seal tool 内部走 tmp 文件 + os.rename 模式（POSIX 原子），消除"写一半被读"的 race
3. **三档 strict mode**：default=`WARN`（log + counter，不抛），灰度 1 周后切 `FAIL_CLOSED`；保留紧急降级文件（`/tmp/disable_strict_handoff`）旁路重启即降级
4. **ChartMakerHandoff schema 以现有 prompt 字段为准**（paradigm/chart_files/failed_charts/summary），不发明新字段名，避免破坏现有 dogfood

**改动**：

- `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` —
  - 新增 `ChartMakerHandoff` 类（**字段 = paradigm/chart_files/failed_charts/summary**，与 chart-maker prompt 对齐）
  - 新增 `FailedChart` 子模型（chart_id, reason）
  - `DataQualityWarning` 增 `code: str`（取自 SAMPLE.*/MOTOR.*/SIGNAL.*/METHOD.* 4 个一级分类）+ `evidence: dict[str, Any]` + `blocks_downstream: bool` 字段
  - `MetricStat` 增 `parameters_used: dict[str, float|str|int]` 字段
  - `CodeExecutorHandoff.inputs` 增 `paradigm: str` + `ev19_template: str` 字段（从 experiment-context.json 冗余过来，handoff 自包含）+ 顶层 `analysis_config_id: str`
  - 所有 4 个 handoff 类增 `analysis_config_id: str` 字段（Sprint 4.5 用）
  - **不在 Sprint 0 加 ExperimentSummary**（Q9/B 锁定：Sprint 6 走 facts 通道，不加新顶层结构）
- `packages/agent/backend/packages/harness/deerflow/subagents/handoff_registry.py` — 注册 `ChartMakerHandoff`
- **`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py` 新建** —
  - 4 个 first-party tool：`seal_code_executor_handoff` / `seal_data_analyst_handoff` / `seal_chart_maker_handoff` / `seal_report_writer_handoff`
  - 共享 helper `_seal_handoff(model_cls, **kwargs, runtime)`：Pydantic 校验 → 写 `.tmp` 文件 → `os.rename` 原子换名 → 写 `.lineage/manifest.json`（Sprint 5.5 加 sha256）→ 返回 `"OK: sealed (sha256=...)"`
  - 每个 tool 用 `@tool` + 函数签名定义参数 schema，LangChain tool_call 自动校验类型/必填
- **`packages/agent/backend/packages/harness/deerflow/subagents/builtins/{code_executor,data_analyst,chart_maker,report_writer}.py` 4 个 subagent prompt** —
  - 工作流末尾 "写 handoff" 步骤改为 "调 seal_X_handoff tool 传入字段"
  - tools 列表 disallow `write_file` 写 `handoff_*.json`（write_file 还保留，但 path 命中 handoff 文件 → 通过 guardrail provider 拒绝）
- `packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py` 扩展 — code-executor 已用 path-allowlist 拒绝越界 write，扩展到拦截 4 个 subagent 用 write_file 直接写 `handoff_*.json`
- `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` — 三档 strict mode：
  - 新增 `HandoffStrictMode` enum：`OFF / WARN / FAIL_CLOSED`
  - `read_handoff()` schema violation：
    - OFF：旧行为，soft + 写 `_schema_violations`
    - WARN：soft + 写 `_schema_violations` + WARNING log + counter（默认）
    - FAIL_CLOSED：raise `HandoffSchemaError`
  - 紧急降级：若 `/tmp/disable_strict_handoff` 存在，强制走 WARN
  - 配置项：`config.handoff_strict_mode`（默认 WARN）
  - 旧会话不兼容：grill 锁定接受（v0.1 频繁变更期）
- `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py`（**前端展示层**）+ 前端 SSE 渲染 — bash 命令文本展示前替换 `python -m ethoinsight.scripts.X.Y` → `ethoinsight: X.Y`（用户看到的不是 Python 调用细节）

**验收**：
1. 所有 4 个 handoff schema 有对应 Pydantic 类（ChartMaker 字段对齐现有 prompt）
2. 4 个 subagent 改用 seal tool，dogfood 跑通 EPM 全流程
3. WARN mode 下，故意写错字段名一个 handoff → 日志出现 WARNING + counter +1，但流程继续
4. FAIL_CLOSED mode 下，同样的错误 → raise HandoffSchemaError
5. `touch /tmp/disable_strict_handoff` → FAIL_CLOSED 立即降级 WARN，无需重启
6. atomic write 测试：mock subagent 在 seal 时强制 process.kill 一半 → 落盘要么没有要么完整 JSON，不存在半截

### Sprint 1（1.5 周）：data_quality_warnings 结构化 + 浮上水面 ← 重估

**动机**：审计发现 dispatcher.py 的 warning 只有 `{severity, metric, message}`，缺 `code` 和 `evidence`。先加结构，再浮出来。

**warning code 4 个一级分类（grill 锁定）**：
- `SAMPLE.*` — 样本量、组数、subject 数（如 `SAMPLE.TOO_SMALL` n<3、`SAMPLE.UNDERPOWERED` n<5）
- `MOTOR.*` — 运动相关混杂（如 `MOTOR.LOW_VELOCITY` velocity 全低、`MOTOR.LOW_DISTANCE` 距离过低、`MOTOR.LOW_ENTRIES` 进臂次数不足）
- `SIGNAL.*` — 信号品质（如 `SIGNAL.TRACKING_LOST` 追踪丢帧、`SIGNAL.LOW_TRANSITION_COUNT` 穿梭次数不足）
- `METHOD.*` — 统计方法不适配（如 `METHOD.SHAPIRO_INAPPLICABLE` n=2 不能跑正态检验，Sprint 3 参数审计也走这）

加新一级前缀（如 `PARADIGM.*`、`USER.*`）需要 review；初版 4 个分类够用。

**改动**：

- `packages/ethoinsight/ethoinsight/metrics/dispatcher.py` — 9 处 warning 全部补：
  - `code`（按上述 4 分类，如 `SAMPLE.TOO_SMALL` / `MOTOR.LOW_VELOCITY`）
  - `evidence`（如 `{"velocity_median_mm_s": 5.2, "threshold_mm_s": 30.0}`）
  - `blocks_downstream: bool`（n<3 / velocity 全低 → true；method 警告 → false）
- `packages/ethoinsight/ethoinsight/scripts/_cli.py` — handoff_code_executor.json 顶层透传 `data_quality_warnings`（映射到 Sprint 0 已改好的 `DataQualityWarning` schema）
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — 读 warnings，critical 的放入 method_warnings，key_findings 前置警告段；handoff schema 增 `quality_warnings` 字段；gate_signals 增 `quality_warnings_critical_count`
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 播报模板加"已收到 data-analyst 结果: N 条 critical 质量警告"
- **`packages/agent/frontend/`** — auto 模式下渲染策略（双级，grill 锁定 β）：
  - `severity=critical` + `blocks_downstream=true` → 红字 + 警示图标
  - `severity=critical` + `blocks_downstream=false` → 橙字
  - `severity=warning` → 灰色提示

**验收**：跑 velocity 全 < 30mm/s 的 case，UI 可见红字警告（含具体数字，不只"有警告"）。

### Sprint 2a（2.5 周）：参数下沉 catalog（catalog 端）← 拆分

**动机**：审计确认 catalog 当前是"半 SSOT"。先把参数从硬编码常量搬进 YAML，建立 catalog 端 SSOT。

**搬迁范围**（grep 实测）：

- `metrics/_common.py`：`_VELOCITY_THRESHOLD_MM_S`（30.0）、`_VELOCITY_MIN_DURATION`（25）
- `metrics/_pendulum.py`：`SMOOTH_WINDOW`、`ANALYSIS_WINDOW`、`PERIOD_MIN`、`PERIOD_MAX`、`PERIODICITY_THRESHOLD`、`ACTIVITY_STRUGGLE_THRESHOLD`、`MIN_STILL_ACTIVITY`、`MODERATE_ACTIVITY_THRESHOLD`、`MIN_STATE_DURATION`、`PENDULUM_GRACE_PERIOD`（共 9 个常量）
- `metrics/dispatcher.py:235`：`_ZM_LOW_DISTANCE_THRESHOLD`（10.0，写在函数内）

合计 12 个可调常量。

**SSOT 设计（grill 锁定 C 混合）**：
- **shared**（跨范式共享）→ `catalog/_common.yaml.shared_parameters`：`velocity_threshold`、`velocity_min_duration`
- **per-paradigm**（范式独有）→ 各自 yaml：
  - pendulum 9 个常量 → fst.yaml + tst.yaml
  - zero_maze_low_distance → zero_maze.yaml
  - 各范式 sample size 阈值（n<5、entries<8、transitions<4）→ 各自 yaml
- **loader 自动校验重复**（grill 锁定）：同名 + default 一致出现在 2+ 范式 → 报错"应该提到 _common.yaml"

**改动**：

- `packages/ethoinsight/ethoinsight/catalog/schema.py` —
  - `MetricEntry` 增 `parameters: dict[str, ParamSpec]` + `parameters_ref: list[str]`（引用 shared）
  - 新增 dataclass `ParamSpec(default, unit, description, tunable_by_user, valid_range)`
  - `PlanMetric` 增 `parameters_in_use: dict[str, float|int|str]`
  - 新增 `SharedParameters` 结构（_common.yaml 顶层）
- `packages/ethoinsight/ethoinsight/catalog/loader.py` —
  - 校验 `parameters` 块（valid_range 合法性检查）
  - 校验 `parameters_ref` ID 在 _common.yaml 中存在
  - **重复参数自动检测**：grep 同名 + default 一致 → ValueError
- `packages/ethoinsight/ethoinsight/catalog/_common.yaml` — 新增 `shared_parameters` 顶层段：
  - `velocity_threshold` (default=30.0, mm/s, valid_range=[1.0, 100.0], tunable_by_user=true)
  - `velocity_min_duration` (default=25, samples, valid_range=[5, 250], tunable_by_user=true)
- `packages/ethoinsight/ethoinsight/catalog/{epm,oft,ldb,zero_maze,fst,tst}.yaml` —
  - 各 metric 加 `parameters_ref: [velocity_threshold, velocity_min_duration]`（按需）
  - fst.yaml + tst.yaml 加 pendulum 9 个 `parameters`
  - zero_maze.yaml 加 `zm_low_distance_threshold`
  - 各范式 sample-size 阈值加 `parameters`

### Sprint 2b（2 周）：参数管线 5 跳封闭（管线端）← 拆分

**动机**：catalog 端有参数定义了，但执行管线 5 跳每一跳都要打洞。

**绑定时机（grill 锁定 A 早绑定）**：catalog.resolve 时一次性把 override 固化进 PlanMetric.parameters_in_use → 所有下游照 plan 跑，不再有运行时参数感知。改参数 = 新 plan = 全链路重跑（这是 feature，不是 bug）。

**override 传递（grill 锁定 b JSON 文件）**：override 通过 JSON 文件路径传给 catalog.resolve，文件由 experiment-context.json 维护（Sprint 4.5 iii 锁定）。

**改动**：

- `packages/ethoinsight/ethoinsight/catalog/resolve.py` —
  - 新增 `--overrides-file <path>` CLI 参数
  - PlanMetric 生成时合并 catalog default + overrides → 透传到 `parameters_in_use`
  - parameters_in_use 完整 dict（含 default 未被 override 的项也写满）
- `packages/ethoinsight/ethoinsight/metrics/_common.py` + `_pendulum.py` —
  - 删除模块常量（Sprint 2a 已搬入 catalog）
  - 函数签名加参数默认值（仅为本地 unit test 用，生产路径必须传参）
- `packages/ethoinsight/ethoinsight/scripts/{fst,tst}/compute_immobility_*.py` —
  - argparse 加 `--velocity-threshold` / `--velocity-min-duration` / pendulum 9 个参数
  - 不读环境变量（grill 锁定排除晚绑定方案）
- `packages/ethoinsight/ethoinsight/scripts/_cli.py` —
  - 执行 metric script 后把实际使用的参数（即 PlanMetric.parameters_in_use）回写到 handoff `MetricStat.parameters_used`
  - **handoff 写入走 Sprint 0 的 seal_code_executor_handoff tool**，不直接 write_file
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` —
  - 派遣 metric script 时按 plan_metrics.json 内 `metric.args` 拼参数（CLI 参数已包含 override 后的值）
  - 不需要 code-executor 自己感知 override

**验收**：
1. `analysis_manifest.json` 里每个 MetricStat 都有 `parameters_used`，与 catalog default（无 override）/ overrides JSON（有 override）可逐项比对
2. 同样的 overrides JSON 跑两次，生成的 plan_metrics.json bit-identical（确定性）
3. 改一项 override → 重新 catalog.resolve → 新 plan_metrics.json + 新 analysis_config_id（Sprint 4.5 衔接）

### Sprint 3（1 周）：data-analyst 参数审计 ← grill 复审（加编号检查 + 热点协调 + 依赖 2b✅）

**动机**：参数可配置了，但没人告诉用户"你的数据用当前参数不合适"。data-analyst 比对参数和实际数据分布，发现 velocity 中位数 5mm/s 但阈值 30mm/s → 生成警告。

**关键原则**：只警告，不调参。调参需用户显式确认。

**改动**：

- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — workflow 加参数适配性检查段（从 `MetricStat.parameters_used` 取参数，从 handoff 数据分布取实际值，比对）；handoff schema 增 `parameter_audit` 字段；gate_signals 增 `parameter_audit_findings_count`

### Sprint 4（1 周）：调参指南进 by-experiment md ← grill 复审 🔴（SSOT 归属：内容由同事写，工程只做通路）

**动机**：agent 警告了阈值不合适，但不知道"调到多少"。paradigm md 提供权威调参指南。knowledge-assistant 回答"为什么是 30"时解释权衡而非背诵权威。

**关键原则**：参数默认值在 catalog YAML，调参权衡解释在 paradigm md。两者不重复。

**改动**：

- `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/{forced_swim,tail_suspension,epm,open_field,zero_maze,light_dark_box}.md` — 末尾加 `## 参数调整指南`
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — workflow 加"read paradigm md 时 grep 参数调整指南段"
- `ev19-dependent-variables.md` 不动（它是公式 SSOT，不混应用指南）

### Sprint 4.5（0.5 周）：analysis_config_id 设计 + 落地 ← ★ 新增

**动机**：当用户改了参数（如"immobility 阈值从 30 改到 5"），需要知道"上次的分析结果"和"今天的分析结果"对应的是哪组参数。每次参数变更产生新的 `analysis_config_id`，所有计算结果关联到此 id。

**设计原则（grill 锁定）**：
- **B deterministic hash**：`analysis_config_id = sha256(canonical_json(catalog_default + overrides))[:16]`。同 input 必然同 id（可重放、可比较），不需要维护 previous_config_ids 列表
- **b 展示位置**：报告末尾 + 假设面板顶部（Sprint 7）
- **iii overrides 位置**：experiment-context.json 顶层加 `parameter_overrides: dict` 字段，prep_metric_plan 自动从 context 读

**改动**：

- `experiment-context.json` schema 增顶层字段：
  - `analysis_config_id: str`（每次新参数自动计算 hash）
  - `parameter_overrides: dict[str, float|int|str]`（默认 `{}`）
- `agents/middlewares/experiment_context.py` —
  - `set_experiment_paradigm` 可选参数 `parameter_overrides: dict | None`
  - 设置/修改 overrides 时：调 `compute_analysis_config_id(catalog_default, overrides)` 生成新 id 写入
  - `compute_analysis_config_id()` helper：normalize（sort keys）+ json.dumps + sha256 → 取前 16 字符
- `handoff_schemas.py` — 所有 handoff 类的 `analysis_config_id: str` 字段（Sprint 0 已加）
- `prep_metric_plan_tool.py` —
  - 读 experiment-context.json 取 `parameter_overrides`
  - 写到 `/mnt/user-data/workspace/overrides_<config_id>.json`
  - 调 `catalog.resolve --overrides-file <path>` 生成 plan
- `lead_agent/prompt.py` — 模板增 `{analysis_config_id}`：
  - 报告末尾附上（"本次分析标识：xxx-xxxx"）
  - 假设面板顶部展示（Sprint 7 联动）
- `seal_*_handoff tool` 内部 — 自动从 experiment-context.json 读 `analysis_config_id` 注入 handoff（subagent 不用手动传）

**验收**：
1. 两次不同参数跑同一份数据，产出的报告末尾 analysis_config_id 不同；可据此反查完整的参数差异
2. 同一份 overrides JSON 两次跑得到完全相同的 analysis_config_id（deterministic 验证）
3. parameter_overrides 为空时（首次跑），analysis_config_id 仍然生成（基于纯 catalog default 的 hash）

### Sprint 5（1 周）：GuardrailProvider 数据质量门 ← 重估（降 2→1 周）+ grill 复审（估期可能 >1 周，依赖未核验）

**动机**：现在 code-executor 生成 warning 后 data-analyst 照样跑。有 `severity=critical` + `blocks_downstream=true` 时，必须先让用户确认再继续。用 DeerFlow 已有 GuardrailMiddleware 机制拦截下游 subagent 派遣。

**代码审计结论**：`Ev19TemplateGuardrailProvider` 是完美的模板——拦截逻辑、ContextVar bridge、含明确指令的 deny 消息模板都在。本质是克隆 + 改读路径 + 改判定逻辑 + 改 deny 文案。**1 周富余。**

**Auto / Manual 双轨设计（grill 锁定）**：

| 模式 | 行为 | 实现 |
|---|---|---|
| `workflow_mode=auto`（默认，日常用户）| 不阻断，UI 红字标注 | **不挂 DataQualityGuardrailProvider**；前端按 `blocks_downstream` 渲染红字/橙字（Sprint 1） |
| `workflow_mode=manual`（飞轮专家）| guardrail 拦下游 subagent，必须 ask_clarification + acknowledge_quality 才放行 | **挂 DataQualityGuardrailProvider**（仿 GateEnforcementMiddleware 在 manual 模式启用） |

**拦截目标（grill 锁定 B）**：`task(data-analyst)` + `task(chart-maker)` + `task(report-writer)`；knowledge-assistant 免拦。

**gate 机制（grill 锁定 D）**：复用 `set_experiment_paradigm(acknowledge_quality=True)` 写 `gate2_quality_acknowledged` → guardrail 放行。

**粒度（grill 锁定）**：DataQualityWarning 加 `blocks_downstream: bool`，**两模式共用此字段**：
- auto：true → 红字 / false → 橙字
- manual：true → guardrail 拦 / false → 不拦

**避免双重提示（Sprint 1 dogfood 教训）**：当前 manual 模式下，`GateEnforcementMiddleware` 已经在 `task(data-analyst)` 派遣前用粗粒度 ToolMessage 列出 critical message 触发 ask_clarification；Sprint 1 又给 lead 播报 AIMessage 注入 `additional_kwargs.quality_warnings`，前端结构化 banner 二次展示同一批 warning。用户因此在 EPM n=1 dogfood thread 里**看到两次同源数据**（粗粒度中断 + 后续 banner 详情）。Sprint 5 重构 `DataQualityGuardrailProvider` 替换 `GateEnforcementMiddleware` 的 gate2 拦截路径时，**拒绝消息应直接包含结构化 warning payload**（code/evidence/blocks_downstream），让前端用同一个 banner 组件渲染——guardrail 拦截只贡献"中断状态"，UI 信息源仍是 banner，不再产生第二条粗粒度文字。

**改动**：

- `packages/agent/backend/packages/harness/deerflow/guardrails/data_quality_provider.py` 新建 — pre-tool-call 拦截：
  - 检查 tool_name == `task` 且 `subagent_type ∈ {data-analyst, chart-maker, report-writer}`
  - 读 handoff 中 `severity=critical AND blocks_downstream=true` 的 warning
  - 检查 experiment-context.json 是否含 `gate2_quality_acknowledged` → 已确认放行
  - 未确认 → deny + 含明确指令："请先 ask_clarification 告知用户以下 N 条数据质量问题：[code+message 列表]，等待用户确认后调 set_experiment_paradigm(acknowledge_quality=True) 再继续"
- `packages/agent/backend/packages/harness/deerflow/guardrails/__init__.py` — 注册
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` —
  - **仅在 `workflow_mode=manual` 时**挂 DataQualityGuardrailProvider（仿 GateEnforcementMiddleware）
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — manual 模式下调度员角色边界加："critical + blocks_downstream warning 出现时，必须先 ask_clarification → 调 set_experiment_paradigm(acknowledge_quality=True) → 才能派下游 subagent"

### Sprint 5.5（0.5 周）：Handoff 内容完整性校验 ← grill 复审降级（原"lineage 封印"2 周 → 内容非空校验 0.5 周）

> **⚠️ 2026-05-29 grill 复审降级**：本 sprint 原设计是 handoff 内容 **sha256 封印 + fail-closed + audit log**（防篡改，2 周）。复审否决，理由见上文「grill 复审」表 5.5 行：威胁不存在（单产品/确定性脚本/自有 subagent 链）、race 已被 Sprint 0 atomic write 覆盖、用户无感、且与真需求「内容非空校验」重叠。**hash 封印推 v0.2 或砍。** 下面是降级后的设计。原 hash 封印设计存档于本节末「附：原 lineage 封印设计（已搁置）」。

**动机（降级后）**：当前 `_validate_handoff_emitted`（Sprint 5.7）+ seal-resume 补轮判定（Sprint 5.8）**只查 handoff 文件存在，不查内容**。LLM（正常调 seal 或 5.8 补轮被催着调）极小概率会调了 seal 但核心字段残缺/空（`key_findings=[]`、`summary=""`）→ 文件产出 → 判 COMPLETED → 下游读到空内容 handoff，**无人报警、一路绿灯产垃圾**。这比漏调更隐蔽（漏调至少 5.7 会响亮 FAILED）。这是 seal tool 的**普遍**问题（非补轮引入），是真实会发生的失败模式。

**设计原则**：
- 增强 `_validate_handoff_emitted`（或新建并列 validator），文件存在性检查**之后**追加"核心字段非空"检查。
- 各 subagent 的"核心字段"判据来自其 handoff 契约（`handoff_schemas.py` 的 Pydantic 类，[[feedback_single_source_of_truth]]：契约单一来源）：
  - code-executor：`metrics_summary` 非空
  - data-analyst：`key_findings` 非空数组
  - chart-maker：`chart_files` 非空 或 `failed_charts` 有说明
  - report-writer：`report_path` 存在且文件非空
- 覆盖**正常路径 + 5.8 补轮路径**（两者复用同一 validator，5.8 补轮判定自动受益）。
- 失败处理：与 5.7 一致——标 FAILED + 诊断（"sealed but core field X empty"）→ lead 重派 / 5.8 补轮。**不做 hash、不做 audit log**（那是被否决的 v0.2 范畴）。

**改动**：

- `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` — `_validate_handoff_emitted` 扩展（或加 `_validate_handoff_nonempty`）：文件存在后读 JSON，按 subagent 查核心字段非空。**核心字段判据从 handoff_schemas.py 的契约推导，不在此硬编码字段名清单**（避免与 schema 双存，违反 single-source）。
- `tests/` — 各 subagent "文件在但核心字段空 → 仍判失败" 的 case；正常路径不退化。
- **依赖**：Sprint 0（seal tool）+ 5.7（_validate_handoff_emitted）已实施 ✅；与 5.8 补轮共用 validator，需在 5.8 之后或协调。

**验收**：
1. 构造 data-analyst handoff 文件存在但 `key_findings=[]` → 派下游时判失败（而非绿灯放行）
2. 正常完整 handoff → 放行
3. 5.8 补轮产出空内容 handoff（模拟）→ 同样被拦

**附：原 lineage 封印设计（已搁置，v0.2 再评估）**：原拟在 seal tool 内写 `.lineage/manifest.json`（sha256）+ 新建 `LineageIntegrityGuardrailProvider` 在 `evaluate()` 拦 task 验 hash + fail-closed + `.lineage/violations.log` audit。grill 复审认定 v0.1 不需要（无篡改威胁），完整设计见 git history 本节修订前版本。

### ~~Sprint 5.5（2 周）：Lineage 封印~~ — 见上（已降级）

### Sprint 5.7（0.5 周）：Handoff Emission Validator ← ★ 新增（兜底 LLM 漏调 seal tool，dogfood 实证驱动）

**动机**：2026-05-28 EPM dogfood 实证——data-analyst 的 LLM 最后一轮只输出 thinking（"现在封存分析结果："）但**没 emit `seal_data_analyst_handoff` tool_call**，executor 见无新 tool_call → 静默标 COMPLETED → handoff 文件根本不存在 → 下游断链，仅靠 lead 事后 read_file 探测 + 运气自救。根因是 Sprint 1 在 data_analyst.py 留下"两个 step 2"编号冲突（已 prompt 修复 commit `2df10880`，但 prompt fix 不能根治 LLM 随机性）。

**本质**：把"概率性的静默断链"转化为"确定性的可恢复失败"。LLM 漏调这一随机事件消不掉（留给后训练/prompt），但其破坏性后果（静默断链）被 harness 层拦死。

**设计原则**：harness 强约束 > prompt 配方（[[feedback_deny_messages_must_direct]] / [[project_2026-05-18_lead_not_reading_handoff]] 教训）。正交于 Sprint 5.5 lineage 封印——5.7 管"handoff 是否存在"，5.5 管"handoff 内容一致性"，前者是后者前提。

**改动**：

- `subagents/executor.py` — 加 `_HANDOFF_EMISSION_REQUIRED` 白名单字典（4 个 ethoinsight subagent，连字符 key）+ `_validate_handoff_emitted` helper（异常安全 fail-open）+ 改 `executor.py:723` 的 `try_set_terminal(COMPLETED)` 为 validate-first：handoff 文件不存在 → 标 FAILED + 明确诊断（含 "terminated without emitting" 关键字）
- `agents/lead_agent/prompt.py` — 加自动 retry 规则：error 含 "terminated without emitting" → 不询问用户，直接 re-dispatch 同 subagent + 末尾追加 seal 强化提示；最多 2 次
- white-list 而非 black-list：general-purpose / bash / knowledge-assistant 不验证（无 seal contract）

**验收**：dogfood 临时削弱 data_analyst step 2.7 复现漏调 → 第一次派遣 task FAILED → lead 自动 retry → 第二次（带 reminder）成功 → 整链跑通且用户无感。

**spec + 实施文档**：[Sprint 5.7 spec](../superpowers/specs/2026-05-28-sprint-5.7-handoff-emission-validator-design.md) / [实施文档（代码核验版）](../superpowers/plans/2026-05-29-sprint-5.7-handoff-emission-validator-impl.md)

### Sprint 6（0.5 周）：跨会话范式 memory ← 修正（复用 deerflow facts 通道，砍掉新顶层结构）+ grill 复审（与 5.8 改同一 seal 路径需协调；PRD 将跨范式划出 MVP）

**动机**：这是"主人"的质变能力。用户上周做 EPM、这周做 OFT——agent 应能跨会话汇聚证据。审计发现 MemoryMiddleware 的 LLM 抽取通道会把数字漂移成字符串——但**不需要新建顶层 ExperimentSummary 结构**，复用 deerflow 现有的 facts 通道即可。

**重新评估**（grill 锁定 B）：deerflow memory 系统已就位的能力：
- per-user 隔离（`{base_dir}/users/{user_id}/memory.json`）✅
- facts 结构化（`{id, content, category, confidence, ...}`）✅
- system prompt 自动注入（top facts 按 confidence 排序）✅
- eviction（max_facts=100，按 confidence 排序）✅
- debounced 异步更新（30s）✅

只需要新增一条**确定性写入路径**——report-writer 完成时直接写一条 `category="experiment_summary"` 的 fact（confidence=1.0），完全不走 LLM 抽取。

**关键原则**：experiment_summary 走**确定性写入**（report-writer seal handoff 时直接 storage.add_fact），不走 LLM 抽取通道。LLM 抽取通道继续做它的事（user preferences/style 提取），但精确数字不再依赖它。

**改动**：

- `packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py` —
  - 若没有 `add_fact()` 公共方法则新增（load → append → save 的 helper）
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`（Sprint 0 建的）—
  - `seal_report_writer_handoff` 内部，atomic write + manifest 之后，**额外调** `memory_storage.add_fact(user_id, fact)`，fact content 形如：
    ```
    "{paradigm} analysis on {date}: n_per_group={n}, key metrics: {metric1}={value1}, {metric2}={value2}; effect_size={es}; analysis_config_id={config_id}"
    ```
    - `category="experiment_summary"`
    - `confidence=1.0`（确定性写入，最高优先级保留）
    - `source=f"thread:{thread_id}, analysis_config_id:{config_id}"`
- **不动**：MemoryMiddleware / memory updater / memory config / memory prompt（所有现有路径不变）

**验收**：
1. 完成一次 EPM 分析后，`memory.json[users/{user_id}].facts` 中出现一条 `category=experiment_summary` 的 fact，content 含精确数字（不是 LLM 抽出来的）
2. 第二个会话做 EPM 时，lead system prompt 的 `<memory>` 段含上次 fact（confidence=1.0 排序最优先）
3. 跑 200 次分析后，memory.json 仍只有 100 条 facts（max_facts 自动 eviction）；confidence=1.0 的 experiment_summary 不被低 confidence 的 user preference 挤掉

### Sprint 7（0.5 周）：用户侧假设暴露面板 ← ★ 新增（轻量版，不强制）+ grill 复审 🟢（最健康，已自我纠偏，仅标依赖）

**动机**：Sprint 3 的参数审计 + Sprint 4 的调参指南生成了 method_warnings 和参数调整推荐，但它们写在 data-analyst handoff 里，是给 lead 看的中间产物。"让本体论承诺显式化"的哲学价值确实存在——但**不该用 LLM 必调 + guardrail 强制的方式实现**（增加决策树复杂度、用户大概率不读冗余卡片）。

**重新评估**（grill 锁定 B）：所有要展示的信息（in-use parameters / 偏离 default / quality warnings / gate 决策 / analysis_config_id）经过前 6 个 sprint 后**已经在各处展示**。Sprint 7 只需提供一个**可主动调用的聚合工具**——lead 在 prompt 引导下"复杂分析时主动调"，而不是 GateProvider 强制每次结束都渲染。

**改动**：

- `packages/agent/backend/packages/harness/deerflow/tools/builtins/present_assumptions.py` 新建 — first-party tool，聚合查询：
  - 从 plan_metrics.json 取 in-use parameters（Sprint 2b 衔接）
  - 从 handoff_data_analyst.json 取 parameter_audit / quality_warnings（Sprint 1/3 衔接）
  - 从 experiment-context.json 取 gate_completed / parameter_overrides / analysis_config_id（Sprint 4.5 衔接）
  - 渲染成 markdown 段（前端按 collapsed 卡片样式展示）
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — final delivery 段加**建议性指引**（非强制）："当分析包含 critical quality warning、参数 override 或假设面板可能对用户有价值时，主动调 `present_assumptions` 工具"
- `packages/agent/frontend/` — 新增假设面板渲染组件（默认 **collapsed** 折叠样式，点开看；标题"分析假设摘要 (config_id=...)"）

**不做（grill 锁定）**：
- ❌ AssumptionPanelGateProvider 强制 guardrail（增加 LLM 决策复杂度 + 用户体验差）
- ❌ 每次 final delivery 都强制渲染（用户看冗余卡片几次后就无视了）

**可观测性**：上线后跟踪 lead 主动调用率 + 用户卡片点开率。30 天后若用户点开率 > 50%，考虑升级为 GateProvider 强制（推迟到 v0.2）。

**验收**：
1. lead 在含 critical warning 或 parameter override 的分析末尾，主动调 `present_assumptions` 并渲染折叠卡片
2. 用户点开卡片可看到结构化假设清单（参数、偏离、警告、决策、config_id）
3. 简单分析（无 warning + 无 override + 全 default）lead 不调，避免噪声

### Sprint 8（2 周）：feedback verdict 回流到 prompt ← 原 Sprint 7

**动机**：专家三按钮反馈已在 SQLite 落盘（correct/needs_fix/wrong + revised_text），但没有回流到 prompt。agent 下次遇到同范式时不知道自己上次被纠正过。这是主人"不重复犯同一个错"的闭环。

**注意**：长期靠微调底模型，Sprint 8 是短期补丁。微调到位后收益减半。

**前置工作（grep 实测）**：当前 `feedback` 表 schema 是 `{feedback_id, thread_id, run_id, user_id, message_id, verdict, revised_text, comment, created_at}`——**没有 paradigm 字段**。"查同 paradigm 上次 verdict ≠ correct"无法直查，必须先解决：

- **方案 A（推荐）**：feedback 表加 `paradigm: str | None` 字段（schema migration + 提交时从 `experiment-context.json` 读 paradigm 一并落盘）。新数据 → 直查
- **方案 B**：feedback 表不动，查询时 thread_id → 反查 `experiment-context.json` 取 paradigm。性能差但兼容历史数据

实施按 A，历史 feedback 数据 paradigm 字段留 null（不参与 prior_corrections 回流）。

**改动**：

- `packages/agent/backend/app/persistence/feedback_repository.py` — schema 加 `paradigm` 字段 + Alembic migration（或等价的 SQLite ALTER）
- `packages/agent/backend/app/gateway/routers/feedback.py` — 提交时读 `experiment-context.json.paradigm` 一并写入；增查询接口 `GET /api/feedback/prior_corrections?paradigm=epm&user_id=X&limit=3` 查同 paradigm 上次 verdict ≠ correct 的 revised_text
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` — `make_lead_agent` 在构建 prompt 前 fetch `prior_corrections`
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 模板增 `{prior_corrections}` 插槽

---

## 实施顺序与依赖

```
Sprint 0 (handoff schema 全面化) ← 最高优先级，一切的下游承接器
    ↓
Sprint 1 (data_quality 结构化) ──────────────────────┐
    ↓                                                  │
Sprint 2a (参数下沉 catalog 端)                         │
    ↓                                                  │
Sprint 2b (参数管线 5 跳封闭)                            │
    ↓                                                  │
Sprint 3 (参数审计) ←── 依赖 2a+2b                       │
    ↓                                                  │
Sprint 4 (调参指南) ←──── 依赖 2a+3                      │
    ↓                                                  │
Sprint 4.5 (analysis_config_id) ←── 依赖 0, 独立实施     │
    ↓                                                  │
Sprint 5 (DataQualityGuardrail) ←── 依赖 1+3 ←─────────┘
    ↓
Sprint 5.5 (lineage 封印) ←── 依赖 0+4.5+2b
    ↓
Sprint 6 (跨会话 memory) ←── 依赖 0+4.5
    ↓
Sprint 7 (假设暴露面板) ←── 依赖 2b+3+5.5
    ↓
Sprint 8 (feedback 回流) ←── 独立实施
```

简化版依赖图：

```
S0 → S1 → ─ ─ ─ ─ ─ ─ → S5 → S5.5 → S7
S0 → S2a → S2b → S3 → S4 ↗
S0 → S4.5 → S5.5 ↗
S0 → S4.5 → S6
S5.7（独立，兜底 seal 漏调，正交于 5/5.5，dogfood 实证后任意时点启动）
S8（独立）
```

Sprint 1 和 Sprint 2a 可以在 S0 完成后并行；Sprint 4.5 可以在 S0 完成后独立启动。

总计 **12 个 sprint**（0/1/2a/2b/3/4/4.5/5/5.5/5.7/5.8/6/7/8 — 含 2026-05-29 新增的 5.7 + 5.8 两个 seal 可靠性 sprint）。**2026-05-29 grill 复审后周数下调**：5.5 从 2 周降到 0.5 周（hash 封印 → 内容非空校验），新增 5.7（0.5）+ 5.8（0.5）。约 **15.5 周**。

**已实施（截至 2026-05-29）**：Sprint 0 / 1 / 2a / 2b / 5.7 / 5.8 ✅。其余待写 spec / 待实施。

## 与原路线图的差异对照

| 原 Sprint | v2 调整（grill 锁定后）| 变更原因 |
|-----------|---------|---------|
| — | **Sprint 0（新增，2 周）** | handoff schema + 4 个 seal_*_handoff first-party tool + atomic write + 三档 strict mode；LLM 不再写 JSON 字符串 |
| Sprint 1（1-2 周） | Sprint 1（1.5 周） | code+evidence+blocks_downstream 三字段；4 一级分类（SAMPLE/MOTOR/SIGNAL/METHOD） |
| Sprint 2（2-3 周） | Sprint 2a+2b（4.5 周） | catalog 端 + 管线端拆分；C 混合 SSOT（_common.yaml shared + 范式独有 + loader 重复校验）；早绑定 + overrides JSON |
| Sprint 3（1 周） | Sprint 3（1 周） | 不变 |
| Sprint 4（1 周） | Sprint 4（1 周） | 不变 |
| — | **Sprint 4.5（新增，0.5 周）** | analysis_config_id：deterministic hash（不 UUID）；experiment-context.json 加 parameter_overrides 字段 |
| Sprint 5（2 周） | Sprint 5（1 周） | auto/manual 双轨：auto 不挂 guardrail+UI 红字、manual 挂 guardrail+ack 机制 |
| — | **Sprint 5.5（新增，2 周）** | lineage 封印：seal tool 内写 manifest + LineageIntegrityGuardrailProvider 验 hash + fail-closed+audit log |
| Sprint 6（3 周） | **Sprint 6（0.5 周）** | 复用 deerflow facts 通道（category=experiment_summary, confidence=1.0），不加新顶层结构 |
| — | **Sprint 7（新增，0.5 周）** | present_assumptions 工具 + collapsed 卡片；**不做** GateProvider 强制 |
| Sprint 7（2 周） | Sprint 8（2 周） | feedback 表先 schema migration 加 paradigm 字段 |
| 7 sprint，12-15 周 | **10 sprint，16.5 周** | 新增 4 个 sprint + 2a/2b 拆分 + S6/S7 压缩 |

## 关键设计原则

1. **参数 SSOT 唯一**（MEMORY.md）：catalog YAML 是参数默认值的 SSOT，skill md 和 prompt 不内嵌参数值；跨范式共享走 `_common.yaml.shared_parameters`，loader 自动校验重复
2. **只警告不调参**：data-analyst 参数审计只生成建议，调参需用户显式确认
3. **科学一致性**：与 CLAUDE.md §9 的判读哲学一致（组间比较，不用绝对阈值）
4. **per-user 隔离**（Tier 4）：跨会话 memory 不跨用户聚合（deerflow 现有机制已就位）
5. **deny 必须含指令**（MEMORY.md）：GuardrailProvider 拒绝时必须告知 lead 下一步做什么
6. **handoff 自包含**（v2 新增）：每个 handoff JSON 是独立可校验的数据包，不依赖外部文件才能解释
7. **确定性层 vs 概率性层严格分离**（v2 新增）：指标计算和统计检验走 Python 函数（bit-identical），解读和报告走 LLM（同义不同字），两层之间用结构化 handoff 接口
8. **假设必须显式化**（v2 新增 + grill 调整）：当前分析的所有参数选择、偏离项、警告、用户决策，通过 `present_assumptions` 工具按需聚合呈现（**不强制每次都渲染**，避免噪声）
9. **lineage 不可篡改**（v2 新增）：handoff 文件写完即冻结（hash），下游读时验 hash，不匹配直接 fail-closed
10. **LLM 不写 JSON 字符串**（v2 + grill 锁定）：所有 4 个 subagent 走 first-party `seal_*_handoff` tool，LangChain tool_call schema + Pydantic 双重校验；LLM 只填参数，Python 负责序列化 + 原子写 + manifest
11. **Auto / Manual 双轨**（grill 锁定）：DataQuality guardrail 仅 manual 模式启用；auto 模式靠 UI 红字/橙字标注 + lead 自然语言复述；deerflow 已有 `workflow_mode` 配置直接复用
12. **deterministic 不 UUID**（grill 锁定）：analysis_config_id 是参数 hash（sha256(catalog+overrides)[:16]），同 input 必同 id，可重放可比较；不维护版本历史列表
13. **复用 deerflow 现成机制优先于自造**（MEMORY.md + grill 反复强调）：memory facts / GuardrailProvider 模板 / workflow_mode / max_facts eviction 都不重造

## 参考文件

| 类别 | 路径 |
|------|------|
| CLAUDE.md | `/home/wangqiuyang/noldus-insight/CLAUDE.md` |
| catalog schema | `packages/ethoinsight/ethoinsight/catalog/schema.py` |
| catalog YAML | `packages/ethoinsight/ethoinsight/catalog/{epm,oft,fst,tst,ldb,zero_maze}.yaml` |
| metrics | `packages/ethoinsight/ethoinsight/metrics/_common.py` + `_pendulum.py` |
| data quality | `packages/ethoinsight/ethoinsight/metrics/dispatcher.py` |
| handoff schemas | `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` |
| handoff registry | `packages/agent/backend/packages/harness/deerflow/subagents/handoff_registry.py` |
| experiment context | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` |
| data-analyst prompt | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` |
| code-executor prompt | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` |
| lead prompt | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| lead agent | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` |
| guardrails | `packages/agent/backend/packages/harness/deerflow/guardrails/` |
| Ev19TemplateGuardrail | `packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py` |
| IntentPostStepAskGate | `packages/agent/backend/packages/harness/deerflow/guardrails/intent_post_step_ask_gate_provider.py` |
| memory | `packages/agent/backend/packages/harness/deerflow/agents/memory/` |
| memory middleware | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py` |
| paradigm md | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/` |
| EV19 公式 | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md` |
| deerflow backend | `packages/agent/backend/CLAUDE.md` |

## 测试基线

```
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/python -m pytest tests/ -q
cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test
```

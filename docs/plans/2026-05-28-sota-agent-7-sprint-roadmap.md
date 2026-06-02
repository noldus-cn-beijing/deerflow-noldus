# SOTA Agent 构建路线图：从"奴隶"到"主人"

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

### 为何基于 DeerFlow

DeerFlow 已有 5 个机制可以支撑这 5 个能力，不需要新架构：

| DeerFlow 机制 | 当前使用 | SOTA 用途 |
|--------------|---------|----------|
| MemoryMiddleware | 只存 user facts | 存实验摘要，跨会话引用 |
| Skills 渐进披露 | 只用于 EV19 模板 | 注入方法论指导（统计选择、图表选择） |
| ClarificationMiddleware | 只用于范式反问 | 反问参数选择、图表选择 |
| GuardrailMiddleware（含 9 个 provider） | 没用 | 数据质量门禁 |
| TrainingDataMiddleware | 已录制但被动 | 每次决策都是 SFT 种子 |

---

## 7 个 Sprint 路线图

### Sprint 1（1-2 周）：data_quality_warnings 浮上水面 ★ 最高 ROI

**动机**：dispatcher.py 已经在生成 9 处 `data_quality_warnings`，但 handoff 流转不规范，UI 上看不到。agent 睁着眼但看不见东西。让现有警告浮出水面，零代码风险，最高感知收益。

**改动**：
- `packages/ethoinsight/ethoinsight/metrics/dispatcher.py` — 标准化 warning 结构 `{code, severity: info|warning|critical, metric, message, evidence}`
- `packages/ethoinsight/ethoinsight/scripts/_cli.py` — handoff_code_executor.json 顶层透传 `data_quality_warnings`
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — 读 warnings，critical 的放入 method_warnings，key_findings 前置警告段；handoff schema 增 `quality_warnings` 字段；gate_signals 增 `quality_warnings_critical_count`
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 播报模板加"已收到 data-analyst 结果:1 条 critical 质量警告"

**验收**：跑 velocity 全 < 30mm/s 的 case，UI 可见红字警告。

### Sprint 2（2-3 周）：参数下沉 catalog ★ 地基

**动机**：`VELOCITY_THRESHOLD_MM_S = 30.0`、`PERIODICITY_THRESHOLD = 0.55` 等硬编码在 `_common.py` / `_pendulum.py` 的模块级常量。agent 不知道这些参数存在，更无法调整。参数先进 catalog，后续审计和调参指南才能跟上。

**关键原则**：catalog YAML 是参数的 SSOT。skill md、prompt、报告都不内嵌默认值。

**改动**：
- `packages/ethoinsight/ethoinsight/catalog/schema.py` — `MetricEntry` 增 `parameters: dict[str, ParamSpec]`，新 dataclass `ParamSpec(default, unit, description, tunable_by_user, valid_range)`
- `packages/ethoinsight/ethoinsight/catalog/loader.py` — 校验 `parameters` 块
- `packages/ethoinsight/ethoinsight/catalog/{epm,oft,ldb,zero_maze,fst,tst}.yaml` — 把硬编码常量搬进 parameters
- `packages/ethoinsight/ethoinsight/metrics/_common.py` + `_pendulum.py` — 删除模块常量，函数签名加参数默认值
- `packages/ethoinsight/ethoinsight/scripts/{fst,tst}/compute_immobility_*.py` — argparse 加参数
- `packages/ethoinsight/ethoinsight/catalog/resolve.py` — PlanMetric 生成时透传参数
- `packages/ethoinsight/ethoinsight/catalog/schema.py` — `PlanMetric` 增 `parameters_in_use: dict`

### Sprint 3（1 周）：data-analyst 参数审计

**动机**：参数可配置了，但没人告诉用户"你的数据用当前参数不合适"。data-analyst 比对参数和实际数据分布，发现 velocity 中位数 5mm/s 但阈值 30mm/s → 生成警告。

**关键原则**：只警告，不调参。调参需用户显式确认。

**改动**：
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — workflow 加参数适配性检查段；handoff schema 增 `parameter_audit` 字段；gate_signals 增 `parameter_audit_findings_count`

### Sprint 4（1 周）：调参指南进 by-experiment md

**动机**：agent 警告了阈值不合适，但不知道"调到多少"。paradigm md 提供权威调参指南。knowledge-assistant 回答"为什么是 30"时解释权衡而非背诵权威。

**关键原则**：参数默认值在 catalog YAML，调参权衡解释在 paradigm md。两者不重复。

**改动**：
- `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/{forced_swim,tail_suspension,epm,open_field,zero_maze,light_dark_box}.md` — 末尾加 `## 参数调整指南`
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — workflow 加"read paradigm md 时 grep 参数调整指南段"
- ev19-dependent-variables.md 不动（它是公式 SSOT，不混应用指南）

### Sprint 5（2 周）：GuardrailProvider 数据质量门

**动机**：现在 code-executor 生成 warning 后 data-analyst 照样跑。有 `severity=critical` 时，必须先让用户确认再继续。用 DeerFlow 已有 GuardrailMiddleware 机制拦截 task(data-analyst) 调用。

**关键原则**：deny 消息必须含明确指令告诉 lead 下一步做什么（参考 MEMORY.md "拒绝消息必须含明确指令 不能只列 Available"）。

**改动**：
- `packages/agent/backend/packages/harness/deerflow/guardrails/data_quality_provider.py` 新建 — pre-tool-call 拦截，读 handoff 中的 critical warnings，强制 ask_clarification
- `packages/agent/backend/packages/harness/deerflow/guardrails/__init__.py` — 注册
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` — 中间件链加 DataQualityGuardrailProvider
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 调度员角色边界加："critical warning 出现时，必须先 ask_clarification"

### Sprint 6（3 周）：跨会话范式 memory

**动机**：这是"主人"的质变能力。用户上周做 EPM、这周做 OFT——agent 应能跨会话汇聚证据。但 MemoryMiddleware 当前只存通用 user facts，无范式专用 schema。

**关键原则**：per-user 隔离（Tier 4 已就位）。"上次会话"指同一研究员的上次，不能跨用户聚合。

**改动**：
- `packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py` — memory.json schema 增顶级字段 `experiment_summaries`（thread_id, paradigm, ev19_template, date, n_per_group, key_effect_sizes, verdict_at_end），限 100 条 LRU
- `packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py` — 增 `_extract_experiment_summary()`，从 report-writer 输出抽取
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — `<memory>` 段注入"过去 5 次同 paradigm experiment_summaries"
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — 派遣 prompt 注入本范式过去 effect_size 历史值

### Sprint 7（2 周）：feedback verdict 回流到 prompt

**动机**：专家三按钮反馈已在 SQLite 落盘（correct/needs_fix/wrong + revised_text），但没有回流到 prompt。agent 下次遇到同范式时不知道自己上次被纠正过。这是主人"不重复犯同一个错"的闭环。

**注意**：长期靠微调底模型，Sprint 7 是短期补丁。微调到位后收益减半。

**改动**：
- `packages/agent/backend/app/gateway/routers/feedback.py` — 增查询接口：查同 paradigm 上次 verdict ≠ correct 的 revised_text
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` — make_lead_agent 在构建 prompt 前 fetch prior_corrections
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 模板增 `{prior_corrections}` 插槽

---

## 实施顺序与依赖

```
Sprint 1 → Sprint 2 → Sprint 3 → Sprint 4
                                    ↓
                              Sprint 5（并行）
                                    ↓
                              Sprint 6（并行）
                                    ↓
                              Sprint 7（并行）
```

- Sprint 1 独立，无依赖，最高优先级
- Sprint 2 是地基，Sprint 3 和 4 都依赖它
- Sprint 5-7 可以在 Sprint 4 完成后并行推进

总计 7 个 sprint，约 12-15 周，35 个文件改动。

## 关键设计原则

1. **参数 SSOT 唯一**（MEMORY.md）：catalog YAML 是参数默认值的 SSOT，skill md 和 prompt 不内嵌参数值
2. **只警告不调参**：data-analyst 参数审计只生成建议，调参需用户显式确认
3. **科学一致性**：这与 CLAUDE.md §9 的判读哲学一致（组间比较，不用绝对阈值）
4. **per-user 隔离**（Tier 4）：跨会话 memory 不跨用户聚合
5. **deny 必须含指令**（MEMORY.md）：GuardrailProvider 拒绝对必须告知 lead 下一步做什么

## 参考文件

| 类别 | 路径 |
|------|------|
| CLAUDE.md | `/home/wangqiuyang/noldus-insight/CLAUDE.md` |
| catalog schema | `packages/ethoinsight/ethoinsight/catalog/schema.py` |
| catalog YAML | `packages/ethoinsight/ethoinsight/catalog/{epm,oft,fst,tst,ldb,zero_maze}.yaml` |
| metrics | `packages/ethoinsight/ethoinsight/metrics/_common.py` + `_pendulum.py` |
| data quality | `packages/ethoinsight/ethoinsight/metrics/dispatcher.py` |
| data-analyst prompt | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` |
| lead prompt | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| guardrails | `packages/agent/backend/packages/harness/deerflow/guardrails/` |
| memory | `packages/agent/backend/packages/harness/deerflow/agents/memory/` |
| paradigm md | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/` |
| EV19 公式 | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md` |
| deerflow backend | `packages/agent/backend/CLAUDE.md` |

## 测试基线

```
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/python -m pytest tests/ -q
cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test
```

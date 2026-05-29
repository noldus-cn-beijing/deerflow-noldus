# Sprint 7 设计骨架 — 用户侧假设暴露面板（present_assumptions 工具）

**类型**：设计骨架版（非代码核验版——实施前需按 §核验清单升级）
**对应**：[roadmap v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 7（★ 新增，轻量版）+ 2026-05-29 grill 复审 🟢（**最健康**，已自我纠偏，仅需标依赖）
**估期**：0.5 周
**前置**：Sprint 2b（in-use parameters，已实施 ✅）+ Sprint 3（parameter_audit）+ Sprint 5.5（lineage 内容校验 / analysis_config_id 衔接）

---

## 0. 目标与原则

**目标**：让"本体论承诺显式化"——把当前分析的参数选择、偏离 default 项、quality warnings、gate 决策、analysis_config_id 聚合成一张用户可看的卡片。哲学价值是"主人不藏假设"。

**核心设计（grill 锁定 B）：可主动调用的聚合工具，不是强制渲染。** 所有要展示的信息经过前 6 个 sprint 后**已经在各处产出**（plan_metrics.json / handoff_data_analyst.json / experiment-context.json）。Sprint 7 只新建一个 **first-party 聚合查询工具** `present_assumptions`，lead 在 prompt 引导下"复杂分析时主动调"——**不是** GateProvider 每次结束强制渲染。

**铁律：不过度工程。** 这是 grill 标记的"最健康"sprint，因为它已经从"LLM 必调 + guardrail 强制"自我纠偏到"建议性指引"。实施 agent 别把它退化回强制路径。

---

## 1. grill 复审发现（实施前必读）

### 🟢 已自我纠偏——保持轻量，别加回强制
roadmap line 468-470 明确**不做**：
- ❌ `AssumptionPanelGateProvider` 强制 guardrail（增加 LLM 决策树复杂度 + 用户体验差）
- ❌ 每次 final delivery 都强制渲染（用户看冗余卡片几次后无视）

实施 agent 若发现自己在写 GuardrailProvider，就是走错了——Sprint 7 只有"工具 + prompt 建议指引 + 前端折叠卡片"三件，没有任何拦截/强制。

### 🟡 present_assumptions 工具不存在，是新建
已核验：`present_assumptions` 在代码里**不存在**，只在 roadmap/milestone 文档出现。所以是 first-party tool 新建（不是复用）。新建 first-party tool 要走 deerflow tool 注册（`tools/builtins/__init__.py` + `tools/tools.py` 的 `BUILTIN_TOOLS`——这俩是 [[feedback_sync_protected_files_registry_loss]] 标记的受保护注册文件，加工具时改它们要小心，别整文件覆盖）。

### 🟡 依赖 3 / 5.5 未实施——工具要容错读
present_assumptions 聚合的数据源里：
- in-use parameters（2b ✅，可读）
- parameter_audit（**Sprint 3** 的 handoff 字段——3 未实施则该段为空）
- analysis_config_id（**Sprint 4.5/5.5** 衔接——未实施则显示 PENDING）

所以工具实现必须**容错**：某个数据源文件不存在/字段缺失 → 那一段显示"（暂无）"或跳过，不报错。Sprint 7 可以先于 3/5.5 上线（卡片少几段而已），但完整体验依赖它们。**实施前确认 3/5.5 状态，决定卡片完整度**。

---

## 2. 设计（骨架）

**核心数据流**：lead 在复杂分析末尾主动调 `present_assumptions` → 工具读多个 workspace 文件聚合 → 返回 markdown 段 → 前端按 collapsed 卡片渲染。

**改动（骨架，行号待实施时核验）**：

- `tools/builtins/present_assumptions.py` **新建** — first-party tool，聚合查询（**每个源容错**）：
  - 从 `plan_metrics.json` 取 in-use parameters（2b 衔接）
  - 从 `handoff_data_analyst.json` 取 parameter_audit / quality_warnings（1/3 衔接）
  - 从 `experiment-context.json` 取 gate_completed / parameter_overrides / analysis_config_id（4.5 衔接）
  - 渲染成 markdown 段（标题含 `config_id=...`）
- `tools/builtins/__init__.py` + `tools/tools.py` — 注册到 `BUILTIN_TOOLS`（⚠️ 受保护文件，surgical 加一行，别覆盖）
- `agents/lead_agent/prompt.py` — final delivery 段加**建议性指引**（非强制）：
  > "当分析包含 critical quality warning、参数 override，或假设面板可能对用户有价值时，主动调 `present_assumptions` 工具。简单分析（无 warning + 无 override + 全 default）不必调。"
- `packages/agent/frontend/` — 新增假设面板渲染组件：默认 **collapsed** 折叠，点开看；标题"分析假设摘要 (config_id=...)"

**工具读路径**：复用现有 workspace resolve 机制（`/mnt/user-data/workspace/*.json`）。**别让工具自己拼路径**——查 deerflow 有没有现成的 workspace 文件读 helper（呼应 [[feedback_subagent_consumption_via_first_party_tool]]：消费包内/workspace 文件走 first-party tool resolve，不硬编码）。

---

## 3. 实施前核验清单（骨架→核验版的升级步骤）

实施 agent 开工前必须核验：
1. first-party tool 的注册范式（看现有 `tools/builtins/` 下某个简单 tool 怎么定义 + 怎么进 `BUILTIN_TOOLS`）——别凭空发明
2. workspace 文件读 helper 是否现成（resolve `/mnt/user-data/workspace/`）
3. plan_metrics.json / handoff_data_analyst.json / experiment-context.json 的真实字段结构（决定怎么取数）
4. Sprint 3 / 4.5 / 5.5 实施状态（决定 parameter_audit / config_id 段是否有内容）
5. 前端现有折叠卡片组件惯例（复用样式，别新造一套）
6. lead prompt 的 final delivery 段当前在哪（插建议指引的位置）

---

## 4. 验收（骨架）

1. lead 在含 critical warning 或 parameter override 的分析末尾，主动调 `present_assumptions` 并渲染折叠卡片
2. 用户点开卡片可看到结构化假设清单（参数、偏离 default、警告、gate 决策、config_id）
3. **简单分析**（无 warning + 无 override + 全 default）lead **不调**，避免噪声
4. 工具对缺失数据源容错（3/5.5 未实施时，对应段显示"暂无"，不报错）
5. **没有任何 GuardrailProvider 强制路径**（grill 锁定：不做强制）
6. 全量测试不退化

## 5. 不在范围
- ❌ `AssumptionPanelGateProvider` 强制 guardrail（grill 锁定否决）
- ❌ 每次 final delivery 强制渲染（grill 锁定否决）
- ❌ 升级为 GateProvider 强制（推迟 v0.2，先观测 30 天点开率 > 50% 再议）
- ❌ 新增 quality/parameter 数据（present_assumptions 只**聚合呈现**已有数据，不产生新分析结论）

## 6. 可观测性（上线后）
跟踪 lead 主动调用率 + 用户卡片点开率。30 天后若点开率 > 50%，考虑升级 GateProvider 强制（v0.2）。

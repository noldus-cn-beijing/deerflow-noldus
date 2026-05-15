# Handoff 双层协议 + Runtime 路径绝缘 架构设计

> **日期**：2026-05-14
> **状态**：草案（draft）—— 待执行 agent 完成 dogfood 修复 + 一次完整端到端测试后定稿
> **前置**：
> - [2026-05-13 Metric Catalog 架构](2026-05-13-metric-catalog-architecture-design.md)（catalog YAML 作 single source of truth）
> - [2026-05-11 Subagent 协作：files are facts](2026-05-11-subagent-file-is-facts-design.md)（handoff 作事实源、消除二手摘要、乐团指挥隐喻）
> **触发**：
> - 2026-05-13 dogfood thread `5046a6e6-4bfc-4ca9-9650-b674ec3734cf` 暴露的故障级联（详见 [dogfood-fix-iteration-plan](../plans/2026-05-13-dogfood-fix-iteration-plan.md)）
> - 同事 PR #7 反馈：[Gap 清单](../../review-packages/2026-05-13-还差哪些with细颗粒度.md) + [指标外分析需求](../../review-packages/2026-05-13-常见行为学指标外分析要求.md)
> - 用户原话："runtime / thread / subagent / lead / run 之间的文件路径都应该绝缘"
> - 用户原话："handoff 文件应该是很简练的"
> - 用户 AgentCore 经验："文件 handoff 文件会很大，但不漂"

---

## 设计原则（贯穿本 spec）

继承 5-11 spec 的"乐团指挥"哲学和 5-13 spec 的"single source of truth"原则，并新增三条硬约束：

> **1. handoff 双层职责分离 — L1 摘要 + L2 hard fact**
> 一个 subagent 工作的产出表达在两个层面：进入 message history 的简练摘要（L1），与落到文件的完整数据（L2）。所有 agent 默认消费 L1，仅在 L1 信息不足以做出工作产出时升级读 L2。

> **2. runtime / thread / subagent / lead / run 之间路径绝缘**
> 任何 agent 实例（包括同名 subagent 的多次调用）的产物必须能在物理路径层面与其他实例物理隔离。同名文件覆盖不应发生。

> **3. 拥抱 deerflow 框架原生通道、不绕开它**
> handoff 在两个通道分别落位：L1 走 subagent return value（被 task tool 自动包成 ToolMessage 进 lead 的 message history），L2 走 sandbox workspace 文件。这两个通道都是 deerflow 原生提供的——本 spec 不引入新的协作机制，只规范如何用对已有的。

本 spec 所有决策先过这三条筛子。

---

## 1. 背景与触发问题

### 1.1 5-11 / 5-13 spec 留下的演进路径

[5-11 files-are-facts spec](2026-05-11-subagent-file-is-facts-design.md) 已经把架构从 "lead 手工捏 code_summary.json 二手摘要" 演进为 "handoff_*.json 作为权威事实源 + lead 仅按需 read_file"。它解决了"指挥不抄谱"的违反，但留下两条遗债：

1. **lead 在 Step 1.5 拦截决策时仍要 read_file 完整 handoff**：当时把它列为"需要看 data_quality_warnings 几条门禁信号但拿到几 KB JSON 全文"的 follow-up，未根治
2. **handoff 文件名按 subagent 类型命名 (`handoff_code_executor.json`)**：未考虑"同一 subagent 被多次调用"的场景

[5-13 metric catalog spec](2026-05-13-metric-catalog-architecture-design.md) 引入了 catalog YAML 作 single source of truth、code-executor 按 catalog 拼乐高。它把"指标定义、范式映射、展示元数据"从 markdown 迁到 YAML，但**没**讨论 catalog 字段在 subagent 间如何高效传递——code-executor 读完 YAML 后，data-analyst 想用 `direction_for_anxiety / display_name_zh` 时还得**自己再读一遍 YAML**。

### 1.2 2026-05-13 dogfood thread 5046a6e6 暴露的故障级联

完整故障链见 [dogfood-fix-iteration-plan](../plans/2026-05-13-dogfood-fix-iteration-plan.md)。本 spec 关心其中**架构层面**的暴露：

| 现象 | 根因 |
|------|------|
| run 5 派第二次 code-executor 时，`handoff_code_executor.json` 覆盖 run 3 第一次版本 | handoff 文件名按 subagent 类型命名，无 run / 实例隔离 |
| lead 自己在派 data-analyst **之前**写了"7.99% 远低于典型低焦虑小鼠（>20%）" + 编 C57BL/6J 品系 + 引"金标准" | lead 在 message history 里看不到结构化决策信号、只能凭印象；同时 lead 的角色边界不强（详见 [Issue #3 in plan](../plans/2026-05-13-dogfood-fix-iteration-plan.md)） |
| 用户问"能否对现有轨迹做时间分段分析？"→ code-executor 把它当"超出施工单"打回 → lead 自己 write_file 写 `time_bin_epm.py` → 触发 uvicorn reload bug → SSE 断 → thinking 字段丢字段 → 会话死 | 缺少"指标外分析"的合法路径，lead 越权写代码 |
| feedback 表里 needs_fix 关联的 run_id `019e20a6-*` 在 langgraph.log 完全找不到 | runtime 重启后 thread state 部分恢复但 run_id 上下文断裂——无路径隔离让回溯困难 |

### 1.3 2026-05-13 同事 PR #7 反馈

同事在 PR #7 提供了：

- [Gap 清单](../../review-packages/2026-05-13-还差哪些with细颗粒度.md)：
  - **系统功能**："进一步**落实**不要参考常模或基线的默认原则"——"落实"两字暗示之前承诺过、但 thread 5046a6e6 没做到
  - **系统使用体验**："输出到第一版分析耗时久。需要压缩在 5 分钟内"
  - **系统使用体验**："编写代码耗时过久"——指 lead 自己 write_file 写脚本那段
- [指标外分析需求](../../review-packages/2026-05-13-常见行为学指标外分析要求.md)：把"指标外"归纳为 3 类——时间分段、分区、事件/状态分段。这是 catalog 之外的合法分析需求，**lead 当前没有合法路径承接**。

### 1.4 用户对架构的核心诉求

用户在本 spec 讨论过程中反复强调的几点（按出现顺序）：

1. handoff 简练、格式化、是 subagent → lead 的标准契约
2. runtime / thread / subagent / lead / run 路径绝缘
3. 在 deerflow 框架内自洽，**不绕开框架**
4. 用户 AgentCore 经验：文件 handoff 体量大也不漂，因为 lead **不消费完整文件、只读摘要**
5. 未来 subagent 类型不会暴增（就 4 种），但**同一 subagent 可能被多次调用**
6. handoff_suffix 由 lead 显式命名（语义清晰）

这些诉求共同推导出本 spec 的核心架构。

---

## 2. 顶层设计：双层 handoff 协议

### 2.1 隐喻延续

5-11 spec 的乐团指挥隐喻继续生效，扩展两条新角色契约：

| 角色 | 5-11 已定 | 本 spec 新增 |
|------|----------|------------|
| lead = 指挥 | 调度 / 倾听 / 替用户把关 / 用户呈现；不演奏 / 不抄谱 | **看简报做决策**（L1），不消费乐谱完整内容（L2）除非用户特别要求 |
| subagent = 乐手 | 演奏 / 写自己的产物 / 用结构化契约报告指挥 | **同时写两份产物**：简报（L1，给指挥）+ 完整谱子（L2，给审计 + 例外升级路径） |

### 2.2 双层契约

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1 (message history)：所有 agent 默认消费的"通用契约"                 │
│  --------------------------------------------------------------     │
│  -- 体量：~1-2 KB（硬上限 5 KB）                                       │
│  -- 通道：subagent return value → task tool → ToolMessage           │
│           → lead AgentState.messages                                │
│  -- 含义：决策信号 + key results + catalog 字段投影                    │
│  -- 谁消费：lead 默认看；下游 subagent 通过 lead 派遣 prompt inline 看  │
├─────────────────────────────────────────────────────────────────────┤
│  L2 (workspace 文件)：完整 hard fact / 审计 / 例外升级                 │
│  --------------------------------------------------------------     │
│  -- 体量：不限（KB 到 MB 都可）                                        │
│  -- 通道：runs/<run_id>/handoff_<type>__<suffix>.json               │
│  -- 含义：完整原始数据 + 详细 trace + chart base64 + 统计模型对象       │
│  -- 谁消费：仅在 L1 信息不足时由 subagent 自治 read_file 升级          │
│           lead 极少消费（除非用户特别问详情）                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 为什么这套设计抗漂移

用户 AgentCore 经验显示"文件 handoff 体量大也不漂"——经反推，这是因为 **lead 实际消费的是文件中的"摘要部分"，而非完整内容**。本设计把这条隐含约束**显式化**：

- L1 摘要在设计时**强制要求**简练（< 5 KB 硬上限），且默认进 message history——lead 不需要主动选择"读哪一部分"
- L2 完整数据**默认不进 lead 的 prompt**——lead 主动 read_file 才进，且仅在用户特别问详情时才发生
- 下游 subagent 看 L1（lead 在派遣 prompt 中 inline 给它）+ 按需读 L2——下游 subagent 是新 context、读 L2 不影响 lead

**用户最初的担忧"message history 累积会让 lead 漂移"**——在 L1 < 5 KB + 未来 subagent 量级有限的前提下，5 次调用累积 ~25 KB 是可承受的；加上 ArchivingSummarizationMiddleware 长 thread 时还会归档，长期可持续。

### 2.4 L1 schema（规范约定，不做强校验）

L1 通过 subagent 的最终消息返回，格式：

```
OK: <subagent_type> completed

[handoff_summary]
{
  "schema_version": "1.0",
  "subagent": "<type>",                          // code-executor / data-analyst / report-writer / knowledge-assistant
  "handoff_suffix": "<lead 显式命名>",            // 如 epm_basic / time_segment_5bins / single_descriptive
  "run_id": "<langgraph run_id>",                // 给 lead 跨 run 引用时定位
  "status": "success" | "partial" | "failed",
  "key_results": [...],                          // 决策必需的数值 + catalog 字段投影
  "gate_signals": {...},                         // 5-11 spec 沿用的门禁信号
  "summary_text": "1-2 句中文，给 lead 转述用",
  "full_handoff_ref": "{{handoff://<type>__<suffix>}}"   // 占位符，升级路径
}
```

#### 2.4.1 `key_results` 字段（最关键）

`key_results` 是 L1 区别于过往 `[gate_signals]` 块的核心差异——它**带 catalog 字段投影**，让下游 subagent 不用读 catalog YAML 也能直接工作。

code-executor 写 L1 时，**必须**从 catalog YAML 取下列字段放进 `key_results[i]`（每个 metric 一项）：

```json
{
  "id": "open_arm_time_ratio",
  "value": 0.0799,
  "unit": "ratio",
  "display_name_zh": "开放臂时间比例",          // ← 从 catalog YAML 投影
  "unit_zh": "比例",                            // ← 从 catalog YAML 投影
  "one_liner": "动物在开放臂中停留时间占总时长的比例",  // ← 从 catalog YAML 投影
  "direction_for_anxiety": "lower_is_anxious",  // ← 从 catalog YAML 投影（解读方向）
  "statistical_default": "groupwise_compare",   // ← 从 catalog YAML 投影
  "is_primary": true                            // ← 该范式是否核心指标（catalog 衍生）
}
```

**为什么 catalog 字段必须投影到 L1**：让 single source of truth 真正生效。catalog YAML 是源，code-executor 读完 YAML 后投影进 L1。下游 subagent（data-analyst / report-writer）**只看 L1**就能拿到 `direction_for_anxiety / display_name_zh`，不必各自再读 YAML。这是 5-13 spec catalog 架构的真正闭环。

**为什么 5-13 spec 没做到这一步**：因为它只设计了"agent 主动读 catalog YAML"，没设计"catalog 字段如何在 agent 间传递"。本 spec 补这条。

#### 2.4.2 `gate_signals` 字段（5-11 spec 沿用、扩展）

继续作为决策信号，但格式从"文本块"严格化为 JSON：

```json
{
  "data_quality": {
    "critical_count": 1,
    "warning_count": 0,
    "critical_items": [
      {
        "severity": "critical",
        "code": "n_too_small",                    // 机器可读 code
        "message": "仅含 1 个被试，无法进行组间统计比较",
        "metric_id": null                          // 关联的具体指标（如有）
      }
    ]
  },
  "statistical_validity": "skip",                 // ok | warning | failed | skip
  "errors_count": 0,
  "needs_user_decision": true                      // 5-11 spec 已有的拦截信号
}
```

#### 2.4.3 `pending_actions[]` 字段（thread b0d3a611 遗漏根因 1 修复）

`pending_actions[]` 列 **subagent 自己处理不了、需要 lead 后续处理** 的未决事项。**与 `errors` 不同**——errors 是"本次任务过程中遇到的错误"，pending_actions 是"lead 必须显式回应的指令"。

```json
"pending_actions": [
  {
    "action_id": "supplement_trajectory_chart",
    "trigger": "user_request_out_of_plan",          // 触发原因
    "requested_by": "user_message_at_run_2",         // 谁要求的
    "description": "用户请求轨迹图，但 plan.charts 未列",
    "resolution_options": [
      "update_plan_and_redispatch_code_executor",    // 推荐路径
      "ask_clarification_with_user"                  // 备选路径
    ],
    "must_acknowledge": true                         // ← 关键：lead 必须 ack
  }
]
```

**为什么从 `errors` 字段分离**：

- thread b0d3a611 的 `handoff_code_executor.json` 把"轨迹图未在 plan"信息埋在 `errors[]`，lead 看完后**忘了**这是强制指令
- `errors[]` 在 prompt 语义上是"已完成的任务遇到的问题、可参考"，没有"lead 必须处理"的语气
- `pending_actions[]` 明确"待办"语义 + `must_acknowledge=true` 字段——配合 §5.5 的 `HandoffPendingActionsProvider` 实现 **机制层强制 ack**

**lead 处理 `pending_actions[]` 的规则**（写进 lead prompt）：

收到含 `pending_actions[]` 的 handoff 后，lead **下一次 task() 前必须** 二选一：

1. **更新 plan 重派**：更新 `metric_plan.json` 把未决事项加入 charts/metrics → 重派 code-executor（推荐）
2. **澄清用户**：调用 `ask_clarification` 问用户是否要做这件事 / 用什么参数（备选）

**机制保证**：`HandoffPendingActionsProvider`（§5.5.2）会 block `task()` 调用直到 lead 用上面两种方式之一 acknowledge。

#### 2.4.4 L1 体量估算

以一次完整 EPM 单只描述为例：

| 字段 | 估算 token |
|------|----------|
| schema_version + subagent + suffix + run_id | ~50 |
| status + summary_text | ~50 |
| key_results × 5（每项 catalog 字段投影） | 5 × ~200 = ~1000 |
| gate_signals 完整结构化 | ~150 |
| full_handoff_ref | ~30 |
| JSON 结构开销 | ~50 |
| **合计** | **~1330 token ≈ 5 KB** |

EPM 5 个 metric 是 catalog 中 default_metrics 的典型规模。如果范式 metric 数大幅扩张（OFT 也大致 5 个、shoaling 3 个），单次 L1 在 5 KB 量级内可控。

**硬上限**：5 KB。超过则要求 subagent 二次精简——保留 id + value + unit，详细 catalog 字段只保留 display_name_zh，其余字段留给 L2。

#### 2.4.5 L1 不做 JSON Schema 强校验（先约定后验证）

用户拍板：先用 prompt 规范约定 L1 字段，**不在中间件层做强校验**。

理由：
- prompt 已经能让 deepseek 输出 90%+ 正确 JSON
- 加 schema 校验 + auto-retry 是另一个工程，可以等下次 dogfood 暴露真问题再做
- 当前关键是落地分层模型，不是完美约束

下次 dogfood 看到 L1 格式错误率高时再升级为 JSON Schema 强校验 + 拦截重写中间件。

### 2.5 L1 在 thread state 中的实际形态

通过 deerflow 的 task tool 自动 wrap 机制，subagent 返回值变成 `ToolMessage`，进 lead AgentState.messages：

```
AgentState.messages = [
  HumanMessage("请分析这个 EPM 数据"),
  AIMessage(tool_calls=[task(subagent_type="code-executor", ...)]),
  ToolMessage(                                             ← subagent return value 在这
    content="OK: code-executor completed\n\n[handoff_summary]\n{...JSON...}",
    tool_call_id="call_00_xxx"
  ),
  AIMessage("..."),   // lead 看到 ToolMessage 后下一轮推理
  ...
]
```

**lead 看 ToolMessage**就**直接**能拿到 L1 内容——这就是 deerflow 原生通道。

---

## 3. Run-scoped 路径绝缘

### 3.1 绝缘层级回顾

DeerFlow 当前已实现的隔离层级（详见 [paths.py:62-87](../../../packages/agent/backend/packages/harness/deerflow/config/paths.py)）：

| 层级 | 隔离方式 | 状态 |
|------|---------|------|
| user | `{base_dir}/users/{user_id}/threads/` | ✅ 已做 |
| thread | `{base_dir}/users/<uid>/threads/{thread_id}/user-data/` | ✅ 已做 |
| sandbox 容器 | 虚拟路径 `/mnt/user-data/...` 映射 | ✅ 已做 |
| subagent 显式授权 | `{{handoff://name}}` + `HandoffIsolationProvider` | ✅ 已做 |
| **run 之间（同 thread 内多 run）** | — | ❌ **本 spec 补这条** |
| **同 subagent 多次调用** | — | ❌ **本 spec 补这条**（通过 handoff_suffix） |
| runtime 之间 | checkpointer 序列化 | ⚠️ 部分（Issue #2 修复后保证 thinking field 不丢） |

### 3.2 新增 run-scoped workspace 目录结构

```
{base_dir}/users/{user_id}/threads/{thread_id}/user-data/
├── uploads/                          ← thread-scoped（持久，跨 run 共享）
│   └── <user 上传文件>
├── outputs/                          ← thread-scoped（用户可见的最终交付物）
│   └── <agent present_files 给用户的产物>
└── workspace/
    ├── _shared/                      ← thread-scoped 跨 run 共享元数据
    │   ├── columns.json              ← parse 一次、所有 run 复用
    │   ├── experiment-context.json   ← 范式 / 分组 / treatment 描述
    │   └── raw_files.json            ← raw 文件清单
    └── runs/
        ├── <run_id_1>/               ← run-scoped 隔离区
        │   ├── metric_plan.json
        │   ├── m_open_arm_time.json
        │   ├── m_open_arm_time_ratio.json
        │   ├── ...（每 metric 的中间数据）
        │   ├── handoff_code_executor__epm_basic.json
        │   ├── handoff_data_analyst__epm_single_descriptive.json
        │   └── handoff_report_writer__epm_single_descriptive.json
        ├── <run_id_2>/
        │   └── ...
        └── <run_id_N>/
            └── ...
```

#### 3.2.1 路径分类说明

| 路径 | 生命周期 | 谁能读写 |
|------|---------|---------|
| `uploads/` | thread-scoped 永久 | 用户上传、agent 读 |
| `outputs/` | thread-scoped 永久 | agent 通过 present_files 写、用户看 |
| `workspace/_shared/` | thread-scoped 永久 | 跨 run 共享元数据（parse 一次、复用）|
| `workspace/runs/<run_id>/` | run-scoped（run 删时整目录可删） | 该 run 内的所有 subagent 读写 |

#### 3.2.2 关键设计

- **每个 run 一个目录**：run_id 由 langgraph 在用户消息触发时分配，subagent 派遣时通过 runtime context 传递给 sandbox
- **subagent 派遣时获得 `RUN_DIR` 环境变量**（如 `/mnt/user-data/workspace/runs/019e209c-702f`），所有 read/write 都基于该路径
- **handoff 文件命名**：`handoff_<subagent_type>__<lead_命名的 suffix>.json`，双下划线分隔类型和 suffix
- **`_shared/` 目录的内容**：`columns.json` 等只在用户第一次上传 raw 时由 parse subagent 写一次，后续所有 run 都从此读

### 3.3 handoff_suffix 由 lead 显式命名

用户拍板：lead 派遣 subagent 时**必须**显式指定 handoff_suffix，框架不自动生成。

#### 3.3.1 命名规范

- 蛇形小写，仅含字母/数字/下划线
- 体现"派遣意图"（不是 subagent 类型，那个已在文件名前缀）
- 例：
  - `epm_basic` — 基础 EPM 指标计算
  - `time_segment_5bins` — 时间分段 5 等分
  - `single_descriptive` — 单只描述性解读
  - `group_comparison` — 组间统计判读
  - `revision_pass_1` — 第一次修订（用户要求 retry）

#### 3.3.2 在 task() 调用中的位置

```python
task(
  subagent_type="code-executor",
  description="EPM 基础 5 指标计算",
  handoff_suffix="epm_basic",                    # ← 新增必填参数
  prompt="..."
)
```

#### 3.3.3 实现要点

- `task()` 工具签名增加 `handoff_suffix: str`（required）
- subagent 启动时通过 prompt / runtime context 接收 suffix
- subagent 写 L2 时构造路径 `${RUN_DIR}/handoff_${SUBAGENT_TYPE}__${SUFFIX}.json`
- subagent 写 L1 时把 suffix 放在 handoff_summary 字段
- 验证：suffix 不合规（空 / 含非法字符）时 task tool 拒绝调用

### 3.4 跨 run 引用历史 handoff

绝缘**不**意味着 lead 无法引用历史 run 的产物。例外场景：

- 用户问 "刚才单只那次的指标和现在时间分段对比怎么样" → lead 需要同时引用 run N（基础）和 run M（时间分段）的 handoff

实现方式：

- lead 自己的 message history **持续保留** L1 摘要（包括历史 run 的）—— deerflow 通过 checkpoint 跨 run 持久化 message history
- 如果 lead 想读历史 run 的 L2 完整 handoff，使用 `{{handoff://<type>__<suffix>}}` 占位符——`HandoffIsolationProvider` 需扩展支持跨 run 路径解析
- 跨 run 引用是**例外路径**，绝大多数 dogfood 不会触发

---

## 4. 用户感知工作流（端到端，含双层 handoff）

### 4.1 完整 happy path

以用户上传 EPM 数据请求分析为例（thread 5046a6e6 的成功理想版）：

```
[Step 1] 用户：请分析这个 EPM 数据
[Step 2] lead Gate 1：识别范式
         → ask_clarification: "EPM？小鼠？分组如何？"
[Step 3] 用户答完反问
[Step 4] lead 通过 bash 跑 catalog.resolve → 生成 metric_plan.json
         （写到 _shared/）
[Step 5] lead 派 code-executor:
           task(
             subagent_type="code-executor",
             handoff_suffix="epm_basic",
             prompt="按 metric_plan.json 计算 5 个 EPM 指标..."
           )
         播报: "🧮 正在计算 5 个 EPM 指标..."
[Step 6] code-executor 工作：
         - 读 metric_plan.json
         - 读对应 catalog/epm.yaml 取字段（display_name_zh / direction_for_anxiety 等）
         - 跑 5 个 compute_*
         - 写 m_open_arm_time.json 等 5 个文件到 runs/<run_id>/
         - 写完整 L2: runs/<run_id>/handoff_code_executor__epm_basic.json
         - 返回 L1 摘要（自动进 lead message history）
[Step 7] lead 看 L1（无需 read_file 任何 L2）：
         - 看 status=success
         - 看 gate_signals.data_quality.critical_count=1 → n=1 警示
         - 看 key_results 拿到 5 个指标值 + display_name_zh + unit_zh
         - 整合呈现给用户：
           "已计算 5 个硬性指标：
           | 指标 | 数值 | 单位 |
           | 开放臂时间比例 | 7.99% | 比例 |
           | ...
           注意：数据质量提示——仅含 1 个被试，无法做组间统计。"
[Step 8] lead ask_clarification:
         "硬性指标已出，是否要做行为学洞察解读？"
[Step 9] 用户："是"
[Step 10] lead 派 data-analyst（inline L1 摘要给它）:
           task(
             subagent_type="data-analyst",
             handoff_suffix="epm_single_descriptive",
             prompt=f"""
             请对 code-executor 的结果做单只描述性解读。

             code-executor 摘要（看这里做大局判断）：
             {l1_summary_from_history}

             需要详细看具体 metric 数据时：
             {{{{handoff://code-executor__epm_basic}}}}

             遵守 EthoInsight 输出宪法（详见 skills/custom/ethoinsight/references/output-constitution.md）：
             - 单只数据不与外部参考比较
             - 不引用未告知的元数据（品系/体重等）
             - 不使用"典型值/常模/金标准"等表述
             """
           )
[Step 11] data-analyst 工作：
          - 看 prompt 里的 L1 摘要 → 拿到 5 个 metric 值 + catalog 字段
          - 大部分判读直接基于 L1 完成（display_name_zh 翻中文，direction_for_anxiety 知道方向，
            statistical_validity=skip 知道不做推断性结论）
          - 仅当需要更深细节（如某 metric 的源列、轨迹分布特征）时
            read_file {{handoff://code-executor__epm_basic}} 升级到 L2
          - 写自己的 L2: handoff_data_analyst__epm_single_descriptive.json
          - 返回 L1 摘要给 lead
[Step 12] lead 看 data-analyst 的 L1 摘要，整合 code-executor + data-analyst 两份 L1 → 完整呈现给用户
[Step 13] lead ask_clarification:
          "是否要生成结构化研究报告（PDF/Markdown）？"
[Step 14] 用户答（视情况派 report-writer，类似流程）
```

### 4.2 关键设计点

#### 4.2.1 Step 7：lead 不 read_file L2

这是本 spec 的核心修正——**lead 看 L1 摘要做决策、整合呈现、ask_clarification，全程无需 read_file 完整 handoff**。

对比 thread 5046a6e6 实际发生：lead 看完 [gate_signals] 块后**仍然** read 了 handoff_code_executor.json（langgraph.log 显示），把几十 KB 灌入自己的 prompt。本 spec 后 lead message history 累积只有 L1 摘要——~5 KB 每次，可控。

#### 4.2.2 Step 8：硬性指标先呈现、再问是否要洞察

这是用户在讨论中明确表达的工作流：硬性指标（数值）属于 code-executor 的责任、不需要 data-analyst 介入。先给用户看数值，由用户决定要不要洞察。

呼应 [Issue #3 lead 角色边界](../plans/2026-05-13-dogfood-fix-iteration-plan.md)：lead 在派 data-analyst **之前****不**自己写"7.99% 偏低"这类判读。

#### 4.2.3 Step 10：lead inline L1 摘要给下游

lead 自己 message history 里有 code-executor 返回的 L1，派遣 data-analyst 时**复制**这段 JSON inline 进派遣 prompt。

为什么需要 inline 而不是让 data-analyst 自己看 lead 的 message history：
- 每个 subagent 是独立 context（5-11 spec 已确立）
- subagent 看不到 lead 的 message history
- 必须由 lead 主动转发关键信息

#### 4.2.4 Step 11：data-analyst 默认看 L1、按需升级 L2

这是用户拍板的 b 选项含义——下游 subagent 也**默认看 L1**（因为 L1 已包含 catalog 字段投影），仅在判读需要更细节时才读 L2。

判断"L1 够不够"由 data-analyst **自治**决定，prompt 给出参考标准：
- L1 的 key_results + gate_signals 够做单 metric 方向性判读 → 不升级
- 需要看每只被试的具体值分布、轨迹特征 → 升级 read_file L2

### 4.3 失败 / 例外流程

#### 4.3.1 L1 体积超 5 KB

subagent 在写 L1 前自查体积。超过 5 KB 时：
- 保留 key_results 的 id + value + unit
- 移除非必需的 catalog 字段（保留 display_name_zh、移除 one_liner）
- summary_text 不超过 100 字
- 在 L1 注明 `"trimmed": true` 让 lead 知道
- 完整内容仍写 L2

#### 4.3.2 L1 信息不足 data-analyst 决定升级

data-analyst 调用 `read_file {{handoff://code-executor__<suffix>}}` 时由 `HandoffIsolationProvider` 验证授权（lead 在派遣 prompt 中使用过此占位符即授权）。

#### 4.3.3 同一 subagent 多次调用

run 3 派了 code-executor 写 `runs/019e209c/handoff_code_executor__epm_basic.json`，run 5 又派 code-executor 写 `runs/019e20b2/handoff_code_executor__time_segment_5bins.json`：

- 物理路径完全不同（run_id 不同）—— 不会覆盖
- 即使在同一 run 内（极少）派两次相同 subagent，suffix 不同也不会覆盖
- 极端情况：lead 派两次相同 type + 相同 suffix（应该禁止）—— task tool 校验时拒绝

---

## 5. EthoInsight 输出宪法（Output Constitution）

### 5.1 动机

同事 Gap 清单 #1："进一步**落实**不要参考常模或基线的默认原则"——"落实"两字暗示之前承诺过但 thread 5046a6e6 没做到。具体在 lead 自己写"远低于典型低焦虑小鼠（>20%）" / 编 C57BL/6J 品系 / 引"金标准"。

[Issue #3 dogfood-fix plan](../plans/2026-05-13-dogfood-fix-iteration-plan.md) 当前在 `lead_agent/prompt.py` 加一段"角色边界硬约束"——但这只覆盖 lead，不覆盖 data-analyst / report-writer / 未来新 subagent。

### 5.2 设计

把"判读哲学 / 元数据 grounding / 表达规范"沉到**一份共享文档**：

```
/home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethoinsight/references/output-constitution.md
```

所有 subagent 的 prompt 包含一句"开工前必须 read_file output-constitution.md"，并在 handoff L1 中加 `"constitution_acknowledged": true` 字段证明已读。

### 5.3 宪法内容草案（10 条以内）

```markdown
# EthoInsight 输出宪法

> 所有 EthoInsight agent（lead 和 subagent）输出时必须遵守。
> 由行为学同事 + 团队共同维护。

## 1. 判读哲学：组间比较 > 绝对阈值

- **不要**将单只或单组数据与"典型值/常模/金标准/参考范围/文献基线"比较
- **只**评估 control vs treatment 之间是否有显著差异
- 单只数据 → 仅描述性表述、不做方向性判读

## 2. 元数据 grounding：忠于输入

- 用户消息和 raw file headers 中**未出现**的字段（品系/性别/体重/年龄/给药剂量等），**禁止**在输出中编造
- 同事如需标注品系等，**先反问用户**，不要凭印象推测

## 3. 表达规范：用 catalog YAML 的中文翻译

- 指标中文名 = catalog YAML 的 `display_name_zh`
- 单位 = catalog YAML 的 `unit_zh`
- 解读方向 = catalog YAML 的 `direction_for_anxiety`（如有）
- 不要凭印象翻译

## 4. 数值精度

- 比例：保留 2-4 位小数（如 0.0799 或 7.99%）
- 时间：保留 2 位小数（如 23.56 秒）
- 计数：整数
- 不要给出超出 raw 数据精度的虚假精确数字

## 5. 引用规范

- 引用具体数据时标注来源（"来自 handoff_code_executor.json 的 open_arm_time_ratio"）
- 引用领域知识时标注来源（"参见 skills/custom/ethoinsight/references/<file>"）
- 不要编造文献引用

## 6. 执行边界（角色不可越权）

**lead 是调度员，不是执行员**。即使用户提出"补充图表 / 补充分析 / 换个参数算"等指标外请求，lead **不能**：

1. **不能 `write_file` 写可执行脚本** — 包括 `.py` / `.sh` / `.ipynb` / `.bash` / `.zsh` 扩展名的文件
2. **不能用 `bash` 跑分析脚本** — bash 仅限：
   - `python -m ethoinsight.parse.*`（解析 EthoVision 文件）
   - `python -m ethoinsight.catalog.*`（生成 metric_plan.json）
   - 安全文件操作：`mkdir / cp / mv / ls / cat / grep / head / tail`
3. **不能 `python -c "..."` heredoc 写分析逻辑** — 同上禁令包含这种内联写法

**当用户提出指标外请求时，lead 必须二选一**：

a) **更新 plan 重派 code-executor**：把请求映射到 metric_plan.json 的 `charts` 或新 metric → 重派
b) **澄清用户**：用 `ask_clarification` 问"该请求需要：i) 走标准 plan 重新计算 / ii) 暂不做 / iii) 走 ad-hoc 路径（未来工作）"

**为什么是机制层禁止、不只是 prompt 约束**：

thread b0d3a611 暴露了 prompt 自觉的脆弱性——commit `24715250` 已经在 lead prompt 加了 3 条角色边界禁令，但 lead 仍然在 msg 52 的 thinking 里说 "I can generate them with a simple Python script. Let me do this directly rather than re-running the full code-executor pipeline"。**LLM 在压力下会自作主张绕开 prompt 禁令**。

§5.5 节定义机制层 GuardrailProvider，把本条从"应当"升级为"做不到"。

## 7. Pending Actions 强制 Acknowledge

任何 subagent 返回的 L1 中含 `pending_actions[]`（详见 §2.4.3）时，lead **必须**在下一次 `task()` 调用前完成 acknowledge：

| Acknowledge 方式 | 实现 |
|---|---|
| 更新 plan 重派 | 更新 `metric_plan.json` 包含未决事项 → 调 task(code-executor) |
| 澄清用户 | 调 `ask_clarification` 问用户决策 |

**机制保证**：§5.5 节的 `HandoffPendingActionsProvider` 会 deny `task()` 调用直到 lead 用上面两种方式之一 ack。

## 8-10：未来扩展（同事 PR 时补）
```

### 5.4 与本 spec 其它部分的协同

- code-executor 派遣 prompt：`"开工前请 read_file <constitution path>，遵守第 1-7 条"`
- data-analyst 派遣 prompt：同上，特别强调第 1 条（判读哲学）
- report-writer 派遣 prompt：同上，特别强调第 3 条（表达规范）+ 第 5 条（引用规范）
- L1 `constitution_acknowledged` 字段：subagent 证明已读
- 宪法第 6-7 条（执行边界 + pending_actions ack）通过 §5.5 机制层 guardrail 强制，不依赖 subagent 自觉

### 5.5 机制层执行边界（GuardrailProvider）

宪法第 6 条（执行边界）和第 7 条（pending_actions ack）通过 deerflow 现成的 [GuardrailProvider 协议](../../../packages/agent/backend/packages/harness/deerflow/guardrails/provider.py) 实现机制层防护。

**新增 2 个 Provider**（仿照 [`ScriptInvocationOnlyProvider`](../../../packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py) 的实现模板）：

#### 5.5.1 `LeadAgentExecutionBoundaryProvider`

**职责**：强制宪法第 6 条——lead 不能写脚本、不能跑非白名单 bash

**新文件**：`packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py`

**逻辑**：

```python
class LeadAgentExecutionBoundaryProvider:
    name = "lead_execution_boundary"

    _LEAD_BASH_ALLOWED = re.compile(
        r"^\s*python\s+-m\s+ethoinsight\.(parse|catalog)\."
    )
    _LEAD_BASH_SAFE_OPS = re.compile(
        r"^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)"
    )
    _FORBIDDEN_SCRIPT_EXTENSIONS = (".py", ".sh", ".ipynb", ".bash", ".zsh")

    def evaluate(self, request):
        # Only gate lead — subagents have their own providers
        if request.agent_id and request.agent_id.startswith("subagent:"):
            return GuardrailDecision(allow=True)

        if request.tool_name == "write_file":
            path = request.tool_input.get("path", "")
            if path.endswith(self._FORBIDDEN_SCRIPT_EXTENSIONS):
                return GuardrailDecision(allow=False, reasons=[
                    GuardrailReason(
                        code="lead_execution_boundary.script_write_forbidden",
                        message=(
                            "lead 是调度员，不写脚本。补充分析/图表请：\n"
                            "  a) 更新 metric_plan.json → 重派 code-executor\n"
                            "  b) ask_clarification 问用户是否要做\n"
                            "执行分析脚本是 code-executor 的工作。"
                        )
                    )
                ])

        if request.tool_name == "bash":
            cmd = request.tool_input.get("command", "")
            if self._LEAD_BASH_ALLOWED.match(cmd) or self._LEAD_BASH_SAFE_OPS.match(cmd):
                return GuardrailDecision(allow=True)
            return GuardrailDecision(allow=False, reasons=[
                GuardrailReason(
                    code="lead_execution_boundary.bash_not_allowed",
                    message=(
                        "lead 的 bash 仅可：\n"
                        "  1. python -m ethoinsight.parse.* （解析数据）\n"
                        "  2. python -m ethoinsight.catalog.* （生成 plan.json）\n"
                        "  3. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
                        "执行分析脚本请走 task(code-executor)。"
                    )
                )
            ])

        return GuardrailDecision(allow=True)

    async def aevaluate(self, request):
        return self.evaluate(request)
```

**Wire 位置**：[`lead_agent/agent.py` line 314](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py) 之后追加：

```python
from deerflow.guardrails.lead_execution_boundary_provider import LeadAgentExecutionBoundaryProvider
middlewares.append(GuardrailMiddleware(
    provider=LeadAgentExecutionBoundaryProvider(),
    fail_closed=True,
))
```

#### 5.5.2 `HandoffPendingActionsProvider`

**职责**：强制宪法第 7 条——`pending_actions[]` 未 ack 时 block `task()`

**新文件**：`packages/agent/backend/packages/harness/deerflow/guardrails/handoff_pending_actions_provider.py`

**逻辑**：

```python
class HandoffPendingActionsProvider:
    """Block task() when prior handoff_*.json has unacknowledged pending_actions."""
    name = "handoff_pending_actions"

    def __init__(self, workspace_resolver):
        # 复用 Ev19WorkspaceBridgeMiddleware 的 contextvar 模式
        self._resolve_workspace = workspace_resolver

    def evaluate(self, request):
        # 只在 lead 调用 task() 时检查
        if request.agent_id and request.agent_id.startswith("subagent:"):
            return GuardrailDecision(allow=True)
        if request.tool_name not in ("task", "ask_clarification"):
            return GuardrailDecision(allow=True)

        # ask_clarification 是合法 ack 方式，pass
        if request.tool_name == "ask_clarification":
            return GuardrailDecision(allow=True)

        workspace = self._resolve_workspace()
        if workspace is None:
            return GuardrailDecision(allow=True)  # fail-open

        # 扫描所有 handoff_*.json 找未 ack 的 pending_actions
        unacked = self._find_unacked_pending_actions(workspace)
        if not unacked:
            return GuardrailDecision(allow=True)

        # 检查本次 task 是否在 ack 范围内
        # （lead 派的 subagent_type 或 prompt 内容覆盖了 unacked.action_id）
        if self._task_acks_pending(request.tool_input, unacked):
            return GuardrailDecision(allow=True)

        action_descs = "\n".join(
            f"  - {a['action_id']}: {a['description']}" for a in unacked
        )
        return GuardrailDecision(allow=False, reasons=[
            GuardrailReason(
                code="handoff_pending_actions.unacknowledged",
                message=(
                    f"前序 handoff 中有未处理的 pending_actions：\n{action_descs}\n\n"
                    "处理方式：\n"
                    "  a) 更新 metric_plan.json 包含这些事项 → 重派 code-executor\n"
                    "  b) ask_clarification 问用户决策\n"
                    "完成后再调用 task()。"
                )
            )
        ], policy_id="handoff_pending_actions")
```

**Wire 位置**：同 `LeadAgentExecutionBoundaryProvider`，加到 lead 中间件链。需要配套加 `WorkspaceBridgeMiddleware`（仿 [`Ev19WorkspaceBridgeMiddleware`](../../../packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py)）注入 workspace contextvar。

**实现细节留给 plan 阶段**：

- `_find_unacked_pending_actions` 的具体匹配规则（按时间戳？按 run_id？）
- `_task_acks_pending` 的匹配启发式（task prompt 包含 action_id？subagent_type=code-executor + plan 包含未决事项？）

---

## 6. Catalog Modifier（指标外分析的合法路径）—— 未来工作

### 6.1 同事需求回顾

同事在 [指标外分析需求](../../review-packages/2026-05-13-常见行为学指标外分析要求.md) 把"指标外"归纳为 3 类：

| 类别 | 同事描述 | 例子 |
|------|---------|------|
| 时间分段分析 | 把总时长切成 N 段 | 5×60s bins、首末 20%、二分前后 |
| 分区分析 | 按空间区域聚合 | 中心区 vs 外周区移动距离 |
| 事件/状态分段分析 | 按硬件状态/阈值切片 | 明暗期、给药前后、活跃度 >10 |

thread 5046a6e6 的 run 5 暴露的问题：用户问"能否对现有轨迹做时间分段分析" → code-executor 当"超出施工单"打回 → lead 选择**自己 write_file** 写 Python 脚本 → 触发故障级联。

### 6.2 修复方向：catalog 引入 modifier 维度

不直接在本 spec 实施，但**指出方向**让下次端到端测试后开新 spec 实施：

```yaml
# packages/ethoinsight/ethoinsight/catalog/epm.yaml
default_metrics:
  - id: open_arm_time
    script: ethoinsight.scripts.epm.compute_open_arm_time
    supports_modifiers: [time_segment]    # ← 声明该 metric 能接哪些模具

# packages/ethoinsight/ethoinsight/catalog/modifiers/time_segment.yaml（跨范式共享）
modifiers:
  - id: time_segment.equal_n
    description: 把总时长等分 N 段
    args_schema: { n_bins: int }
  - id: time_segment.first_last_pct
    description: 前 X% 和后 Y% 各取一段
    args_schema: { first_pct: float, last_pct: float }
```

`metric_plan.json` 扩展：

```json
{
  "metrics": [
    {
      "id": "open_arm_time",
      "modifier": {
        "id": "time_segment.equal_n",
        "args": { "n_bins": 5 }
      },
      "output": "runs/<run_id>/m_open_arm_time__tbin5.json"
    }
  ]
}
```

`compute_*` 脚本必须支持 `--modifier <id>` + `--modifier-args <json>` 参数。

### 6.3 为什么不在本 spec 落实施细节

- 涉及面大：catalog YAML schema 改、metric_plan.json schema 改、所有 compute_* 脚本改
- 是独立项目：与 handoff 双层协议正交，先做 handoff 协议、dogfood 测试稳定后再做 modifier
- 同事文档刚提交（5-13）：需要团队/同事一起 review 设计选择

本 spec 仅作为方向预留——未来 plan 时引用本节作为前置。

---

## 7. 与现有 Issue / Plan 的关系

### 7.1 与 [dogfood-fix-iteration-plan](../plans/2026-05-13-dogfood-fix-iteration-plan.md) 的关系

dogfood plan 是**当前正在执行的 bug 修复**（执行 agent 在另一会话跑）。本 spec 是**架构演进设计**，二者关系：

| dogfood plan Task | 与本 spec 的关系 |
|------------------|----------------|
| Task 1 修 uvicorn reload glob | 独立 bug，不在本 spec 范围 |
| Task 2 lead 角色边界 prompt | **过渡方案**——commit 24715250 修了 3 条输出禁令（不写判读 / 不引未告知元数据 / 不用绝对参考术语），但 thread b0d3a611 验证 prompt 自觉不够。本 spec §5.5 用机制层 `LeadAgentExecutionBoundaryProvider` 根治"lead 越权写脚本"这一执行层漏洞。Task 2 的 3 条输出禁令保留并迁入 §5 宪法第 1-3 条。 |
| Task 3-4 修 thinking 字段丢失 | runtime 绝缘的基础保证，本 spec 第 3.1 节列在"已部分做" |
| Task 5 阶段播报 | 独立 UX，不在本 spec 范围 |
| Task 6 reasoning autoClose | 独立前端，不在本 spec 范围 |
| Task 7 catalog plan 路径虚拟化 | **本 spec 第 3 章路径绝缘的子集**——dogfood 修这一处、本 spec 系统化到 run 层 |
| Task 8 code-executor 不重跑 | 独立行为，不在本 spec 范围 |
| Task 9 dogfood 验证 | 本 spec 后续端到端测试的前置 |
| Task 10 checkpointer adelete_for_runs | run 取消时清理——本 spec run-scoped 隔离的配套 |

**dogfood plan 修紧急 bug、稳定基线**；**本 spec 演进架构、补 5-11/5-13 spec 的下一阶段**。两者不冲突、不相互阻塞。

### 7.2 与 [5-11 files-are-facts](2026-05-11-subagent-file-is-facts-design.md) 的关系

继承：handoff 作事实源、subagent 间需 lead 显式授权、乐团指挥隐喻。

扩展：handoff 分层（L1 摘要 + L2 完整）、catalog 字段投影到 L1、run-scoped 路径绝缘、同 subagent 多次调用通过 suffix 隔离。

修订：5-11 spec 留下的 follow-up "lead 在 Step 1.5 read_file 完整 handoff 膨胀上下文" 由本 spec L1 摘要根治。

### 7.3 与 [5-13 metric-catalog](2026-05-13-metric-catalog-architecture-design.md) 的关系

继承：catalog YAML 作 single source of truth、code-executor 按 catalog 拼乐高、Q6 白名单反退化测试。

扩展：catalog 字段如何在 subagent 间高效传递——通过 L1 投影，data-analyst / report-writer 不必各自 read catalog YAML。

未涉及：catalog modifier（指标外分析）—— 本 spec 第 6 章预留方向、不实施。

### 7.4 与 thread b0d3a611 E2E 失败的关系

本 spec draft v2 修订（§2.4.3 pending_actions / §5 宪法第 6-7 条 / §5.5 机制层 guardrail）直接响应 thread b0d3a611 E2E 失败的根因 A 和遗漏根因 1。详细诊断材料见 [docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md](../../problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md)。

**根因 B（sandbox 路径不对称）+ 触发点（`//` 误报）的处置**：详见本 spec §10.3 长期 backlog。在 §5.5 的执行边界 guardrail 落地后，这两个 bug 在 lead 路径上**不再可触发**（lead 跑不了 `write_file *.py` 和 `python heredoc` 了），降级为非紧急维护项。

---

## 8. 实施路径概览（仅指方向、不写 plan）

本 spec 定稿后开 plan 时，预期分阶段：

### 阶段 1：双层 handoff 协议（核心，~1-2 周）

1. 改 task tool 签名加 `handoff_suffix: str` 必填参数
2. 改 4 个 subagent prompt（code-executor / data-analyst / report-writer / knowledge-assistant）：
   - 接收 `handoff_suffix` 参数
   - 写 L2 时构造路径 `${RUN_DIR}/handoff_<type>__<suffix>.json`
   - 返回 L1 摘要（统一 schema）
   - catalog 字段投影逻辑（仅 code-executor）
3. 改 lead prompt：
   - 必须为每次 task() 显式指定 handoff_suffix
   - 派 data-analyst / report-writer 时 inline 上一个 subagent 的 L1
   - 决策时只看 L1、避免 read_file L2

### 阶段 1.5：执行边界 GuardrailProvider（~2-3 天，独立可先做）

**优先级**：高——直接修 thread b0d3a611 暴露的根因 A，可独立于阶段 1 落地

1. 新建 `guardrails/lead_execution_boundary_provider.py`（~80 行，照 `script_invocation_only_provider.py` 模板）
2. 新建 `guardrails/handoff_pending_actions_provider.py`（~120 行，照 `ev19_template_provider.py` 模板含 WorkspaceBridge）—— **本步依赖阶段 1 的 L1 schema 落地（pending_actions 字段）；如阶段 1 未完成，先只上 (1)、(3) 关于 LeadAgentExecutionBoundaryProvider 的部分**
3. Wire 到 `lead_agent/agent.py` 中间件链（line 314 之后）
4. 单测覆盖：
   - lead write_file `.py` → deny + 错误信息检查
   - lead write_file `.md` → allow
   - lead bash `python -m ethoinsight.catalog.resolve ...` → allow
   - lead bash `python -c "..."` → deny
   - lead bash `python /path/to/file.py` → deny
   - 含 pending_actions[] 的 handoff 存在 → lead task() deny
   - lead ask_clarification → 始终 allow（合法 ack 方式）
   - subagent (passport="subagent:...") 的 bash/write_file → pass through（不受 lead 边界限制）
5. 集成测试：用 thread b0d3a611 的复现流跑一遍——确认 lead 在 msg 52 想 write_file `gen_charts.py` 时被 deny + 看到错误信息后转向 update plan 或 ask_clarification

**阶段 1.5 的 `LeadAgentExecutionBoundaryProvider` 部分**可以与阶段 1 并行；`HandoffPendingActionsProvider` 部分需要阶段 1 的 L1 schema（含 pending_actions[] 字段）落地。

### 阶段 2：run-scoped 路径绝缘（~1 周）

1. 改 `paths.py` 加 `runs/<run_id>/` 子目录支持
2. SandboxMiddleware 给 subagent 注入 `RUN_DIR` 环境变量
3. catalog.resolve 支持新路径（呼应 dogfood Task 7）
4. `HandoffIsolationProvider` 支持跨 run 路径解析
5. 加 `adelete_for_runs` checkpointer（与 dogfood Task 10 合并）

### 阶段 3：输出宪法 + Constitution acknowledged 机制（~3-5 天）

1. 落 `output-constitution.md` 初版
2. 4 个 subagent prompt 加"开工前 read_file"
3. L1 schema 加 `constitution_acknowledged` 字段
4. 同事 PR 进一步完善宪法内容

### 阶段 4：端到端验证 + spec 定稿

跑完整 dogfood，验证：
- 同 subagent 多次调用不冲突
- lead 不 read_file L2、决策准确度不下降
- 用户感知工作流（硬性指标 → ask → 洞察）流畅
- 5 个 metric 的 EPM 单只完整链路在 5 分钟内（同事 Gap #4）

### 后续

阶段 5：catalog modifier（指标外分析）—— 独立 spec + plan
阶段 6：微调数据生产管道借用 L2 完整 handoff —— 训练飞轮升级

---

## 9. 验收标准（草案）

本 spec 实施完成后，应能通过以下端到端验证：

1. **同 subagent 多次调用**：thread 内派 code-executor 两次（不同 suffix），两次产物物理隔离、互不覆盖
2. **跨 run handoff 引用**：lead 在 run N+1 能引用 run N 的 L1 摘要做决策
3. **lead 不 read_file L2**：完整 dogfood 走完，langgraph.log 显示 lead 没有 read 任何 `handoff_*.json`
4. **catalog 字段不打架**：data-analyst 报告中的 `display_name_zh / direction_for_anxiety` 与 catalog YAML 一致（通过 grep 对比验证）
5. **下游 subagent L1 自给**：data-analyst 在 90%+ 场景下完成判读不升级读 L2（通过 SandboxAudit 统计 read_file 频率）
6. **输出宪法生效**：dogfood 中 lead / data-analyst / report-writer 都不再出现"典型值/常模/金标准/未告知品系"等违规话术（呼应同事 PR Gap #1）
7. **5 分钟体感**（同事 Gap #4）：EPM 单只完整链路（lead → code-executor → data-analyst）在 5 分钟内完成

---

## 10. 未决项与风险

### 10.1 未决项

- **宪法内容**：6-10 条留给同事 PR 完善——5/14 spec 评审时同步邀请同事
- **L1 schema 演进**：本 spec 用规范约定不做强校验。下次 dogfood 看到错格式高频时升级为 JSON Schema 强校验 + 自动重写 middleware
- **跨 thread handoff 引用**：本 spec 假设跨 thread 引用是反模式（不应允许）。如果未来用户期望"thread A 学到的知识自动应用到 thread B"，需要另设计 thread 间通讯通道（不是 handoff）

### 10.2 风险

| 风险 | 缓解 |
|------|------|
| L1 摘要太精简导致 lead 决策偏差 | 阶段 4 dogfood 时重点验证；准备 fallback 方案（lead 在特定 gate_signals 下强制 read_file L2） |
| handoff_suffix 命名失控（lead 起重复或冲突的 suffix） | task tool 校验：同 run 内同 type+suffix 拒绝；下次 dogfood 看到再加更强约束 |
| 同事 PR 来更多反馈推翻 L1 schema 设计 | 本 spec 标 draft；端到端测试 + 同事 review 后才进 plan |
| catalog 字段投影增加 code-executor 输出体积 | 阶段 1 实施后单测体积；超 5 KB 触发二次精简（详见 4.3.1） |
| 跨 run 引用的 HandoffIsolationProvider 扩展引入安全漏洞 | code review 时重点审计跨 run 解析逻辑；保留默认拒绝兜底 |

---

### 10.3 长期 backlog（不在本 spec 实施范围）

#### 10.3.1 sandbox 路径不对称（thread b0d3a611 根因 B）

**问题**：[`sandbox/tools.py:1568-1607`](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py) `write_file_tool` 写文件时不替换 content 里的 `/mnt/user-data/*` 字面量；而 `bash_tool` 的命令字符串会被 `replace_virtual_paths_in_command` 替换。**语义不对称**：

- `bash python -c "open('/mnt/user-data/uploads/x.txt')"` ✅ 工作（命令字符串被替换）
- `write_file '/mnt/.../x.py'` 写 `open('/mnt/user-data/uploads/x.txt')` 然后 `bash python /mnt/.../x.py` ❌ FileNotFoundError（脚本内容不被替换）

**为何不修**：在 §5.5 `LeadAgentExecutionBoundaryProvider` 落地后，lead 不能 `write_file *.py`，code-executor 又只能 `python -m ethoinsight.scripts.*`（预置模块自己读 `DEERFLOW_PATH_*` env var）——**两条都堵了之后这个 bug 在生产路径上无法触发**。

**论证缺陷修正（thread 8ff3be6d dogfood 暴露）**：上述判断**不完整**。lead 调 `python -m ethoinsight.catalog.resolve` 等 ethoinsight CLI 仍然合法走 sandbox replace 通道——如果 CLI 接受"虚拟路径作为 metadata 字符串"的参数（例如 `--virtual-workspace-dir /mnt/user-data/workspace` 想让该字符串被写进 plan.json 而非作为文件操作目标），sandbox replace 会把该字符串翻译成物理路径，CLI 收到时语义已丢失。

**已采取的修复模式（G5 修复 plan）**：CLI 改用 env var `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` 反推虚拟路径，绕过被替换的命令行参数值。这是 §10.3.1 修复哲学（"用 env var 旁路 sandbox 替换"）的**早期落地**——本 spec 阶段未来如有同类需求（CLI 接受虚拟路径作 metadata），应复用同一模式：在 CLI 端用 env var key 的存在性判断"是否在 sandbox 模式下"，并用代码硬编码的虚拟路径常量作为 fallback 返回值。

**仍然不动 sandbox 本身**：本论证缺陷的修正不改变"不修 sandbox 路径替换逻辑"的结论——只是说明根因 B 的处置不能只靠"lead 路径堵死"，还需要在所有"虚拟路径作 metadata"的 CLI 接受口加 env var fallback。

**未来何时该修**：当我们允许新的 subagent 自由写 `.py` 脚本（如未来 ad-hoc analysis subagent）时，需要回头根治：在 sandbox write_file_tool 中按文件扩展名白名单跑路径替换，或注入引导脚本头自动读 `DEERFLOW_PATH_*`。

#### 10.3.2 `_ABSOLUTE_PATH_PATTERN` 正则误报（thread b0d3a611 触发点）

**问题**：[`sandbox/tools.py:24`](../../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py) 的 `_ABSOLUTE_PATH_PATTERN` 是纯正则扫描，**不识别 shell 上下文**（heredoc body / `python -c "..."` 引号内容 / Python 整除 `//`）。thread b0d3a611 的 `n // 5000` 被误判为绝对路径。

**为何不修**：

- lead 路径上不再可触发（执行边界 guardrail 堵死）
- code-executor 路径上用 `python -m ...`，不进 heredoc，也不会写 `n // 5000` 这类表达式作为命令字符串

**未来何时该修**：当某个合法 bash 调用确实被误报（如 grep 含 `//` 的 URL pattern）时，根治方向：删 `_ABSOLUTE_PATH_PATTERN` 全局扫描，所有路径校验走 `_split_shell_tokens` 的 token-level 判断 + 加 heredoc-aware tokenizer（识别 `<<DELIM` 边界，body 整段视为一个 token 不扫描）。这是 `shlex` 标准库的局限，需要手写 bash parser 或引入 `bashlex`。

#### 10.3.3 ArchivingSummarizationMiddleware 压缩 pending_actions

**问题**：lead 看完 handoff 后过几轮触发 archive，pending_actions 内容可能被压缩出 lead 视野。

**目前缓解**：

- L2 文件不删，lead 任何时候可 read_file 找回
- `HandoffPendingActionsProvider`（§5.5.2）会**主动扫描 workspace 的 handoff_*.json**，不依赖 message history——所以即使被 archive 压掉，guardrail 仍然能 block 未 ack 的 task()

**因此本问题在 §5.5 落地后自然消解**，不需独立修复。

---

## 11. 修订记录

| 日期 | 版本 | 修订说明 |
|------|------|---------|
| 2026-05-14 | draft v1 | 初稿——用户 + Claude 对话沉淀；待执行 agent 完成 dogfood 修复 + 一次完整端到端测试后定稿 |
| 2026-05-14 | draft v2 | 根据 thread b0d3a611 E2E 失败 + deepseek co-analysis + deerflow GuardrailProvider 机制发现更新：§2.4 加 `pending_actions[]`、§5 加宪法第 6-7 条 + §5.5 机制层 guardrail（两个新 provider）、§7 标 dogfood Task 2 为过渡方案 + 新增 §7.4、§8 加阶段 1.5、§10.3 加长期 backlog 论证。复用 deerflow GuardrailProvider 机制实现根治。 |

---

## 12. 评审 / 反馈邀请

本 spec 草案预期接受以下 audience 的 review：

- **用户**：架构判断、与产品愿景匹配度
- **行为学同事**（PR #7 作者）：output-constitution 内容、判读哲学落实方式
- **下一阶段执行 agent**：实施可操作性、是否需要更细的 plan

评审反馈通过 PR comment、handoff 文档、或新一轮讨论沉淀，最终更新本 spec 进入 plan 阶段。

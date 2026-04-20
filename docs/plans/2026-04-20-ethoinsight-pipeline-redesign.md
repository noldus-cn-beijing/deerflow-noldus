# EthoInsight 流水线重设计

> 日期: 2026-04-20
> 作者: 与 Opus 4.7 协作 brainstorm
> 状态: 设计已对齐，待实施
> 背景会话: 2026-04-20 斑马鱼 shoaling E2E 测试暴露多个问题

---

## 1. 动机

2026-04-20 第一次真实端到端 E2E 测试（上传 5 个斑马鱼 shoaling 轨迹文件）暴露三类问题：

1. **流水线死板** — 强制 code-executor → data-analyst → report-writer 三步，即使用户只想看分析结果不要 APA 报告也要等几分钟生成
2. **Subagent 自身鲁棒性不足** — 非外部原因（LLM 超时、429 限流）的失败：数据边界处理缺失（n=2 样本、单鱼文件计算群体指标产生零方差伪数据）、handoff 契约模糊、失败返回格式不统一
3. **前端显示反了** — 用户看不到 agent 的专家思考，却看得到 `Tool: write_file`、`Tool: task` 这种底层调用

本设计针对这三类问题。**外部因素（LLM 超时/429/网络）不在本次范围**。

---

## 2. 核心理念

**从"流水线工厂"变成"交互式助手"**。

- 分析先行，报告按需
- 失败不静默 bypass，统一走 ask_clarification 问用户
- Subagent 自身设计要鲁棒，靠契约和数据边界防御，而不是靠 lead agent 兜底
- 前端最大化复用 DeerFlow 上游已有能力，只在真正缺失处补丁

---

## 3. 新流水线

```
用户上传数据 + 分析请求
  ↓
[code-executor]  解析 + 统计 (写 handoff_code_executor.json)
  ↓
[data-analyst]   专家解读 (写 analysis_summary.md)
  ↓
Lead 用自然语言整合呈现给用户
  ├─ 2-3 段中文文本
  ├─ 指标表格
  ├─ 关键洞察
  └─ 数据质量警告
  ↓
ask_clarification 三选一:
  ① 需要 APA 格式报告
  ② 不需要，谢谢
  ③ 先帮我解释 XX
  ↓
选① → [report-writer] 生成 APA 报告
选② → 结束
选③ → [knowledge-assistant] 回答
```

**关键变化**:

- `report-writer` **不再是默认步骤**，用户明示才调用
- 用户"已有分析数据 + 要求写报告"场景 → 跳过 code-executor，直连 report-writer
- 用户"只分析看看" → 停在呈现洞察，节省 2-3 分钟报告生成时间

**prompt 改动位置**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- L230 派遣顺序从 3 步改为 2 步 + ask_clarification
- L249 "只帮我重新写个报告"分支保留并强化

---

## 4. 失败处理统一化

**现状（问题）**: data-analyst / report-writer 失败时，prompt 指令让 lead 静默 bypass 自己硬写，导致流水线质量劣化、专家视角缺失。

**新设计**: 所有 subagent 失败统一走 ask_clarification。

| 失败点 | 新策略 |
|---|---|
| code-executor 失败 | 已有 ask_clarification（保留） |
| data-analyst 失败 | `ask_clarification(options=["重试", "直接展示 code-executor 原始统计结果（跳过专家解读）", "中止"])` |
| report-writer 失败 | `ask_clarification(options=["重试", "只要分析洞察就够了（不要报告）", "中止"])` |

**prompt 改动位置**: `lead_agent/prompt.py` L278-L284 失败处理表格重写；L285-L313 失败处理规则从"只处理 code-executor"扩展到全部 subagent。

---

## 5. Subagent 自身鲁棒性 — 四层防御

### L1 — Handoff 契约强约束

新建 `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`，用 Pydantic 定义所有 subagent 返回的 handoff JSON schema:

```python
class CodeExecutorHandoff(BaseModel):
    status: Literal["completed", "partial", "failed"]
    output_files: list[Path]
    metrics_summary: dict[str, MetricStat]
    data_quality_warnings: list[DataQualityWarning]
    confidence: float
    errors: list[str] = []

class DataAnalystHandoff(BaseModel):
    status: Literal["completed", "failed"]
    analysis_summary_path: Path
    key_findings: list[str]
    excluded_metrics: list[str]
    recommendations: list[str]
```

Lead 读 handoff 时用 Pydantic 验证，schema 错 = 明确的"契约失败"，不是模糊的"subagent 失败"。

### L2 — 数据边界防御

修改 `packages/ethoinsight/ethoinsight/metrics.py` 和 `assess.py`:

- **n<3 组**: 返回 `DataQualityWarning(severity="critical", metric="all", message="样本量过小...")`，不阻塞计算（继续输出描述统计）
- **单鱼文件 × 群体指标**: IID / polarity 计算函数内部检测到"只有 1 个 subject"时返回 `{"applicable": false, "reason": "group metric requires ≥2 simultaneously tracked subjects"}`，**而不是返回零方差伪数据**

本次 E2E 暴露最严重的 bug 在这里——IID / polarity 那些"零方差"本不该算出来。

### L3 — Subagent prompt 明确化

审查以下位置:

- `packages/agent/skills/custom/ethoinsight/SKILL.md`
- `packages/agent/skills/custom/ethoinsight-analysis/SKILL.md`
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/*/prompt.py`

确保每个 subagent 的 prompt 都明确:

- 输入格式（带例子）
- 输出格式（Pydantic schema 的 JSON 例子）
- 失败时如何返回 `{"status": "failed", "errors": [...]}`

### L4 — 工具契约测试

新建 `packages/agent/backend/tests/test_subagent_contracts.py`，用 mock LLM（按脚本返回 tool_call）跑 subagent 子流水线，验证:

- code-executor: 给定 n=2/n=3 shoaling 数据 → 产出合法 `CodeExecutorHandoff`
- data-analyst: 给定 `code_summary.json` → 产出合法 `DataAnalystHandoff`
- n=1 文件（单鱼）→ IID / polarity 被标 `applicable: false`

**不测 LLM 智能，只测契约和工具调用序列。**

---

## 6. write_file 大文件鲁棒性

**问题**: Sonnet 写 10K+ 字符 APA 报告时 `write_file` 报 `content: Field required`，根因是 `parse_docstring=True` 的 schema 对 Sonnet 不够明确 + 单次 content 太大。

**改造** (`packages/agent/backend/packages/harness/deerflow/sandbox/tools.py` L1279):

1. **显式 Pydantic args_schema** 替换 `parse_docstring=True`
   - `content: str = Field(..., description="...", min_length=0)` 明确必填
2. **阈值保护** — 超过 8000 字符直接 fail-fast，返回明确指引:
   ```
   Error: Content exceeds 8000 chars (got 10236).
   Please split and call write_file multiple times:
   1. First call: append=False with first chunk
   2. Subsequent calls: append=True with remaining chunks
   ```
3. **不新增 streaming tool** (YAGNI) — 现有 append 参数已够用，只是模型不知道要分段

**测试**:

- `test_write_file_schema.py` 验证 Pydantic schema 接受/拒绝输入
- `test_write_file_large_content.py` 验证 >8000 字符返回正确 fail-fast 消息
- E2E: 让 Sonnet 写 15K 字符报告，验证它会分段

---

## 7. 前端显示 — 复用 DeerFlow 上游

**重大发现**: DeerFlow 上游已经有 4/5 个我们需要的能力，不要造轮子。

### 已有能力（直接复用）

| 能力 | DeerFlow 实现 |
|---|---|
| Todo list UI（折叠式、状态图标） | `frontend/src/components/workspace/todo-list.tsx` L14-99 |
| 文件卡片 + 下载预览 | `frontend/src/components/workspace/artifacts/artifact-file-list.tsx` L40-193 |
| Subagent 卡片 + running/completed/failed 状态 | `frontend/src/components/workspace/messages/subtask-card.tsx` L32-177 |
| Tool call 语义化标签 | `frontend/src/components/workspace/messages/message-group.tsx` L186-424（按 name 分支渲染） |

### 唯一缺失（需要新做）

**`ask_clarification` 交互按钮** — 现只显示静态 "Need your help" 标签，无法点选 options 直接回复。

**新增工作**:
- React 组件渲染 options 按钮组（~100 行）
- 点击按钮时填充用户输入框并触发发送（复用现有 send-message API）

### 调整工作

在 `message-group.tsx` L186-424 的 if-else 链中:

- **隐藏**: `read_file` / `write_file` / `str_replace` / `bash` / `ls`（I/O 工具，用户不关心）
- **隐藏**: `get_analysis_template` / `parse_trajectories` / `compute_metrics` / `run_statistics` / `generate_charts` / `assess_and_handoff`（ethoinsight 细粒度工具，过程透明但用户不关心）
- **保留**: `task`（显示为 subtask 卡片）、`ask_clarification`（新增按钮组）、`present_files`（文件卡片）、`write_todos`（todo list）

### Lead prompt 补"过程透明"章节

`lead_agent/prompt.py` 新增:

```
### 过程透明原则
每次派遣 subagent / 调用 ask_clarification / 呈现文件之前，用 1-2 句中文告诉用户：
- 正在做什么（"正在解读统计结果..."）
- 发现了什么（"IID 零方差，很可能是单鱼文件的计算问题..."）
- 下一步（"接下来会问你要不要生成 APA 报告..."）

不要说 "我会调用 data-analyst"，说 "正在请专家解读结果"。
用户视角无需知道内部 subagent 名字。
```

---

## 8. 实施路径（6 个 commit）

| # | Commit | 影响文件 | 风险 |
|---|---|---|---|
| 1 | Handoff Pydantic schemas + 契约测试 | 新建 `handoff_schemas.py`、`test_subagent_contracts.py` | 低，纯新增 |
| 2 | ethoinsight 数据边界防御（单鱼检测 / n<3 警告） | `ethoinsight/metrics.py`、`assess.py` + 测试 | 中，改现有计算逻辑 |
| 3 | write_file 显式 schema + 8000 字符阈值 | `sandbox/tools.py` + 测试 | 中，改工具行为 |
| 4 | Subagent SKILL.md / prompt 明确化审查 | `skills/custom/*/SKILL.md`、`subagents/builtins/*/prompt.py` | 低，文档为主 |
| 5 | Lead prompt 改造：流水线按需、失败 ask_clarification、过程透明 | `lead_agent/prompt.py` + 测试 | 中高，核心改动 |
| 6 | 前端：`ask_clarification` 按钮组件 + `message-group.tsx` tool 过滤 | `frontend/` 相关组件 | 中，需要 QA 视觉验证 |

**建议顺序**: 1 → 2 → 3 → 4 → 5 → 6（先地基后上层；prompt 放第 5 因依赖前面契约和边界都就位）

**验证**:

- 每个 commit 前后跑 `make test`
- commit 5 后跑 E2E 场景验证（同样 shoaling 数据，期望 lead 在 data-analyst 完成后停下来问是否要 APA 报告）
- commit 6 后人工验证前端聊天流不再出现 `Tool: write_file` 这种原始字样，`ask_clarification` 选项可点击

---

## 9. 六个关键决策（brainstorm 对齐记录）

| # | 问题 | 决策 |
|---|---|---|
| 1 | data-analyst 完成后如何呈现结果 | Lead agent 整合为自然语言输出 |
| 2 | 如何触发 report-writer | ask_clarification 三选一（按钮点击） |
| 3 | subagent 失败时策略 | 全部走 ask_clarification 问用户 |
| 4 | write_file 大文件鲁棒性方式 | 显式 schema + 8000 字符阈值保护 |
| 5 | 前端显示策略 | 渐进：隐藏 I/O 类工具，保留语义工具 |
| 6 | lead agent 思考过程可见方式 | prompt 里要求 lead 主动"说出来" |
| 补 | subagent 自身鲁棒性范围 | 工具契约测试 + Prompt 明确化 + Handoff 强约束 + 数据边界防御（四项全做） |

---

## 10. 不做什么（YAGNI 记录）

- ❌ 新增 `write_file_streaming` tool（现有 append 够用）
- ❌ 前端事件过滤白名单配置文件（DeerFlow 现有 if-else 分支够用）
- ❌ 改 SSE 通道流式输出 thinking tokens（prompt 要求说出来更可控）
- ❌ 给 lead agent 加"重试 N 次"的自动重试机制（统一 ask_clarification 问用户意愿，避免静默重试浪费 token）

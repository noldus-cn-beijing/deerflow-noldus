# EV19 列语义确认 — 架构设计讨论文档

> 本文档汇总了 2026-06-05 关于"列语义 HITL 确认"的完整设计讨论，供外部 agent 分析最佳实践。

## 1. 背景

### 1.1 项目简介

EthoInsight — 面向行为学研究员的 AI 分析助手。基于 DeerFlow（LangGraph agent 框架 fork）。用户上传 EthoVision XT (EV19) 导出的轨迹数据，Agent 自动完成统计分析、图表生成、报告撰写。

### 1.2 触发事件

用户上传了一批真实的 OFT（旷场）数据（34 个 XLSX 文件，英文版 EV19 导出），agent 判定"不是 EV19 数据"拒绝分析。

### 1.3 直接根因

`detect_ethovision()` 只检查 XLSX 文件第一格是否含 `"标题行数"`（EV19 中文版 header marker）。英文版第一格是 `"Number of header lines:"` → 返回 False → agent 报 `format_unrecognized`。

### 1.4 深层问题

EV19 导出数据的列名有三层确定性：

| 层级 | 确定性 | 列名特征 | 示例 | 数量 |
|------|--------|----------|------|------|
| **L1 固定列** | EV19 导出必定存在，名称固定 | 标准化 tracking 列 | `Trial time`, `Recording time`, `X center`, `Y center`, `Area`, `Areachange`, `Elongation` | 7 列 |
| **L2 伪固定列** | 语义强相关但用户可改名 | 英文版 `Distance moved` vs 中文版 `移动距离`；zone 列（用户自定义区域名称） | `distance_moved`, `velocity`, `mobility_state`, `in_zone_*`, `center` | ~15-30 列 |
| **L3 自由列** | 用户完全自定义 | 自定义 result 变量、JS 状态机输出、实验者手动添加的列 | `result_1`, `边缘区到center`, `CustomVariable_JS` | 不定 |

当前系统在面对 L2/L3 的不确定性时，缺乏机制与用户确认——agent 看到不认识的东西就放弃了。**应该做的是：承认这是 EV19 数据（基于 L1 骨架），列出识别到的 + 不确定的，进入 Human-in-the-Loop 确认流程。**

## 2. 业务需求

1. Agent 能基于 L1 固定列签名识别出 EV19 数据，不因不认识的列名放弃
2. 对于不确定的列（zone 列语义、unknown 列含义），进入 HITL 流程让用户确认
3. 用户确认的列语义全局可用（当前 session 的 data-analyst、report-writer subagent 能消费）
4. 这是一个**主动的协作仪式**（"我看到这些列，我的理解是 X，对吗？"），不是被动的纠错机制（"resolve 报错了所以问你"）
5. 对标准数据（所有列被识别）不增加额外步骤

## 3. 现有架构

### 3.1 Agent 流水线

```
upload → identify_ev19_template (parse + zone detect + template candidate lookup)
       → [ambiguous?] ask_clarification
       → set_experiment_paradigm (guardrail checks user_confirmed_template)
       → prep_metric_plan
       → task(code-executor) → handoff_code_executor.json
       → task(data-analyst) → handoff_data_analyst.json
       → [Gate 2 quality warnings?] ask_clarification → acknowledge_quality
       → task(chart-maker) → task(report-writer)
```

### 3.2 关键 infra（全部已存在，可复用）

| Infra | 作用 | 复用方式 |
|-------|------|----------|
| `inspect_uploaded_file` tool | 解析文件 → 返回 columns + data preview + metadata | **增强返回值**：加 column_assessment |
| `ask_clarification` + `ClarificationMiddleware` | 反问用户 + Command(goto=END) 中断等待回复 | **直接复用**：合并反问 |
| `set_experiment_paradigm` | 写 experiment-context.json | **加模式**：`confirm_column_semantics=True`（对标已有的 `acknowledge_quality`） |
| `experiment-context.json` | 实验配置全局状态文件 | **加字段**：`column_semantics` |
| `Ev19TemplateGuardrailProvider` | 拦截未确认模板的 task(code-executor) | **加子检查**：column_semantics 是否已确认 |
| skill 渐进披露 | lead agent 按需 read_file 方法论 | **新增 skill**：ethoinsight-column-confirmation |
| prompt.py "反问合并规则" | 教 lead 如何合并多个待确认事项为一条 ask_clarification | **加范例**：列确认的合并示例 |
| `ethovision-paradigm-knowledge` skill | 范式/模板的领域知识 | **补充**：各范式常见列的默认语义（如 "OFT 中 in_zone 通常代表中心区"） |

### 3.3 已有的 zone HITL 先例

`catalog/resolve.py` 中 `_detect_anonymous_zone` + `anonymous_zone_is` override + `parameter_overrides` 已经是一条列语义 HITL 的微缩版：数据里只有裸 `in_zone` 而 metric 需要 `in_zone_center_*` → resolve 报 `zone_unnamed` → prompt 驱动 lead 反问 → 用户答 → `parameter_overrides` 存决议。**这条链路跑通了，但被设计为被动纠错而非主动确认，且硬编码只认 `in_zone` 一种列。**

## 4. 设计方案

### 4.1 修正后的流水线（Gate 1.5 插入）

```
identify_ev19_template → paradigm_key + template candidates
    ↓
inspect_uploaded_file → columns + data preview + [新] column_assessment + open_questions
    ↓
ask_clarification（合并：模板选择 + zone 含义 + unknown 列 + 分组）
    ↓
set_experiment_paradigm → lock paradigm + template + column_semantics
    ↓
prep_metric_plan（column_semantics 已知，zone_unnamed 不触发或已有 override）
    ↓
task(code-executor)
```

`column_assessment` 不新建独立工具，而是 `inspect_uploaded_file` 的自然增强——该工具已经 parse header + 展示列结构，加列分类是同一职责的延伸。

### 4.2 三层职责切分

| 层 | 职责 | 放哪里 | 内容 |
|---|------|--------|------|
| **工具层** | 产出 data（列分类结果 + open_questions） | `inspect_uploaded_file` 增强，内部调 ethoinsight 库 | 每列的 category（fixed/recognized/zone/unknown）、confidence、inferred_meaning |
| **Prompt 层** | 定义触发时机 + 合并规则 | `lead_agent/prompt.py` | 1-2 行触发条件 + skill 指针；在现有的"反问合并规则"段加列确认范例 |
| **Skill 层** | 交互方法论（thin） | 新 `ethoinsight-column-confirmation` SKILL.md + references | 展示模板、正例反例、答案映射、跳过条件 |

### 4.3 Skill 边界（关键：遵循 SSOT 铁律）

**存**（操作手册——"怎么交互"）：
- 触发条件：inspect 返回 column_assessment 中有 `needs_confirmation=true` 的列
- 跳过条件：所有列被识别 → 不触发（标准数据零额外开销）
- 展示模板：用户可见的 ask_clarification 话术正例
- 反例：常见错误（"逐列列出所有 23 列"、"对 fixed 列问含义"、"假设 unknown 不重要而跳过"）
- 答案映射：用户自然语言 → `set_experiment_paradigm(column_semantics={...})` 参数
- 多文件场景处理

**不存**（ethoinsight 库/工具的职责）：
- L1/L2/L3 列分类规则（`ethoinsight/utils.py` 的 `normalize_column_name` + COLUMN_MAP）
- Zone 列检测模式（`inspect_uploaded_file` 内的 `_compute_anonymous_zone_evidence`）
- `inspect_uploaded_file` 返回值的字段结构描述（反脑补铁律：`feedback_skill_describing_tool_output_enables_hallucination`）
- 范式/模板知识（`ethovision-paradigm-knowledge` skill 的职责）

### 4.4 三个 skill 的协作关系

```
ethovision-paradigm-knowledge          ethoinsight-lead-interaction
("OFT 中 in_zone 通常代表中心区")      ("如何合并多个待确认事项为一条反问")
         |                                       |
         +───────────┬───────────────────────────+
                     |
          ethoinsight-column-confirmation (新)
          ("怎么让用户确认这些列的含义、答完怎么落盘")
```

### 4.5 建议的 skill 文件结构

```
skills/custom/ethoinsight-column-confirmation/
├── SKILL.md                        # ~60 行，触发条件 + 方法论概述 + 落盘方式
└── references/
    ├── presentation-template.md     # 展示模板正例（1 个）+ 反例（3 个）
    └── answer-mapping.md            # 用户回答 → column_semantics 映射表
```

### 4.6 Gate 1.5 guardrail

复用 `Ev19TemplateGuardrailProvider`，在 `task(code-executor)` 检查中：
- 现有：`ev19_template` 未设置 → 拦截
- 新增：`column_semantics` 的 `open_questions` 未全部确认 → 拦截（提示信息列出待确认的列）

注意：只有 `open_questions` 非空时才拦截。标准数据不触发。

### 4.7 experiment-context.json column_semantics schema

```json
{
  "paradigm": "open_field",
  "ev19_template": "OpenFieldRectangle-AllZones",
  "column_semantics": {
    "confirmed_at": "2026-06-05T14:30:00Z",
    "columns": {
      "in_zone": {
        "raw_name": "In zone(Center / Center point)",
        "meaning_zh": "中心区位置标记",
        "category": "zone"
      },
      "边缘区到center": {
        "raw_name": "边缘区到center",
        "meaning_zh": "用户自定义：边缘区到中心区距离",
        "category": "user_defined"
      }
    }
  }
}
```

- 只存 `needs_confirmation=true` 的列
- `meaning_zh` 直接用于 data-analyst 报告引用

## 5. 实施清单

| 文件 | 改动 | 估计量 |
|------|------|--------|
| `ethoinsight/parse/_core.py` | `detect_ethovision` 加固：英文 header marker（已修）+ L1 固定列签名兜底检测 | ~20 行 |
| `ethoinsight/utils.py` | 新增 `_FIXED_COLUMNS` 常量 + `assess_column_confidence()` 函数 | ~80 行 |
| `inspect_uploaded_file` tool | 增强返回 column_assessment + open_questions | ~60 行 |
| `skills/.../ethoinsight-column-confirmation/SKILL.md` | **新建** thin SKILL.md | ~60 行 |
| `skills/.../ethoinsight-column-confirmation/references/` | **新建** 2 个 reference 文件 | ~80 行 |
| `lead_agent/prompt.py` | 加 1-2 行触发条件 + skill 指针 + 反问合并规则加范例 | ~15 行 |
| `experiment_context.py` | `set_experiment_paradigm` 加 `confirm_column_semantics` 模式 | ~30 行 |
| `ev19_template_provider.py` | guardrail 加 column_semantics 子检查 | ~25 行 |
| `extensions_config.json` | 注册新 skill | 3 行 |
| `test_*.py` | 测试覆盖 | ~200 行 |

## 6. 待分析的设计问题

1. **L1 固定列阈值**：6/7 还是 7/7？需要实测英文版 EV19 所有 locale/版本下的固定列是否一致。
2. **Skill 的"三件一起做"**：创建文件 + extensions_config 注册 + prompt 引用，是否有遗漏？
3. **ask_clarification 合并的最大信息密度**：模板选择 + zone 含义 + unknown 列 + 分组，四个问题合并一次反问的体验是否好？是否需要分两次？
4. **已有 `_detect_anonymous_zone` 的处理**：在 Gate 1.5 主动确认后，prep_metric_plan 里的 zone_unnamed 检测是否应该从"单独反问"降级为"兜底"？还是完全移除？
5. **是否应该在 catalog YAML 中为每个范式声明"常见列"**，让 column_assessment 的 inferred_meaning 有范式感知（如 OFT 的 in_zone 默认 meaning="中心区"，而 EPM 的 in_zone 可能是"开臂"）

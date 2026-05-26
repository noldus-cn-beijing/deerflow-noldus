# Metric Catalog 架构设计

> 日期：2026-05-13
> 状态：brainstorming 完成、待用户审 spec、再进入 writing-plans
> 前置：[2026-05-12 Script-Per-Metric 架构设计](2026-05-12-script-per-metric-architecture-design.md)（本 spec 的直接前身、已 approved）
> 触发：2026-05-13 同事对 5-12 review 包反馈（[`docs/review-packages/2026-05-12-feedback.md`](../../review-packages/2026-05-12-feedback.md)）+ 用户在 brainstorming 中提出的 single source of truth 硬要求

---

## 设计原则（贯穿本 spec）

继承 5-12 spec 的总原则："**agent 架构由上游 deerflow 操心，我们专注数据分析洞察**"，并新增一条硬约束：

> **Single source of truth — 同一份知识（指标定义 / 范式映射 / 展示元数据 / 判读方向性）绝对不能在 skill 文档和代码库里各维护一份。**

用户原话："千万不要有两份知识打架"。本 spec 所有决策先过这条筛子：

- 凡是想往 skill markdown 里塞"指标 id 列表 / 中文名 / 单位 / 范式默认指标"等结构化知识的提议 → 否决，沉到代码库 YAML
- 凡是 agent 主动 read 的知识源 → 必须有"单测可校验、golden-case 可消费"的代码层落点
- skill 文档**只能写"如何读 catalog + 按 role 关注什么字段"的指引**，不能内嵌结构化数据

---

## 1. 背景与触发问题

### 1.1 5-12 spec 解决了什么、留下了什么

[2026-05-12 Script-Per-Metric 架构](2026-05-12-script-per-metric-architecture-design.md) 把 ethoinsight 从"agent 拼胶水代码"改造成"每个指标一个独立可执行脚本、agent 是编排者"。它解决了 e2e 测试暴露的 `python -c help(...)` 反复试错问题，并通过 `ScriptInvocationOnlyProvider` Guardrail 把"agent 只调脚本"从软约定升级为硬约束。

但它留下两条遗债：

1. **范式知识仍在 skill markdown 里**：每个范式一份 `references/by-paradigm/<paradigm>.md`，含"脚本清单 + 决策树 + handoff schema 局部约定"。这些**应该 agent + 代码 + 单测 + golden-case 同时消费的事实**，写成自由文本 markdown 无法被代码层校验。
2. **决策权混在 code-executor**：code-executor 自己 read by-paradigm md → 自己裁剪脚本清单 → 自己跑 → 自己聚合。这一切都发生在 LLM 推理内部，决策不可审计、不可单测、训练数据切分困难。

### 1.2 2026-05-13 同事反馈进一步暴露的问题

同事 5-13 对 5-12 review 包给出 [feedback](../../review-packages/2026-05-12-feedback.md)，核心冲击：

| 反馈点 | 暴露的架构问题 |
|--------|----------------|
| "不要参考常模或基线，所有数据指标要和对照组比" | `ethoinsight/assess.py` 的 `_DEFAULT_THRESHOLDS`（normal_range / high_anxiety）是死代码 + 错误哲学 |
| Q4 整题作废、Q5 推断错误（Mobility_1/10 是采样率不是阈值） | review 包对"指标含义和判读"的描述跟代码事实并行存在两份独立文档 |
| Q6 给出明确指标白名单（EPM 5 个、OFT 5 个、FST 3 个） | 当前 OFT 缺 `center_distance` 和 `center_time`、FST 命名对齐但需要 catalog 化 |
| Q2 "列名歧义时不要猜，要问" | OFT silent fallback（裸 `in_zone` 当 center）违反此原则 |
| Q3 "raw 最后一列是 result block 命名，不是 user-defined variable" | `parse.py` 自动分组推断逻辑需重写 |

这些问题的根本病灶是 **"知识源分散、agent 自行裁剪"**。Q6 白名单如果只写在 review 包 markdown 里、不沉到代码层 YAML，下次同事改主意 → 改 review 包 → 但 agent 跑的指标没变 → 同事再来一轮反馈。

### 1.3 本次任务承载的关键决策

1. **catalog 沉到 ethoinsight 库**：`packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml` 作为"范式→指标清单 + 展示元数据 + 判读方向"的 single source
2. **资源解析交给 Python 函数 + CLI**：`ethoinsight.catalog.resolve` 函数读 catalog + 列名 + 用户定制 → 生成 `metric_plan.json`，agent 只触发 bash、不在 LLM 推理里做查表
3. **职责切分 A1**：lead 亲自 bash 调 `parse.dump_headers` + `catalog.resolve` 生成 plan.json → 派遣 code-executor 读 plan 逐条执行；不派遣 subagent "勘察"
4. **双 skill 切分**：`ethoinsight-code`（瘦身后，工程执行手册）+ `ethoinsight-metric-catalog`（新建，领域知识接口）
5. **多消费者共读**：lead / data-analyst / report-writer 共享 `ethoinsight-metric-catalog`，按 role 读 catalog 的不同字段子集
6. **现状清扫**：删 7 份 `by-paradigm/<paradigm>.md`、删 `assess.py` 的死阈值代码、`error-recovery.md` 和 `quality-checks.md` 中 `assess_and_handoff` 旧引用

---

## 2. 三个候选 + 决策

### 2.1 候选 A：catalog 只给 lead 用、lead 在派遣 prompt 内嵌完整脚本清单

- lead 读 `ethoinsight-metric-catalog` skill → 读 catalog YAML → 自己拼脚本调用清单 → 写进 code-executor 派遣 prompt
- code-executor 极简，只跑 bash + 写 handoff
- **被否决的理由**：lead 在 prompt 里拼 N 行 `python -m ethoinsight.scripts.<paradigm>.<script> --input ... --output ...` 字符串容易引入幻觉（路径写错、参数漏掉），且 lead context 在 Qwen3-8B 工程预算下已接近临界，再多 200 tokens 不安全

### 2.2 候选 B：code-executor 读 catalog 自己选脚本

- lead 只传 paradigm + 用户定制需求（自然语言或 JSON）
- code-executor 持有 catalog skill，读 catalog 自己做"指标筛选 + 路径拼接"
- **被否决的理由**：把"中文意图 → catalog 查表 + set 运算"这种确定性操作交给 LLM 多一跳。Qwen3-8B 在 yaml lookup + set difference 上的可靠性显著低于代码 + 单测。每多一个 LLM 决策节点 = 多一个失败源、多一份训练数据切分复杂度。违反"不要让 AI 写胶水代码不停试错"原则

### 2.3 候选 C'（**最终选择**）：catalog 函数化、lead 触发确定性 bash 生成 plan.json

- catalog YAML 是数据源、`ethoinsight.catalog.resolve` 是数据消费函数（Python + 单测）
- lead 触发 `python -m ethoinsight.catalog.resolve` 生成 `metric_plan.json` 文件
- code-executor 读 plan.json 逐条执行 —— "知道我要做什么" = "plan.json 里写的"
- 决策路径上 LLM 节点只有 lead 一个；查表 / 路径拼接 / 列名校验全在确定性 Python 代码里
- 通过 files-are-facts 机制传递事实（plan.json 是 lead → code-executor 的硬契约文件）
- **优势汇总**（对照本仓库已有最佳实践）：
  - files-are-facts ✓（[2026-05-11 spec](2026-05-11-subagent-file-is-facts-design.md)）
  - script-invocation-only Guardrail 零改动 ✓（[5-12 spec §4](2026-05-12-script-per-metric-architecture-design.md)）
  - lead bash 已有 affordance 复用 ✓
  - 减少 LLM 决策跳数（对 Qwen3-8B 友好）✓
  - 反问链路从 3 跳缩到 1 跳（对齐同事"列名歧义不要猜要问"）✓
  - skill 渐进披露完整性 ✓（lead 一侧一次走完）

### 2.4 决策与理由

**走 C'**。它在 9 项核心维度上对 A / B 是 8-2 完胜（参见 brainstorming 过程记录），不存在接近的判断。

C' 的本质是**用代码替代 LLM 推理**做确定性操作，把 LLM 留给真正需要语义理解的事（识别范式、收口用户定制、判断是否反问）。

---

## 3. catalog 数据层架构

### 3.1 目录结构

```
packages/ethoinsight/ethoinsight/catalog/
├── __init__.py          # public API: load_catalog(paradigm) -> dict, resolve(...) -> dict
├── loader.py            # YAML 读取 + schema 校验
├── resolve.py           # 主逻辑：读 catalog + columns + 用户偏差 → 生成 plan
├── cli.py               # python -m ethoinsight.catalog.resolve 入口
├── schema.py            # pydantic 或 dataclass 模型，agent 端 / 测试端共享
├── epm.yaml
├── oft.yaml
├── fst.yaml
├── tst.yaml
├── ldb.yaml
├── zero_maze.yaml
└── shoaling.yaml
```

### 3.2 catalog YAML schema

每份 `<paradigm>.yaml` 的字段：

```yaml
paradigm: epm
ev19_templates:
  - Elevated Plus Maze XT190
  - Elevated Plus Maze (custom)

default_metrics:
  - id: open_arm_time_ratio
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns: [in_zone_open_arms_*]
    output_unit: ratio
    display_name_zh: 开放臂时间比例
    unit_zh: 比例
    one_liner: 动物在开放臂中停留时间占总时长的比例，用于评估焦虑样回避行为
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_arm_time
    script: ethoinsight.scripts.epm.compute_open_arm_time
    requires_columns: [in_zone_open_arms_*]
    output_unit: seconds
    display_name_zh: 开放臂时间
    unit_zh: 秒
    one_liner: 动物在开放臂内的累计停留时间
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_arm_entry_count
    script: ethoinsight.scripts.epm.compute_open_arm_entry_count
    requires_columns: [in_zone_open_arms_*]
    output_unit: count
    display_name_zh: 开放臂进入次数
    unit_zh: 次
    one_liner: 进入开放臂的累计次数
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_arm_entry_ratio
    script: ethoinsight.scripts.epm.compute_open_arm_entry_ratio
    requires_columns: [in_zone_open_arms_*, in_zone_closed_arms_*]
    output_unit: ratio
    display_name_zh: 开放臂进入比例
    unit_zh: 比例
    one_liner: 进入开放臂次数占总进入次数的比例
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: total_entry_count
    script: ethoinsight.scripts.epm.compute_total_entry_count
    requires_columns: [in_zone_open_arms_*, in_zone_closed_arms_*]
    output_unit: count
    display_name_zh: 总进入次数
    unit_zh: 次
    one_liner: 进入所有臂的累计次数，反映动物的整体探索活动量
    direction_for_anxiety: null
    statistical_default: groupwise_compare

optional_metrics: []

charts:
  - id: trajectory
    script: ethoinsight.scripts.epm.plot_trajectory
    when: always
  - id: box_open_arm
    script: ethoinsight.scripts.epm.plot_box_open_arm
    when: n_per_group >= 3

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.epm.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

字段语义：

| 字段 | 用途 | 谁读 |
|------|------|------|
| `paradigm` | 范式 key，与 `ev19_facts.py` 中 paradigm 对齐 | resolve / 测试 |
| `ev19_templates` | 该范式对应的 EV19 模板名（事实表），跟 `ethovision-paradigm-knowledge` skill 的 by-experiment 互不重复 —— catalog 只放映射、不放识别决策树 | resolve（用于断言一致性） |
| `default_metrics[].id` | 指标唯一标识，handoff 中作为 key | code-executor / data-analyst / report-writer |
| `default_metrics[].script` | dotted import 路径，resolve 拼出 `python -m <path> --input ... --output ...` | code-executor |
| `default_metrics[].requires_columns` | 需要的数据列模式（含 glob `*`），resolve 跟 `columns.json` 对照决定是否可执行 | resolve |
| `default_metrics[].output_unit` | 机器可读单位（`ratio`/`seconds`/`count`/`cm`/...） | dispatcher / 测试 |
| `default_metrics[].display_name_zh` | 中文展示名 | report-writer |
| `default_metrics[].unit_zh` | 中文单位 | report-writer |
| `default_metrics[].one_liner` | 一句话领域解释 | report-writer §4 / knowledge-assistant 追问 |
| `default_metrics[].direction_for_anxiety` | enum: `lower_is_anxious` / `higher_is_anxious` / `null`（非焦虑范式或方向无关） | data-analyst |
| `default_metrics[].statistical_default` | enum: `groupwise_compare` / `paired_compare`（v0.2+ 可扩展） | dispatcher / data-analyst |
| `optional_metrics` | 同 default 结构，但需用户显式 include 或数据列恰好存在 | resolve |
| `charts[].when` | enum: `always` / `n_per_group >= K` / `n_groups >= K` 等条件 | resolve |
| `statistics.default.when` | 何时跑统计；不满足时 plan 中标 `skip_reason` | resolve |

### 3.3 catalog 加载与校验

`loader.py` 启动时（或单测中）校验每份 YAML：

| 校验项 | 失败行为 |
|--------|----------|
| YAML 语法正确 | 测试 fail |
| 必填字段齐全 | 测试 fail |
| `script` 字段指向的 Python 模块可 import | 测试 fail |
| `id` 在同一范式内唯一 | 测试 fail |
| `direction_for_anxiety` 是允许的 enum 值 | 测试 fail |
| `statistical_default` 是 ethoinsight 已实现的入口 | 测试 fail |

校验在 CI 跑（`pytest packages/ethoinsight/tests/test_catalog.py`），保证 "catalog 说有 → 脚本一定存在 → 单测可消费"。

### 3.4 catalog.resolve 函数契约

**Python API**：

```python
def resolve(
    paradigm: str,
    columns: list[str],              # 来自 columns.json
    raw_files: list[str],            # raw 文件绝对路径列表
    groups_file: str | None = None,  # groups.json 绝对路径
    columns_file: str | None = None, # columns.json 绝对路径（用于 plan.inputs 字段回填）
    workspace_dir: str = ".",        # plan.json 中 metrics[].output 等路径的 base
    include: list[str] = (),         # 用户额外要的 metric ids
    exclude: list[str] = (),         # 用户排除的 metric ids
    n_per_group: int | None = None,  # lead 给的实验设计信息
    n_groups: int | None = None,
) -> dict:                            # 返回 plan dict（结构见 §4）
    """生成 metric_plan.json 数据结构。

    失败模式（抛 ResolveError，含 code 字段供 lead 反问）:
    - unknown_paradigm
    - unknown_metric: include 里的 id 不在 catalog
    - empty_plan: 所有 default + include 都因列缺失或 exclude 被剪光
    - columns_missing: default 或 user-include 指标的 requires_columns 在 columns 中找不到（含 ambiguous 子类）
    - schema_violation: catalog YAML 损坏
    """
```

**CLI 接口**（`python -m ethoinsight.catalog.resolve`）：

```
--paradigm PARADIGM             必填
--columns-file PATH             必填（dump_headers 产物）
--raw-files-json PATH           必填（指向 JSON 数组的文件，单文件场景也用单元素数组包装）
--groups-file PATH              可选；不分组时省略
--workspace-dir PATH            必填；plan 中 metrics[].output / charts[].output 的 base 目录
--include METRIC_ID             可重复
--exclude METRIC_ID             可重复
--n-per-group INT               可选；charts/statistics 的 when 条件用
--n-groups INT                  可选
--output PATH                   plan.json 写盘位置
```

出错时：
- exit 0 + 写 plan.json：正常
- exit 1 + stderr 写结构化错误：lead 看到 stderr → 触发 ask_clarification

错误 JSON 格式：

```json
{
  "code": "unknown_metric",
  "message": "Metric 'foo_bar' not found in epm catalog.",
  "details": {"requested": "foo_bar", "available": ["open_arm_time", ...]}
}
```

---

## 4. metric_plan.json schema

```json
{
  "schema_version": "1.0",
  "paradigm": "epm",
  "ev19_template": "Elevated Plus Maze XT190",
  "generated_at": "2026-05-13T11:23:00Z",
  "inputs": {
    "raw_files": ["/mnt/user-data/uploads/raw.txt"],
    "groups_file": "/mnt/user-data/workspace/groups.json",
    "columns_file": "/mnt/user-data/workspace/columns.json"
  },
  "metrics": [
    {
      "id": "open_arm_time_ratio",
      "script": "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
      "input": "/mnt/user-data/uploads/raw.txt",
      "output": "/mnt/user-data/workspace/m_open_arm_time_ratio.json",
      "required": true,
      "reason": "epm.default"
    }
  ],
  "statistics": {
    "id": "groupwise_compare",
    "script": "ethoinsight.scripts.epm.run_groupwise_stats",
    "input": "/mnt/user-data/workspace/handoff_code_executor.json",
    "output": "/mnt/user-data/workspace/stats.json",
    "skip_reason": null
  },
  "charts": [
    {
      "id": "trajectory",
      "script": "ethoinsight.scripts.epm.plot_trajectory",
      "input": "/mnt/user-data/uploads/raw.txt",
      "output": "/mnt/user-data/workspace/plot_trajectory.png"
    }
  ],
  "skipped": [
    {
      "id": "distance_moved",
      "reason": "user.exclude",
      "detail": "User explicitly excluded distance_moved."
    }
  ],
  "notes": [
    "用户定制：+center_distance, -distance_moved",
    "数据列 27 个，3 个 optional 指标因缺列跳过"
  ]
}
```

字段语义详见 §3.4 + brainstorming 记录。关键约定：

- `metrics[].script` 是 dotted 路径字符串，**code-executor 自行 format 成 bash 命令**，catalog / plan 都不嵌完整 shell 字符串（避免转义噩梦）
- `metrics[].input` / `output` 是 per-entry 绝对路径，resolve 函数预先计算（shoaling 多文件时 input 指向 wrapper JSON）
- `metrics[].required = true` 失败时 code-executor 必须停下来报 lead；`required = false` 失败时记 warning 继续
- `metrics[].reason` enum：`{paradigm.default, paradigm.required, user.include, paradigm.optional.applicable}`
- `skipped[].reason` enum：`{user.exclude, columns.missing, paradigm.not_applicable, catalog.unknown}`

---

## 5. 端到端流程（A1）

```
Turn 1 (lead) — 收用户上传 + 中文请求
  ├─ skill: ethoinsight-planning（已有）
  ├─ skill: ethovision-paradigm-knowledge（识别 ev19_template + paradigm）
  └─ 如 paradigm/定制需求不清 → ask_clarification → 退出 turn

Turn 2 (lead) — paradigm 确定 + 用户定制需求收口
  ├─ skill: ethoinsight-metric-catalog（新；~300 tokens SKILL.md）
  ├─ bash: python -m ethoinsight.parse.dump_headers \
  │          --input /mnt/user-data/uploads/raw.txt \
  │          --output /mnt/user-data/workspace/columns.json
  │     失败 → 列名解析问题 → ask_clarification
  ├─ bash: python -m ethoinsight.catalog.resolve \
  │          --paradigm epm \
  │          --columns-file /mnt/user-data/workspace/columns.json \
  │          --include center_distance \
  │          --exclude distance_moved \
  │          --raw-files-json /mnt/user-data/workspace/raw_files.json \
  │          --groups-file /mnt/user-data/workspace/groups.json \
  │          --output /mnt/user-data/workspace/metric_plan.json
  │     失败 → 解析 stderr JSON code → ask_clarification
  └─ task(subagent="code-executor",
          prompt="按 metric_plan.json 执行。plan 路径: /mnt/user-data/workspace/metric_plan.json")

Turn 3 (code-executor)
  ├─ read_file /mnt/user-data/workspace/metric_plan.json
  ├─ for entry in plan.metrics:
  │    bash python -m <entry.script> --input <entry.input> --output <entry.output>
  ├─ if plan.statistics.skip_reason is null:
  │    bash python -m <plan.statistics.script> ...
  ├─ for chart in plan.charts:
  │    bash python -m <chart.script> ...
  ├─ 聚合所有 metrics[].output JSON → handoff_code_executor.json
  └─ 输出 [gate_signals] 块

Turn 4+ (lead → data-analyst → report-writer): 不变
  ├─ data-analyst: read handoff + 按 metric id cat catalog YAML 取 direction_for_anxiety / statistical_default
  └─ report-writer: read handoff + 按 metric id cat catalog YAML 取 display_name_zh / unit_zh / one_liner
```

### 5.1 反问链路示例（同事 Q2 "列名歧义不要猜要问"）

**情景**：OFT 数据只有单列 `in_zone`（无后缀），catalog 要求 `requires_columns: [in_zone_center_*]`。

| 步骤 | 发生位置 | 内容 |
|------|----------|------|
| 1 | lead bash | `catalog.resolve --paradigm oft ...` |
| 2 | resolve 函数 | 检测 columns.json 中无任何匹配 `in_zone_center_*` 的列，但有裸 `in_zone` |
| 3 | resolve exit 1 | stderr: `{"code": "columns.ambiguous", "details": {"metric": "center_time_ratio", "found": "in_zone", "expected_pattern": "in_zone_center_*", "hint": "该列可能是 center 区也可能是 periphery 区，需用户确认"}}` |
| 4 | lead | 解析 stderr → ask_clarification 中文反问 |
| 5 | 用户回答 | "in_zone 就是 center 区" |
| 6 | lead | bash resolve 加 `--column-alias in_zone=in_zone_center` 参数重试 |

反问链路 = **1 跳**（resolve → lead → 用户）。对比候选 B 的 3 跳（resolve → code-executor → lead → 用户）。

---

## 6. skill 切分

### 6.1 `ethoinsight-code`（瘦身后 — 工程执行手册）

**保留**：
- SKILL.md：工作流改写为 "read plan.json → loop bash → 聚合 handoff → 输出 gate_signals"
- `references/error-recovery.md`：脚本错误处理（范式无关），删第 46-51 行 `assess_and_handoff` 旧引用
- `references/quality-checks.md`：handoff 自检清单，删第 44 行 `assess_and_handoff` 旧引用
- `templates/output-contract.md`：handoff JSON + gate_signals 格式约定

**删除**：
- `references/by-paradigm/<paradigm>.md` × 7（范式知识全部沉到 catalog YAML）

挂载：仅 `code-executor`。

### 6.2 `ethoinsight-metric-catalog`（新建 — 领域知识接口）

**结构**：

```
packages/agent/skills/custom/ethoinsight-metric-catalog/
├── SKILL.md            # 主入口，按 role 分段（~50 行）
└── references/
    ├── resolve-cli.md  # lead 专用：resolve CLI 参数 + 错误码
    └── field-guide.md  # 各 role 关注的字段映射表
```

**SKILL.md 框架**：

```markdown
---
name: ethoinsight-metric-catalog
description: EthoInsight 范式指标 catalog 读取手册。lead 用 resolve 生成 plan；data-analyst 取 direction_for_anxiety/statistical_default；report-writer 取 display_name_zh/unit_zh/one_liner。
---

# 范式指标 catalog 入口

catalog 物理位置：`packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml`
（可由 `python -c "import ethoinsight.catalog as c; print(c.__file__)"` 定位）

## 你是哪种 role？

### lead
触发 resolve 流程时按以下步骤：
1. bash: python -m ethoinsight.parse.dump_headers --input <raw> --output <columns.json>
2. bash: python -m ethoinsight.catalog.resolve [参数见 references/resolve-cli.md]
3. resolve 报错时按 stderr JSON 的 code 字段 ask_clarification

不要直接读 catalog YAML —— resolve 函数会替你做。

### data-analyst
拿到 handoff_code_executor.json 后，按 metric id 读 catalog 字段：
- `cat .../catalog/<paradigm>.yaml | grep -A 8 "id: <metric_id>"` 取
  - `direction_for_anxiety`: 判断"差异方向是否符合实验假设"
  - `statistical_default`: 验证 code-executor 用了正确的统计入口

### report-writer
写 "Results / Discussion" 段时，按 metric id 取展示字段：
  - `display_name_zh`: 中文展示名
  - `unit_zh`: 单位
  - `one_liner`: 一句话解释（仅首次提及该指标时引用）
```

**挂载**：`lead`（通过 lead_agent system prompt 的 skill description 段）+ `data-analyst` + `report-writer`。

**不挂 `code-executor`**：它只读 plan.json，挂着也不会读 catalog，反而给 LLM 多一个偷懒空间。

### 6.3 SubagentConfig 变更

```python
# code_executor.py
skills=["ethoinsight-code", "ethoinsight-charts"],   # ethoinsight-code 已瘦身

# data_analyst.py
skills=["ethoinsight", "ethoinsight-metric-catalog"],   # 新增 catalog

# report_writer.py
skills=["ethoinsight", "ethoinsight-metric-catalog"],   # 新增 catalog

# lead 端：在 lead_agent/prompt.py 的 skill description 段加 ethoinsight-metric-catalog 入口描述
```

---

## 7. 新增 Python 模块

### 7.1 ethoinsight.parse.dump_headers（CLI）

`packages/ethoinsight/ethoinsight/parse/__init__.py` 已有；新增 `dump_headers.py`（或作为 `parse.py` 的 `__main__` 子命令）：

```
python -m ethoinsight.parse.dump_headers \
    --input <raw_file.txt> \
    --output <columns.json>
```

行为：
1. read EthoVision header（复用 `parse.parse_header`）
2. 取列名清单
3. 写 `{"file": ..., "columns": ["col1", ...], "n_subjects": N, "duration_s": ...}` 到 output JSON

错误码（exit 1 + stderr JSON）：
- `file.not_found`
- `header.parse_failed`
- `format.unrecognized`（非 EthoVision 导出）

### 7.2 ethoinsight.catalog 模块

```
packages/ethoinsight/ethoinsight/catalog/
├── __init__.py     # public: load_catalog, resolve, ResolveError
├── loader.py       # YAML → dict + schema 校验
├── schema.py       # MetricEntry / CatalogEntry / Plan dataclass
├── resolve.py      # 主逻辑
├── cli.py          # __main__ 入口
└── <paradigm>.yaml × 7
```

### 7.3 单测覆盖

新增 `packages/ethoinsight/tests/test_catalog.py`：

- `test_all_yamls_load_without_error` — 7 个范式 YAML 全部能 load
- `test_all_scripts_importable` — catalog 中每个 `script` dotted path 都能 import 到一个有 `main` 入口的模块
- `test_unique_metric_ids_per_paradigm` — 同范式内 id 唯一
- `test_enum_values_valid` — `direction_for_anxiety` / `statistical_default` 在允许列表
- `test_resolve_default_path` — paradigm=epm 默认调用产 5 个 metric 的 plan
- `test_resolve_with_user_include_outside_catalog` — exit 1 + code=`unknown_metric`
- `test_resolve_with_missing_required_columns` — 必需列缺失 → `columns.missing`
- `test_resolve_empty_plan` — 所有 default 都被剪 → `empty_plan`
- `test_plan_json_schema_valid` — 输出符合 plan_schema 定义
- `test_q6_whitelist_alignment` — EPM/OFT/FST 的 default_metrics 完全等于同事 5-13 Q6 白名单（这是反退化测试，未来谁改 catalog 都必须显式动这个测试）

---

## 8. 现状清扫范围

### 8.1 ethoinsight 库

| 文件 | 改动 |
|------|------|
| `assess.py` | 删 `_DEFAULT_THRESHOLDS`、`_load_thresholds`、`_assess_thresholds`、`assess_results` 中阈值分支；改 module docstring 说明"判读基于组间统计比较"；删 epm/open_field 范式相关 recommendation；保留 `_infer_phenotype`（基于显著差异的方向性表型推断）和组间比较主流程 |
| `metrics/oft.py` | 删 `_find_center_zone_column` 裸 `in_zone` silent fallback；改为抛 `AmbiguousZoneError`，在 dispatcher 层冒泡到 resolve 函数（resolve 通过 `requires_columns` 模式匹配预先检测） |
| `parse.py` | 重写自动分组推断逻辑：从 raw 最后一列读 "result block 命名"（不是 user-defined variable 列） |
| `tests/test_statistics.py` | 现有 `TestAssessResults` 3 个测试不动（都走 shoaling 路径不触阈值分支） |
| `metrics/epm/__init__.py` 等 | 命名 / 函数签名以同事 Q6 白名单对齐（EPM 5 个、OFT 加 `compute_center_time`+`compute_center_distance`、FST 命名 `_time/_latency/_count` 收口） |
| `scripts/epm/` etc | 加缺失的脚本（OFT `compute_center_time` / `compute_center_distance`）；脚本接口跟 catalog YAML 的 `script` 字段一对一 |

### 8.2 agent skill

| 路径 | 改动 |
|------|------|
| `skills/custom/ethoinsight-code/SKILL.md` | 工作流段改写（read plan → loop bash → 聚合） |
| `skills/custom/ethoinsight-code/references/by-paradigm/*.md` × 7 | **删除** |
| `skills/custom/ethoinsight-code/references/error-recovery.md` | 删第 46-51 行（`assess_and_handoff` 旧引用） |
| `skills/custom/ethoinsight-code/references/quality-checks.md` | 删第 44 行（同上） |
| `skills/custom/ethoinsight-metric-catalog/` | **新建** —— SKILL.md + references/resolve-cli.md + references/field-guide.md |

### 8.3 deerflow agent 端

| 路径 | 改动 |
|------|------|
| `subagents/builtins/code_executor.py` | system prompt workflow 段改写（read plan.json）；skills 列表保持 `["ethoinsight-code", "ethoinsight-charts"]` |
| `subagents/builtins/data_analyst.py` | skills 加 `"ethoinsight-metric-catalog"`；system prompt 加一段"判读差异方向时查 catalog" |
| `subagents/builtins/report_writer.py` | skills 加 `"ethoinsight-metric-catalog"`；system prompt 加一段"翻译指标展示元数据时查 catalog" |
| `agents/lead_agent/prompt.py` | skill description 段加 `ethoinsight-metric-catalog`；Gate 2 段加"派遣 code-executor 前调 dump_headers + resolve"的工作流提示 |

### 8.4 文档

| 路径 | 改动 |
|------|------|
| `docs/handoffs/2026-05/2026-05-13-feedback-corrections-handoff.md` | 新建（执行完本 spec 后写） |
| `CLAUDE.md` §第 7 条 | 更新流水线描述（code-executor 不再"依次调用 5 个细粒度 tool"，改为"按 lead 生成的 metric_plan.json 逐条执行脚本"） |

---

## 9. 不影响范围

明确不动的部分（避免 scope creep）：

| 模块 | 不动原因 |
|------|----------|
| deerflow harness（agents / middlewares / sandbox / tools） | 5-12 spec 设计原则："agent 架构由上游 deerflow 操心" |
| `ScriptInvocationOnlyProvider` Guardrail | catalog / parse 不在 code-executor 的白名单内（lead 跑），零改动 |
| `HandoffIsolationProvider` Guardrail | 跟本 spec 正交 |
| code-executor 工具集（bash + read_file + write_file + ls + str_replace） | 不变 |
| `ethoinsight.statistics` 决策树 | Shapiro-Wilk → t / Mann-Whitney 是学界标准，留在代码硬编码，catalog 只用 `statistical_default` enum 指针 |
| `ev19_facts.py` + `ethovision-paradigm-knowledge` skill | EV19 模板识别地基（5-08 spec），跟 catalog 是上下游关系，互不重复 |
| 前端 reasoning 渲染 | 5-12 spec 已处理 |
| 训练数据飞轮 | TrainingDataMiddleware 自动收集，不需要改 |

---

## 10. 已知风险与未决问题

### 10.1 风险

| 风险 | 缓解 |
|------|------|
| catalog YAML 字段在未来扩展时 breaking change | schema_version 字段 + loader 兼容 fallback |
| Qwen3-8B 在 lead 派遣 prompt 引用 plan.json 路径时拼错绝对路径 | metric-catalog SKILL.md 提供精确派遣 prompt 模板；lead 派遣 prompt 字符极短（只含 plan.json 路径），出错面有限 |
| resolve 函数错误码与 lead 反问指令不匹配 | resolve 输出的 stderr JSON 是结构化 enum；lead 通过 metric-catalog skill 的 `resolve-cli.md` 学到 enum → 反问话术的映射 |
| 同事改 catalog 后忘记加 / 删脚本 | CI `test_all_scripts_importable` 测试在每个 PR 必跑 |
| catalog 跟 ev19_facts.py 的 paradigm 名不一致（如 zero_maze vs o_maze） | loader 启动校验：catalog `paradigm` 字段必须在 `ev19_facts.SUPPORTED_PARADIGMS` 列表 |
| dump_headers 失败模式（如非 EthoVision 文件、损坏 header） | dump_headers exit 1 + 结构化错误；lead 看到 → ask_clarification "这个文件看起来不像 EthoVision 导出，能再确认一下吗？" |

### 10.2 未决问题（不阻塞本 spec，留给 writing-plans 决定）

- **shoaling 多文件场景下 resolve 怎么生成 wrapper JSON 路径** —— 沿用 5-12 spec §3.2 的 `--inputs <wrapper.json>` 约定，resolve 函数预先写入 `inputs.json` 数组文件
- **catalog YAML 的 i18n** —— 当前只有 `display_name_zh` / `unit_zh`。如果未来支持英文 user，需要并列 `display_name_en` 等字段；本期不做
- **catalog 的 hot reload** —— 改 YAML 后是否需要 agent 重启？v0.1 阶段简单粗暴：改 YAML = 重启 agent 服务；v0.2 可加 mtime 监听
- **catalog 之外的"高级用户自定义指标"** —— 用户上传自己的脚本？v0.1 明确否决（违反 ScriptInvocationOnlyProvider 白名单原则）

---

## 11. 与前置 spec 的关系

| spec | 关系 |
|------|------|
| [2026-04-29 EV19 模板范式重定位](../../plans/2026-04-29-ev19-template-paradigm-design.md) | 上游（识别）→ 本 spec（执行编排）。catalog `ev19_templates` 字段引用 EV19 模板名 |
| [2026-05-08 EV19 模板 skill 地基](2026-05-08-ev19-template-skill-foundation-design.md) | 上游。`ethovision-paradigm-knowledge` skill 提供识别，本 spec 提供执行 |
| [2026-05-11 files-are-facts](2026-05-11-subagent-file-is-facts-design.md) | 协同。metric_plan.json 是 lead → code-executor 的 facts file |
| [2026-05-12 Script-Per-Metric](2026-05-12-script-per-metric-architecture-design.md) | **直接前身**。本 spec 是它的延伸：把 §3.4 的 markdown skill md 升级为 YAML catalog + Python resolve |

本 spec **不替代** 5-12 spec，而是它的 evolution。Guardrail / 脚本接口约定 / handoff schema 都继承自 5-12 spec。

---

## 12. 实施次序（writing-plans 时细化）

粗粒度阶段：

1. **catalog 库内骨架**：`catalog/` 模块 + EPM/OFT/FST 三份 YAML（同事 5-13 反馈的范式优先）+ `resolve.py` + `cli.py` + 单测全绿
2. **dump_headers CLI**：极小动作，先打通 lead → resolve 的数据流
3. **resolve CLI 错误码闭环**：把 6 种错误码全跑通 + 对应测试
4. **agent 端 skill 切分**：建 `ethoinsight-metric-catalog` skill + 删 `by-paradigm/*.md` + 改 3 个 subagent config
5. **lead_agent prompt 改造**：Gate 2 段加 dump_headers + resolve 工作流提示
6. **code-executor prompt 改造**：workflow 段改为 read plan.json
7. **data-analyst / report-writer prompt 微调**：加"查 catalog 字段"指引
8. **现状清扫**：assess.py 死代码 + parse.py 分组推断重写 + oft.py silent fallback 退役
9. **剩余 4 个范式 YAML**（TST / LDB / Zero-maze / Shoaling）
10. **e2e**：用 5-12 真数据跑 EPM/OFT/FST 三套流程，确认 catalog → resolve → plan.json → code-executor → handoff → data-analyst → report-writer 全链路通

---

## 13. 关键参考

- [CLAUDE.md](../../../CLAUDE.md) §9（组间比较不用绝对阈值）、§10（Memory event-loop fix）、§12（复用 deerflow 现成功能）
- [2026-05-12 同事反馈](../../review-packages/2026-05-12-feedback.md) — 本 spec 的直接触发
- [2026-05-12 Script-Per-Metric 架构](2026-05-12-script-per-metric-architecture-design.md) — 前身
- [2026-05-11 files-are-facts](2026-05-11-subagent-file-is-facts-design.md) — metric_plan.json 的机制基础
- [packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py](../../../packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py) — Guardrail 现状（确认 lead pass through）

---

## Addendum 2026-05-27 — subagent 消费路径修正

原 spec 设想 catalog YAML "被 lead / data-analyst / report-writer 多方共读"。这条
契约在运行时不成立 — deerflow sandbox 的 read_file 白名单不放行 Python 包内路径
(`/app/backend/packages/ethoinsight/ethoinsight/catalog/*.yaml` 会被
PermissionError 拒)。

修正后契约:catalog YAML **仅** 由 lead 通过 `prep_metric_plan` 工具在沙箱外消费,
所有 subagent 需要的字段(`unit_zh` / `one_liner` / `output_unit` /
`direction_for_anxiety` / `statistical_default` 等)由 resolve.py 透传到
plan_metrics.json,subagent 走 `/mnt/user-data/workspace/plan_metrics.json`
(白名单允许)一次读出。

详细设计:[2026-05-27-catalog-fields-into-plan-design.md](2026-05-27-catalog-fields-into-plan-design.md)

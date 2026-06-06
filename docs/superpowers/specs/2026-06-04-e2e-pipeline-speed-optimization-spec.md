# Spec: E2E 流水线加速 — Prompt/Skill 层优化

> 基于 2026-06-04 OFT 旷场 E2E dogfood 瓶颈分析（总耗时 ~21 min, ~80 LLM turns）

## TL;DR

四项零代码/轻量优化，不改架构、不动后端 harness，只改 prompt 和 SKILL.md 文本。预期将 E2E 从 21 min 降到 10-12 min。

| # | 优化 | 省时 | 类型 | 目标 |
|---|------|------|------|------|
| 1 | n=1 快速路径 | -3 min | lead prompt | n_per_group < 2 时跳过 statistics + data-analyst |
| 2 | 反问合并 | -2 min | lead prompt | 一次性收集全部缺失信息，不逐轮反问 |
| 3 | 并行 bash | -1.5 min | code-executor SKILL.md | 独立脚本用 `& wait` 一个 bash 调用跑完 |
| 4 | batch 读文件 | -1.5 min | 所有 subagent SKILL.md | `bash cat f1 f2 f3` 替代逐个 `read_file` |

---

## 优化 1: n=1 快速路径（省 ~3 min）

### 问题

当每组只有 n=1 时：
- statistics.py 会设置 `skip_reason = "n_per_group >= 2 and n_groups >= 2 条件不满足"`
- code-executor 仍然跑 10 个 bash 脚本（指标计算本身不受影响）
- data-analyst 仍然被派遣，读取多个文件后得出结论"n=1 无法做统计检验"
- 用户等待 4 分钟后看到的结论是"样本太小，仅供参考"

**data-analyst 的大部分工作（判读效应量、排查混杂、推荐检验方法）在 n=1 时没有价值**——没有统计检验就谈不上效应量判断、方法选择。

### 方案

在 lead prompt 的 E2E 流水线段增加 n=1 快速路径判定：

```
规则: 派遣 code-executor 前，检查 plan_metrics.json 中的 subject_count 信息。
如果任一组 n < 2（即无法做组间统计检验）:
  1. 正常派遣 code-executor（指标计算仍需要）
  2. code-executor 完成后，**跳过 data-analyst**
  3. 直接用 lead 自己写一份简短的描述性摘要（基于 handoff_code_executor.json 的 metrics_summary）
  4. 直接反问用户：是否需要图表 + 报告（E2E_FULL_ASKVIZ 路径）
  5. **不要派遣 data-analyst**，在 lead 消息中明确告知："由于每组仅 n=1，无法进行统计检验，已跳过专业判读环节。以下为描述性对比："
```

### 实施位置

- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 流水线段（~line 999）

### 判定条件

从 `prep_metric_plan` 的返回值或 `plan_metrics.json` 的 `groups` 字段获取每组的 subject 数量。如果 `min(n_per_group) < 2` → 触发快速路径。

### 注意边界

- n=1 时仍然要跑 code-executor（指标计算有价值，用户可以看到描述性对比）
- chart-maker 和 report-writer 仍然可以派遣（图表和报告在 n=1 时仍有用）
- 只是跳过 data-analyst（专业判读在 n=1 时没有统计基础）

---

## 优化 2: 反问合并（省 ~2 min）

### 问题

当前 E2E 的 3 轮反问是串行的：
```
Lead: identify → 发现候选模板有2个 → ask_clarification "选 A 还是 B?"
用户: 选 A
Lead: set_paradigm → prep_metric_plan → 发现 zone 未命名 → ask_clarification "in_zone=1 是中心区吗?"
用户: 是
Lead: 发现分组信息缺失 → ask_clarification "Trial 1 和 Trial 2 各是什么组?"
用户: control / treatment
```

每轮 = 2-4 个 LLM turns。3 轮 = ~9-12 turns ≈ 2-3 分钟。

### 方案

在 lead prompt 中增加指令：**在派遣 code-executor 之前，一次性扫描所有缺失信息，构造一个包含全部问题的 ask_clarification**。

```
规则: 在 identify_ev19_template → inspect_uploaded_file → set_experiment_paradigm → prep_metric_plan
这条链上，不要每发现一个缺失信息就问一个。累积所有发现后，构造一个 ask_clarification:
- 如果模板有歧义（候选>1）: 列入问题
- 如果 zone 未命名: 列入问题
- 如果分组信息缺失: 列入问题
- 把多个问题合并为一个 ask_clarification 的 options 列表或自由文本区
```

### 实施位置

- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — Gate 1 段 + 流水线段

### 示例

```
⚠️ 在开始分析前，需要确认以下信息：

1. EV19 模板: A. OpenFieldRectangle / B. OpenFieldCircle（推荐 A）
2. 分组: Trial 1 (dark_mice) 和 Trial 2 (white_mice) 各属于什么组?
3. 区域: in_zone=1 是中心区吗?

请一次性回复，例如: "A, Trial 1=control, Trial 2=treatment, in_zone=1=中心区"
```

---

## 优化 3: 并行 bash（省 ~1.5 min）

### 问题

code-executor 逐个执行 10 个 bash 脚本：
```
bash compute_center_distance --input s1 --output s1.json   → 1 turn
bash compute_center_distance --input s2 --output s2.json   → 1 turn
bash compute_distance_ratio --input s1 --output s1.json    → 1 turn
... (共 10 个独立调用)
```

每个 bash 调用 = 1 个 LLM turn。但这 10 个脚本之间完全独立（不同的 subject × 不同的指标），可以并行。

### 方案

修改 `ethoinsight-code/SKILL.md`，在工作流步骤中指示：**如果多个脚本处理不同 subject 且彼此独立，用一个 bash 调用并行执行**。

```markdown
## 并行执行规则

当多个脚本满足以下全部条件时，用一个 bash 调用并行跑：
1. 脚本之间互不依赖（不同 subject / 不同指标）
2. 输出文件不同
3. 输入文件不同

示例（OFT 旷场，2 subjects × 5 指标 = 10 个独立脚本）：
```bash
bash -c "
python -m ethoinsight.scripts.oft.compute_center_distance --input s1.xlsx --output cdist_s1.json --parameters-json '{\"center_zone\":\"in_zone\"}' &
python -m ethoinsight.scripts.oft.compute_center_distance --input s2.xlsx --output cdist_s2.json --parameters-json '{\"center_zone\":\"in_zone\"}' &
python -m ethoinsight.scripts.oft.compute_distance_ratio --input s1.xlsx --output dratio_s1.json --parameters-json '{\"center_zone\":\"in_zone\"}' &
...
wait
"
```

**效果**: 10 个 turns → 1 个 turn。省 ~9 turns ≈ 1-1.5 分钟。

### 实施位置

- `packages/agent/skills/custom/ethoinsight-code/SKILL.md` — 工作流段

### 注意

- 仅在脚本间无依赖时并行
- `&` 后台运行 + `wait` 等待全部完成
- 如果有脚本失败，`wait` 后检查退出码，逐个排查

---

## 优化 4: Batch 读文件（省 ~1.5 min）

### 问题

data-analyst 和 report-writer 逐个读取文件：
```
read_file handoff_code_executor.json    → 1 turn
read_file plan_metrics.json             → 1 turn
read_file OFT范式文档.md                 → 1 turn
read_file output_1.json                 → 1 turn
read_file output_2.json                 → 1 turn
...
```

每个 `read_file` = 1 个 LLM turn。但这些文件内容互不依赖，可以一次性读取。

### 方案

修改 data-analyst 和 report-writer 的 SKILL.md，在工作流开头指示：**用 `bash cat` 把需要同时看的文件一次性输出到一个临时文件，然后 `read_file` 那一个临时文件**。

```markdown
## Batch read 规则

在工作流开始时，如果你需要读多个文件（这些文件之间互不依赖），一次性 cat 它们：

```bash
bash cat /mnt/user-data/workspace/handoff_code_executor.json \
         /mnt/user-data/workspace/plan_metrics.json \
         /mnt/user-data/workspace/handoff_planning.json \
         > /tmp/context_bundle.txt
```

然后 `read_file /tmp/context_bundle.txt` 一次拿到全部上下文。
```

### 实施位置

- `packages/agent/skills/custom/ethoinsight-data/SKILL.md`
- `packages/agent/skills/custom/ethoinsight-charts/SKILL.md`
- `packages/agent/skills/custom/ethoinsight-code/SKILL.md`（如有类似模式）
- report-writer 的 SKILL.md

### 注意

- 不要 batch 读超大文件（如原始轨迹 txt，5MB+）— 这些保持单独 `read_file`
- 只 batch 读 JSON 配置文件、handoff JSON、范式文档等小文件
- 如果文件数 ≤ 2，直接读即可，batch 优势不大

---

## 改动文件汇总

| 文件 | 优化 # | 改动类型 |
|------|--------|---------|
| `agents/lead_agent/prompt.py` | #1 n=1 快速路径 | 新增流水线分支规则 |
| `agents/lead_agent/prompt.py` | #2 反问合并 | 修改 Gate 1 反问策略 |
| `skills/custom/ethoinsight-code/SKILL.md` | #3 并行 bash | 新增并行执行规则 |
| `skills/custom/ethoinsight-code/SKILL.md` | #4 batch 读文件 | 新增 batch read 规则 |
| `skills/custom/ethoinsight-data/SKILL.md` | #4 batch 读文件 | 新增 batch read 规则 |
| `skills/custom/ethoinsight-charts/SKILL.md` | #4 batch 读文件 | 新增 batch read 规则 |
| report-writer SKILL.md | #4 batch 读文件 | 新增 batch read 规则 |

## 验证方案

| 优化 | 验证方法 |
|------|---------|
| #1 n=1 快速路径 | 用 OFT 旷场数据（n=1）跑 E2E，确认 data-analyst 未被派遣，lead 直接出描述性摘要 |
| #2 反问合并 | 发 "分析数据"，确认只有 1 轮反问（含全部缺失信息），而非 3 轮 |
| #3 并行 bash | 检查 code-executor 的 bash 调用是否包含 `& wait`，10 个脚本是否在同一个 bash 中 |
| #4 batch 读文件 | 检查 subagent 是否用 `bash cat` 一次性读多个文件 |

## 预期效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| OFT n=1（本次 dogfood） | ~21 min | ~13 min |
| +subagent flash 模型 | ~17 min | ~9-10 min |
| 正常 n≥5 E2E | ~21 min | ~14-15 min |

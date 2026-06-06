# E2E 耗时瓶颈分析与加速方案

> 基于 2026-06-04 OFT 旷场 E2E dogfood 实测数据（deepseek-v4-pro, 4文件, 21分钟）

## 1. 时间分配

```
总耗时: ~21 min, ~80 LLM turns
每个 turn ≈ 10-15 秒（deepseek-v4-pro 推理 8-12s + tool 执行 1-3s）

┌─────────────────────────────────────────┬────────┬──────────────────┐
│ 阶段                                    │ 耗时    │ LLM turns        │
├─────────────────────────────────────────┼────────┼──────────────────┤
│ Lead: identify + inspect + 规划          │ ~2 min │ 5-8              │
│ 反问 ×3 (模板/分组/zone)                 │ ~3 min │ 9-12 (含我的回复) │
│ code-executor (max_turns=40)            │ ~3 min │ ~15-20           │
│ data-analyst (max_turns=12)             │ ~4 min │ ~10              │
│ Lead: 汇总 + 反问是否出图                 │ ~2 min │ 3-5              │
│ chart-maker (max_turns=15)             │ ~2 min │ ~8               │
│ Lead: 展示 + 反问是否出报告              │ ~2 min │ 3-5              │
│ report-writer (max_turns=15)           │ ~3 min │ ~10              │
├─────────────────────────────────────────┼────────┼──────────────────┤
│ 合计                                    │ ~21 min│ ~80              │
└─────────────────────────────────────────┴────────┴──────────────────┘
```

## 2. 瓶颈分析（按影响排序）

### 瓶颈 #1: LLM 推理延迟（占 60%+）

每个 turn = deepseek-v4-pro 推理 + tool 执行。推理本身占 8-12 秒。

- **根因**: 当前所有 agent (lead + 4 个 subagent) 都使用 `deepseek-v4-pro`（thinking enabled）
- **影响**: 80 turns × 10s 推理 = ~13 分钟纯推理时间
- **当前 subagent max_turns 配置**:
  - code-executor: 40 (timeout 900s) — 过度分配，实际只用 ~15
  - data-analyst: 12 (timeout 600s)
  - chart-maker: 15 (timeout 600s)
  - report-writer: 15 (timeout 600s)

### 瓶颈 #2: 串行 Subagent 链

当前流程: `code-executor → data-analyst → chart-maker → report-writer` 严格串行。

**但实际上**:
- `chart-maker` 只需要 `plan_charts.json`（lead 在派遣 code-executor 前已生成），不依赖 code-executor 的输出
- `code-executor` 和 `chart-maker` **可以并行派遣**

### 瓶颈 #3: 反问轮次过多（3 轮 → 可合为 1 轮）

本次 E2E 有 3 轮反问:
1. 选 OFT 模板 (A/B)
2. 分组确认 (control/treatment)
3. Zone 确认 (in_zone=1 是中心区)

每轮反问 = 2-4 个 LLM turns。3 轮 → 9-12 turns ≈ 2-3 分钟。

**根因**: lead 的 prompt 没有指示"一次性收集所有缺失信息"。每次只问一个问题，发现新的缺失信息再问下一个。

### 瓶颈 #4: Subagent 逐文件读取

data-analyst 的实际工作:
```
read_file handoff_code_executor.json     → 1 turn
read_file plan_metrics.json              → 1 turn
read_file OFT范式文档.md                  → 1 turn
read_file 输出JSON × N                   → 3-5 turns
reasoning + write handoff                → 3-4 turns
```

**问题**: 每个 `read_file` 调用 = 1 个 LLM turn。如果改为一个 `bash cat file1 file2 file3`，3-5 个 turns 合并为 1 个。

### 瓶颈 #5: Code-executor 逐脚本执行

code-executor 的 10 个 bash 调用逐个执行。但这些脚本之间相互独立（不同 subject 的不同指标），完全可以并行。

当前: `bash python -m ethoinsight.scripts.oft.compute_center_distance --input ... --output ...` × 10 次 = 10 turns

改进: 一个 bash 调用中 `& wait` 并行跑 10 个脚本 = 1 turn。

### 瓶颈 #6: Subagent 不必要的 thinking

data-analyst 和 report-writer 的 `supports_thinking: true` 让它们在每个 turn 都产生冗长的 thinking block。对于"读文件 → 提取字段"这种机械操作，thinking 没有价值却显著增加推理时间。

## 3. 加速方案（按投入产出比排序）

### Tier 1: 零代码改动（Skill/Prompt 优化）

| 优化 | 预期节省 | 实施方式 |
|------|---------|---------|
| **反问合并** | -2 min | 修改 lead_agent/prompt.py 的 Gate 1 段，指示"一次性列出所有缺失信息" |
| **code-executor 并行 bash** | -1.5 min | 修改 ethoinsight-code/SKILL.md，指示用 `& wait` 批量执行独立脚本 |
| **subagent batch 读文件** | -2 min | 修改各 subagent SKILL.md，指示用 `bash cat f1 f2 f3` 替代逐个 read_file |
| **subagent 禁 thinking** | -3 min | 对 code-executor/data-analyst/chart-maker/report-writer 设置 `supports_thinking: false`（纯工具调用不需要 deep thinking） |

**小计: ~8.5 min 节省，目标 E2E < 12 min**

### Tier 2: 轻量代码改动（1-2 天）

| 优化 | 预期节省 | 实施方式 |
|------|---------|---------|
| **并行派遣 code-executor + chart-maker** | -2 min | 修改 lead prompt，在 pre-dispatch 阶段同时调 `task(code-executor)` + `task(chart-maker)` |
| **Subagent 使用更快模型** | -3 min | 在 config.yaml 中为 subagent 配置 `deepseek-v4-flash`（无 thinking，推理更快） |

**小计: +5 min 节省，目标 E2E < 7 min**

### Tier 3: 架构改动（1-2 周）

| 优化 | 预期节省 | 实施方式 |
|------|---------|---------|
| **减少 subagent 链长度** | -2 min | 合并 data-analyst + report-writer 为一个 subagent（减少 handoff 序列化 + 重复读文件） |
| **"快速路径"跳过统计** | -3 min | 当 n=1 时，跳过 data-analyst（只做描述性汇总），直接到 report |
| **Pre-fetch 共享文件** | -1 min | lead 在派遣 subagent 前把 handoff JSON、范式文档等公共文件写入 workspace，subagent 直接读写而不反复 read_file |

**小计: +6 min 节省，目标 E2E < 2 min（此目标需要 Tier 1+2+3 全上）**

## 4. 立即可执行的最小改动集

### 4.1 反问合并（改 prompt）

`lead_agent/prompt.py` Gate 1 段，当前逻辑:
```
发现缺失信息 A → ask_clarification
用户回答 → 发现缺失信息 B → ask_clarification
用户回答 → 发现缺失信息 C → ask_clarification
```

改为:
```
一次性扫描所有缺失信息 → 构造一个包含所有问题的 ask_clarification
```

### 4.2 子代理 batch read（改 SKILL.md）

`ethoinsight-data/SKILL.md` 和 `ethoinsight-code/SKILL.md`:

将:
```
step 1: read_file handoff_code_executor.json
step 2: read_file plan_metrics.json
step 3: read_file OFT文档.md
```

改为:
```
step 1: bash cat handoff_code_executor.json plan_metrics.json OFT文档.md
        → 输出到 /tmp/context.txt
step 2: read_file /tmp/context.txt (需要时)
```

### 4.3 Code-executor 并行脚本（改 SKILL.md + lead prompt）

`ethoinsight-code/SKILL.md`:
```
for each metric entry:
  如果脚本之间相互独立（不同 subject），用一个 bash 调用并行执行:
  bash -c "
    python -m ethoinsight.scripts.oft.compute_center_distance --input s1 --output s1.json &
    python -m ethoinsight.scripts.oft.compute_center_distance --input s2 --output s2.json &
    ...
    wait
  "
```

### 4.4 Subagent 禁用 thinking（改 config.yaml 或 subagent config）

```yaml
# 在 subagent 的 when_thinking_enabled 中设为 false
# 或直接给 subagent 用不支持 thinking 的模型
```

## 5. 实验验证建议

建议按以下顺序验证，每步测一次 E2E 耗时:

1. **基线**: 当前 21 min
2. **+反问合并**: 目标 < 19 min
3. **+batch read + 并行 bash**: 目标 < 16 min
4. **+禁用 subagent thinking**: 目标 < 13 min
5. **+subagent 用 flash 模型**: 目标 < 10 min
6. **+并行派遣**: 目标 < 8 min

# PR-2 最终 spec —— chart-maker workflow 收敛 + skill 渐进披露重塑

> **生成于**：2026-05-22
> **依赖**：PR-1（catalog schema 升级）合入后实施
> **范围**：chart_maker.py prompt + 两个 skill 文档 + skill 注册
> **来源**：[2026-05-22-chart-maker-grill-corrected-handoff.md](../handoffs/2026-05/2026-05-22-chart-maker-grill-corrected-handoff.md) §PR-2 + grill 真因 §1（α + β）+ Q3-c skill 分工决策

## 0. 背景与目标

### 0.1 grill 现场核实的 2 处契约错乱（本 PR 解决）

| 真因 | PR-2 解决 |
|---|---|
| **α**：`ethoinsight-charts` 是孤儿 skill（zero reference），但 SKILL.md 自称服务 chart-maker；同时 `ethoinsight-chart-maker/SKILL.md:30-31` "通用图必须追加 --paradigm" 强制 LLM 对 plot_trajectory（不接 --paradigm）传错参数 | ✅ chart-maker `skills=[...]` 加 `ethoinsight-charts`；删 SKILL.md 那条错指示；按"L2 know-what / L3 how-to-execute"分工重写两个 SKILL |
| **β**：`chart_maker.py:43/67` prompt 模板 `python -m ethoinsight.scripts.<paradigm>.<chart_script>` 与 catalog yaml fallback 注册的 `ethoinsight.scripts._common.plot_trajectory` 路径冲突 → LLM 必须违反模板才能跑通 fallback | ✅ prompt 改 `bash python -m <entry.script> <entry.args 拼接>`（files-are-facts，与 code_executor.py prompt 对齐）|

### 0.2 PR-2 不做什么

- 不改 catalog yaml / schema（PR-1）—— PR-2 **依赖 PR-1**：PlanChart 已含 `args` 字段
- 不改 lead prompt / 不动 guardrail（PR-3）
- 不动 chart-maker subagent 注册的 tools/disallowed_tools/max_turns 等配置（不在范围）
- 不改各范式 yaml 的 chart entry display_name_zh 内容（PR-1 已做）

### 0.3 与 PR-1 的契约依赖

PR-2 实施前必须确认 PR-1 已合入 dev：

- `PlanChart` 含 `args: list[str]` 字段（PR-1 §2.1）
- `PlanChart.output` 已是 `/mnt/user-data/outputs/...png`（PR-1 §2.4）
- `plan_charts_to_dict` 序列化 args 字段（PR-1 §2.4）

如果 PR-1 未合，PR-2 改的 chart_maker prompt 用 `<entry.args>` 拼接会拼出空字符串，导致脚本调用缺参数。

## 1. 核心原则

| 原则 | 应用 |
|---|---|
| **files-are-facts**（与 code_executor 对齐） | chart-maker prompt 信 plan_charts.json 的 entry.script + entry.args，不让 LLM 推导 |
| **L2 know-what / L3 how-to-execute 分工** | `ethoinsight-charts` = L2 图种知识；`ethoinsight-chart-maker` = L3 执行工作流 |
| **渐进披露最佳实践**（anthropic skill 哲学） | SKILL.md 主文 system prompt 注入（L2 overview）；references/ 按需 read_file（L3 深细节）|
| **lead 完全不持图种 know-how** | 已对齐 lead-interaction:48 + lead-prompt:185 W16 注释 |

## 2. 改动清单

### 2.1 `packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py`

#### 2.1.1 第 152 行 `skills=[...]` 加 ethoinsight-charts

**Before**：
```python
skills=["ethoinsight", "ethoinsight-chart-maker"],
```

**After**：
```python
skills=["ethoinsight", "ethoinsight-chart-maker", "ethoinsight-charts"],
```

#### 2.1.2 workflow Step 2 后插入 Step 2.5（L2 渐进披露入口）

**Before**（chart_maker.py:31-33）：
```
1. **开工前必读执行宪法**: read_file `/mnt/skills/custom/ethoinsight/references/execution-conventions.md`
2. **读 chart-maker skill**: read_file `/mnt/skills/custom/ethoinsight-chart-maker/SKILL.md`，获取绘图决策树和 fallback 规则
3. **读 handoff_code_executor.json**: ...
```

**After**：
```
1. **开工前必读执行宪法**: read_file `/mnt/skills/custom/ethoinsight/references/execution-conventions.md`
2. **读 chart-maker 执行手册**: read_file `/mnt/skills/custom/ethoinsight-chart-maker/SKILL.md`，获取工作流、fallback 决策树、handoff schema
3. **读 chart-maker 图种知识**: read_file `/mnt/skills/custom/ethoinsight-charts/SKILL.md`，获取「图种 → 适用场景」对照表（决策时按用户意图选图用）
4. **读 handoff_code_executor.json**: read_file `/mnt/user-data/workspace/handoff_code_executor.json`，获取 paradigm、metrics 输出、统计结果
...（之后步骤序号 +1）
```

整体重排：原来 Step 3-13 全部 +1，变成 4-14。

#### 2.1.3 第 43 行（重排后是 Step 11，原 Step 10）prompt 模板改为 files-are-facts

**Before**（原 Step 10）：
```
10. **执行绘图脚本**（最多 4 个）: bash `python -m ethoinsight.scripts.<paradigm>.<chart_script> --input <metrics_output> --output /mnt/user-data/outputs/<chart_filename>.png`。**注意 --output 必须是 /mnt/user-data/outputs/ 下的路径**，不要写到 workspace/。
```

**After**（新 Step 11）：
```
11. **执行绘图脚本**（最多 4 个）: 遍历 plan_charts.json 的 `charts[]` 数组，对每个 entry 执行 bash `python -m <entry.script> <entry.args 用空格拼接>`。
    - `entry.script` 是完整模块路径（`ethoinsight.scripts._common.plot_trajectory` 或 `ethoinsight.scripts.fst.plot_box_immobility`，**不要自己拼 paradigm 前缀**）
    - `entry.args` 是 resolve 阶段已经按脚本签名拼好的参数数组（含 --input / --output / 可能含 --paradigm），**直接拼接成命令字符串即可，不要额外加任何参数**
    - 例：entry.script=`ethoinsight.scripts._common.plot_timeseries`, entry.args=`["--input", "/path/to/raw.txt", "--output", "/mnt/user-data/outputs/plot_timeseries_plot.png", "--paradigm", "fst"]`
      → bash `python -m ethoinsight.scripts._common.plot_timeseries --input /path/to/raw.txt --output /mnt/user-data/outputs/plot_timeseries_plot.png --paradigm fst`
    - PR-1 已让 PlanChart.output 直接是 `/mnt/user-data/outputs/...png`，**不需要后续移动文件**
```

#### 2.1.4 第 67 行 `<bash_constraints>` 段对应模板同步更新

**Before**（chart_maker.py 第 67 行附近）：
```
- 绘图脚本: python -m ethoinsight.scripts.<paradigm>.<chart_script> --input ... --output ...
```

**After**：
```
- 绘图脚本: python -m <entry.script> <entry.args 拼接>（entry.script + entry.args 来自 plan_charts.json，不要自己拼 paradigm 路径）
```

#### 2.1.5 第 9 步决策树（重排后 Step 10）引用 charts skill references

**Before**（原 Step 9 一句话）：
```
9. **决策树（按 ethoinsight-chart-maker skill 指示）**:
   - catalog 中有对应图表 → 执行对应绘图脚本（catalog 路径）
   - catalog 无对应但指标数据存在 → 执行 fallback 通用绘图脚本
   - 指标数据缺失或脚本报错 → 记入 failed_charts[]，继续处理下一个图表
```

**After**（新 Step 10）：
```
10. **决策树**（按 user_intent 选哪些图执行）:
    - 用户意图明确（"轨迹图" / "箱线图" / "时序图" 等）→ 按 ethoinsight-charts skill 的图种 → 函数对照表选匹配子集；细节见 references/distribution-charts.md / association-charts.md / spatial-temporal-charts.md
    - 用户意图模糊（"再画几个图" / "画一下"）→ 按 ethoinsight-chart-maker skill 的 fallback 决策树；细节见 references/fallback-decision-tree.md
    - 指标数据缺失或脚本报错 → 记入 failed_charts[]，继续处理下一个图表
```

### 2.2 `packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md`

#### 2.2.1 第 30-31 行整段删除（α 修复）

**Before**：
```
5. for each selected chart: bash 跑脚本
   - 通用图(`_common.plot_timeseries` / `_common.plot_trajectory`)必须追加 `--paradigm <p>`,让脚本按范式选默认 y_col。漏传会让 timeseries 退到 `distance_moved` 列;FST/LDB 等范式数据里没有这列,会出空图。
```

**After**（脚本签名已下沉到 yaml + resolve）：
```
5. for each entry in plan_charts.json.charts: bash 跑脚本
   - 用 `python -m <entry.script> <entry.args 拼接>` 形式调用
   - **不要自己拼 paradigm 前缀或追加 --paradigm 参数**——entry.script 和 entry.args 已经是 resolve 阶段按 catalog yaml `accepts_paradigm` 字段拼好的（PR-1 落地)
```

#### 2.2.2 frontmatter description 调整

**Before**：
```yaml
description: >
  chart-maker subagent 的可视化决策手册。基于 catalog plan_charts.json
  和用户语义,决策跑哪些图脚本。
  Use when chart-maker receives a visualization task with handoff_code_executor.json.
```

**After**：
```yaml
description: >
  chart-maker subagent 的执行工作流手册（how-to-execute）。
  包含：bash 调用模板、fallback 决策树、handoff schema、用户意图模糊时的处理。
  图种知识（图种 → 适用场景，what-to-pick）见姐妹 skill `ethoinsight-charts`。
  Use when chart-maker receives a visualization task with handoff_code_executor.json.
```

#### 2.2.3 "Fallback 决策树" 段保留

第 42-47 行的 Fallback 决策树是本 skill 的 L2 核心（"用户意图模糊时怎么选图"），保留原文。

### 2.3 `packages/agent/skills/custom/ethoinsight-charts/SKILL.md`

#### 2.3.1 frontmatter description 调整

**Before**：
```yaml
description: >
  服务对象:**chart-maker subagent**。图种 → 适用场景对照表,chart-maker
  决策选哪些图时的查询源。Lead 不读本 skill — capability-exposure 后
  "用户语义 → 图种" 归 chart-maker,lead 不持图选择 know-how。
```

**After**：
```yaml
description: >
  chart-maker subagent 的图种知识库（what-to-pick）。
  8 种图表 × 适用场景对照表 + 选择决策树。
  执行工作流（how-to-execute）见姐妹 skill `ethoinsight-chart-maker`。
  Lead 不读本 skill — capability-exposure 模式下，"用户语义 → 图种" 归 chart-maker。
```

#### 2.3.2 "变更说明 (2026-05-18 W9)" 段更新

**Before**（第 11-16 行）：
```markdown
**变更说明 (2026-05-18 W9)**:
- 服务对象从"lead agent"变为"chart-maker subagent"
- lead 不再读本 skill 决策图种
- "用户语义 → 图种" 决策树移到 `ethoinsight-chart-maker` skill (W21)
- 本 skill 保留作为图种适用性查询源 (chart-maker 在 W21 skill 决策时 reference 这里)
```

**After**（按 PR-2 决策更新）：
```markdown
**变更说明 (2026-05-22 PR-2)**:
- chart-maker subagent 通过 `skills=[..., "ethoinsight-charts"]` 把本 SKILL.md 主文注入 system prompt（L2 知识）
- 选图决策树（图种 → 适用场景）留在本 skill；fallback 决策树（catalog 命中 / 模糊语义时的 fallback）在姐妹 `ethoinsight-chart-maker` skill
- references/ 是 L3 深细节，按需 read_file（chart-maker prompt workflow Step 10 触发）
```

#### 2.3.3 现有"选择决策树"段保留

第 20-33 行的"需要展示什么？"决策树是本 skill 的 L2 核心，保留原文。

### 2.4 验证

#### 2.4.1 改动后必须跑

```bash
cd /home/wangqiuyang/noldus-insight
# subagent config 自检（导入时 fail-fast 校验 required_upstream_handoffs）
cd packages/agent/backend && source .venv/bin/activate && python -c "
from deerflow.subagents.builtins.chart_maker import CHART_MAKER_CONFIG
print('skills:', CHART_MAKER_CONFIG.skills)
assert 'ethoinsight-charts' in CHART_MAKER_CONFIG.skills, 'PR-2 §2.1.1 not applied'
print('OK')
"

# 测试整体未坏
cd packages/agent/backend && make test 2>&1 | tail -20
```

#### 2.4.2 dogfood 验证用例（用 PR-1 已合入 dev 后）

**用 thread 843fe2b8 重跑 FST 流程**，期望：

- chart-maker workflow Step 2/3 应该 read 两个 SKILL（chart-maker + charts）
- chart-maker bash 调用形如 `python -m ethoinsight.scripts._common.plot_timeseries --input ... --output /mnt/user-data/outputs/... --paradigm fst`（args 数组来自 plan_charts.json，不是 LLM 拼的）
- PlanChart.output 已是 outputs/（PR-1 落地），chart-maker 不需要后续 cp/mv
- chart-maker handoff 的 chart_files[] 全部是 outputs/ 路径
- failed_charts=[]

### 2.5 跨文件影响检查

实施后必须 grep 验证以下查询返回零结果（确认旧模板/旧错指示被清除）：

```bash
cd /home/wangqiuyang/noldus-insight

# 1) chart-maker prompt 不应再含旧拼接模板
grep -n "ethoinsight.scripts.<paradigm>.<chart_script>" packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py
# 期望：0 行

# 2) SKILL 不应再含 "plot_trajectory" + "--paradigm" 的连体指示
grep -B1 -A1 "plot_trajectory" packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md
# 期望：仅出现在 "5. for each entry..." 那段提示语境，且**不含"必须追加 --paradigm"**

# 3) ethoinsight-charts 不再是孤儿
grep -rn "ethoinsight-charts" packages/agent/backend/packages/harness/deerflow/
# 期望：至少 chart_maker.py 一处引用（PR-2 §2.1.1 落地）
```

## 3. 实施顺序（建议）

1. **§2.1.1**：chart_maker.py skills=[...] 加 ethoinsight-charts（孤儿挂回）
2. **§2.1.2**：workflow 加 Step 2.5（read ethoinsight-charts SKILL）+ 后续步骤序号 +1
3. **§2.1.3 + §2.1.4 + §2.1.5**：prompt 模板改 files-are-facts 风格 + decision tree 引用更新
4. **§2.2**：chart-maker SKILL.md 删 plot_trajectory --paradigm 错指示 + description 改 how-to-execute 定位
5. **§2.3**：charts SKILL.md description 改 L2 know-what 定位 + 变更说明更新
6. **§2.4.1**：跑 subagent config 自检 + make test
7. **§2.4.2**：dogfood 用 thread 843fe2b8 复跑验证
8. **§2.5**：grep 验证清除项
9. commit + push

## 4. 关键文件速查

| 路径 | 用途 |
|---|---|
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py` | 主要改动文件（prompt + skills 注册）|
| `packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md` | L3 执行手册（删错指示 + 改 description）|
| `packages/agent/skills/custom/ethoinsight-charts/SKILL.md` | L2 图种知识库（改 description）|
| `packages/agent/skills/custom/ethoinsight-chart-maker/references/` | fallback-decision-tree.md / user-intent-parsing.md（不改）|
| `packages/agent/skills/custom/ethoinsight-charts/references/` | distribution-charts.md / association-charts.md / spatial-temporal-charts.md（不改）|
| `packages/ethoinsight/ethoinsight/catalog/_common.yaml` | trajectory_plot / timeseries_plot 注册位置（不改，PR-1 已处理）|

## 5. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 改 prompt workflow 步骤号让 LLM 困惑（之前是 1-13，现在 1-14） | 步骤重排后**逐步号 read**，序号紧密相邻；dogfood 验证 chart-maker 流程未崩 |
| PR-1 未合入但 PR-2 已合：plan_charts.json 没 args 字段，prompt 模板 `<entry.args>` 拼空字符串 | **依赖 PR-1 必须先合**；CI 加单测验证 plan_charts.json schema 含 args 字段 |
| ethoinsight-charts SKILL 注入后 chart-maker context 变长 | 两个 SKILL 主文 < 100 行，注入到 system prompt 影响小；references/ 按需 read 不进入 system prompt |
| Step 2.5 read 两个 SKILL 多花 turn | chart-maker max_turns=15 充足；turn 预算受影响 < 1 turn |

## 6. 不在本 PR 范围

- catalog schema 升级（PR-1）
- PlanChart.output 改 outputs/（PR-1）
- ChartEntry yaml accepts_paradigm 字段（PR-1）
- lead bash disallow / IntentPostStepAskGateProvider / task_tool Literal / 删 ethoinsight-planning（PR-3）
- 修 IntentClassificationGuardrailProvider 假阳 bug（PR-4）

## 7. commit 建议

按 §3 实施顺序分 4 commit：

1. `feat(chart-maker): skills=[..., ethoinsight-charts] 挂载孤儿 skill + workflow Step 2.5 渐进披露入口`
2. `feat(chart-maker): prompt 模板改 files-are-facts <entry.script> <entry.args> 风格，与 code_executor 对齐`
3. `docs(skill): ethoinsight-chart-maker SKILL.md 删除 plot_trajectory --paradigm 错指示（PR-1 已下沉到 yaml）`
4. `docs(skill): chart-maker/charts SKILL.md description 改 L3/L2 分工定位 + 变更说明`

总计 4 commit，可在一个 PR 里。

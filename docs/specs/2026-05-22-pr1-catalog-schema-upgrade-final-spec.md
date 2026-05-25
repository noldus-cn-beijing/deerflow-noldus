# PR-1 最终 spec —— catalog schema 升级 + display_name_zh 透传 + accepts_paradigm 声明性元数据

> **生成于**：2026-05-22
> **范围**：纯 ethoinsight 库改动 + 单测，**不动 agent 代码**
> **独立可测**：不依赖 PR-2 / PR-3
> **schema_version 升级**：1.0 → 1.1
> **来源**：[2026-05-22-chart-maker-grill-corrected-handoff.md](../handoffs/2026-05/2026-05-22-chart-maker-grill-corrected-handoff.md) §PR-1 + grill 真因 §1（B4 决策）/ §3（display_name_zh 链条）

## 0. 背景与目标

### 0.1 grill 现场核实的 4 处契约错乱

PR-1 解决 4 处中的 2 处（剩 2 处由 PR-2/PR-3 处理）：

| 真因 | PR-1 解决 |
|---|---|
| `ethoinsight-chart-maker/SKILL.md:28` "通用图必须追加 --paradigm" 但 `_common/plot_trajectory.py` argparse 不接 --paradigm | ✅ **B4 方案**：脚本签名声明性下沉到 ChartEntry yaml `accepts_paradigm` 字段，resolve 据此生成 args 数组；LLM 不再需要猜哪个脚本接什么参数 |
| `plan_metrics.json` schema 不含 `display_name_zh`，只有英文 id，PR-2 前端 plan_labels 取不到中文名 | ✅ **γ 数据契约**：catalog yaml display_name_zh → PlanMetric/PlanChart → JSON 序列化字段，single source of truth 链条完整 |
| `chart_maker.py:43/67` prompt 模板 `python -m ethoinsight.scripts.<paradigm>.<chart_script>` 与 fallback `ethoinsight.scripts._common.plot_*` 路径冲突 | PR-2 负责（chart_maker prompt 改 `python -m <entry.script> <entry.args 拼接>`） |
| 78ccb52b lead 越过 ScriptInvocationOnlyProvider 直接 bash cp 移文件 + 派 general-purpose | PR-3 负责（lead disallow bash + task_tool Literal）|

### 0.2 PR-1 不做什么

- 不改任何 chart_maker subagent / lead prompt（PR-2 / PR-3）
- 不动 `_common/plot_trajectory.py` 等脚本签名本身（脚本现状已经是对的：trajectory 不接 paradigm，timeseries 接 paradigm；问题在 catalog 没声明）
- 不引入 chart purpose 字段（用户 Q3-a 明确拒绝）—— PlanChart.output 直接锁 outputs/，不引入"中间产物 chart"枚举

## 1. 核心原则

| 原则 | 应用 |
|---|---|
| **single source of truth** | display_name_zh / 脚本签名声明 都只在 catalog yaml 一份，下游 PlanMetric/PlanChart 透传 |
| **声明性元数据优于命令式拼参数** | `accepts_paradigm: true` 替代 chart-maker LLM 推断 |
| **保持 ethoinsight 库纯净** | 不让脚本依赖 plan_charts.json schema；保持库可独立调用 |
| **schema 升级走 backward-compatible** | 新字段加默认值，旧 yaml 不需要全量改 |

## 2. 改动清单（按依赖顺序）

### 2.1 `packages/ethoinsight/ethoinsight/catalog/schema.py`

**改动**：

1. `MetricEntry` 已有 `display_name_zh` —— ✅ 无需改

2. `ChartEntry` 加两个字段：
```python
@dataclass(frozen=True)
class ChartEntry:
    id: str
    script: str
    when: ChartCondition
    display_name_zh: str = ""          # 新增，默认空串保 backward-compat
    accepts_paradigm: bool = False      # 新增，默认 False
```

3. `PlanMetric` 加字段：
```python
@dataclass
class PlanMetric:
    id: str
    script: str
    input: str
    output: str
    required: bool
    reason: str
    subject_index: int = 0
    display_name_zh: str = ""           # 新增
```

4. `PlanChart` 加字段：
```python
@dataclass
class PlanChart:
    id: str
    script: str
    input: str
    output: str
    subject_index: int = 0
    display_name_zh: str = ""           # 新增
    args: list[str] = field(default_factory=list)  # 新增；resolve 阶段填充
```

5. `Plan` 的 `schema_version` 常量更新为 `"1.1"`（如果有顶层常量）

**注意**：所有新字段都是 default 值，单测和已有调用方不需要改签名。

### 2.2 `packages/ethoinsight/ethoinsight/catalog/loader.py`

**改动**：

1. `_parse_chart_list` 添加可选字段读取：
```python
def _parse_chart_list(raw: dict, source: Path) -> list[ChartEntry]:
    items = raw.get("charts", []) or []
    if not isinstance(items, list):
        raise CatalogError(f"{source}: 'charts' must be a list")
    out: list[ChartEntry] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise CatalogError(f"{source}: charts[{i}] must be a mapping")
        for f in ("id", "script", "when"):
            if f not in it:
                raise CatalogError(f"{source}: charts[{i}] missing '{f}'")
        # 新增：display_name_zh 必填、accepts_paradigm 可选
        display_name_zh = it.get("display_name_zh", "")
        if not display_name_zh:
            raise CatalogError(f"{source}: charts[{i}] missing 'display_name_zh'")
        accepts_paradigm = bool(it.get("accepts_paradigm", False))
        out.append(ChartEntry(
            id=it["id"],
            script=it["script"],
            when=it["when"],
            display_name_zh=display_name_zh,
            accepts_paradigm=accepts_paradigm,
        ))
    return out
```

2. `_parse_chart_list_under_key`（`common_charts:` 段使用）同样应用上面变更——把同一段逻辑共享或复制一次。

**Backward-compat**：`display_name_zh` 改为**必填**——这意味着所有 yaml 都需要先加这个字段。变更前先做 yaml 同步（§2.3）。

### 2.3 `packages/ethoinsight/ethoinsight/catalog/*.yaml`

逐个 yaml 加 `display_name_zh` 必填字段；只有少数需要加 `accepts_paradigm`。

#### `_common.yaml`

```yaml
common_charts:
  - id: trajectory_plot
    script: ethoinsight.scripts._common.plot_trajectory
    when: total_subjects >= 1
    display_name_zh: "轨迹图"
    # accepts_paradigm 留空 / 不写（默认 false）—— plot_trajectory.py argparse 不接受 --paradigm
    rationale: 单被试或组间数据不全时的轨迹可视化

  - id: timeseries_plot
    script: ethoinsight.scripts._common.plot_timeseries
    when: total_subjects >= 1
    display_name_zh: "时间序列图"
    accepts_paradigm: true          # plot_timeseries.py 接受 --paradigm 决定默认 y_col
    rationale: 单被试或组间数据不全时的时间动态
```

#### `fst.yaml` / `oft.yaml` / `epm.yaml` / `zero_maze.yaml` / `shoaling.yaml`

每个 chart entry 加 `display_name_zh`。各范式脚本是否接 `--paradigm` 要现场核验：

- 各范式自家 `plot_box_*` / `plot_*_bar` / `plot_*_summary` 等脚本通常**不接** --paradigm（因为已经是范式特定的脚本）
- 实施时按以下命令逐脚本确认：
```bash
for yaml in packages/ethoinsight/ethoinsight/catalog/{fst,oft,epm,zero_maze,shoaling}.yaml; do
  echo "=== $yaml ==="
  grep -E "script: ethoinsight" $yaml | awk -F'script: ' '{print $2}'
done
# 然后每个 script 文件：
grep -nE "add_argument.*paradigm|--paradigm" packages/ethoinsight/ethoinsight/scripts/<paradigm>/*.py
```

display_name_zh 命名参考（MetricEntry 现有命名风格）：
- `plot_box_immobility` → "不动时间箱线图"
- `plot_box_center` → "中心时间箱线图"
- 其他遵循「中文指标名 + 图种」格式

**实施建议**：写一个临时脚本扫所有 catalog yaml，先 dry-run 列出"缺 display_name_zh 的 chart entry"，请用户审核中文名再 commit。

### 2.4 `packages/ethoinsight/ethoinsight/catalog/resolve.py`

**改动**：

1. `_metric_to_plan` 传 display_name_zh 到 PlanMetric：
```python
plans.append(
    PlanMetric(
        id=m.id,
        script=m.script,
        input=raw_file,
        output=output_path,
        required=required,
        reason=reason,
        subject_index=idx,
        display_name_zh=m.display_name_zh,   # 新增
    )
)
```

2. `_chart_to_plan` 重写：
   - PlanChart.output 改为 `/mnt/user-data/outputs/` （**不再写 workspace**）
   - 按 `accepts_paradigm` 决定 args 是否含 `--paradigm <p>`
   - 透传 display_name_zh

```python
def _chart_to_plan(
    ch: ChartEntry, raw_files: list[str], workspace_dir: str,
    paradigm: str,                          # 新增参数：当前 paradigm 名
    virtual_workspace_dir: str | None = None,
    virtual_outputs_dir: str | None = None, # 新增参数：outputs 虚拟路径
) -> list[PlanChart]:
    if not raw_files:
        return []
    # PlanChart.output → outputs/ (与 chart_maker.py "PNG 必须写到 outputs/" 一致)
    effective_outputs = virtual_outputs_dir or "/mnt/user-data/outputs"
    multi = len(raw_files) > 1
    plans: list[PlanChart] = []
    for idx, raw_file in enumerate(raw_files):
        suffix = f"_s{idx}" if multi else ""
        output_path = str(Path(effective_outputs) / f"plot_{ch.id}{suffix}.png")
        # args 列表：base args + 按 accepts_paradigm 加 --paradigm
        args = ["--input", raw_file, "--output", output_path]
        if ch.accepts_paradigm:
            args.extend(["--paradigm", paradigm])
        plans.append(
            PlanChart(
                id=ch.id,
                script=ch.script,
                input=raw_file,
                output=output_path,
                subject_index=idx,
                display_name_zh=ch.display_name_zh,
                args=args,
            )
        )
    return plans
```

3. `_chart_to_plan` 调用方需要更新签名传 paradigm 和 outputs_dir：grep `_chart_to_plan(` 找所有调用点，逐个补参数。

4. `plan_metrics_to_dict` / `plan_charts_to_dict` 序列化新字段：

```python
"metrics": [
    {
        "id": m.id,
        "script": m.script,
        "input": m.input,
        "output": m.output,
        "required": m.required,
        "reason": m.reason,
        "subject_index": m.subject_index,
        "display_name_zh": m.display_name_zh,    # 新增
    }
    for m in pm.metrics
],
```

`plan_charts_to_dict` 类似，charts 数组添加 `display_name_zh` 和 `args`。

5. schema_version 顶层从 `"1.0"` 改 `"1.1"`（找 `schema_version` 字面量赋值处）。

### 2.5 测试

#### 必须新增的单测：

**`packages/ethoinsight/tests/test_catalog_schema_v11.py`**（新建）：

1. `test_chart_entry_display_name_zh_required` — yaml 缺 display_name_zh 应该 raise CatalogError
2. `test_chart_entry_accepts_paradigm_default_false` — 不写 accepts_paradigm 时 ChartEntry.accepts_paradigm == False
3. `test_plan_chart_output_in_outputs_dir` — resolve 生成的 PlanChart.output 路径以 `/mnt/user-data/outputs/` 开头
4. `test_plan_chart_args_includes_paradigm_when_accepts_paradigm` — accepts_paradigm=true 的 chart entry 生成的 PlanChart.args 含 `["--paradigm", "<paradigm_name>"]`
5. `test_plan_chart_args_excludes_paradigm_otherwise` — accepts_paradigm=false 的 chart entry 生成的 PlanChart.args 不含 --paradigm
6. `test_plan_metrics_to_dict_includes_display_name_zh` — 序列化结果含中文标签
7. `test_plan_charts_to_dict_includes_args_and_display_name_zh`
8. `test_schema_version_is_1_1` — Plan / PlanCharts / PlanMetrics 顶层 schema_version 字面 "1.1"

#### 必须更新的现有单测：

- `test_catalog_resolve_paths.py`：PlanChart.output 路径断言从 `/mnt/user-data/workspace/...` 改 `/mnt/user-data/outputs/...`
- `test_catalog_resolve_multi_subject.py`：多 subject 场景 output 路径同上
- 任何用 ChartEntry 构造的单测：需要传 `display_name_zh=...`
- `test_catalog.py` / `test_common_catalog.py`：catalog yaml fixture 如果是字符串内嵌的，加 display_name_zh

#### 验证命令（实施后跑）

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/ -v 2>&1 | tail -40
```

## 3. 实施顺序（建议）

1. **schema.py** 加新字段（带默认值，老代码不破坏）
2. **catalog yaml** 同步加 display_name_zh + accepts_paradigm（参考 §2.3 实施建议先 dry-run 列举待加项）
3. **loader.py** display_name_zh 必填校验 + accepts_paradigm 可选读取
4. **resolve.py** _chart_to_plan 接 paradigm + args 生成 + outputs/ 路径；_metric_to_plan / dict 序列化字段传递
5. 跑现有测试，逐个修预期路径断言
6. 新建 §2.5 单测
7. `pytest tests/` 全绿
8. 最后 commit + push

## 4. 关键文件速查

| 路径 | 用途 |
|---|---|
| `packages/ethoinsight/ethoinsight/catalog/schema.py` | dataclass 定义 |
| `packages/ethoinsight/ethoinsight/catalog/loader.py` | yaml→dataclass 解析 |
| `packages/ethoinsight/ethoinsight/catalog/resolve.py` | catalog→Plan / PlanMetrics 生成 + JSON 序列化 |
| `packages/ethoinsight/ethoinsight/catalog/_common.yaml` | fallback chart 定义（trajectory + timeseries）|
| `packages/ethoinsight/ethoinsight/catalog/{fst,oft,epm,zero_maze,shoaling}.yaml` | 各范式 metric + chart + statistics |
| `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py` | 不接 --paradigm（已对）|
| `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py` | 接 --paradigm（已对）|
| `packages/ethoinsight/tests/` | 单测目录 |

## 5. 验证 thread

**用现有 thread 843fe2b8 的 plan_charts.json 做手工对比**：

实施后跑：
```bash
cd /home/wangqiuyang/noldus-insight
python -c "
from ethoinsight.catalog.resolve import resolve_charts, plan_charts_to_dict
# 用 843fe2b8 的 columns.json / raw_files 重生成 plan_charts.json
# 期望：每个 PlanChart 含 args 数组 + display_name_zh + output 在 /mnt/user-data/outputs/
"
```

或者写一个 e2e 测试调用 resolve 后断言输出 JSON 含期望字段。

**期望输出对比**（与现有 843fe2b8 plan_charts.json）：

| 字段 | 当前 | PR-1 后 |
|---|---|---|
| `output` (trajectory_plot s0) | `/mnt/user-data/workspace/plot_trajectory_plot_s0.png` | `/mnt/user-data/outputs/plot_trajectory_plot_s0.png` |
| `display_name_zh` | 不存在 | `"轨迹图"` |
| `args` | 不存在 | `["--input", "...", "--output", "..."]`（trajectory 不接 paradigm）|
| timeseries_plot 的 args | 不存在 | `["--input", "...", "--output", "...", "--paradigm", "fst"]` |

## 6. Backward-compat 风险

| 风险 | 缓解 |
|---|---|
| display_name_zh 必填会让旧 catalog yaml 加载失败 | 实施时**先加 yaml**（§2.3），后改 loader.py（§2.2）—— commit 顺序保证不会有"loader 严了但 yaml 还没加"的中间态 |
| PlanChart.output 改路径会让 chart-maker subagent（PR-2 前）找不到 PNG | **PR-2 实施前**：chart-maker 仍读 plan_charts.json，看到 outputs/ 路径会照拼正确（chart_maker.py:43 已说"--output 必须是 outputs/"），现在二者一致；如果 chart-maker LLM 改写路径，那是 PR-2 工作流问题，不归本 PR |
| args 字段加上后 chart-maker（PR-2 前）不知道用 | **PR-2 前**：chart-maker prompt 模板仍是 `python -m <paradigm>.<script>`，会照常用 input/output 拼参数，无视 args 字段——**不会破坏**当前行为 |

实施后 dogfood 1 个 thread 验证 plan_charts.json schema 正确即可（不需要等 PR-2/PR-3）。

## 7. 不在本 PR 范围

- chart_maker.py prompt 模板修复 → PR-2
- chart-maker `skills=[...]` 加 ethoinsight-charts → PR-2
- SKILL.md:28 "通用图必须追加 --paradigm" 整行删除 → PR-2
- lead 边界硬约束 / IntentPostStepAskGateProvider / 删 planning skill → PR-3

## 8. 跑测试命令清单

实施过程中和完成后：

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight

# 完整测试
pytest tests/ -v

# 仅相关测试
pytest tests/test_catalog* tests/test_common_catalog.py -v

# 新增测试
pytest tests/test_catalog_schema_v11.py -v
```

## 9. commit 建议

按 §3 顺序分 commit：

1. `feat(catalog): 升级 schema 1.0→1.1, ChartEntry/PlanMetric/PlanChart 加 display_name_zh + accepts_paradigm + args`
2. `feat(catalog): _common.yaml + 各范式 yaml 添加 display_name_zh 字段; timeseries_plot 加 accepts_paradigm=true`
3. `feat(catalog): loader.py 校验 display_name_zh 必填, 读 accepts_paradigm 可选字段`
4. `feat(catalog): resolve._chart_to_plan 按 accepts_paradigm 注入 --paradigm; PlanChart.output 改 outputs/; 序列化新字段`
5. `test(catalog): 升级现有单测路径断言 + 新增 test_catalog_schema_v11.py`

总计 5 commit，可以在一个 PR 里。

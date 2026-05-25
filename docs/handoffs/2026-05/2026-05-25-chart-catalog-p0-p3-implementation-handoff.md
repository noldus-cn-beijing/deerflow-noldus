# 2026-05-25 P0-P3 chart catalog 实施 handoff

> **状态**: 5/25 已完成 fix(subagent) recursion_limit + frontend B2+B3 (HEAD `232f12c8` on dev)。chart catalog 18 张图缺口仍未实施。本 handoff 给下一位 agent，按 P0→P3 顺序补图，**每个 P 一个 commit + push**，让用户能逐个验证。
>
> **不要重新走调研流程**：本文档已含同事拍板的全部参数和实施细节，下手前**先读完整份**再开工。SSOT/参考脚本/单测模板/命名规范全在这里，不要再去翻 review-packages 反复确认。

---

## 当前会话已完成（context for you）

本会话前两段在 dev 已 push：

1. **`8cea306a` fix(subagent): recursion_limit 按 middleware hook 动态算** — data-analyst max_turns=12 配 8 个 middleware hook 算下来 recursion_limit 远超旧 `*2+1=25`，根因是 5/25 上游 sync `11fee2f9` 加 `SafetyFinishReasonMiddleware` 让 subagent middleware 链多 1 个 after_model 节点，叠加同日 `3834a022` 给 data-analyst prompt 加 step 2.5 让它多 1 个 turn，刚好顶破。`calculate_subagent_recursion_limit(middlewares, max_turns)` 按实际 hook 数动态算。
2. **`232f12c8` fix(frontend): subagent 完成后 lead 汇报被折叠的 5/25 回归** — `cd512536` (5/21) 让 lead 把 narrative + ask_clarification 打进同一 AIMessage，前端 `groupMessages` 把这种 AIMessage 分到 `assistant:processing` group，message-list.tsx 没显式 case 落到 fallback `<MessageGroup>` 把 content 当 thinking 渲到 CoT 折叠区，用户看不见。修复: (B3) message-list.tsx 加 `assistant:processing` 分支让 content 走主气泡 narrative + reasoning 留独立折叠；(B2) subtask-card.tsx 把 task.result 挪出 ChainOfThoughtContent 放卡片顶部永久可见。

这两条**跟你本次 P0-P3 任务正交**，不要回头改它们。

---

## 任务范围

按 [docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md](../../review-packages/2026-05-25-chart-catalog-implementation-plan.md) 补 chart catalog：

| 优先级 | 范围 | 图数 | commit 单位 |
|---|---|---|---|
| **P0** | fst + tst 各补：柱状图（均值±SEM）/ 活动强度图 / 放弃挣扎分布图 | 6 | 1 commit |
| **P1** | zero_maze + ldb 各补：轨迹图 / 热区图 / 箱线图（zero_maze/ldb 各有孤儿脚本 `plot_box_open_zone.py` / `plot_box_light.py` 直接挂注册）/ 柱状图 | 8 张 + 2 注册 | 1 commit |
| **P2** | epm + oft 各补：轨迹图 / 热区图（通用脚本下沉 `scripts/_common/`） | 4 | 1 commit |
| **P3** | oft 时间进程图（5 分钟 bin，2 条折线：运动距离 + 中心区滞留） | 1 | 1 commit |

总计 **19 张图 + 2 注册**，分 **4 个 commit**。

---

## 关键决定（同事 + 用户已拍板，不要再问）

### 已拍板 1：柱状图（均值±SEM）在 n<3 时
- n ≥ 3：画柱体 + SEM 误差线
- **n < 3 (n=1/n=2)：柱体照出，不画误差线，柱顶标数值**
- catalog `when` 用 `total_subjects >= 1`

### 已拍板 2：OFT 时间进程图（P3）
- catalog `when`: `total_duration_seconds > 300`
- bin 长度**固定 300 秒**，bin 数 = `floor(总时长_秒 / 300)`
- 末尾不足 5 min 并入最后一个 bin
- 每 bin 画 2 条折线：**运动距离** + **中心区滞留**

### 已拍板 3：活动强度图代理列（用户 2026-05-25 拍板）
- **用 `velocity` 列作为活动强度代理**
- 不用 `mobility_continuous`（部分文件没这列），不用 `mobility_state` (0/1 二值)
- 画 velocity 随 trial_time 变化的时序面积图/折线图，title 标 "Activity intensity (velocity proxy)"

### 已拍板 4：放弃挣扎分布图
- "挣扎 vs 放弃挣扎时间分布对比"
- 数据源：`mobility_state` 列（mobile = 挣扎，immobile = 放弃挣扎）
- 用 matplotlib `eventplot` 每个 subject 一行，画 immobile bouts 的横条
- 多 subject 时按 subject ID 纵向排列

### 已拍板 5：命名 + 国际化
- `display_name_zh` 用 SSOT 文档中文名（轨迹图 / 热区图 / 箱线图 / 柱状图 / 区域进入分布图 / 中心区进入汇总图 / 时间进程图 / 活动强度图 / 放弃挣扎分布图）
- `chart id` 照现有 catalog 风格延续（`<指标>_<图种>` 或 `<图种>_<范畴>`）

### 不要做的
- ❌ 不加 SSOT 未列的图（散点图、相关性热图、小提琴图、回归图）
- ❌ 不改已有 `box_*` 图的 `n_per_group >= 3` 阈值
- ❌ 不改 chart_id 命名规范
- ❌ 不动今天已合的 recursion_limit fix / frontend B2+B3

---

## 现状速查（你直接照搬，省得自己探）

### 现有 chart 函数（`packages/ethoinsight/ethoinsight/charts.py`）
- ✅ `box_plot(metrics, metrics_to_plot, significance, output_path)` - line 76
- ✅ `bar_chart(metrics, metrics_to_plot, error_type="sem", significance, output_path)` - line 128
- ✅ `trajectory_plot(df, color_by="subject", output_path)` - line 239
- ✅ `timeseries_plot(timeseries_df, y_col, x_col="trial_time", output_path)` - line 298
- ✅ `add_significance_markers` / `raincloud_plot` / `beeswarm_plot` / `violin_plot` / `correlogram` (现有但 SSOT 没要求用)

**需要新增的 chart 函数**（在 charts.py 加）：
1. `heatmap_plot(df, output_path)` — 2D KDE 或 hexbin 热区图（输入 x_center/y_center）
2. `activity_intensity_plot(df, output_path)` — velocity 时序面积/折线图
3. `struggle_distribution_plot(per_subject_bouts, output_path)` — eventplot 多 subject 不动 bout 分布
4. `time_progress_plot(per_bin_data, output_path)` — OFT 时间进程双折线图（5min bin，distance + center_time）

zone_entry_distribution / center_entry_summary 这俩**单样本场景不需要新 chart 函数**，照现有 `plot_zone_entry_distribution.py` / `plot_center_entry_summary.py` 直接用 matplotlib 双柱即可。

### 现有 plot 脚本（每范式目录下）

```
scripts/fst/ : compute_*.py + plot_box_immobility.py + run_groupwise_stats.py
scripts/tst/ : compute_*.py + plot_box_immobility.py + run_groupwise_stats.py
scripts/epm/ : compute_*.py + plot_box_open_arm.py + plot_open_arm_time_ratio_bar.py + plot_zone_entry_distribution.py + run_groupwise_stats.py
scripts/oft/ : compute_*.py + plot_box_center.py + plot_center_entry_summary.py + plot_center_time_ratio_bar.py + run_groupwise_stats.py
scripts/zero_maze/ : compute_*.py + plot_box_open_zone.py (孤儿 - 未注册) + run_groupwise_stats.py
scripts/ldb/ : compute_*.py + plot_box_light.py (孤儿 - 未注册) + run_groupwise_stats.py
scripts/_common/ : plot_trajectory.py (已存在) + plot_timeseries.py (已存在) + compute_*.py
```

### 现有 catalog（每范式 yaml 的 charts: 块）

| 范式 | 已注册 charts | 需新增 |
|---|---|---|
| `epm.yaml` | box_open_arm, open_arm_time_ratio_bar, zone_entry_distribution | trajectory, heatmap |
| `oft.yaml` | box_center, center_time_ratio_bar, center_entry_summary | trajectory, heatmap, time_progress |
| `zero_maze.yaml` | （空） | box_open_zone（挂孤儿）, bar, trajectory, heatmap |
| `ldb.yaml` | （空） | box_light（挂孤儿）, bar, trajectory, heatmap |
| `fst.yaml` | box_immobility | bar, activity_intensity, struggle_distribution |
| `tst.yaml` | box_immobility | bar, activity_intensity, struggle_distribution |

### 现有的 chart catalog YAML 条目格式

```yaml
charts:
  - id: box_immobility
    script: ethoinsight.scripts.fst.plot_box_immobility
    when: n_per_group >= 3
    display_name_zh: "不动时间箱线图"
```

字段：`id`（snake_case） / `script`（python module 路径） / `when`（条件表达式，用 `total_subjects` / `n_per_group` / `n_groups` / `total_duration_seconds` 等） / `display_name_zh`（同事 SSOT 文档中文名）

### 现有 plot 脚本结构（**严格仿照**，不要发明新结构）

`packages/ethoinsight/ethoinsight/scripts/fst/plot_box_immobility.py`:

```python
"""FST: 不动行为组间对比箱线图。

CLI: python -m ethoinsight.scripts.fst.plot_box_immobility \
       --inputs <inputs.json> --groups <groups.json> --output <png>
"""
from __future__ import annotations
import sys
from ethoinsight.charts import box_plot
from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_groups_json, read_inputs_json

METRICS_TO_PLOT = ["immobility_time", "immobility_bout_count"]

def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=True).parse_args(argv)
    if not args.inputs:
        print("error: ... requires --inputs (multi-file)", file=sys.stderr)
        return 2
    paths = read_inputs_json(args.inputs)
    groups = read_groups_json(args.groups) if args.groups else None
    parsed = parse_batch(paths)
    metrics = compute_paradigm_metrics(parsed, paradigm="forced_swim", groups=groups)
    output_path = box_plot(metrics, metrics_to_plot=METRICS_TO_PLOT, output_path=args.output)
    emit_result({"plot": "box_immobility", "path": output_path, "metrics": METRICS_TO_PLOT})
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

单样本柱状图模板见 `scripts/epm/plot_open_arm_time_ratio_bar.py`。
分布柱图模板见 `scripts/epm/plot_zone_entry_distribution.py` 和 `scripts/oft/plot_center_entry_summary.py`。

**make_plot_parser 提供的参数**：`--input`（单文件） / `--inputs`（json 多文件） / `--groups`（groups.json） / `--output`（PNG path）。
**emit_result** 把成功结果按统一格式打印到 stdout（catalog.resolve 依赖此格式判断脚本成功）。

### 单测模板（**严格仿照**）

参考 `packages/ethoinsight/tests/test_plot_epm_single_subject_cli.py`：

```python
# 关键 helper: _df_to_ethovision_file(df, path, subject="Subject 1")
# 关键 fixture: 给每个测试 paradigm 写一个 minimal trajectory file
# 测试方式: subprocess.run([sys.executable, "-m", "ethoinsight.scripts.<paradigm>.<script>", ...])
#           断言 returncode == 0, output file exists, file size > 0
# 覆盖至少 2 个分支: n=1 (单样本) + n=3 (多样本/组间)
```

`_df_to_ethovision_file` helper 写到测试文件本地（每个测试文件复制一份，**不要重复抽象到 conftest.py**——其它 plot 测试也是这样独立 helper 的）。

### 范式 ↔ paradigm key 映射

`paradigm` 字段在 `compute_paradigm_metrics(paradigm="...")` 调用要用**学术名**：

| catalog yaml | metrics 用的 paradigm 字符串 |
|---|---|
| `epm.yaml` | `"epm"` |
| `oft.yaml` | `"open_field"` |
| `zero_maze.yaml` | `"zero_maze"` |
| `ldb.yaml` | `"light_dark"` |
| `fst.yaml` | `"forced_swim"` |
| `tst.yaml` | `"tail_suspension"` |

`metrics.dispatcher.compute_paradigm_metrics` 看 `compute_paradigm_metrics()` 当前 paradigm 列表自己确认。

---

## 工作流（每张图都走一遍）

每张图 4 步：

1. **写脚本** `packages/ethoinsight/ethoinsight/scripts/<paradigm>/plot_<chart_id>.py`
2. **catalog 注册** 在对应 `packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml` 的 `charts:` 块加条目
3. **新 chart 函数**（如需要）加到 `packages/ethoinsight/ethoinsight/charts.py`
4. **单测** `packages/ethoinsight/tests/test_<paradigm>_<chart_id>_cli.py`（覆盖 n=1 + n=3 两条分支）

每个 P 完成后跑：
```bash
cd packages/ethoinsight && uv run pytest tests/ -q
cd ../agent/backend && .venv/bin/python -m pytest tests/ -q   # 防回归
```
两边都 0 failed 再 commit + push。

---

## P0 详细清单（先做这个）

### 新增 chart 函数（charts.py）

#### `activity_intensity_plot(df, output_path) -> str`
```python
def activity_intensity_plot(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """Activity intensity time-series (velocity as proxy).

    For FST/TST where mobility_state alone is binary; velocity captures
    continuous activity changes. Plots a filled area chart of velocity
    vs trial_time.

    Args:
        df: parsed trajectory with columns ``trial_time`` and ``velocity``.
        output_path: defaults to a temp file.
    """
    _setup_style()
    output_path = _resolve_output_path(output_path, "activity_intensity")
    fig, ax = plt.subplots(figsize=(10, 4))
    if "trial_time" in df.columns and "velocity" in df.columns:
        v = pd.to_numeric(df["velocity"], errors="coerce").fillna(0)
        t = pd.to_numeric(df["trial_time"], errors="coerce")
        ax.fill_between(t, v, color="#4C9F70", alpha=0.5)
        ax.plot(t, v, color="#2D5F3F", linewidth=0.7)
        ax.set_xlabel("Trial time (s)")
        ax.set_ylabel("Velocity (cm/s)")
        ax.set_title("Activity intensity (velocity proxy)")
    else:
        ax.text(0.5, 0.5, "velocity column missing", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
```

#### `struggle_distribution_plot(bouts_by_subject, output_path) -> str`
```python
def struggle_distribution_plot(
    bouts_by_subject: dict[str, list[tuple[float, float]]],
    output_path: str | None = None,
) -> str:
    """eventplot of immobility bouts per subject.

    Each subject = one row of horizontal bars marking [start_time, end_time]
    of immobility bouts. Reveals temporal pattern of giving-up vs struggle.

    Args:
        bouts_by_subject: {"Subject 1": [(start_sec, end_sec), ...], ...}
        output_path: defaults to a temp file.
    """
    _setup_style()
    output_path = _resolve_output_path(output_path, "struggle_distribution")
    subjects = list(bouts_by_subject.keys())
    fig, ax = plt.subplots(figsize=(10, max(2, len(subjects) * 0.6)))
    for i, sub in enumerate(subjects):
        bouts = bouts_by_subject[sub]
        for start, end in bouts:
            ax.broken_barh([(start, end - start)], (i - 0.35, 0.7), facecolors="#B33A3A")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_xlabel("Trial time (s)")
    ax.set_title("Immobility (giving-up) bouts over time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
```

### 6 个 P0 plot 脚本

| 文件 | id | chart 函数 | metrics 模式 |
|---|---|---|---|
| `scripts/fst/plot_bar_immobility.py` | `bar_immobility` | `bar_chart` | groups mode + 多 metric (immobility_time, immobility_bout_count, immobility_latency) |
| `scripts/fst/plot_activity_intensity.py` | `activity_intensity` | `activity_intensity_plot` | 单 subject 直接读 velocity |
| `scripts/fst/plot_struggle_distribution.py` | `struggle_distribution` | `struggle_distribution_plot` | 多 subject 时画 1 行/subject，单 subject 也画 1 行 |
| `scripts/tst/plot_bar_immobility.py` | `bar_immobility` | 同 fst | 同 fst（paradigm="tail_suspension"） |
| `scripts/tst/plot_activity_intensity.py` | `activity_intensity` | 同 fst | 同 fst |
| `scripts/tst/plot_struggle_distribution.py` | `struggle_distribution` | 同 fst | 同 fst |

**柱状图脚本 n<3 行为**: 调用 `bar_chart` 前自己检查 `metrics["group_summary"]` 每组 n；如果有任一组 n<3，临时把 `error_type` 改成空或修脚本内 fallback（推荐方案：脚本里判断后传 `bar_chart(..., suppress_errorbars=True)`——需要 `bar_chart` 支持这个 kwarg；如果不想改 charts.py 就在脚本里用 matplotlib 直接画——见同事 SSOT "柱顶标数值"）。

**推荐实现路径**: 在 `bar_chart` 加 `suppress_errorbars: bool | None = None` kwarg，默认 None = auto（任一组 n<3 自动 suppress），True/False = 强制。这样脚本完全不用判断。同时改动 charts.py + 6 个脚本一次到位。

**bouts 提取**: 写一个 helper `extract_immobility_bouts(df) -> list[tuple[float, float]]`，从 `mobility_state` 列做 run-length encoding；放在 `ethoinsight/metrics/utils.py` 或类似 utility 模块。如果 metrics 包已经有类似函数（看 `metrics/fst.py` 或 `metrics/tst.py`），**先复用别重复实现**。

### 6 个 catalog 注册（fst.yaml / tst.yaml）

```yaml
# fst.yaml charts: 块（追加）
  - id: bar_immobility
    script: ethoinsight.scripts.fst.plot_bar_immobility
    when: total_subjects >= 1
    display_name_zh: "不动行为柱状图（均值±SEM）"

  - id: activity_intensity
    script: ethoinsight.scripts.fst.plot_activity_intensity
    when: total_subjects >= 1
    display_name_zh: "活动强度图"

  - id: struggle_distribution
    script: ethoinsight.scripts.fst.plot_struggle_distribution
    when: total_subjects >= 1
    display_name_zh: "放弃挣扎分布图"
```

tst.yaml 一模一样（除了 `ethoinsight.scripts.tst.*`）。

### 6 个单测

- `tests/test_plot_fst_bar_immobility_cli.py`（n=1 单样本不画误差线 + n=3 多组带误差线 + 不画误差线两条断言）
- `tests/test_plot_fst_activity_intensity_cli.py`（产物 PNG 存在、文件非空）
- `tests/test_plot_fst_struggle_distribution_cli.py`（产物 PNG 存在 + multi-subject 输入也能跑）
- 三个 tst 测试同形

### P0 commit message 模板

```
feat(charts): P0 — fst + tst 各补 3 张图（柱状/活动强度/放弃挣扎）

按 review-packages/2026-05-25-chart-catalog-implementation-plan.md P0 优先级,
补 fst + tst 缺失的 6 张图。

- 新增 charts.py 函数:
  - bar_chart 加 suppress_errorbars kwarg, 任一组 n<3 自动 suppress
  - activity_intensity_plot (velocity 时序面积图)
  - struggle_distribution_plot (immobility bouts eventplot)
- 新增 6 个 plot 脚本: fst/tst.{bar_immobility, activity_intensity, struggle_distribution}
- 新增 helper extract_immobility_bouts (mobility_state run-length encoding)
- catalog 注册: fst.yaml / tst.yaml 各加 3 个 chart 条目
- 6 个单测覆盖 n=1 + n=3

依据: SSOT docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md
同事已拍板参数: n<3 不画误差线柱顶标数值; velocity 作活动强度代理

测试: ethoinsight pytest XXX passed; agent backend pytest 3017 passed
```

---

## P1 详细清单

### 工作内容

zero_maze + ldb 各补 4 类共 8 张图 + 2 个孤儿脚本挂注册：

#### zero_maze (`scripts/zero_maze/`, `catalog/zero_maze.yaml`)
- ✏️ `plot_box_open_zone.py` **已存在但未注册** → 加 catalog 条目即可
- 新增 `plot_bar_open_zone.py` — 柱状图（用 `bar_chart`，metrics: open_zone_time_ratio, open_zone_distance, hesitation_count）
- 新增 `plot_trajectory_zero_maze.py` — 调 `_common.plot_trajectory`（thin wrapper：parse_trajectory → trajectory_plot）
- 新增 `plot_heatmap_zero_maze.py` — 调新加的 `heatmap_plot`

#### ldb (`scripts/ldb/`, `catalog/ldb.yaml`)
- ✏️ `plot_box_light.py` **已存在但未注册** → 加 catalog 条目即可
- 新增 `plot_bar_light.py` — 柱状图（metrics: light_time_ratio, transition_count, light_latency）
- 新增 `plot_trajectory_ldb.py` — 同上
- 新增 `plot_heatmap_ldb.py` — 同上

### 新增 chart 函数（charts.py）

#### `heatmap_plot(df, output_path) -> str`
```python
def heatmap_plot(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """2D KDE heatmap of x_center/y_center positions.

    SSOT 宽高比固定 4:3.

    Args:
        df: parsed trajectory with x_center / y_center columns.
        output_path: defaults to a temp file.
    """
    _setup_style()
    output_path = _resolve_output_path(output_path, "heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))  # 4:3
    if "x_center" in df.columns and "y_center" in df.columns:
        x = pd.to_numeric(df["x_center"], errors="coerce").dropna()
        y = pd.to_numeric(df["y_center"], errors="coerce").dropna()
        hb = ax.hexbin(x, y, gridsize=40, cmap="YlOrRd", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Frame count")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Position density heatmap")
    else:
        ax.text(0.5, 0.5, "x_center/y_center missing", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
```

### catalog 注册示例（zero_maze.yaml）

```yaml
charts:
  - id: box_open_zone
    script: ethoinsight.scripts.zero_maze.plot_box_open_zone
    when: n_per_group >= 3
    display_name_zh: "开放区时间箱线图"

  - id: bar_open_zone
    script: ethoinsight.scripts.zero_maze.plot_bar_open_zone
    when: total_subjects >= 1
    display_name_zh: "开放区指标柱状图（均值±SEM）"

  - id: trajectory
    script: ethoinsight.scripts.zero_maze.plot_trajectory_zero_maze
    when: total_subjects >= 1
    display_name_zh: "轨迹图"

  - id: heatmap
    script: ethoinsight.scripts.zero_maze.plot_heatmap_zero_maze
    when: total_subjects >= 1
    display_name_zh: "热区图"
```

ldb 一样。

### P1 8 个单测

每个新脚本一个 CLI 测试（n=1 + n=3 各一条断言）。

---

## P2 详细清单

epm + oft 各补轨迹图 + 热区图 (4 张)。

**关键**：通用脚本下沉 `_common/`，每个范式只写 thin wrapper：

```python
# scripts/epm/plot_trajectory_epm.py
"""EPM: 轨迹图 thin wrapper, 调 _common.plot_trajectory。"""
from ethoinsight.scripts._common import plot_trajectory
if __name__ == "__main__":
    import sys
    sys.exit(plot_trajectory.main())
```

但其实 catalog 可以直接指向 `_common.plot_trajectory`：

```yaml
# epm.yaml charts: 块（追加）
  - id: trajectory
    script: ethoinsight.scripts._common.plot_trajectory
    when: total_subjects >= 1
    display_name_zh: "轨迹图"

  - id: heatmap
    script: ethoinsight.scripts._common.plot_heatmap
    when: total_subjects >= 1
    display_name_zh: "热区图"
```

oft.yaml 一样。

但是要先确认 `_common.plot_trajectory` 接口跟 catalog 期望的 CLI 参数兼容（catalog.resolve 跑脚本是 `python -m <script> --input <path> --output <path>`，看 `_common/plot_trajectory.py` 是不是支持）。trace 中已经看到它能跑 `--input "/mnt/.../<file>" --output /mnt/user-data/outputs/...`，所以兼容。

新增 `_common/plot_heatmap.py`（用 `heatmap_plot` 函数）。

zero_maze / ldb 在 P1 时也可以直接指向 `_common.plot_trajectory` 和 `_common.plot_heatmap`，**省去 4 个 thin wrapper**。**这是更好的做法**——把 P2 提前到 P1 一起做。但用户的 commit 切分要求是 P1/P2 分两次 push，所以仍保持分开 commit，**只是 P2 的工作量会比预想小**（如果 P1 已经把 thin wrapper 省了直接用 `_common.*`，P2 就只剩 epm/oft 各加 trajectory + heatmap 2 个 catalog 条目，4 行 yaml）。

---

## P3 详细清单

OFT 时间进程图 1 张。

### 新增 chart 函数（charts.py）

#### `time_progress_plot(per_bin_data, output_path) -> str`
```python
def time_progress_plot(
    per_bin_data: list[dict],
    output_path: str | None = None,
) -> str:
    """OFT 时间进程双折线图.

    Args:
        per_bin_data: [{"bin_start_sec": 0, "bin_end_sec": 300, "distance": 12.3,
                        "center_time": 45.2}, ...]
        output_path: defaults to a temp file.
    """
    _setup_style()
    output_path = _resolve_output_path(output_path, "time_progress")
    fig, ax_left = plt.subplots(figsize=(10, 4))
    bin_centers = [(b["bin_start_sec"] + b["bin_end_sec"]) / 2 / 60.0 for b in per_bin_data]
    distance = [b["distance"] for b in per_bin_data]
    center_time = [b["center_time"] for b in per_bin_data]
    ax_left.plot(bin_centers, distance, "o-", color="#2D5F3F", label="Distance moved")
    ax_left.set_xlabel("Time bin center (min)")
    ax_left.set_ylabel("Distance moved (cm)")
    ax_right = ax_left.twinx()
    ax_right.plot(bin_centers, center_time, "s--", color="#B33A3A", label="Center time")
    ax_right.set_ylabel("Center zone time (s)")
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")
    ax_left.set_title("Time-progress: distance + center time per 5-min bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
```

### plot 脚本

`scripts/oft/plot_time_progress.py`:
1. parse_trajectory(input)
2. 计算 `total_duration_seconds = df["trial_time"].max() - df["trial_time"].min()`（或从 metadata）
3. n_bins = `floor(total_duration_seconds / 300)`，bin 长度 5 min（300 s）
4. 末尾不足并入最后一个 bin（即第 N 个 bin 从 (N-1)*300 到 total_duration_seconds）
5. 每 bin 计算：`distance_moved`（用 `compute_distance_moved` 或在 bin 内 sum velocity*dt）+ `center_time`（bin 内 `in_zone_center==1` 的累计秒数）
6. 调 `time_progress_plot(per_bin_data, output_path)`

### catalog 注册

```yaml
# oft.yaml charts: 块（追加）
  - id: time_progress
    script: ethoinsight.scripts.oft.plot_time_progress
    when: total_duration_seconds > 300
    display_name_zh: "时间进程图（5min bin）"
```

注意 `total_duration_seconds` 必须是 `evaluate_when` 已支持的变量。**先 grep 确认**：
```bash
grep -n "total_duration_seconds\|total_subjects\|n_per_group" packages/ethoinsight/ethoinsight/catalog/resolve.py
```
如果没支持，需要在 resolve.py 加（评估上下文里塞这个 key），但这是潜在的额外工作量——**先 grep 看看再决定**。

### 单测
`tests/test_plot_oft_time_progress_cli.py`：构造 600 秒（10min）的 minimal trajectory，断言出 2 bins；构造 350 秒（< 6min）的，断言报错或被 catalog `when` 排除（取决于设计）。

---

## 验证流程（每个 P 完成都要走）

```bash
# 1. ethoinsight 单测
cd packages/ethoinsight && uv run pytest tests/ -q
# 期望: 全过, 没有新失败

# 2. agent backend 回归
cd ../agent/backend && .venv/bin/python -m pytest tests/ -q
# 期望: 3017 passed, 18 skipped, 0 failed (跟 232f12c8 基线一致)

# 3. lint (ethoinsight 没有强制 lint, agent 用 ruff)
# 跳过, agent 这边 ruff lint 跟本任务无关

# 4. 手测一张图（任选一）
cd packages/ethoinsight && uv run python -m ethoinsight.scripts.fst.plot_bar_immobility --inputs /tmp/test_inputs.json --groups /tmp/test_groups.json --output /tmp/test.png
ls -la /tmp/test.png  # 文件存在且 > 5KB
```

通过后：

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/
git commit -m "..."  # 用上面的 commit message 模板
git push origin dev
```

---

## 通用注意事项

1. **paradigm key vs catalog key**: `catalog/fst.yaml` 里 `paradigm: fst`，但 `compute_paradigm_metrics(paradigm="forced_swim")`。两者**不一样**！每个 plot 脚本里要写**正确的 metrics paradigm 字符串**（`forced_swim` / `tail_suspension` / `open_field` / `light_dark` / `epm` / `zero_maze`）。
2. **不要默认参数 paradigm 错乱**: 看 `compute_paradigm_metrics` 的 paradigm 列表确认。
3. **n=1 → bar_chart 不画误差线**: 通过 `bar_chart(..., suppress_errorbars=True/None)` 控制，不要在 plot 脚本里直接调 matplotlib 重新画。
4. **emit_result 必须输出**: catalog.resolve 依赖 `OK: ...` 前缀判断脚本成功。看 `scripts/_cli.py:emit_result` 确认格式。
5. **make_plot_parser 参数**: `supports_groups=True` 给多文件脚本，`supports_groups=False` 给单样本脚本。
6. **绝对不读 raw EthoVision txt**: 全部走 `parse_trajectory(path)` / `parse_batch(paths)`。
7. **测试用 helper `_df_to_ethovision_file`**: 从 `test_plot_epm_single_subject_cli.py` 复制一份到每个新测试文件，**不要抽到 conftest**。
8. **不要碰 `lead_agent/prompt.py`**: 你的工作是 catalog/scripts/charts/tests，跟 lead prompt 完全无关。
9. **不要碰 `subagents/`**: 同上。
10. **不要碰前端**: 同上。

---

## 风险点

- **`evaluate_when` 不支持 `total_duration_seconds`**：P3 要先确认。先 grep 再开工 P3。如果不支持，要么扩 evaluate_when 上下文，要么改 catalog `when` 用其他可用字段。
- **`mobility_state` 列名**：FST/TST 文件实际列名是 `mobility_state_*`（带 dynamic suffix），看 `metrics/fst.py` 实现确认。
- **velocity 列在 EthoVision XT 不一定每个 paradigm 都有**：FST/TST 在圆柱容器内可能没 velocity（看 `Mobility (.+)` dynamic pattern）。先 grep 一个真实 FST 文件确认 velocity 列存在。如果没有 velocity，活动强度图改用 `mobility_continuous_*`。
- **n=1 在 `bar_chart` 的 metrics["group_summary"]` 字段结构**：可能没有 SEM 字段。看 `metrics/dispatcher.py` 输出确认。

---

## 关键文件速查

| 任务 | 文件 |
|---|---|
| SSOT | `docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md` |
| 实施清单 | `docs/review-packages/2026-05-25-chart-catalog-implementation-plan.md` |
| 现有 chart 函数 | `packages/ethoinsight/ethoinsight/charts.py` |
| catalog yaml | `packages/ethoinsight/ethoinsight/catalog/{fst,tst,epm,oft,zero_maze,ldb}.yaml` |
| catalog resolve | `packages/ethoinsight/ethoinsight/catalog/resolve.py` |
| plot 脚本入口模板 | `packages/ethoinsight/ethoinsight/scripts/fst/plot_box_immobility.py` |
| 单样本 plot 模板 | `packages/ethoinsight/ethoinsight/scripts/epm/plot_open_arm_time_ratio_bar.py` |
| 分布柱图模板 | `packages/ethoinsight/ethoinsight/scripts/epm/plot_zone_entry_distribution.py` |
| 单测模板 | `packages/ethoinsight/tests/test_plot_epm_single_subject_cli.py` |
| make_plot_parser | `packages/ethoinsight/ethoinsight/scripts/_cli.py` |
| metrics dispatcher | `packages/ethoinsight/ethoinsight/metrics/dispatcher.py` |
| FST metrics | `packages/ethoinsight/ethoinsight/metrics/fst.py` |
| column normalize | `packages/ethoinsight/ethoinsight/utils.py` (COLUMN_MAP + _DYNAMIC_PATTERNS) |

## 本会话产物速查

- HEAD: `232f12c8` (frontend B2+B3)
- 上一条 commit: `8cea306a` (recursion_limit fix)
- 这两条 commit 内容**跟你工作正交**，不要回头看

---

## 接手第一步（建议）

1. **Read** 本文件全部
2. **Read** SSOT: `docs/review-packages/2026-0521-feedbacks/tstYoyo/范式-图表对应关系.md`
3. **Read** 一个现有 plot 脚本: `packages/ethoinsight/ethoinsight/scripts/fst/plot_box_immobility.py`
4. **Read** 一个现有单测: `packages/ethoinsight/tests/test_plot_epm_single_subject_cli.py`
5. **grep** 风险点 (`total_duration_seconds` / velocity 列存在性 / paradigm key 列表)
6. **开 P0**: 新加 chart 函数 → 6 个脚本 → 2 个 yaml 注册 → 6 个单测 → 跑测试 → commit + push
7. 用户重启服务跑 FST dogfood 验证 P0 → 反馈通过 → 开 P1
8. 重复 P1 → P2 → P3

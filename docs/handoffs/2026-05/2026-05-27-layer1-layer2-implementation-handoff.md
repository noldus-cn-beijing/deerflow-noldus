# 2026-05-27 Layer 1+2 本体补齐 handoff

## 核心认知

我们系统的所有 P0 bug 有一个共同根因：**Python 实现（Layer 3）脱离了 Noldus 官方定义（Layer 1/2）**。

```
Layer 0: EV19 硬件/软件 — 产生原始像素和坐标数据
Layer 1: Noldus 官方因变量定义 — Activity/Mobility/Distance/Heading 等怎么从原始数据算
Layer 2: Noldus 官方 JS 脚本 — 48 个基于因变量的行为学分析方法（immobility/velocity bins/...）
Layer 3: 我们的 Python 实现（_common.py / charts.py）— 对 Layer 2 的翻译
Layer 4: Agent 系统 — 调 Layer 3 代码，给用户出报告
```

**现状**：只有 Layer 3 和 Layer 4。Layer 1 和 Layer 2 不存在于系统中。
**后果**：所有 bug（曼哈顿距离、mobility_state 二值假设、velocity 当 activity）都是 Layer 3 脱离 Layer 1/2 导致的。
**方案**：把 Layer 1 和 Layer 2 物化进系统——不是加新功能，是补缺失的本体层次。

## 已有基础

| Layer | 物化形式 | 状态 |
|-------|---------|------|
| Layer 1 | `ev19-dependent-variables.md`（EV19 因变量公式，14 章） | ✅ 已完成 |
| Layer 1→3 追溯 | `_common.py` docstring 标注 EV19 公式引用 | ✅ 已完成 |
| Layer 2 知识 | Noldus 48 个 JS 算法源码 | ❌ 只下载了 6 个，其余 42 个未获取 |
| Layer 2 目录 | `noldus_js_algorithms.yaml` SSOT | ❌ 未创建 |
| Layer 2→3 追溯 | catalog metric 标注实现的是哪个 Noldus 算法 | ❌ 未做 |
| Layer 4 发现 | lead 能查 Layer 2 的 tool | ❌ 未做 |

## 要做什么

### 核心目标

让系统**知道自己知道什么、知道自己不知道什么、每个实现有权威来源可追溯**。

### 不需要的

- 不改 agent 工作流（主线还是 code→data→chart→report）
- 不新建 subagent
- 不新建 skill
- 不新建 catalog 系统

---

## 实施：Phase 0（1 个 PR）

### 第 1 步：批量抓取 48 个 Noldus JS 源码

Noldus 仓库：`https://github.com/noldus/EthoVision-JavaScriptCustomAnalysis`

3 个目录需要逐个 fetch：
- `Activity (pixel change)/` — 3 个 .txt 文件
- `Single-subject analysis/` — 32 个 .txt 文件
- `Multi-subject analysis/` — 13 个 .txt 文件

方法：对每个文件名，fetch `https://raw.githubusercontent.com/noldus/EthoVision-JavaScriptCustomAnalysis/main/<目录>/<文件名>`

保存路径：`packages/ethoinsight/ethoinsight/noldus_js_algorithms/raw/<id>.js`

共 48 个文件，每个 ≤ 2KB。

### 第 2 步：创建 noldus_js_algorithms.yaml（SSOT）

路径：`packages/ethoinsight/ethoinsight/noldus_js_algorithms.yaml`

一个 YAML 文件，包含 48 条 entry。每条最小字段：

```yaml
algorithms:
  - id: non_movement_bouts_min_duration
    noldus_name: "Non-movement bouts with a minimum duration"
    noldus_dir: "Single-subject analysis"
    category: "immobility"
    output_kind: "state"  # continuous / state / event
    summary_zh: "速度低于阈值且持续 N 帧才记为静止"
    inputs: [x_center, y_center, trial_time]
    applicable_paradigms: [forced_swim, tail_suspension]
    port_status: "not_ported"  # not_ported / ported / wont_port
    ported_in: null
    priority_tier: 1
    js_source_file: "noldus_js_algorithms/raw/non_movement_bouts_min_duration.js"
```

48 个算法的完整清单如下（按 category 分组）：

**Activity（3）**: activity_pixel_change, activity_pixel_change_smoothed, number_of_pixels_changed

**Immobility（1）**: non_movement_bouts_min_duration

**Locomotion（7）**: acceleration_smoothed, distance_moved_cumulative_k, turn_angle_center_point_running_total, turn_angle_head_direction, turn_angle_head_direction_running_total, turn_angle_distance_filter, velocity_bins

**Body（6）**: body_area, body_area_running_average, body_length_direct, body_length_sum_of_segments, center_nose_length, angle_center_nose

**Heading（4）**: head_direction_absolute, heading_smoothed, heading_relative_to_vector, heading_relative_to_vector_binned

**Zone（8）**: find_proportion_zones_visited, percentage_correct_choices, percentage_correct_choices_reached, time_to_reach_percentage_correct, xy_coordinates, xy_coordinates_zone_center, xy_coordinates_rescaled, zone_transitions_sum

**Zone_constraint（3）**: zone_visits_min_duration, zone_visits_two_body_points, non_movement_bouts_min_duration (also in immobility)

**Mobility_device（1）**: mobility_in_device_interval

**Fish_body_orientation（3）**: body_orientation_fish, fish_head_down, fish_head_up

**Multi_distance（4）**: iid_continuous, nnd_continuous, social_contact_any_body_point, social_contact_each_body_point

**Multi_social（4）**: approaching, following, leaving, aggregation_5_subjects

**Multi_arena（5）**: largest_subject_coordinates, more_than_k_in_zone, subjects_in_zone_count, focal_orientation_relative, velocity_high_two_subjects

标记 ported 的已有项（我们代码已经实现了对应功能）：activity_pixel_change（EV19 内置列）、body_area（parser 列映射）、head_direction_absolute（P3 compute_head_direction_stats）、xy_coordinates（parser 列映射）、turn_angle_head_direction（P3 compute_turn_angle_stats）

### 第 3 步：创建 noldus_algorithm_search 工具

**新建文件**: `packages/agent/backend/packages/harness/deerflow/tools/builtins/noldus_algorithm_search_tool.py`

```python
@tool("noldus_algorithm_search")
def noldus_algorithm_search_tool(runtime, query: str, paradigm: str | None = None) -> dict:
    """搜索 Noldus 官方 JS 算法库。
    
    返回匹配的算法条目，写入 workspace 供 subagent read_file。
    """
```

工具逻辑：
1. 加载 `noldus_js_algorithms.yaml`
2. 对 query 做模糊匹配（id / noldus_name / summary_zh 含关键词）
3. 如果有 paradigm，加分匹配 applicable_paradigms 的条目
4. 返回最多 5 条 match
5. 把结果写入 `/mnt/user-data/workspace/noldus_algorithm_matches.json`

**注册工具**:
- `tools/builtins/__init__.py` — 加 import + `__all__`
- `tools/tools.py` — 加 BUILTIN_TOOLS 注册（⚠ 此文件含 Noldus 注册集，sync 上游时严禁整文件覆盖）

### 第 4 步：lead prompt 加 Noldus 咨询场景

**修改文件**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

在 `noldus_rules` 的意图路由段追加：

```
### Noldus 算法库咨询

用户问"能算 X 吗""Noldus 有没有 Y""EthoVision 怎么算 Z"，或请求的指标不在当前 catalog：
  1. 调 noldus_algorithm_search(query=用户关键词, paradigm=当前范式)
  2. matches 全 ported → 告知用户这是已支持的指标（metric id），问要不要算
  3. matches 部分/全 not_ported → 描述算法，告知状态，问要不要详细解释
  4. 用户追问细节 → 派 knowledge-assistant（附 noldus_algorithm_matches.json 路径）
```

### 第 5 步：knowledge-assistant 的 SKILL.md 引导

**修改文件**: `packages/agent/skills/custom/ethoinsight/SKILL.md`

末尾加一段（薄，不复制 SSOT 内容）：

```markdown
## Noldus 官方算法库

当 lead 派遣时附带 `noldus_algorithm_matches.json` 路径，read_file 该文件获取匹配结果。
若需深入解释某算法的公式和用法，read_file 对应的 js_source_file 路径（见 matches 中每条 entry）。
```

### 第 6 步：顺手补 P3 基础注册（前一 handoff 遗留）

4 个 P3 函数（body_elongation_stats, head_direction_stats, turn_angle_stats, rose_plot）已有函数和测试，缺 catalog 注册和 CLI 入口。

**修改文件**:
- `catalog/epm.yaml`: optional_metrics 加 body_elongation_stats, head_direction_stats, turn_angle_stats；charts 加 rose_plot (confidence: optional)
- `catalog/oft.yaml`: optional_metrics 加 body_elongation_stats, turn_angle_stats
- `catalog/zero_maze.yaml`: optional_metrics 加 head_direction_stats；charts 加 rose_plot (confidence: optional)

**新建 CLI 脚本**（参考 `scripts/tst/compute_immobility_time.py` 的结构，每个脚本 ~30 行）：

| 脚本 | 路径 |
|------|------|
| compute_body_elongation_stats | `scripts/epm/` + `scripts/oft/` |
| compute_head_direction_stats | `scripts/epm/` + `scripts/zero_maze/` |
| compute_turn_angle_stats | `scripts/epm/` + `scripts/oft/` |
| plot_rose | `scripts/epm/` + `scripts/zero_maze/` |

每个脚本结构：
```python
from ethoinsight.metrics._common import compute_xxx
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json

def main():
    args = make_compute_parser(description=__doc__).parse_args()
    df = parse_trajectory(args.input)
    value = compute_xxx(df)
    payload = {"metric": "xxx", "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
```

---

## Phase 0 产出清单

| # | 文件 | 操作 |
|---|------|------|
| 1 | `packages/ethoinsight/ethoinsight/noldus_js_algorithms/raw/*.js` | 新建 48 个 |
| 2 | `packages/ethoinsight/ethoinsight/noldus_js_algorithms.yaml` | 新建 48 entries |
| 3 | `packages/ethoinsight/ethoinsight/noldus_js_algorithms/__init__.py` | 新建 |
| 4 | `packages/ethoinsight/ethoinsight/noldus_js_algorithms/fetcher.py` | 新建（YAML 加载 + 模糊搜索） |
| 5 | `packages/agent/backend/packages/harness/deerflow/tools/builtins/noldus_algorithm_search_tool.py` | 新建 |
| 6 | `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py` | 修改（加 import） |
| 7 | `packages/agent/backend/packages/harness/deerflow/tools/tools.py` | 修改（加注册） |
| 8 | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 修改（加咨询场景） |
| 9 | `packages/agent/skills/custom/ethoinsight/SKILL.md` | 修改（加引导段） |
| 10 | `packages/ethoinsight/ethoinsight/catalog/epm.yaml` | 修改（P3 补充） |
| 11 | `packages/ethoinsight/ethoinsight/catalog/oft.yaml` | 修改（P3 补充） |
| 12 | `packages/ethoinsight/ethoinsight/catalog/zero_maze.yaml` | 修改（P3 补充） |
| 13 | `packages/ethoinsight/ethoinsight/scripts/epm/compute_body_elongation_stats.py` | 新建 |
| 14 | `packages/ethoinsight/ethoinsight/scripts/epm/compute_head_direction_stats.py` | 新建 |
| 15 | `packages/ethoinsight/ethoinsight/scripts/epm/compute_turn_angle_stats.py` | 新建 |
| 16 | `packages/ethoinsight/ethoinsight/scripts/epm/plot_rose.py` | 新建 |
| 17 | `packages/ethoinsight/ethoinsight/scripts/oft/compute_body_elongation_stats.py` | 新建 |
| 18 | `packages/ethoinsight/ethoinsight/scripts/oft/compute_turn_angle_stats.py` | 新建 |
| 19 | `packages/ethoinsight/ethoinsight/scripts/zero_maze/compute_head_direction_stats.py` | 新建 |
| 20 | `packages/ethoinsight/ethoinsight/scripts/zero_maze/plot_rose.py` | 新建 |
| 21 | `packages/ethoinsight/tests/test_noldus_algorithms_consistency.py` | 新建（CI lint） |

---

## 验证标准

Phase 0 完成后的验收：

1. `python -c "from ethoinsight.noldus_js_algorithms.fetcher import search; print(len(search('immobility')))"` → 返回 ≥1 条 match
2. `python -m pytest tests/test_noldus_algorithms_consistency.py` → passed
3. lead prompt 中搜索 `noldus_algorithm_search` 能找到咨询场景段
4. knowledge-assistant 的 SKILL.md 搜索 `Noldus 官方算法库` 能找到引导段
5. `python -m pytest tests/ -q` → ethoinsight 436+ passed
6. `make test` → agent backend 3043+ passed

## 后续 Phase 1

Phase 1 开始 port B1-B12 算法（见前一 handoff）。每 port 一个算法：
1. 写 Python 函数 + 测试
2. 在 catalog YAML 注册
3. 在 `noldus_js_algorithms.yaml` 更新 `port_status: ported`

## 注意事项

1. **git 操作必须用 `git -C /home/wangqiuyang/noldus-insight`**（CWD 在符号链接内）
2. **`tools/tools.py` 是 PROTECTED_FILE**（含 Noldus 注册集，sync 上游时严禁整文件覆盖，只做 surgical merge）
3. **Noldus raw 文件 URL 中的空格需要编码**：`Activity%20(pixel%20change)`
4. **SSOT YAML 遵循 MEMORY 的单一权威原则**：不要在 skill 文档里复制算法列表
5. **subagent 不能直接读 Python 包内文件**：所有知识必须通过 lead 工具落到 `/mnt/user-data/workspace/` 后再让 subagent read_file

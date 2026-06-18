# Spec: 2026-06-16 统一 ethoinsight 脚本 I/O 边界 + 修聚合累加器（statistics 读不到 inputs.json + metrics_summary mean/std 恒错）

> 日期：2026-06-16
> 起点：本日 EPM dogfood 复跑（thread `158187ef`，gateway.log 15:36–16:04）。前序六 bug（#1–#6）全合后**首次走到 data-analyst 正常路径**，逐层剥出两个潜伏 bug，外加它们触发的 subagent 连锁失败（data-analyst / report-writer 各三连 `terminated without emitting handoff`）。
> 基线：origin/dev `f5752d65`（含 PR#135 虚拟路径族根治）。
> 实施者：新 worktree（`git worktree add -b feat/ethoinsight-io-boundary-and-aggregator <path> origin/dev`，独立分支）。
> 关联：本 spec 与 PR#135（路径族根治）**正交**——PR#135 让 `resolve_sandbox_path` 在进程内被调时能兜底；本 spec 修的是"该调 resolve 的地方没调"和"聚合器半成品"，两个问题 PR#135 都不涉及也不解决。

---

## 0. 这次 dogfood 暴露了什么（先读——两个潜伏 bug + 连锁）

### 现象（gateway.log 证据）

| 阶段 | 结果 | log 锚点 |
|---|---|---|
| code-executor `run_metric_plan` | 140/140 compute 成功，但 **statistics 失败** → `statistics: {}` | gateway.log:161-162 |
| data-analyst ×3 | **全部 `terminated without emitting handoff`** | gateway.log:275 / 304 / 334 |
| report-writer ×3 | **全部 `terminated without emitting handoff`** | gateway.log:400 / 432 / 494 |
| chart-maker | ✅ 4 张图成功 | gateway.log:262 |

workspace 实测确认：`handoff_data_analyst.json` 与 `handoff_report_writer.json` **都不存在**（从未落盘）。

### 因果链（根因 → 连锁）

```
缺陷 1（read_inputs_json/read_groups_json 不 resolve 自身 path 参数）
   └─ run_metric_plan Step7 statistics 读 /mnt/.../inputs.json → FileNotFoundError
       └─ statistics: {} 落进 handoff_code_executor.json
           └─ data-analyst fast-fail 1a 触发"描述性 partial"路径（这条 prompt 规则正确，不致命）
               │
               └─ 但缺陷 2 让它在 partial 路径上仍手算螺旋 ↓

缺陷 2（metric_aggregation 累加器是半成品：n 对、mean 恒=首个 subject、std 恒=null）
   └─ metrics_summary 报的 mean 与 per_subject 真实 mean 矛盾（实测 0.099 vs 0.253）
       └─ data-analyst 发现数据自相矛盾 → 陷入"验证/重算/怀疑"螺旋
           └─ 12 turn 用完仍漏调 seal → terminated without emitting
               └─ lead 重派 ×3 同样失败 → 放弃，反问用户
                   └─ report-writer 拿不到 handoff_data_analyst.json → 同样三连失败
```

**关键判断**：data-analyst 这次的 seal 失败**不是** `feedback_subagent_seal_deadlock_is_prompt_not_budget` 说的"纯 prompt 不可靠"——LLM 的怀疑**有道理**（上游数据真矛盾）。若只加更强的 seal 兜底而不修上游，data-analyst 会"准时 seal 但基于错误 mean 给错误结论"，危害更大。**治本在 ethoinsight 层让数据自洽**，而非在 prompt 层教 LLM 忍耐矛盾。

---

## 1. 缺陷 1：`read_inputs_json` / `read_groups_json` 不 resolve 自身的 path 参数

### 现状（不对称的 I/O 边界）

`packages/ethoinsight/ethoinsight/scripts/_cli.py` 的三个 I/O helper：

| 函数 | 行 | 对 `path` 参数 resolve？ | 对内容条目 resolve？ |
|---|---|---|---|
| `save_output_json` | :146 | ✅ `Path(resolve_sandbox_path(path))` | —（写入） |
| `read_inputs_json` | :170 | ❌ `Path(path)` 直接读 | ✅ 条目经 resolve |
| `read_groups_json` | :188 | ❌ `Path(path)` 直接读 | ✅（但格式错，见缺陷 1b） |

**#2（PR#131）当时只修了 `parse_trajectory` + `save_output_json`，漏了这两个读函数**。`test_cli_resolves_sandbox_paths.py` 的 spec 注释自己写"与 read_inputs_json/read_groups_json 对称"，但代码没兑现、测试也没覆盖——注释承诺对称、代码漏改、测试假绿。

### 为什么 compute 成功、statistics 炸

- compute 脚本（如 `compute_open_arm_time.py:33`）用 `parse_trajectory(args.input)`——单文件 uploads，`parse_trajectory` 内部已 resolve（#2 修过）。
- statistics 脚本（`run_groupwise_stats.py:39`）用 `read_inputs_json(args.inputs)`——读 workspace 下的 **inputs.json 这个文件本身**，而 inputs.json 自身的 `/mnt` 路径**没人 resolve** → `Path('/mnt/user-data/workspace/inputs.json').read_text()` → FileNotFoundError。

两条路径用不同函数，#2 只修了 compute 那条。

### 实测复现（已坐实）

设好与 `_build_path_env` 完全一致的 env（含 `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`），`resolve_sandbox_path('/mnt/user-data/workspace/inputs.json')` 正确解析、文件存在；但 `read_inputs_json` 仍 `FileNotFoundError: '/mnt/user-data/workspace/inputs.json'`（虚拟路径）。给 `path` 也包 `resolve_sandbox_path` 后读成功（28 个 raw_files）。

> 注：statistics 在父进程内跑、`_scoped_path_env` 已设 env（run_metric_plan_tool.py:237），所以**不是 env 缺失**——是函数根本没对 `path` 参数调 resolve。PR#135 的 `workspace_base` 兜底也救不了"不调用"。

### 缺陷 1a 修复：对称化（path 参数也 resolve）

`packages/ethoinsight/ethoinsight/scripts/_cli.py`，两个读函数入口各加一行：

```python
def read_inputs_json(path: str | Path) -> list[str]:
    p = Path(resolve_sandbox_path(path))          # ← 改：原 Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    ...

def read_groups_json(path: str | Path) -> dict[str, list[str]]:
    p = Path(resolve_sandbox_path(path))          # ← 改：原 Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    ...
```

**行为等价性**：`resolve_sandbox_path` 是 fail-safe 幂等的——真实路径原样返回、`/mnt` 匹配不到 env 也原样返回。对喂真实路径 / bash-mounted 路径 / 已 resolve 路径的现有调用方零变化（守 `test_resolve_is_idempotent_and_failsafe` 既有契约）。

### 缺陷 1b：`read_groups_json` 期望的 groups 格式与 SSOT 不符

即便修了 1a，statistics 还会错。三处对 `groups.json` 的格式认知**不一致**：

| 消费方 | 位置 | 认为的格式 |
|---|---|---|
| **`prep_metric_plan` 写**（SSOT 来源） | prep_metric_plan_tool.py:235 | `dict[str, str]`（`subject_file → group_name`），且有类型校验 |
| **`metric_aggregation.py` 读** | metric_aggregation.py:182-183 | `dict[str, str]`（`subject_file → group_name`），与写一致 ✅ |
| **`read_groups_json` 读**（statistics 链） | _cli.py:188 docstring | `dict[str, list[str]]`（`group_name → [files]`），与实际文件**不符** ❌ |

实测 groups.json 内容：`{"/mnt/.../Trial 1.xlsx": "control", ...}`（28 条 file→group）。`read_groups_json` 透传成 `{k: [resolve(s) for s in v]}`，但 `v` 是字符串（组名）不是 list——会抛 `TypeError`（迭代字符串）或产出错误结构。即便不抛，喂给下游 `compute_paradigm_metrics(groups)`（dispatcher.py:88/169 期望 `{group: [subjects]}`）会把**文件路径当组名、组名当 subject**，彻底错乱。

**SSOT 裁定**：格式以**写方 + 主消费方**为准 = `dict[str, str]`（`subject_file → group_name`）。错的 `read_groups_json`：它应读 SSOT 格式并**反转**成下游期望的 `{group: [files]}`，而非期望文件本身就是反转后的形态。

### 缺陷 1b 修复：`read_groups_json` 读 SSOT 格式 + 反转

```python
def read_groups_json(path: str | Path) -> dict[str, list[str]]:
    """读 groups.json 并返回 {group_name: [subject_path, ...]}。

    SSOT 文件格式（prep_metric_plan 写）：``{subject_file: group_name}``（flat map）。
    本函数读入后反转成下游 compute_paradigm_metrics/compare_groups 期望的
    ``{group_name: [subject_path, ...]}``。subject 路径经 resolve_sandbox_path 解析。
    """
    p = Path(resolve_sandbox_path(path))
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object, got {type(data).__name__}")

    # 兼容两种输入：① SSOT {file: group}（反转）② 遗留 {group: [files]}（直通）。
    # 用首个 value 的类型判别：str → SSOT flat map，反转；list → 已是目标形态，直通。
    inverted: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            # 遗留形态 {group: [files]}：直通 + resolve
            inverted[k] = [str(resolve_sandbox_path(s)) for s in v]
        else:
            # SSOT 形态 {file: group}：反转
            group = str(v)
            inverted.setdefault(group, []).append(str(resolve_sandbox_path(k)))
    return inverted
```

> **为什么不直接统一文件格式成 `{group:[files]}`**：会破坏 prep（写方）+ metric_aggregation（主读方）已对齐的 `{file:group}` 契约，且 prep 的 `dict[str,str]` 类型校验、聚合器的 `groups.get(subject_file)` 查询都依赖 flat map。改文件格式 = 三处同改、回归面大；只改 `read_groups_json` 一处反转 = 局部、SSOT 不动（守"绝不双存"——格式只存一份，反转是函数内的派生视图）。

---

## 2. 缺陷 2：`metric_aggregation` 累加器是半成品（mean 恒=首个 subject、std 恒=null）

### 现状

`packages/agent/backend/packages/harness/deerflow/subagents/metric_aggregation.py:230-242`：

```python
if metric_name not in metrics_summary[group_name][metric_name]:
    metrics_summary[group_name][metric_name] = {"mean": value, "std": None, "n": 1, ...}  # 首个 subject
else:
    existing = metrics_summary[group_name][metric_name]
    existing["n"] = (existing.get("n") or 0) + 1   # ← 只 n+=1
    # ← mean 从不更新！std 从不计算！
```

### 实测（control 组 open_arm_time_ratio，7 个 subject）

| 量 | 值 |
|---|---|
| per_subject 值 | `[0.099, 0.084, 1.0, 0.258, 0.133, 0.138, 0.059]` |
| 真实算术 mean | **0.253** |
| handoff 报的 mean | **0.099**（恰好 = 首个 subject Trial 1 的值） |
| handoff 报的 std | **null** |
| handoff 报的 n | 7（对） |

`mean` 恒等于该组**第一个被处理 subject 的值**，`std` 恒 `null`——累加器骨架写了但没填。这是 data-analyst 手算螺旋的直接诱因：它读到 `mean=0.099` 但自己算出 `0.253`，两者矛盾。

### 修复：用列表累加，末尾算 mean/std

聚合循环改成收集 values，循环结束后算统计量。最小改动：在 `existing` 里维护一个 `values` 临时列表（或 running sum/sumsq），最后回填 `mean`/`std` 并删掉临时字段。

```python
# 循环内（替换 :233-242）
if metric_name not in metrics_summary[group_name]:
    metrics_summary[group_name][metric_name] = {
        "_values": [value],          # 临时累加，末尾清除
        "mean": None, "std": None, "n": 1,
        "parameters_used": params,
    }
else:
    existing = metrics_summary[group_name][metric_name]
    existing["_values"].append(value)
    existing["n"] = len(existing["_values"])

# 循环结束后（return 前，新增一步）
import statistics as _stats
for group_metrics in metrics_summary.values():
    for stat in group_metrics.values():
        vals = stat.pop("_values", None)
        if vals:
            vals_clean = [v for v in vals if v is not None]
            stat["n"] = len(vals_clean)
            stat["mean"] = _stats.mean(vals_clean) if vals_clean else None
            stat["std"] = _stats.stdev(vals_clean) if len(vals_clean) >= 2 else None
```

**语义裁定**（spec 显式锁定，避免实施者自由发挥）：
- **mean** = 算术平均（`statistics.mean`），忽略 `None`（不适用的 subject）。
- **std** = **样本标准差**（`statistics.stdev`，n−1 分母），`n<2` 时 `None`。与 scipy/pandas 的 `ddof=1` 默认一致，是行为学统计惯例。
- **n** = 非 None 值的个数（applicable subject 数）。
- `None` 值（compute 脚本报"不适用"）不计入 mean/std/n——与下游 `MetricStat.applicable` 语义对齐。
- 保留 `parameters_used`（首个 subject 的，不变）。

> **为何不抽成纯函数先单测**：聚合逻辑当前内联在 `aggregate_metrics_to_handoff` 的循环里。本次保持最小改动（内联修复 + 契约测试），不强行重构。若实施中发现 values 累加在多 metric 时有边界问题，**可**抽 `_compute_stat(values) -> {mean,std,n}` 纯函数测试，但非强制。

---

## 3. 边界（不做什么——守根因隔离纪律）

1. **不动 PR#135 的 `workspace_base` 兜底**——本 spec 的缺陷 1 是"没调 resolve"，env/兜底都救不了"不调用"；修法是补调用，不是加兜底。
2. **不改 groups.json 文件格式**——SSOT 是 `{file:group}`（写方+主读方已对齐），只改 `read_groups_json` 在函数内反转（派生视图，非双存）。
3. **不动 data-analyst / report-writer 的 prompt**——它们的 seal 失败是上游数据矛盾的连锁后果，上游修好即消。不在根因未隔离前叠加 prompt 兜底（守 `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`）。
4. **不删 `_scoped_path_env`**（Step7/Step8 包 env）——那是 PR#135 双保险的 layer③，与本 spec 正交。
5. **不改 statistics 的"读 handoff_code_executor.json"plan 遗留 `input` 字段**——run_metric_plan_tool.py:218-230 已手构 argv 绕过它（注释写明），与本 spec 无关。
6. **不碰 compute 脚本**（它们用 `parse_trajectory`，#2 已修，本次 140/140 全成功）。

---

## 4. 测试（全 importlib 加载 worktree 源，守 worktree 共享主仓 venv editable 指向主仓的铁律）

### 层 A — 缺陷 1a（read 函数 resolve path 参数）

新文件 `packages/ethoinsight/tests/test_cli_read_helpers_resolve.py`：

- `test_read_inputs_json_resolves_mnt_path`：写真实 `real_ws/inputs.json`，设 `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`，调 `read_inputs_json('/mnt/user-data/workspace/inputs.json')` → 读成功、返回的条目也已 resolve。
  - **red 锚点**：修复前 `FileNotFoundError: '/mnt/user-data/workspace/inputs.json'`（虚拟路径直接读）。
- `test_read_groups_json_resolves_mnt_path`：同上，对 groups.json。
- `test_read_helpers_idempotent_on_real_path`：喂真实路径（无 `/mnt`）零变化，守护 fail-safe。

### 层 B — 缺陷 1b（groups 格式反转）

同文件或 `test_cli_read_groups_inverts_format.py`：

- `test_read_groups_inverts_flat_map`：写 `{"/mnt/.../Trial 1.xlsx": "control", "/mnt/.../Trial 8.xlsx": "treatment"}`，读回 → `{"control": [<resolved Trial1>], "treatment": [<resolved Trial8>]}`。
  - **red 锚点**：修复前要么抛 `TypeError`（迭代字符串），要么产出 `{"<file>": ["c","o","n","t","r","o","l"]}`（把组名字符串当可迭代拆字符）。
- `test_read_groups_passes_through_legacy_group_to_list`：喂遗留 `{"control": ["f1", "f2"]}` → 直通成 `{"control": [<resolved f1>, <resolved f2>]}`（兼容性守护）。
- `test_read_groups_empty_dict`：`{}` → `{}`（不炸）。

### 层 C — 缺陷 2（聚合累加器）

新文件 `packages/agent/backend/tests/test_metric_aggregation_stats.py`（importlib 加载 worktree `metric_aggregation.py`）：

构造 plan + 3 个 m_*.json（同组、同 metric、值 `[2.0, 4.0, 6.0]`）+ groups.json，调 `aggregate_metrics_to_handoff`，断言：
- `mean == 4.0`（算术平均）
- `std == 2.0`（样本标准差，ddof=1）
- `n == 3`
- **red 锚点**：修复前 `mean == 2.0`（首个值）、`std is None`、`n == 3`（n 对、mean/std 错）。

多组守护：
- `test_aggregation_multi_group`：control `[1,2,3]`、treatment `[4,5,6]` 两组 → 各自 mean/std/n 正确、互不污染。
- `test_aggregation_skips_none_values`：某 subject metric 值为 `None` → 不计入 mean/std/n。
- `test_aggregation_single_subject_std_none`：组内仅 1 个值 → `std is None`、`mean == 该值`、`n == 1`。
- `test_aggregation_per_subject_unchanged`：per_subject dict 仍逐 subject 记原值（聚合不破坏明细，守护 data-analyst 仍能读 per_subject）。

### 层 D — 端到端 statistics 链（回归证明 1a+1b 联合修好）

`packages/agent/backend/tests/test_run_metric_plan_statistics.py`（复用 `test_run_metric_plan.py` 的 fixture 构造法）：

- 构造一份**有 statistics 段**（`skip_reason=None`）的 plan + groups.json（SSOT `{file:group}` 格式）+ 若干 compute 产物，调 `run_metric_plan`（或直接 `_run_metric_task` 跑 `run_groupwise_stats`），断言：
  - `statistics != {}`（不再是空，statistics 真跑了）
  - `statistics["comparisons"]` 非空、含 EPM 5 指标
  - **red 锚点**：修复前 `failures` 含 `{"id": "statistics", ...}`、`statistics == {}`。

> 这层是本次的**核心价值证明**——它把"statistics 从未跑通"变成一条常驻回归。任何未来让 statistics 再断的改动都会红。

### 层 E — 回归守护

- 跑 ethoinsight 全量（`pytest packages/ethoinsight/tests/`）——预期 844+ passed 不回归（基线 origin/dev）。
- 跑 backend `test_run_metric_plan*.py`、`test_cli_resolves_sandbox_paths.py`——不回归。
- **裸导入**（守 harness 闭环）：`PYTHONPATH=. python -c "import app.gateway"` + `from deerflow.agents import make_lead_agent`（本次只动 ethoinsight + metric_aggregation 纯聚合，预期不触发闭环，但守铁律必跑）。

---

## 5. 实施顺序与验收

1. **缺陷 2 先行**（聚合器）——它是 data-analyst 连锁失败的根因，且纯函数、测试最干净。先红→绿层 C。
2. **缺陷 1a**（read resolve）——红→绿层 A。
3. **缺陷 1b**（groups 反转）——红→绿层 B。
4. **层 D 端到端**——证明 1a+1b 联合让 statistics 跑通。
5. 全量回归（层 E）+ 裸导入。

**验收 = dogfood 复跑 EPM（merge 后）**：
- ✅ `run_metric_plan` 的 `statistics` 非空（不再 FileNotFoundError）。
- ✅ `metrics_summary[group][metric]` 的 `mean`/`std` 与 per_subject 重算一致（data-analyst 不再有理由手算螺旋）。
- ✅ data-analyst 正常 seal（不再 terminated without emitting）。
- ✅ 走到 report-writer 并产出报告（本批首次跑通到 report-writer）。

> 回归探针：metrics_summary 正确性应作为常驻断言纳入层 C——它是"数据自洽"这一 harness 不变量的守住点，把"mean 对不对"从 LLM 责任变成代码不变量。

---

## 6. 关联 memory（实施者先读）

- `feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path`（#2 原始 spec，本 spec 是它漏改的补全）
- `feedback_subagent_seal_deadlock_is_prompt_not_budget`（seal 死锁历史根因——本批 data-analyst 同症状但异根因：上游数据矛盾，先修上游）
- `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`（测试加载铁律）
- `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（§3 边界依据）
- `feedback_single_source_of_truth`（缺陷 1b 不改文件格式、只函数内反转的依据）

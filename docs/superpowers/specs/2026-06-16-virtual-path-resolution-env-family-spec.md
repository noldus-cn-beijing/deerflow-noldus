# Spec: 2026-06-16 根治"虚拟路径解析依赖 env"故障族 — 让 ethoinsight 在进程内被调时不再静默读不到 /mnt 文件

> 日期：2026-06-16
> 起点：review #6a 时发现 `_derive_group_counts` 对主调用方 prep（虚拟路径 + 无 env）失效（已点修，origin/dev `868ed712`/PR#134）。用户判断："这个缺陷很可能导致之后 agent 运行中 handoff 出问题"——**正确**。本 spec 不再点修，而是**收口整个故障族的根因**，防止它在未来新代码里反复复发。
> 基线：origin/dev `868ed712`（含 #2/#3/#5/#6a/#6b 全部点修）。
> 实施者：新 worktree（`git worktree add -b feat/virtual-path-resolution-family <path> origin/dev` 独立分支，别共享 dev）。

---

## 0. 故障族的本质（先读——这不是某一个 bug，是一类 bug）

EthoInsight 有**两套**"虚拟路径 `/mnt/...` → 真实路径"的解析机制，依赖的输入不同：

| 机制 | 位置 | 依赖 | 可靠性 |
|---|---|---|---|
| **A. `replace_virtual_path(path, thread_data)`** | harness `sandbox/tools.py:486` | `thread_data`（含 workspace_path 等，**总是有**） | ✅ 可靠，无 env 依赖 |
| **B. `resolve_sandbox_path(path)`** | ethoinsight `scripts/_cli.py:77` | **`DEERFLOW_PATH_*` 环境变量** | ⚠️ env 没设就静默 fail-safe 原样返回 `/mnt`（读不到） |

机制 B 的设计前提是"ethoinsight 脚本运行在**沙箱/进程池子进程内**，那里 env 已被设好"（bash sandbox 经 mount、run_metric_plan worker 经 `_worker_init`）。这个前提在脚本被当**子进程**跑时成立。

**故障族 = 机制 B 的代码被"进程内直接调用"（不经子进程、不经沙箱），而调用方没设 `DEERFLOW_PATH_*` env。** 此时 `resolve_sandbox_path` 静默原样返回 `/mnt` 路径 → `read_text` 失败 → 该读的文件读不到 → 下游静默退化（统计跳过 / 校验误报 / handoff 残缺）。

**已发生的两次（都点修了，是本族的两个实例，不是孤例）**：
- **#5**：`run_metric_plan` Step8 在父进程调 `aggregate_metrics_to_handoff(run_validation=True)` → `validate_catalog.validate_plan_results` → `resolve_sandbox_path` 读 plan 的 `/mnt` output → 父进程无 env → 全指标误报 `result_file_unreadable` → 毒化 data-analyst。点修=Step8 包 `_scoped_path_env`。
- **#6a**：`prep_metric_plan` 在进程内调 `resolve_metrics(groups_file=虚拟路径)` → `_derive_group_counts` → `resolve_sandbox_path` 读 `/mnt` groups.json → prep 无 env → 读不到 → 自派生失效。点修=`_derive_group_counts` 加 `workspace_dir` 兜底。

**为什么会一再复发**：机制 B 的函数（`resolve_sandbox_path`、`parse_trajectory`、`save_output_json`、`read_inputs_json`、`read_groups_json`、`validate_plan_results`）**签名上看不出它依赖 env**——调用方以为"传个路径就能读"，不知道还要先设 env。每个新的"harness 进程内直接调 ethoinsight"的调用点，都是一颗潜伏雷。用户担心的"未来 handoff 出问题"就是这个：**只要有人在 harness 里进程内调一个读 /mnt 的 ethoinsight 函数而忘了设 env，就静默炸**，且 pytest 喂真实路径测不出来（#6a 的 layer-A 测试就喂了真实路径、假性证明覆盖）。

---

## 1. 修复目标与策略（用户裁决：根治路径不一致族）

**目标**：让"ethoinsight 在进程内被调时读 /mnt 文件"**不再依赖调用方记得设 env**——要么解析不依赖 env，要么没设 env 时**响亮失败**（不静默原样返回造成下游哑退化）。

**策略：双管（防御 + 可观测），不改机制 B 的 env 优先路径（兼容沙箱子进程）**：

### 改动 ①（根治）：`resolve_sandbox_path` 增加"无 env 兜底基址"参数

让 `resolve_sandbox_path` 接受一个可选的 `workspace_base`（真实 workspace 物理路径），当 `/mnt/user-data/workspace/...` 匹配不到 env 时，用 `workspace_base` 兜底解析。这把 #6a 的兜底从"_derive_group_counts 一处的补丁"上升为"机制 B 本身的能力"——所有进程内调用方只要能拿到 thread_data/workspace_dir，传给它即可可靠解析，不依赖 env。

```python
# scripts/_cli.py
def resolve_sandbox_path(path: str | Path, workspace_base: str | Path | None = None) -> Path:
    """把 /mnt/<x>/... 虚拟沙箱路径解析成真实路径。

    解析优先级：
      1. DEERFLOW_PATH_* env（沙箱子进程/进程池 worker 内：env 已设，原路径）。
      2. workspace_base 兜底（进程内被 harness 直接调、未设 env 时：用真实 workspace
         物理路径拼 /mnt/user-data/workspace 下的后缀）。仅对 workspace 前缀生效——
         其他前缀（uploads/outputs）若无 env 且无对应兜底，仍 fail-safe 原样返回。
      3. 都没有 → 原样返回（fail-safe，与历史等价）。
    """
    p = str(path)
    if not p.startswith("/mnt/"):
        return Path(p)
    for prefix in _KNOWN_SANDBOX_PREFIXES:
        if p.startswith(prefix + "/") or p == prefix:
            env_key = _sandbox_env_key_for_prefix(prefix)
            real_base = os.environ.get(env_key)
            if real_base is not None:
                suffix = p[len(prefix):].lstrip("/")
                return Path(real_base) / suffix if suffix else Path(real_base)
            # env 未设：对 workspace 前缀用 workspace_base 兜底
            if workspace_base and prefix == "/mnt/user-data/workspace":
                suffix = p[len(prefix):].lstrip("/")
                return Path(workspace_base) / suffix if suffix else Path(workspace_base)
    return Path(p)
```

> **为什么只对 workspace 前缀兜底**：harness 进程内调 ethoinsight 读的虚拟路径几乎都在 workspace（groups.json / plan / m_*.json / inputs.json 都在 workspace）。uploads 的兜底由 prep 已经做的 `replace_virtual_path` 在传入前解决（prep 传真实 uploads 路径给 parse）。保守只扩 workspace，不过度泛化。

### 改动 ②（把已点修的下游补丁，改成走机制 B 的新能力，去重）

- `resolve._derive_group_counts(groups_file, workspace_dir)` 已有的 `workspace_dir/basename` 兜底，**改成调 `resolve_sandbox_path(groups_file, workspace_base=workspace_dir)`**——同一处方收口到机制 B，不在 resolve 里维护第二套兜底逻辑（消除双存）。行为等价（实测同样 (7,2)）。
- 同理把 `_derive_group_counts` 里"直接当物理路径读 + resolve_sandbox_path + workspace_dir 拼"三候选简化为一次 `resolve_sandbox_path(groups_file, workspace_dir)` + 一次 `Path(groups_file)` 物理兜底。

### 改动 ③（防御信号——让无 env 静默退化变可观测，覆盖整族）

机制 B 的"无 env → 原样返回 /mnt"是静默的。加一条**可观测信号**：当 `resolve_sandbox_path` 收到 `/mnt` 路径、既无 env 又无 workspace_base 兜底、最终原样返回时，`logger.debug`（不是 warning——正常沙箱外测试也会走这条，不该刷 warning）记一行"虚拟路径未解析（无 env/无兜底），原样返回 %s"。这不改变行为，但给未来排查"读不到 /mnt 文件"提供 grep 锚点。

> **不把它做成 raise/warning**：因为非沙箱环境（如直接跑测试、CLI 本地调试）合法地走"原样返回"路径，raise 会破坏它们。debug 级日志是"可观测但不打扰"的正确档位（守 `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`：响亮要分场景，这里合法静默路径不该响亮）。

---

## 2. 不做什么（划清边界，避免过度收口）

- **不动机制 A（`replace_virtual_path`）**：它本来就可靠（thread_data 无 env 依赖）。harness 工具继续用它在"传给 ethoinsight 前"预解析 uploaded_files——这是对的模式，prep/inspect/identify 都这么做。
- **不强行统一 A/B 成一套**：A 在 harness 层（有 thread_data）、B 在 ethoinsight 层（库不该知道 harness 的 thread_data 类型）。两者分属不同包、依赖不同输入，强行合并会让 ethoinsight 反向依赖 harness（违分层）。**正确收口点是"让 B 不依赖 env 也能用"（改动①），不是"让 B 变成 A"。**
- **不改 run_metric_plan Step8 / Step7 的 `_scoped_path_env`（#5 点修）**：那里 env 设置是对的、且 worker/stats 路径也需要 env 给真子进程脚本用。#5 点修保留——它和本 spec 正交（本 spec 让父进程内的调用即便忘了 _scoped_path_env 也能靠 workspace_base 兜底，是双保险，但不删 #5 的 _scoped_path_env）。
- **不动 data-analyst #6b prompt**：与路径族无关。

---

## 3. red 测试锚点

**层 A（机制 B 本体，ethoinsight，importlib 加载 worktree 源）**：
```python
def test_resolve_sandbox_path_env_priority(monkeypatch, tmp_path):
    """env 设了 → 用 env（沙箱子进程路径不变）。"""
    real = tmp_path / "ws"; real.mkdir()
    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", str(real))
    assert resolve_sandbox_path("/mnt/user-data/workspace/g.json") == real / "g.json"

def test_resolve_sandbox_path_workspace_base_fallback_no_env(monkeypatch, tmp_path):
    """red 锚点：无 env + 有 workspace_base → 兜底解析（进程内被 harness 调的路径）。
    修复前：无 env → 原样返回 /mnt（读不到）。"""
    monkeypatch.delenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", raising=False)
    real = tmp_path / "ws"; real.mkdir()
    got = resolve_sandbox_path("/mnt/user-data/workspace/g.json", workspace_base=str(real))
    assert got == real / "g.json"

def test_resolve_sandbox_path_no_env_no_base_passthrough(monkeypatch):
    """守护：无 env 无兜底 → 原样返回（非沙箱/测试合法路径，不引入新失败）。"""
    monkeypatch.delenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", raising=False)
    assert str(resolve_sandbox_path("/mnt/user-data/workspace/g.json")) == "/mnt/user-data/workspace/g.json"

def test_resolve_sandbox_path_real_path_unchanged(tmp_path):
    """守护：真实路径原样返回（fail-safe 幂等）。"""
    f = tmp_path / "x.json"
    assert resolve_sandbox_path(str(f)) == f
```

**层 B（_derive_group_counts 收口后仍对，且生产保真路径仍绿）**：复用 #6a 已有的 `test_resolve_self_derives_with_virtual_groups_file_no_env`（虚拟 groups_file + 无 env + workspace_dir）——改动②后必须仍绿（证收口到机制 B 行为等价）。

> 回退验证：去掉改动① 的 workspace_base 兜底分支 → 层 A 的 fallback 测试 + 层 B 生产保真测试应红（与 #6a review 实测一致）。

**层 C（族回归探针 + 文档）**：在 `resolve_sandbox_path` docstring 写明"harness 进程内直接调 ethoinsight 读 /mnt 文件时，必须传 workspace_base（或先用 replace_virtual_path 预解析），不能依赖 env"——把隐性契约变显性，防未来调用方再踩。

---

## 4. 验证 + 红线
- `cd packages/ethoinsight && pytest tests/`（层 A/B）；`cd packages/agent/backend && PYTHONPATH=. pytest tests/`（确认 #5/#6a 既有测试不回归）。
- 改 `_cli.py`（机制 B 核心，被 parse/validate/resolve 全调）后跑 **ethoinsight 全量 + backend 既有 run_metric_plan/prep/validate 相关**，确认无回归。
- 裸导入 `import ethoinsight` + harness 两入口（`_cli` 改动面广）。
- **基线债**：backend ~6 failed 是 origin/dev 同 venv 既有（守 `feedback_known_full_suite_test_pollution_4_tests`），别归因本次。
- **行为等价红线**：改动① 的 env 优先路径**字节不变**（沙箱子进程、run_metric_plan worker 行为不能变）；workspace_base 只在"无 env"时才生效——确认现有所有"有 env"的调用方零行为变化。
- **不删 #5 的 `_scoped_path_env`、不删 #6a 的 prep 显式派生**（layer③ 双保险保留）。

## 5. 为什么这次是"根治"而非又一个点修
- #5/#6a 是"在每个出血点各打一个补丁"（Step8 包 env、_derive 加 workspace_dir）。本 spec 把**机制 B 本身**升级成"不依赖 env 也能可靠解析 workspace 路径"，于是：
  - 已有两个补丁收口到一处（改动②去重）。
  - **未来**任何"harness 进程内调 ethoinsight 读 workspace /mnt 文件"的新代码，只要传 workspace_base（thread_data 里就有），自动可靠——不再是潜伏雷。
  - 隐性契约显性化（docstring + debug 信号），新调用方有迹可循。
- 这正是用户"防止之后 agent 运行 handoff 出问题"的诉求：不是修当前这一个，是让这一类不再能悄悄发生。

## 6. 关键文件/行号
- 改动①③：`packages/ethoinsight/ethoinsight/scripts/_cli.py:77`（`resolve_sandbox_path`）
- 改动②：`packages/ethoinsight/ethoinsight/catalog/resolve.py:1079`（`_derive_group_counts`，收口到机制 B）
- 族证据/不动点：`run_metric_plan_tool.py:57`（`_scoped_path_env`，#5，保留）、`prep_metric_plan_tool.py:272`（传 groups_file_virtual，保留——靠 resolve 兜底读）、`sandbox/tools.py:486`（机制 A，不动）
- 机制 B 全调用面（验证无回归）：`parse/_core.py:89`、`_cli.py:122`（save_output_json）、`validate_catalog.py:364`、`resolve.py:1110`

## 7. 关联 memory
- `feedback_run_metric_plan_step8_validation_not_scoped_path_env`（#5，本族实例）
- `feedback_prep_metric_plan_stats_skip_none_counts_poisons_data_analyst_seal`（#6a，本族实例 + review-fix）
- `feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path`（#2，机制 B 在脚本侧的早期实例）
- `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`（测试必须 importlib 加载 worktree 源 + 喂主调用方真实入参形态，别喂真实路径假性证明覆盖）
- `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（信号分场景：合法静默路径不该 raise/warning）

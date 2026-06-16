# Spec: 2026-06-15 run_metric_plan Step8 validation 路径 resolve 修复（#5，dogfood 复跑暴露）

> 日期：2026-06-15（前序四 bug 修复 PR merge 后 dogfood 复跑暴露）
> 起点：EPM dogfood（`Raw data-EPM-Xuhui-28`），#1/#2/#3 修复已验证生效，但 `run_metric_plan` 仍报 5 条 `result_file_unreadable` critical 误报，毒化下游 data-analyst。
> 关联：本 bug 是 `2026-06-15-epm-dogfood-four-bug-fixes-spec.md` 的 #2（路径 resolve）**同根源家族的另一个站点**，但站点不同、独立修复。
> 实施者：新 worktree（`git worktree add -b feat/run-metric-plan-step8-env <path> dev` 独立分支，别共享 dev）。

---

## 1. 现象（dogfood 复跑实证）

`run_metric_plan` 返回 **自相矛盾**的结果：
- `status=completed`、`n_total=140 n_completed=140 n_failed=0`、`failures=[]`
- `metrics_summary` / `per_subject` **真实数据齐全**（对照组开臂时间比 9.9% vs 实验组 19.8%，per-subject 全有值）
- **同时** gate_signals 有 **5 条 `result_file_unreadable` critical 警告**，覆盖全部 5 个指标（open_arm_time_ratio / open_arm_time / open_arm_entry_count / open_arm_entry_ratio / total_entry_count），且 `blocks_downstream: true`

下游连锁：data-analyst 的 fast-fail 规则（`data_analyst.py:84-87`）把"critical METRIC_VALIDATION 覆盖所有指标"判为 "all metrics invalid → status=failed"，于是 data-analyst 长篇纠结后 seal 失败。**整条分析链被这 5 条误报毒死。**

## 2. 根因（已逐层坐实）

- `run_metric_plan_tool.py:251` 的 Step 8 `aggregate_metrics_to_handoff(plan, workspace, run_validation=True)` 在**父进程**执行（ProcessPoolExecutor 已关闭后）。
- `run_validation=True` → `metric_aggregation._collect_validation_warnings` → `ethoinsight.validate_catalog.validate_plan_results(plan)` → `validate_catalog.py:364`：
  ```python
  data = json.loads(resolve_sandbox_path(output_path).read_text(encoding="utf-8"))
  ```
  其中 `output_path` 是 plan 里的**虚拟路径** `/mnt/user-data/workspace/m_*.json`。
- `resolve_sandbox_path` 需要 `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` env 才能把 `/mnt` 翻成真实路径。**但 Step 8 没有包在 `_scoped_path_env(path_env)` 里**——该 env scope 在本文件只用于：① worker（`_worker_init` initializer，`tool.py` 进程池）② statistics runner（`tool.py:237` `with _scoped_path_env(path_env):`）。Step 8 的 aggregate 在父进程**裸跑、无 env** → `resolve_sandbox_path` 匹配不到 env → fail-safe 原样返回 `/mnt/...`（不存在）→ `read_text` 抛 → catalog.py:369 记 `result_file_unreadable`，每个指标一条。

- **为何 aggregation 主体却成功（数据齐全）——症状诡异的来源**：aggregate 主体读 `m_*.json` 是**直接 glob 传入的真实 `workspace` Path**（`Path(output).name` 配真实目录里的文件），用真实宿主路径、不需要 resolve → 成功。只有 validation 子步用 plan 的**虚拟**路径重读 → 需要 resolve → 失败。两套读路径只有一套需要 env，于是"数据在但全标 unreadable"。

**实测坐实点**：
- `tool.py:237` statistics runner 包了 `_scoped_path_env`，`tool.py:251` aggregate **没包**（diff 一目了然）。
- `validate_catalog.py:364` 用 `resolve_sandbox_path(output_path)`，`output_path` 来自 plan 的虚拟 output。
- `_collect_validation_warnings` 的注释（`metric_aggregation.py:64-67`）**错误假设** "in run_metric_plan the workspace env is inherited by the process pool"——但 aggregate 不在 pool 里跑，在父进程跑，env 没继承。

## 3. 修复方案（治本，单点，一行缩进）

把 Step 8 的 aggregate 调用包进已有的 `_scoped_path_env`，与 statistics runner 同款。`path_env` 已在 `tool.py:182` `path_env = _build_path_env(thread_data)` 建好、在 scope 内可用。

`tool.py:250-251`：
```python
    # ---- Step 8: aggregate from disk artifacts (SSOT, run_validation=True) ----
    # validation 子步（validate_plan_results）在父进程内经 resolve_sandbox_path 读 plan 的
    # /mnt 虚拟 output 路径，需 DEERFLOW_PATH_* env——与 statistics runner 同样必须包
    # _scoped_path_env（否则全指标误报 result_file_unreadable，毒化 data-analyst fast-fail；
    # 2026-06-15 dogfood 实证）。aggregate 主体 glob 真实 workspace 不受影响，但 validation
    # 走虚拟路径必须有 env。
    with _scoped_path_env(path_env):
        agg = aggregate_metrics_to_handoff(plan, workspace, run_validation=True)
    status = _derive_status(agg["status"], n_total, n_failed)
```

> **为什么包整个 aggregate 而非只包 validation**：`_scoped_path_env` 是临时设 env + 退出即还原，零副作用；aggregate 主体不读 env（glob 真实路径）所以包住它无害，但能确保内部 validation 子步拿到 env。比改 `metric_aggregation` 内部签名传 env 更小、更内聚（env 管理留在 tool 层，与 statistics 一致）。
> **不动 `metric_aggregation.py` / `validate_catalog.py`**：它们的 resolve 逻辑本身对——给它 env 就能 resolve。问题纯粹在调用方没喂 env。

## 4. red 测试锚点

放 `packages/agent/backend/tests/test_run_metric_plan.py`（已存在，追加）或新建 `test_run_metric_plan_validation_env.py`。用 importlib 加载 worktree 源（守 `feedback_worktree_shares_main_venv_editable_link`）。

**锚点 1（单元，直接打 aggregate + 真实 workspace + 虚拟 plan 路径，不设 env → 修复前误报）**：
```python
def test_aggregate_validation_with_scoped_env_no_false_unreadable(tmp_path, monkeypatch):
    """red 锚点：plan 用 /mnt 虚拟 output 路径、真实 m_*.json 在真实 workspace、
    validation 在 _scoped_path_env 内跑 → 不应误报 result_file_unreadable。

    修复前：run_metric_plan Step8 不包 _scoped_path_env → validate 读 /mnt 失败 →
    5 条 result_file_unreadable critical。修复后：包了 env → 读到真实文件 → 0 误报。
    """
    # 1. 造真实 workspace + 写真实 m_*.json（值合法，过 catalog 范围校验）
    # 2. 造 plan：output 字段用 /mnt/user-data/workspace/m_*.json 虚拟路径
    # 3. 设 DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE = 真实 workspace
    #    （模拟 _scoped_path_env 包住的效果；或直接调被 _scoped_path_env 包的 run_metric_plan 路径）
    # 4. agg = aggregate_metrics_to_handoff(plan, real_workspace, run_validation=True)
    # 5. assert 没有 result_file_unreadable 的 data_quality_warnings
    dq = [w for w in agg["data_quality_warnings"] if "result_file_unreadable" in w.get("message","")]
    assert dq == [], f"误报 result_file_unreadable: {dq}"
```

**锚点 2（守护：env 缺失时确实会误报 → 证明 env 是必要的，锚定根因）**：
```python
def test_aggregate_validation_without_env_does_report_unreadable(tmp_path, monkeypatch):
    """对照：不设 env（模拟修复前 Step8 裸跑）→ 确实误报 result_file_unreadable。
    证明问题根因在 env 缺失，而非 validate 逻辑本身。"""
    monkeypatch.delenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", raising=False)
    # plan 用 /mnt 虚拟路径、真实文件在 real_workspace
    agg = aggregate_metrics_to_handoff(plan_mnt, real_workspace, run_validation=True)
    dq = [w for w in agg["data_quality_warnings"] if "result_file_unreadable" in w.get("message","")]
    assert dq, "无 env 时应误报（证明 env 是修复关键）"
```

> 锚点 1 ⊕ 锚点 2 一起钉死：**有 env→无误报、无 env→误报**，证明 `_scoped_path_env` 是 load-bearing。run_metric_plan 整体的端到端测试可选（更重，需 mock thread_data + 进程池），单元层锚点已足够锚定根因。

## 5. 验证 + 红线

- `cd packages/agent/backend && PYTHONPATH=. pytest tests/test_run_metric_plan*.py -q`（修复后绿）。
- **回退验证 red→green**：把 `with _scoped_path_env(path_env):` 去掉重跑锚点 1 → 应红（出现 result_file_unreadable）。
- **裸导入**（改了 `tool.py` 核心）：`PYTHONPATH=. python -c "import app.gateway"` + `python -c "from deerflow.agents import make_lead_agent"`。
- **既有 run_metric_plan 测试全绿**（`_scoped_path_env` 对 aggregate 主体无副作用，不应破坏既有 28+17 测试）。
- **基线债**：backend 全量 ~39 failed 是基线债（父 commit 同 venv 复现），别归因本次。
- **不动** `metric_aggregation.py` 的 validation 逻辑 / `validate_catalog.py` 的 resolve / `_scoped_path_env` 实现——它们都对，只是调用方漏喂 env。

## 6. dogfood 验收（修复后复跑）

EPM `Raw data-EPM-Xuhui-28` 复跑：`run_metric_plan` 应 `completed` + **`data_quality.critical_count: 0`**（无 result_file_unreadable）→ data-analyst 不再 fast-fail "all metrics invalid"，正常走判读 → 全链路到 report-writer。

## 7. 独立 follow-up（本次不做，记录）

- **data-analyst fast-fail 健壮性**：当 metrics_summary **有真实值**时，"critical METRIC_VALIDATION 覆盖所有指标"是否应直接 hard-fail？本次 #5 修好后误报消失、不触发，所以**先不改**（守 `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`：根因未隔离前别叠兜底）。但 data-analyst 这轮 fast-fail 后仍长篇试图分析、最终 seal 失败的行为，疑与历史 seal 故障族相关，值得单独诊断。
- **#1 加重因素复核**：本次 identify 阶段 lead 仍在反问里把模板候选连同 open/closed 列一起呈现（虽然最终判断对了），可顺手看 identify candidates 措辞是否仍偏向——非阻塞。

## 8. 关键文件/行号
- 修复点：`packages/agent/backend/packages/harness/deerflow/tools/builtins/run_metric_plan_tool.py:251`（Step 8 aggregate）+ `:237`（statistics runner 的同款样板）+ `:182`（path_env 来源）
- 根因链：`subagents/metric_aggregation.py:55/261`（`_collect_validation_warnings` 调 validate）→ `ethoinsight/validate_catalog.py:364`（`resolve_sandbox_path(output_path).read_text`）/ `:369`（result_file_unreadable）
- 下游毒化：`subagents/builtins/data_analyst.py:84-87`（fast-fail all metrics invalid）

## 9. 关联 memory
- `feedback_run_metric_plan_step8_validation_not_scoped_path_env`（本 bug 完整诊断）
- `feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path`（#2，同根源家族）
- `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（为何 data-analyst follow-up 不本次做）
- `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`（测试加载 worktree 源）
- `feedback_known_full_suite_test_pollution_4_tests`（基线债对照）

"""run_chart_plan — 一步确定性跑完 plan_charts.json 的全部绘图脚本。

Spec 2026-06-24-chart-maker-run-chart-plan-deterministic-execution：用确定性 first-party
工具替换 chart-maker subagent 里的 bash LLM 编排（对标 run_metric_plan）。工具内
ProcessPoolExecutor 进程内调绘图脚本（无 subprocess 冷启开销）+ 直接核每个 chart 的
output png 真落盘（磁盘是唯一真相）+ 经 _seal_handoff_to_workspace 原子落盘
handoff_chart_maker.json（sealed_by="run_plan"）。

LLM 在工具边界只传「哪些」(`only_chart_ids`) 和「怎么对待」(`on_error`)，永远
不传「是什么」(args)——内容从 plan artifact 读，LLM 只做指向（守 run_metric_plan 同款红线）。

价值（spec §0 立论，已取证坐实）：消除 LLM bash 编排脆弱（手拼 args 易漏
--parameters-json，dogfood thread 339512dd 真栽过）+ 产物经确定性工具不经 LLM →
chart_files 天生是磁盘真相（顺带堵 ETHO-10 reward hacking）+ 架构对齐 code-executor。
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# Spec §三 step 6：per-task 超时（与 run_metric_plan 同值，不魔法常数）。
PER_TASK_TIMEOUT_SECONDS = 120


def _available_cpus() -> int:
    """进程**实际被允许使用**的 CPU 核数（尊重 cgroup/affinity，照抄 run_metric_plan）。

    生产是 elastic 容器，``os.cpu_count()`` 返回宿主机物理核数而非容器配额。
    ``os.sched_getaffinity(0)`` 返回当前进程被允许调度到的 CPU 集合（Linux 上尊重
    cgroup/affinity），是「实际可用核」的正解。非 Linux 回退 ``os.cpu_count()``。
    """
    try:
        return len(os.sched_getaffinity(0))  # Linux: 尊重 cgroup/affinity
    except AttributeError:
        return os.cpu_count() or 4  # 非 Linux 回退


# 进程池并行度 = 实际可用核数（不封顶，吃满 elastic 配额）。照抄 run_metric_plan。
MAX_WORKERS = _available_cpus()

# 测试钩子：注入同步 runner 绕开 ProcessPoolExecutor 的 fork/pickle（spec §五 单测用）。
# 生产恒为 None → 走进程池。monkeypatch 本变量即可让 _execute_tasks 串行执行。
_TASK_RUNNER_OVERRIDE = None


# ============================================================================
# Worker — 必须模块级（供 ProcessPoolExecutor pickle）。
# 进程内调绘图脚本：importlib.import_module(script).main(args)，无 subprocess 冷启。
# ============================================================================


def _worker_init(path_env: dict[str, str]) -> None:
    """ProcessPoolExecutor initializer — 在**每个 worker 进程内**设置 DEERFLOW_PATH_* env。

    与 run_metric_plan 同构（守其多线程 Gateway 安全理由），额外设 MPLBACKEND=Agg：
    26 个绘图脚本里 5 个（epm/plot_open_arm_time_ratio_bar、epm/plot_zone_entry_distribution、
    ldb/plot_zone_entry_distribution、oft/plot_center_entry_summary、oft/plot_center_time_ratio_bar）
    裸 ``import matplotlib.pyplot`` 不经 charts.py、不设 Agg。本机无 $DISPLAY 默认侥幸是 agg，
    但进程池 worker 从 Gateway 父进程 fork，若父进程曾初始化非 Agg backend（某部署有 $DISPLAY
    / 其他工具先 touch pyplot），fork 的 worker 继承交互式 backend → 非主线程 savefig 崩。
    initializer 在 fresh 子进程入口设 MPLBACKEND（在任何 import matplotlib.pyplot 之前生效），
    免费保险、确定性安全（spec §二 matplotlib backend 坑）。
    """
    if path_env:
        os.environ.update(path_env)
    # setdefault：worker 内不覆盖已显式配置的 backend，但 fresh worker 默认无此键 → 落 Agg。
    os.environ.setdefault("MPLBACKEND", "Agg")


def _run_chart_task(
    script: str, args: list[str], task_id: str
) -> tuple[str, int, str]:
    """Run one plot script in-process. Returns (task_id, returncode, error).

    returncode 0 = success; non-zero or exception = failure with error string.
    Worker 的 DEERFLOW_PATH_* env + MPLBACKEND 由 _worker_init initializer 在进程内设置。

    必须模块级 + 顶层无 deerflow/ethoinsight import（pickle 友好 + 守 harness 闭环纪律）。
    与 run_metric_plan 的 _run_metric_task 完全同构（绘图脚本同样 ``def main(argv)->int``）。
    """
    import importlib

    try:
        mod = importlib.import_module(script)
        rc = mod.main(args)
        return (task_id, int(rc or 0), "")
    except SystemExit as e:  # argparse 解析失败 / 脚本 sys.exit
        return (task_id, int(e.code) if isinstance(e.code, int) else 1, f"SystemExit({e.code})")
    except BaseException as e:  # noqa: BLE001 — worker 必须吞掉一切异常，逐个记失败不杀池
        return (task_id, 1, f"{type(e).__name__}: {e}")


# ============================================================================
# Tool
# ============================================================================


@tool("run_chart_plan", parse_docstring=True)
def run_chart_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    plan_path: str = "/mnt/user-data/workspace/plan_charts.json",
    only_chart_ids: list[str] | None = None,
    on_error: str = "continue",
) -> dict[str, Any]:
    """一步跑完 plan_charts.json 的所有绘图脚本，确定性核磁盘落盘并封存 handoff。

    不需要逐条拼 bash、不需要手构 handoff。工具内 ProcessPoolExecutor 进程内并行跑
    所有绘图脚本（每 worker import 一次），逐个核 output png 真落盘（磁盘真相），
    经 _seal_handoff_to_workspace 原子落盘 handoff_chart_maker.json（sealed_by="run_plan"）。
    chart_files 直接列真落盘的 png 虚拟路径，零重拼 args、零 LLM 自报。

    Args:
      plan_path: plan_charts.json 的虚拟路径（默认 /mnt/user-data/workspace/plan_charts.json）。
      only_chart_ids: 选择器——只画这些 chart id 的子集（重跑子集用）；None = 画全部。
      on_error: 失败策略。"continue" = 遇失败继续画剩余（默认）；"abort" = 遇首个失败停止提交后续。

    Returns:
      紧凑结果（成功明细落盘不进 context）：
        {"status": "completed"|"partial"|"failed",
         "handoff_path": "/mnt/user-data/workspace/handoff_chart_maker.json",
         "n_total": int, "n_rendered": int, "n_failed": int,
         "failures": [{"chart_id","reason"}],     # 仅失败 chart 的摘要
         "gate_signals": {...}}                     # lead 决策信号
    """
    # 惰性 import：守 harness 顶层 import 闭环纪律（sandbox/tools、handoff_schemas、
    # seal_handoff_tools 都在环上）。
    import json

    from deerflow.sandbox.tools import _build_path_env, replace_virtual_path
    from deerflow.subagents.handoff_schemas import ChartMakerHandoff
    from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace

    # ---- Step 1: resolve workspace from thread_data ----
    thread_data = runtime.state.get("thread_data") if runtime.state else None
    if not thread_data or not thread_data.get("workspace_path"):
        return _error_result("workspace_missing", "thread_data.workspace_path 未设置（基础设施 bug）")
    workspace_str = thread_data["workspace_path"]
    workspace = Path(workspace_str)

    # ---- Step 2: read plan ----
    plan_real_path = replace_virtual_path(plan_path, thread_data)
    if not Path(plan_real_path).exists():
        # plan 缺失 → seal failed，让 lead 补 plan（对齐 run_metric_plan plan-missing 路径）。
        _seal_failed(workspace, ChartMakerHandoff, f"plan_charts.json 不存在: {plan_path}")
        return _error_result("plan_missing", f"plan_charts.json 不存在: {plan_path}")
    try:
        plan = json.loads(Path(plan_real_path).read_text(encoding="utf-8"))
    except Exception as e:
        _seal_failed(workspace, ChartMakerHandoff, f"plan_charts.json 解析失败: {e}")
        return _error_result("plan_unparseable", f"plan_charts.json 解析失败: {e}")

    plan_charts: list[dict] = plan.get("charts", [])
    if not plan_charts:
        _seal_failed(workspace, ChartMakerHandoff, "plan 无 charts[]")
        return _error_result("empty_plan", "plan 无 charts[]")

    # ---- Step 3: apply only_chart_ids selector（重跑子集用）----
    only_set = set(only_chart_ids) if only_chart_ids else None
    if only_set is not None:
        plan_charts = [c for c in plan_charts if c.get("id") in only_set]
        if not plan_charts:
            _seal_failed(workspace, ChartMakerHandoff, f"only_chart_ids 过滤后无 chart: {sorted(only_set)}")
            return _error_result("empty_after_filter", f"only_chart_ids 过滤后无 chart: {sorted(only_set)}")

    # ---- Step 4: build DEERFLOW_PATH_* env so workers resolve /mnt virtual paths ----
    # 不在父进程 mutate 全局 os.environ（Gateway 多线程并发跑多 thread subagent，全局
    # mutate 会让一个 thread 的 workspace 路径泄漏成全局污染其他 thread）。worker 经
    # ProcessPoolExecutor initializer 在子进程内设置（_worker_init），父进程 env 不变。
    path_env = _build_path_env(thread_data)

    # ---- Step 5: build task list ----
    # 绘图脚本的 args（plan_charts.json 的 entry.args）已含完整 argv（--input --output
    # --parameters-json ...）。
    #
    # F1（spec 2026-06-24-run-chart-plan-permissionerror）：argv 里的 /mnt 虚拟路径必须
    # 在喂 worker 前预解析成真实物理路径。根因：plot 脚本 fig.savefig(args.output) 不调
    # resolve_sandbox_path（与 compute 脚本走 save_output_json→自 resolve 不同），进程池
    # 透传原始 /mnt → PermissionError/FileNotFoundError（112/113 图全废，见 problem doc）。
    # bash 路径靠 replace_virtual_paths_in_command 重写命令字符串；进程池没有这步——F1 把
    # 等价的 argv 重写提到工具内，让两条路径语义对齐（守「用确定性结构约束行为」）。
    # replace_virtual_path 对非 /mnt 项（--input / --parameters-json / JSON 字符串 / 数字）
    # 原样返回（幂等无害），故可对整个 args 数组逐项跑。
    tasks: list[tuple[str, list[str], str]] = []  # (script, args, task_id)
    # 保留 output_mode 用于 status 派生（aggregate 是组间对比 must_have）。
    chart_meta: dict[str, dict[str, Any]] = {}  # task_id -> {output, output_mode, chart_id}
    for c in plan_charts:
        script = c.get("script", "")
        args = list(c.get("args", []) or [])
        # F1: 预解析 argv 里的 /mnt 虚拟路径 → 真实物理路径（对齐 bash 重写语义）。
        args = [replace_virtual_path(a, thread_data) for a in args]
        chart_id = c.get("id", c.get("output", ""))
        output = c.get("output", "")
        output_mode = c.get("output_mode", "per_subject")
        if not script or not args:
            # 缺 script/args 的 entry 是 plan 生成 bug，响亮失败（对齐 run_metric_plan §3 风险 1）。
            logger.warning("[run_chart_plan] entry 缺 script/args，跳过: id=%s", chart_id)
            tasks.append(("", [], chart_id))  # 空 script → worker 立即 ImportError 失败
        else:
            tasks.append((script, args, chart_id))
        chart_meta[chart_id] = {"output": output, "output_mode": output_mode}

    # ---- Step 6: execute via ProcessPoolExecutor ----
    results, failures = _execute_tasks(
        tasks,
        on_error=on_error,
        timeout=PER_TASK_TIMEOUT_SECONDS,
        runner=_TASK_RUNNER_OVERRIDE,
        path_env=path_env,
    )

    n_total = len(tasks)
    # ---- Step 7: 核磁盘 + 构造 chart_files / failed_charts（run_chart_plan 核心）----
    # 对每个 chart：output 虚拟路径 → replace_virtual_path 物理路径 → exists()。
    # 落盘 → chart_files（虚拟路径）；未落盘（rc≠0 或 rc=0 但 png 缺失=脚本半成功）→ failed_charts。
    chart_files: list[str] = []
    failed_charts: list[dict[str, str]] = []
    for tid, (rc, err) in results.items():
        meta = chart_meta.get(tid, {})
        output_virtual = meta.get("output", "")
        output_mode = meta.get("output_mode", "per_subject")
        on_disk = False
        if output_virtual:
            output_real = replace_virtual_path(output_virtual, thread_data)
            try:
                on_disk = Path(output_real).exists()
            except OSError:
                on_disk = False
        if on_disk:
            chart_files.append(output_virtual)
        else:
            if rc == 0:
                # rc=0 但 png 缺失 = 脚本半成功（rc 不可信，磁盘是唯一真相）。
                reason = "rendered rc=0 but png missing"
            else:
                reason = (err or "")[:500] or f"rc={rc}"
            failed_charts.append({"chart_id": tid, "reason": reason})

    # 失败 task 不在 results 命中（runner error 路径）也兜底进 failed_charts。
    seen_failed_ids = {fc["chart_id"] for fc in failed_charts}
    for f in failures:
        fid = f.get("id", "")
        if fid and fid not in seen_failed_ids:
            failed_charts.append({"chart_id": fid, "reason": (f.get("error", "") or "")[:500]})
            seen_failed_ids.add(fid)

    n_rendered = len(chart_files)
    n_failed = n_total - n_rendered

    # ---- Step 7.5: remaining_charts（P5 预算降级指纹，透传 plan.charts_budget_remaining）----
    remaining_charts: list[dict[str, str]] = []
    for c in plan.get("charts_budget_remaining") or []:
        if isinstance(c, dict) and c.get("id"):
            remaining_charts.append({"chart_id": str(c["id"]), "reason": "chart_budget_truncated"})

    # ---- Step 8: derive status（纯函数，对齐 run_metric_plan _derive_status 语义）----
    # 完成度纯由磁盘产物决定（落盘 png 数）。
    #   0 落盘 → failed；全部落盘 → completed；部分落盘 → partial。
    # 「aggregate 图是组间对比 must_have」由 ETHO-10 的 2.2 reconcile 门在 seal 时把关
    # （completed 且缺 aggregate → ValueError）——本函数只看完成度计数，职责单一。
    # 部分落盘的 conservative partial 已隐含「aggregate 可能缺失」（计数未满）。
    status = _derive_status(n_total, n_rendered)

    # ---- Step 9: assemble + seal handoff (deterministic, sealed_by="run_plan") ----
    payload: dict[str, Any] = {
        "status": status,
        "paradigm": plan.get("paradigm", ""),
        "summary": _default_summary(status, n_total, n_rendered, n_failed),
        "chart_files": chart_files,
        "failed_charts": failed_charts,
        "remaining_charts": remaining_charts,
        "gate_signals": _build_gate_signals(status, n_rendered, n_failed),
        "sealed_by": "run_plan",
    }

    try:
        _seal_handoff_to_workspace(ChartMakerHandoff, "handoff_chart_maker.json", payload, workspace)
    except ValueError as e:
        # seal schema 校验失败 = 产物与 schema 不匹配 / 2.2 reconcile aggregate 门拒绝（回归探针）。
        logger.error("[run_chart_plan] seal schema 校验失败: %s", e)
        return _error_result("seal_failed", f"handoff schema 校验失败: {e}")

    result = {
        "status": status,
        "handoff_path": "/mnt/user-data/workspace/handoff_chart_maker.json",
        "n_total": n_total,
        "n_rendered": n_rendered,
        "n_failed": n_failed,
        "failures": failed_charts,
        "gate_signals": payload["gate_signals"],
    }
    logger.info(
        "[run_chart_plan] done: status=%s n_total=%d n_rendered=%d n_failed=%d",
        status, n_total, n_rendered, n_failed,
    )
    return result


# ============================================================================
# Helpers
# ============================================================================


def _execute_tasks(
    tasks: list[tuple[str, list[str], str]],
    *,
    on_error: str = "continue",
    timeout: int = PER_TASK_TIMEOUT_SECONDS,
    runner=None,
    path_env: dict[str, str] | None = None,
) -> tuple[dict[str, tuple[int, str]], list[dict[str, str]]]:
    """Execute all plot tasks, return (results, failures).

    与 run_metric_plan _execute_tasks 同构：per-task timeout（不杀整池）、on_error policy、
    _TASK_RUNNER_OVERRIDE 测试钩子注入同步串行 runner（绕 fork/pickle）。

    Args:
      tasks: list of (script, args, task_id).
      on_error: "continue" (跑完全部) | "abort" (遇首个失败即停，取消未启动).
      timeout: per-task wall-clock seconds.
      runner: callable(script, args, task_id) -> (task_id, rc, error). None = default
        _run_chart_task via ProcessPoolExecutor. **测试注入同步 runner 绕开 fork/pickle**。
      path_env: DEERFLOW_PATH_* env，经 initializer 在每个 worker 进程内设置（不污染父进程）。

    Returns:
      results: {task_id: (rc, error_str)}.
      failures: [{"id", "error"}] 仅失败 task。
    """
    results: dict[str, tuple[int, str]] = {}
    failures: list[dict[str, str]] = []

    if not tasks:
        return results, failures

    if runner is not None:
        # 同步串行（测试路径）：on_error=abort 时遇首个失败即停，后续 task 标 aborted
        # （镜像生产 ProcessPoolExecutor 路径的 cancel 语义——被取消项进 failures 带 aborted 标记）。
        for idx, (script, args, tid) in enumerate(tasks):
            try:
                _tid, rc, err = runner(script, args, tid)
            except BaseException as e:  # noqa: BLE001
                rc, err = 1, f"runner error: {type(e).__name__}: {e}"
            results[tid] = (rc, err)
            if rc != 0:
                failures.append({"id": tid, "error": (err or "")[:500]})
                if on_error == "abort":
                    for other_script, other_args, other_tid in tasks[idx + 1:]:
                        results[other_tid] = (1, "aborted (on_error=abort 前序失败)")
                        failures.append({"id": other_tid, "error": "aborted"})
                    break
        return results, failures

    # 生产路径：ProcessPoolExecutor。worker env 经 initializer 在子进程内设置（不污染父进程）。
    pool = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_worker_init,
        initargs=(path_env or {},),
    )
    try:
        future_to_tid = {pool.submit(_run_chart_task, sc, ar, tid): tid for (sc, ar, tid) in tasks}
        ordered = list(future_to_tid.items())  # 提交顺序
        # 按提交顺序逐个 result(timeout=)——每个 future 有独立 per-task 超时预算（同 run_metric_plan）。
        for idx, (fut, tid) in enumerate(ordered):
            try:
                _tid, rc, err = fut.result(timeout=timeout)
                results[tid] = (rc, err)
                if rc != 0:
                    failures.append({"id": tid, "error": (err or "")[:500]})
            except FuturesTimeoutError:
                results[tid] = (1, f"timeout (>{timeout}s)")
                failures.append({"id": tid, "error": f"timeout (>{timeout}s)"})
            except BaseException as e:  # noqa: BLE001
                results[tid] = (1, f"pool error: {e}")
                failures.append({"id": tid, "error": f"pool error: {type(e).__name__}: {e}"})

            if on_error == "abort" and results[tid][0] != 0:
                logger.warning("[run_chart_plan] on_error=abort，遇失败即停，取消未启动 task")
                for other_fut, other_tid in ordered[idx + 1:]:
                    other_fut.cancel()
                    results[other_tid] = (1, "aborted (on_error=abort 前序失败)")
                    failures.append({"id": other_tid, "error": "aborted"})
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return results, failures


def _derive_status(n_total: int, n_rendered: int) -> str:
    """Map render outcome to final status (pure function, spec §三 step 8).

    完成度纯由磁盘产物决定（落盘 png 数）：
      - 0 落盘 → failed
      - 全部落盘 → completed
      - 部分落盘 → partial

    「aggregate 图是组间对比 must_have」约束不在本函数——由 ETHO-10 的 2.2 reconcile
    门在 seal 时把关（completed 且缺 aggregate → ValueError）。本函数职责单一：完成度计数。
    """
    if n_rendered == 0:
        return "failed"
    if n_rendered == n_total:
        return "completed"
    return "partial"


def _default_summary(status: str, n_total: int, n_rendered: int, n_failed: int) -> str:
    """Mechanical default summary（spec §三 step 9：机械默认）。"""
    if status == "completed":
        return f"{n_rendered}/{n_total} charts rendered (run_chart_plan)"
    if status == "partial":
        return f"{n_rendered}/{n_total} charts rendered, partial (run_chart_plan)"
    return f"{n_rendered}/{n_total} charts rendered, {n_failed} failed (run_chart_plan)"


def _build_gate_signals(status: str, n_rendered: int, n_failed: int) -> dict[str, Any]:
    """Build gate_signals for lead decision（镜像 chart-maker [gate_signals] 形态）。

    run_chart_plan 据真实落盘计数确定性构造（不靠 LLM 自报）。
    """
    stat_validity = "ok"
    if status == "failed":
        stat_validity = "failed"
    elif status == "partial":
        stat_validity = "warning"
    return {
        "constitution_acknowledged": True,
        "charts_generated": n_rendered,
        "failed_charts": n_failed,
        "data_quality": {"critical_count": 0, "warning_count": 0, "critical_items": []},
        "statistical_validity": stat_validity,
        "errors_count": n_failed,
    }


def _seal_failed(workspace: Path, model_cls: type, message: str) -> None:
    """Best-effort seal a minimal failed handoff (plan missing/unparseable paths)."""
    # 惰性 import 避免闭环；守独立可调。
    import json

    payload = {
        "status": "failed",
        "paradigm": "",
        "summary": message,
        "chart_files": [],
        "failed_charts": [],
        "remaining_charts": [],
        "gate_signals": {
            "constitution_acknowledged": True,
            "charts_generated": 0,
            "failed_charts": 0,
            "data_quality": {"critical_count": 0, "warning_count": 0, "critical_items": []},
            "statistical_validity": "failed",
            "errors_count": 1,
        },
        "sealed_by": "run_plan",
    }
    try:
        from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace

        _seal_handoff_to_workspace(model_cls, "handoff_chart_maker.json", payload, workspace)
    except Exception as e:  # noqa: BLE001
        logger.error("[run_chart_plan] _seal_failed 也失败: %s", e)
        # 最后兜底：直接写裸 JSON（不经 schema 校验），保证磁盘上有 handoff 文件。
        # failed status + 空 chart_files 不触发 _completed_requires_core_output（仅 completed 拦）。
        try:
            (workspace / "handoff_chart_maker.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass


def _error_result(code: str, message: str) -> dict[str, Any]:
    """Build a standardised error return dict (tool-level)."""
    return {
        "status": "failed",
        "error_code": code,
        "message": message,
        "handoff_path": "/mnt/user-data/workspace/handoff_chart_maker.json",
        "n_total": 0,
        "n_rendered": 0,
        "n_failed": 0,
        "failures": [],
        "gate_signals": {
            "constitution_acknowledged": True,
            "charts_generated": 0,
            "failed_charts": 0,
            "data_quality": {"critical_count": 0, "warning_count": 0, "critical_items": []},
            "statistical_validity": "failed",
            "errors_count": 1,
        },
    }

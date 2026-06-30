"""run_metric_plan — 一步确定性跑完 plan_metrics.json 的全部 compute + statistics。

Spec S4 (2026-06-12): 用确定性的 first-party 工具替换 code-executor subagent 里的
bash LLM 编排。工具内 ProcessPoolExecutor 进程内调脚本（无 subprocess 冷启开销）+
确定性聚合（复用 metric_aggregation SSOT）+ 原子落盘 handoff。

LLM 在工具边界只传「哪些」(`only_metric_ids`) 和「怎么对待」(`on_error`)，永远
不传「是什么」(args)——内容从 plan artifact 读，LLM 只做指向（Spec S4 §1.1 红线）。
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Literal

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# Spec S4 §1.3: per-task 超时（不魔法常数，spec 锁定 120s）+ 进程池大小。
PER_TASK_TIMEOUT_SECONDS = 120


def _available_cpus() -> int:
    """进程**实际被允许使用**的 CPU 核数。

    生产是 elastic（弹性伸缩）容器，目标是「能多快多快」吃满可用核。但
    ``os.cpu_count()`` 在容器里返回**宿主机物理核数**，不是容器的 cgroup CPU 配额 /
    CPU affinity——elastic 调度器给容器分 N 核、容器却跑在更大宿主机上时，
    ``os.cpu_count()`` 会高报，导致 fork 远超配额的进程、上下文切换爆炸甚至 OOM。

    ``os.sched_getaffinity(0)`` 返回当前进程被允许调度到的 CPU 集合（Linux 上尊重
    cgroup/affinity），是「实际可用核」的正解：dev 8 vCPU → 8；elastic 32 核容器 → 32；
    本机 48 核无限制 → 48。非 Linux（无 sched_getaffinity）回退 ``os.cpu_count()``。
    """
    try:
        return len(os.sched_getaffinity(0))  # Linux: 尊重 cgroup/affinity
    except AttributeError:
        return os.cpu_count() or 4  # 非 Linux 回退


# 进程池并行度 = 实际可用核数（不封顶，吃满 elastic 配额）。
#
# 回归修复（2026-06-18）：run_metric_plan 工具（commit 63793162, 2026-06-15）出生时把
# MAX_WORKERS 写死为 4。它之前的 bash 编排模式（ethoinsight-code skill）用
# `python -m ... & python -m ... & wait` 把同指标不同 subject **全部 `&` 并行**起，
# 并行度只受可用核限制。工具化（确定性聚合、消除 subprocess 冷启、消除 bash LLM 编排
# 脆弱）是对的，但 MAX_WORKERS=4 这个保守初始值把并行度从「≈可用核」悄悄降到 4——
# dev 服务器 8 vCPU 下 140 任务挤 4 worker ≈107s，本可用满 8 核砍半。这是换工具时
# 漏网的并行度回归。
#
# compute 任务是 CPU 密集（xlsx 解析 + 指标计算），可用核数即合理上限；单 worker 峰值
# ~180MB（dev 16 GiB 下 8 worker ≈1.5GB，余量充足）。不人为封顶，让 elastic 大容器
# 吃满；_available_cpus() 已尊重 cgroup 配额，不会在小容器上 fork 过多。
MAX_WORKERS = _available_cpus()

# 测试钩子：注入同步 runner 绕开 ProcessPoolExecutor 的 fork/pickle（spec §4 单测用）。
# 生产恒为 None → 走进程池。monkeypatch 本变量即可让 _execute_tasks 串行执行。
_TASK_RUNNER_OVERRIDE = None


# ============================================================================
# Worker — 必须模块级（供 ProcessPoolExecutor pickle）。
# 进程内调脚本：importlib.import_module(script).main(args)，无 subprocess 冷启。
# ============================================================================


def _worker_init(path_env: dict[str, str]) -> None:
    """ProcessPoolExecutor initializer — 在**每个 worker 进程内**设置 DEERFLOW_PATH_* env。

    Spec S4 §1.3：worker 需要 DEERFLOW_PATH_* 才能让脚本内 resolve_sandbox_path 解
    /mnt 虚拟路径。通过 initializer 在子进程内 os.environ.update，而非在父进程（Gateway）
    mutate 全局 os.environ——父进程多线程并发跑多个 thread 的 subagent，全局 mutate 会让
    一个 thread 的 workspace 路径泄漏成全局、污染其他 thread（潜在跨线程路径错配）。
    initializer 把副作用关在 worker 进程里，父进程 env 不变。
    """
    if path_env:
        os.environ.update(path_env)


@contextlib.contextmanager
def _scoped_path_env(path_env: dict[str, str] | None) -> Iterator[None]:
    """临时把 path_env 设进父进程 os.environ，退出即恢复原值（含删除新增的键）。

    用于在父进程内跑 statistics（_run_metric_task 直接调 mod.main，脚本内
    resolve_sandbox_path 读 DEERFLOW_PATH_* env）。比永久 os.environ.update 安全：
    多线程 Gateway 下副作用只在 with 块内可见、退出即还原，不泄漏给其他 thread。
    （注：理论上 with 块内的并发线程仍可能瞥见这些键——statistics 是单 task 短调用，
    窗口极小；彻底隔离需把 statistics 也移进进程池，本次保持最小改动。）
    """
    if not path_env:
        yield
        return
    _sentinel = object()
    saved: dict[str, Any] = {k: os.environ.get(k, _sentinel) for k in path_env}
    os.environ.update(path_env)
    try:
        yield
    finally:
        for k, old in saved.items():
            if old is _sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _run_metric_task(
    script: str, args: list[str], task_id: str
) -> tuple[str, int, str]:
    """Run one compute/statistics script in-process. Returns (task_id, returncode, error).

    returncode 0 = success; non-zero or exception = failure with error string.
    Worker 的 DEERFLOW_PATH_* env 由 _worker_init initializer 在进程内设置（Spec S4 §1.3），
    所以脚本内 resolve_sandbox_path 能解 /mnt 虚拟路径。

    必须模块级 + 顶层无 deerflow/ethoinsight import（pickle 友好 + 守 harness 闭环纪律）。
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


@tool("run_metric_plan", parse_docstring=True)
def run_metric_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    plan_path: str = "/mnt/user-data/workspace/plan_metrics.json",
    only_metric_ids: list[str] | None = None,
    on_error: str = "continue",
) -> dict[str, Any]:
    """一步跑完 plan_metrics.json 的所有 compute + statistics，确定性聚合并落盘 handoff。

    不需要逐条拼 bash、不需要手构 handoff。工具内 ProcessPoolExecutor 进程内并行跑
    所有 compute 脚本（每 worker import 一次）+ statistics（如未 skip）+ 确定性聚合
    （含 L-A/L-B 校验）+ 原子落盘 handoff_code_executor.json（sealed_by="run_plan"）。

    Args:
      plan_path: plan_metrics.json 的虚拟路径（默认 /mnt/user-data/workspace/plan_metrics.json）。
      only_metric_ids: 选择器——只跑这些 metric id 的子集（重跑子集用）；None = 跑全部。
      on_error: 失败策略。"continue" = 遇失败继续跑剩余（默认）；"abort" = 遇首个失败停止提交后续。

    Returns:
      紧凑结果（成功明细落盘不进 context）：
        {"status": "completed"|"partial"|"failed",
         "handoff_path": "/mnt/user-data/workspace/handoff_code_executor.json",
         "n_total": int, "n_completed": int, "n_failed": int,
         "failures": [{"id", "error"}],     # 仅失败 task 的摘要
         "gate_signals": {...}}              # lead 决策信号
    """
    # 惰性 import：守 harness 顶层 import 闭环纪律（subagents/metric_aggregation、
    # sandbox/tools、tools/builtins/seal_handoff_tools 都在环上）。
    import json

    from deerflow.sandbox.tools import _build_path_env, replace_virtual_path
    from deerflow.subagents.handoff_schemas import CodeExecutorHandoff
    from deerflow.subagents.metric_aggregation import aggregate_metrics_to_handoff
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
        # plan 缺失 → 同 code_executor 现状：seal failed，让 lead 补 plan。
        _seal_failed(workspace, CodeExecutorHandoff, f"plan_metrics.json 不存在: {plan_path}")
        return _error_result("plan_missing", f"plan_metrics.json 不存在: {plan_path}")
    try:
        plan = json.loads(Path(plan_real_path).read_text(encoding="utf-8"))
    except Exception as e:
        _seal_failed(workspace, CodeExecutorHandoff, f"plan_metrics.json 解析失败: {e}")
        return _error_result("plan_unparseable", f"plan_metrics.json 解析失败: {e}")

    plan_metrics: list[dict] = plan.get("metrics", [])
    if not plan_metrics:
        _seal_failed(workspace, CodeExecutorHandoff, "plan 无 metrics[]")
        return _error_result("empty_plan", "plan 无 metrics[]")

    # ---- Step 3: apply only_metric_ids selector ----
    only_set = set(only_metric_ids) if only_metric_ids else None
    if only_set is not None:
        plan_metrics = [m for m in plan_metrics if m.get("id") in only_set]
        if not plan_metrics:
            _seal_failed(workspace, CodeExecutorHandoff, f"only_metric_ids 过滤后无 metric: {sorted(only_set)}")
            return _error_result("empty_after_filter", f"only_metric_ids 过滤后无 metric: {sorted(only_set)}")

    # ---- Step 4: build DEERFLOW_PATH_* env so workers resolve /mnt virtual paths ----
    # 不在父进程 mutate 全局 os.environ（Gateway 多线程并发跑多 thread subagent，全局
    # mutate 会让一个 thread 的 workspace 路径泄漏成全局污染其他 thread）。改为：
    #   - compute 池 worker：经 ProcessPoolExecutor initializer 在子进程内设置（_worker_init）。
    #   - statistics（在父进程内跑）：用 _scoped_path_env 临时设置 + 退出即恢复（见 Step 7）。
    path_env = _build_path_env(thread_data)

    # ---- Step 5: build task list (compute metrics) ----
    # compute 脚本的 args 已含完整 argv（--input --output --parameters-json ...），直接透传。
    tasks: list[tuple[str, list[str], str]] = []  # (script, args, task_id)
    for m in plan_metrics:
        script = m.get("script", "")
        args = list(m.get("args", []) or [])
        mid = m.get("id", m.get("output", ""))
        if not script or not args:
            # 缺 script/args 的 entry 是 plan 生成 bug，响亮失败（Spec S4 §3 迁移风险 1）。
            logger.warning("[run_metric_plan] entry 缺 script/args，跳过: id=%s", mid)
            tasks.append(("", [], mid))  # 空 script → worker 立即 ImportError 失败
        else:
            tasks.append((script, args, mid))

    # ---- Step 6: execute compute via ProcessPoolExecutor ----
    results, failures = _execute_tasks(
        tasks,
        on_error=on_error,
        timeout=PER_TASK_TIMEOUT_SECONDS,
        runner=_TASK_RUNNER_OVERRIDE,
        path_env=path_env,
    )

    n_total = len(tasks)
    n_failed = len(failures)
    n_completed = n_total - n_failed

    # ---- Step 7: statistics (single task, skip if skip_reason set) ----
    statistics_payload: dict[str, Any] | None = None
    # P2 (spec 2026-06-17-statistics-loud-failure): statistics 子步骤三态可机读降级信号。
    # 权威受校验定义在 handoff_schemas.GateSignals.statistics_status（Pydantic Literal）；下面的
    # 注解是与之对齐的镜像 type hint（不能在此 top-level import 那个类型——handoff_schemas 经
    # subagents.__init__ 闭环，本文件所有 deerflow import 都惰性）。crashed 损害可复现性，交由
    # DegradationCircuitBreakerMiddleware 熔断（自救一次 → HITL）；absent_by_design 是单组/单样本
    # 合理 skip，正常描述性 partial，不熔断。
    statistics_status: Literal["ok", "crashed", "absent_by_design"] = "absent_by_design"
    statistics_error: str | None = None
    stats_obj = plan.get("statistics")
    if isinstance(stats_obj, dict) and stats_obj.get("skip_reason") is None:
        stats_script = stats_obj.get("script", "")
        stats_output = stats_obj.get("output", "")
        if stats_script and stats_output:
            # PlanStatistics 无 args 字段，手构 argv（Spec S4：input 是遗留字段指向错误文件）。
            # inputs.json 内容 = plan.inputs.raw_files；groups.json 来自 plan.inputs.groups_file。
            raw_files = (plan.get("inputs", {}) or {}).get("raw_files", []) or []
            inputs_json_path = workspace / "inputs.json"
            try:
                inputs_json_path.write_text(json.dumps(raw_files), encoding="utf-8")
            except OSError as e:
                logger.warning("[run_metric_plan] 写 inputs.json 失败: %s", e)

            groups_virtual = (plan.get("inputs", {}) or {}).get("groups_file")
            groups_arg = groups_virtual if groups_virtual else "/mnt/user-data/workspace/groups.json"
            inputs_arg = "/mnt/user-data/workspace/inputs.json"
            stats_argv = ["--inputs", inputs_arg, "--groups", groups_arg, "--output", stats_output]

            # 列对齐透传（spec 2026-06-16-statistics-path-column-alignment）：把 plan
            # statistics 段的 parameters（resolve 投影自 metrics 段同源 zone_aliases_overrides）
            # 序列化成 --parameters-json 传给 run_groupwise_stats → dispatcher.zone_overrides。
            # 与 compute step 传 parameters 的方式对称（compute 用 plan_metric.args 内嵌的
            # --parameters-json）。无 column_aliases 时 parameters={} → 等价于不传，行为不变。
            stats_parameters = stats_obj.get("parameters") or {}
            stats_argv.append("--parameters-json")
            stats_argv.append(json.dumps(stats_parameters, ensure_ascii=False))

            # 统计单独提交（不进上面的池：单 task，避免 inputs.json/groups.json 写竞争）。
            # 在父进程内跑（_run_metric_task 直接调 mod.main），脚本内 resolve_sandbox_path
            # 读 DEERFLOW_PATH_* env——用 _scoped_path_env 临时设置 + 退出即恢复，不留全局副作用。
            # 感知测试 runner override（与 compute 路径一致，mock 出同一脚本行为）。
            _stats_runner = _TASK_RUNNER_OVERRIDE if _TASK_RUNNER_OVERRIDE is not None else _run_metric_task
            with _scoped_path_env(path_env):
                _tid, src_rc, src_err = _stats_runner(stats_script, stats_argv, "statistics")
            if src_rc == 0:
                # 读 statistics 产物（output 是虚拟路径，resolve 后读）。
                stats_real = replace_virtual_path(stats_output, thread_data)
                try:
                    statistics_payload = json.loads(Path(stats_real).read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning("[run_metric_plan] 读 statistics 产物失败: %s", e)
                # P2: runner 成功（rc=0）但产物空/不可读也算 crashed（脚本半成功，损害可复现性）。
                if statistics_payload:
                    statistics_status = "ok"
                else:
                    statistics_status = "crashed"
                    statistics_error = "statistics payload empty or unreadable"[:500]
            else:
                logger.warning("[run_metric_plan] statistics 失败 (rc=%d): %s", src_rc, src_err)
                failures.append({"id": "statistics", "error": src_err[:500]})
                statistics_status = "crashed"
                statistics_error = (src_err or "")[:500]
        else:
            # skip_reason is None 但缺 script/output：statistics 段声明了"应该跑"却无法跑
            # （catalog/resolve 投影出了残缺的 statistics 段）→ crashed（非设计内缺席）。
            # 这堵住一个红线一静默降级口子：残缺段不能伪装成 absent_by_design 走正常 partial。
            logger.warning(
                "[run_metric_plan] statistics 段 skip_reason=None 但缺 script/output "
                "(script=%r, output=%r) — 标记 crashed",
                stats_script,
                stats_output,
            )
            statistics_status = "crashed"
            statistics_error = "statistics segment declared (skip_reason=None) but missing script/output"[:500]
    # stats_obj 缺失或 skip_reason 非空：statistics_status 保持默认 absent_by_design（合理 skip）。

    # ---- Step 8: aggregate from disk artifacts (SSOT, run_validation=True) ----
    # validation 子步（validate_plan_results）在父进程内经 resolve_sandbox_path 读 plan 的
    # /mnt 虚拟 output 路径，需 DEERFLOW_PATH_* env——与 statistics runner 同样必须包
    # _scoped_path_env（否则全指标误报 result_file_unreadable，毒化 data-analyst fast-fail；
    # 2026-06-15 dogfood 实证）。aggregate 主体 glob 真实 workspace 不受影响，但 validation
    # 走虚拟路径必须有 env。
    with _scoped_path_env(path_env):
        agg = aggregate_metrics_to_handoff(plan, workspace, run_validation=True)
    status = _derive_status(agg["status"], n_total, n_failed)
    # P2: statistics 崩溃单独降级 completed→partial（不加新枚举，复用现有三态；failed 保持 failed，
    # 更严重的优先）。完成度仍由 compute 决定（n_failed 不含 statistics）——区分靠 statistics_status。
    if statistics_status == "crashed" and status == "completed":
        status = "partial"

    # ---- Step 9: assemble + seal handoff (deterministic, sealed_by="run_plan") ----
    payload: dict[str, Any] = {
        "status": status,
        "summary": _default_summary(status, n_total, n_completed, n_failed),
        "paradigm": agg.get("paradigm", plan.get("paradigm", "")),
        "metrics_summary": agg["metrics_summary"],
        "per_subject": agg["per_subject"],
        "output_files": agg["output_files"],
        "data_quality_warnings": agg["data_quality_warnings"],
        "errors": list(agg.get("errors", [])),
        "confidence": 1.0 if status == "completed" else (0.5 if status == "partial" else 0.0),
        "ev19_template": agg.get("ev19_template", plan.get("ev19_template")),
        "inputs": plan.get("inputs", {}),
        "sealed_by": "run_plan",
        "gate_signals": _build_gate_signals(
            agg, status, n_failed,
            statistics_status=statistics_status,
            statistics_error=statistics_error,
        ),
    }
    if statistics_payload is not None:
        payload["statistics"] = statistics_payload

    # ---- Step 9.5: 拆旁路（体积控制，spec 2026-06-18）----
    # outlier_diagnostics（28K）+ output_files 完整路径列表（8.9K）是「大但偶尔需要」
    # 的细节，把主 handoff 顶过 sandbox read_file 50K 截断线，斩掉尾部 gate_signals /
    # data_quality_warnings（data-analyst fast-fail 必读）。拆到旁路文件，主 handoff 留
    # 摘要（count）+ 虚拟路径引用（_ref）。data-analyst 需 outlier 细节时单独读旁路。
    _slim_payload_into_sidecars(payload, workspace)

    try:
        _seal_handoff_to_workspace(CodeExecutorHandoff, "handoff_code_executor.json", payload, workspace)
    except ValueError as e:
        # seal schema 校验失败 = 聚合产物与 schema 不匹配（回归探针）。响亮失败。
        logger.error("[run_metric_plan] seal schema 校验失败: %s", e)
        return _error_result("seal_failed", f"handoff schema 校验失败: {e}")

    # ---- Step 9.6: 确定性导出指标结果表（CSV + JSON）到 outputs/（spec 2026-06-30 C1 模块1）----
    # best-effort 用户产物：handoff 已 sealed（关键产物），导出失败不得中断 run。
    # 从 agg 读（SSOT、预内脏）——不从 sealed payload 读，剥内脏由构造保证。
    # 惰性 import（cycle-safety，同其他 subagents 惰性 import 习语）。
    try:
        from deerflow.subagents.metrics_table_export import export_metrics_table

        export_metrics_table(
            metrics_summary=agg["metrics_summary"],
            per_subject=agg["per_subject"],
            subject_groups=agg.get("subject_groups", {}),
            outputs_dir=workspace.parent / "outputs",
            paradigm=agg.get("paradigm", "") or plan.get("paradigm", ""),
        )
    except Exception:
        logger.warning("[run_metric_plan] metrics table export failed (non-fatal)", exc_info=True)

    result = {
        "status": status,
        "handoff_path": "/mnt/user-data/workspace/handoff_code_executor.json",
        "n_total": n_total,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "failures": failures,
        "gate_signals": payload["gate_signals"],
    }
    logger.info(
        "[run_metric_plan] done: status=%s n_total=%d n_completed=%d n_failed=%d",
        status, n_total, n_completed, n_failed,
    )
    return result


# ============================================================================
# Helpers
# ============================================================================


# Spec 2026-06-18：旁路文件命名 + 主 handoff 内引用字段。
_OUTLIERS_SIDECAR = "handoff_code_executor_outliers.json"
_OUTPUTS_SIDECAR = "handoff_code_executor_outputs.json"
_OUTLIERS_REF = f"/mnt/user-data/workspace/{_OUTLIERS_SIDECAR}"
_OUTPUTS_REF = f"/mnt/user-data/workspace/{_OUTPUTS_SIDECAR}"


def _write_sidecar_json(workspace: Path, filename: str, data: Any) -> None:
    """原子写旁路 JSON（tmp + rename + chmod 0o644，同 seal 模式）。

    旁路文件不进 seal 的 manifest hash 链（主 handoff 的 hash 已覆盖 _ref 路径字符串）。
    """
    import json

    final = workspace / filename
    tmp = workspace / f"{filename}.tmp"
    tmp.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
    os.replace(tmp, final)  # POSIX atomic
    os.chmod(final, 0o644)


def _slim_payload_into_sidecars(payload: dict[str, Any], workspace: Path) -> None:
    """把 outlier_diagnostics + output_files 完整列表从主 handoff 拆到旁路文件。

    就地 mutate ``payload``：移除内嵌大数据，设 *_count + *_ref 摘要字段。空数据不写旁路
    （守护降级路径：line ~514 的 output_files={"metrics": []} 不产空旁路文件）。

    Spec 2026-06-18 §3.3。旁路 _ref 必须是虚拟路径（/mnt/user-data/workspace/...），
    data-analyst 在 sandbox 内 read_file 用虚拟路径（schemas._VIRTUAL_USER_DATA_PREFIX）。
    """
    # --- outlier_diagnostics: 在 payload["statistics"]["outlier_diagnostics"] ---
    stats = payload.get("statistics")
    if isinstance(stats, dict):
        outliers = stats.get("outlier_diagnostics")
        if outliers:  # 非空 list 才拆
            _write_sidecar_json(workspace, _OUTLIERS_SIDECAR, outliers)
            payload["outlier_diagnostics_count"] = len(outliers)
            payload["outlier_diagnostics_ref"] = _OUTLIERS_REF
            # 主 handoff 的 statistics 移除内嵌完整数组（保留 key 为 [] 兼容旧读取方）。
            stats["outlier_diagnostics"] = []
        else:
            payload["outlier_diagnostics_count"] = 0
            payload["outlier_diagnostics_ref"] = None
    else:
        payload.setdefault("outlier_diagnostics_count", 0)
        payload.setdefault("outlier_diagnostics_ref", None)

    # --- output_files: 完整产物路径列表 ---
    output_files = payload.get("output_files")
    if output_files:  # 非空 dict
        # 统计产物文件数（聚合 metrics 路径列表长度，回退到 dict 内所有 str/list 元素数）。
        metrics_list = output_files.get("metrics") if isinstance(output_files, dict) else None
        if isinstance(metrics_list, list):
            count = len(metrics_list)
        else:
            count = sum(
                len(v) if isinstance(v, list) else (1 if isinstance(v, str) else 0)
                for v in output_files.values()
            ) if isinstance(output_files, dict) else 0
        if count > 0:
            _write_sidecar_json(workspace, _OUTPUTS_SIDECAR, output_files)
            payload["output_files_count"] = count
            payload["output_files_ref"] = _OUTPUTS_REF
            # 主 handoff 清空 output_files（spec §3.2 默认：清空，无下游 subagent 消费）。
            payload["output_files"] = {}
        else:
            payload.setdefault("output_files_count", 0)
            payload.setdefault("output_files_ref", None)
    else:
        payload.setdefault("output_files_count", 0)
        payload.setdefault("output_files_ref", None)


def _execute_tasks(
    tasks: list[tuple[str, list[str], str]],
    *,
    on_error: str = "continue",
    timeout: int = PER_TASK_TIMEOUT_SECONDS,
    runner=None,
    path_env: dict[str, str] | None = None,
) -> tuple[dict[str, tuple[int, str]], list[dict[str, str]]]:
    """Execute all compute tasks, return (results, failures).

    Spec S4 §1.3 / §1.5: per-task timeout (不杀整池), on_error policy。

    Args:
      tasks: list of (script, args, task_id).
      on_error: "continue" (跑完全部) | "abort" (遇首个失败即停，不再等待 / 取消未启动).
      timeout: per-task wall-clock seconds.
      runner: callable(script, args, task_id) -> (task_id, rc, error). None = default
        _run_metric_task via ProcessPoolExecutor. **测试注入同步 runner 绕开 fork/pickle**。
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
        # 同步串行（测试路径）：on_error=abort 时遇首个失败即停。
        for script, args, tid in tasks:
            try:
                _tid, rc, err = runner(script, args, tid)
            except BaseException as e:  # noqa: BLE001
                rc, err = 1, f"runner error: {type(e).__name__}: {e}"
            results[tid] = (rc, err)
            if rc != 0:
                failures.append({"id": tid, "error": (err or "")[:500]})
                if on_error == "abort":
                    break
        return results, failures

    # 生产路径：ProcessPoolExecutor。worker env 经 initializer 在子进程内设置（不污染父进程）。
    pool = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_worker_init,
        initargs=(path_env or {},),
    )
    try:
        future_to_tid = {pool.submit(_run_metric_task, sc, ar, tid): tid for (sc, ar, tid) in tasks}
        ordered = list(future_to_tid.items())  # 提交顺序
        # 按提交顺序逐个 result(timeout=)——每个 future 有独立 per-task 超时预算。真正挂死的
        # task 会在它自己那轮触发 timeout；若改用 as_completed，挂死 task 永不被 yield、其
        # 超时永不触发，整轮 wait 会卡死，故坚持提交序逐个等。
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
                # 真正停止：取消所有尚未启动的 future（排队未启动的会被 cancel；运行中的进程
                # 无法取消，这是进程池固有性质）。被取消项标 aborted 计入 failures。
                logger.warning("[run_metric_plan] on_error=abort，遇失败即停，取消未启动 task")
                for other_fut, other_tid in ordered[idx + 1:]:
                    other_fut.cancel()
                    results[other_tid] = (1, "aborted (on_error=abort 前序失败)")
                    failures.append({"id": other_tid, "error": "aborted"})
                break
    finally:
        # cancel_futures：回收时取消仍排队未启动的 future；wait=True 等运行中的退出。
        pool.shutdown(wait=True, cancel_futures=True)
    return results, failures


def _derive_status(agg_status: str, n_total: int, n_failed: int) -> str:
    """Map aggregation status to final status (pure function, Spec S4 §1.5).

    完成度纯由磁盘产物对账决定（plan 期望 vs 实际 m_*.json），不从 task_results 推——
    task 失败（脚本异常）通常意味着产物没写出，聚合器 glob 时自然把它计进 missing →
    status 降级。所以这里只透传聚合器算出的 status，不另用 n_failed 篡改（守 §1.5：
    完成度是 (plan, artifacts) 的纯函数，不是 (plan, task_results)）。

    唯一例外：n_failed>0 但磁盘对账却 completed（脚本 rc≠0 但产物写成功了，罕见）——
    仍报 failed。确定性工具下任何响亮失败都该如实上报（§3 迁移风险 1），静默吞掉
    会让「脚本半成功」污染下游。
    """
    if agg_status == "failed":
        return "failed"
    if n_failed > 0:
        return "failed"
    return agg_status  # completed or partial


def _default_summary(status: str, n_total: int, n_completed: int, n_failed: int) -> str:
    """Mechanical default summary (Spec S4 §1.6: LLM 增润可选，缺席不致 handoff 无效)."""
    if status == "completed":
        return f"{n_completed}/{n_total} metrics completed (run_metric_plan)"
    if status == "partial":
        return f"{n_completed}/{n_total} metrics completed, partial (run_metric_plan)"
    return f"{n_completed}/{n_total} metrics completed, {n_failed} failed (run_metric_plan)"


def _build_gate_signals(
    agg: dict[str, Any], status: str, n_failed: int,
    *,
    statistics_status: Literal["ok", "crashed", "absent_by_design"] = "absent_by_design",
    statistics_error: str | None = None,
) -> dict[str, Any]:
    """Build gate_signals for lead decision (mirrors code_executor [gate_signals] format)."""
    warnings = agg.get("data_quality_warnings", []) or []
    critical = [w for w in warnings if w.get("severity") == "critical"]
    warn = [w for w in warnings if w.get("severity") == "warning"]
    critical_items = [
        (w.get("message", "") or "")[:80] for w in critical[:5]
    ]
    errors = agg.get("errors", []) or []

    # statistical_validity: skipped 当 statistics skip_reason 非空（单样本/n_per_group<2）。
    stat_validity = "ok"
    if status == "failed":
        stat_validity = "failed"
    elif critical:
        stat_validity = "warning"

    return {
        "constitution_acknowledged": True,
        "data_quality": {
            "critical_count": len(critical),
            "warning_count": len(warn),
            "critical_items": critical_items,
        },
        "statistical_validity": stat_validity,
        "errors_count": len(errors),
        # P2: statistics 子步骤三态降级信号。SSOT: handoff_schemas.GateSignals.statistics_status。
        "statistics_status": statistics_status,
        "statistics_error": statistics_error,
    }


def _seal_failed(workspace: Path, model_cls: type, message: str) -> None:
    """Best-effort seal a minimal failed handoff (plan missing/unparseable paths)."""
    # 惰性 import 避免闭环；本函数在工具主流程惰性 import 之后被调，但守独立可调。
    import json

    payload = {
        "status": "failed",
        "summary": message,
        "paradigm": "",
        "metrics_summary": {},
        "per_subject": {},
        "output_files": {"metrics": []},
        "data_quality_warnings": [],
        "errors": [message],
        "confidence": "low",
        "ev19_template": None,
        "inputs": {},
        "sealed_by": "run_plan",
        "gate_signals": {
            "constitution_acknowledged": True,
            "data_quality": {"critical_count": 0, "warning_count": 0, "critical_items": []},
            "statistical_validity": "failed",
            "errors_count": 1,
        },
    }
    try:
        from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace

        _seal_handoff_to_workspace(model_cls, "handoff_code_executor.json", payload, workspace)
    except Exception as e:  # noqa: BLE001
        logger.error("[run_metric_plan] _seal_failed 也失败: %s", e)
        # 最后兜底：直接写裸 JSON（不经 schema 校验），保证磁盘上有 handoff 文件。
        try:
            (workspace / "handoff_code_executor.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass


def _error_result(code: str, message: str) -> dict[str, Any]:
    """Build a standardised error return dict (tool-level, no seal side effect beyond _seal_failed)."""
    return {
        "status": "failed",
        "error_code": code,
        "message": message,
        "handoff_path": "/mnt/user-data/workspace/handoff_code_executor.json",
        "n_total": 0,
        "n_completed": 0,
        "n_failed": 0,
        "failures": [],
        "gate_signals": {
            "constitution_acknowledged": True,
            "data_quality": {"critical_count": 0, "warning_count": 0, "critical_items": []},
            "statistical_validity": "failed",
            "errors_count": 1,
        },
    }

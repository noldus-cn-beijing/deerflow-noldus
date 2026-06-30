#!/usr/bin/env python3
"""手段 c 复现：确定性重放 run_chart_plan 的 ProcessPoolExecutor 渲染路径。

目标：用与生产 run_chart_plan 完全相同的代码路径，对 thread 89be7344 的真实
plan_charts.json + inputs 跑渲染，回答：
  1. PermissionError 的精确 traceback 栈
  2. 为何 s0 成功、s1..s27 失败
  3. bash 路径 vs 进程池路径的 argv 差异

用法：
  cd packages/agent/backend
  PYTHONPATH=. python3 scripts/repro/run_chart_plan_repro.py

只读取证，不改生产代码、不写 thread workspace（输出到 /tmp/repro-out）。
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ---- 真实 thread 数据 ----
THREAD_ID = "89be7344-17fc-4fa1-a1ef-ce469d761c2a"
USER_ID = "e281f251-59cc-4dd4-a1d2-66f2f0ebce01"
BACKEND = Path("/home/wangqiuyang/noldus-insight/packages/agent/backend")
WS_REAL = BACKEND / ".deer-flow/users" / USER_ID / "threads" / THREAD_ID / "user-data"
WORKSPACE = WS_REAL / "workspace"
UPLOADS = WS_REAL / "uploads"
OUTPUTS_REAL = WS_REAL / "outputs"

REPRO_OUT = Path("/tmp/repro-out-89be7344")


# ---- 复制生产 env 构造逻辑（_build_path_env 的等价）----
def build_path_env() -> dict[str, str]:
    """镜像 sandbox/tools.py:_build_path_env + _thread_virtual_to_actual_mappings。"""
    mappings = {
        "/mnt/user-data/workspace": str(WORKSPACE),
        "/mnt/user-data/uploads": str(UPLOADS),
        "/mnt/user-data/outputs": str(OUTPUTS_REAL),
        "/mnt/user-data": str(WS_REAL),  # common parent
    }
    env = {}
    for container, local in mappings.items():
        key = "DEERFLOW_PATH_" + container.strip("/").replace("/", "_").replace("-", "_").upper()
        env[key] = local
    return env


# ---- 镜像 _worker_init（run_chart_plan_tool.py:64-79）----
def worker_init(path_env: dict[str, str]) -> None:
    if path_env:
        os.environ.update(path_env)
    os.environ.setdefault("MPLBACKEND", "Agg")


# ---- 镜像 _run_chart_task（run_chart_plan_tool.py:82-102），但带完整 traceback ----
def run_chart_task_trace(script: str, args: list[str], task_id: str, repro_outputs: str):
    """同生产 _run_chart_task，但失败时返回完整 traceback（生产只返 type+msg）。"""
    try:
        mod = importlib.import_module(script)
        rc = mod.main(args)
        return (task_id, int(rc or 0), "", "")
    except SystemExit as e:
        return (task_id, int(e.code) if isinstance(e.code, int) else 1, f"SystemExit({e.code})", "")
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        return (task_id, 1, f"{type(e).__name__}: {e}", tb)


def main():
    print("=" * 70)
    print("手段 c 复现：run_chart_plan ProcessPoolExecutor 渲染路径")
    print("=" * 70)

    # 准备隔离的 repro outputs 目录（不污染真实 outputs）
    if REPRO_OUT.exists():
        shutil.rmtree(REPRO_OUT)
    REPRO_OUT.mkdir(parents=True)

    plan_path = WORKSPACE / "plan_charts.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    charts = plan["charts"]
    print(f"plan_charts.json: {len(charts)} charts")
    print(f"distinct scripts: {sorted(set(c['script'] for c in charts))}")
    print()

    # ---- 诊断 1: 先看一个 per_subject chart 的真实 args（关键）----
    bar_charts = [c for c in charts if c["id"] == "open_arm_time_ratio_bar"]
    print(f"=== open_arm_time_ratio_bar: {len(bar_charts)} entries ===")
    for c in bar_charts[:2]:
        print(f"  subject_index={c.get('subject_index')} output={c['output']}")
        print(f"  script={c['script']}")
        print(f"  args={c['args']}")
        # 看 inputs JSON 内容
        inputs_arg = next((a for i, a in enumerate(c['args']) if a == '--inputs' and i+1 < len(c['args'])), None)
        if inputs_arg:
            inputs_path = c['args'][c['args'].index('--inputs') + 1]
            real_inputs = inputs_path
            # resolve via env logic
            if real_inputs.startswith("/mnt/user-data/workspace"):
                real_inputs = str(WORKSPACE) + real_inputs[len("/mnt/user-data/workspace"):]
            p = Path(real_inputs)
            if p.exists():
                print(f"  inputs.json ({real_inputs}): {p.read_text()[:200]}")
            else:
                print(f"  inputs.json MISSING: {real_inputs}")
        print()

    # ---- 诊断 2: s0 vs s1 args diff（回答"为何 s0 成功"）----
    s0 = next(c for c in bar_charts if c.get("subject_index") == 0)
    s1 = next((c for c in bar_charts if c.get("subject_index") == 1), None)
    print("=== s0 vs s1 args diff ===")
    print(f"s0 args: {s0['args']}")
    if s1:
        print(f"s1 args: {s1['args']}")
        print(f"args identical except subject? {s0['args'][:2] == s1['args'][:2] if len(s0['args'])>2 else 'n/a'}")
    print()

    # ---- 诊断 3: 真实进程池渲染（重放生产路径）----
    # 改 output 指向 REPRO_OUT（不污染真实 outputs）
    print("=== 进程池渲染重放（改 output → /tmp/repro-out，其余原样）===")
    path_env = build_path_env()
    print(f"path_env keys: {sorted(path_env.keys())}")
    print()

    # 选每类前 2 个 subject（s0, s1）+ aggregate，共 9 个 task，足以暴露 s0/s1 差异
    test_charts = []
    for cid in ["box_open_arm", "open_arm_time_ratio_bar", "zone_entry_distribution", "trajectory", "heatmap"]:
        matches = [c for c in charts if c["id"] == cid]
        test_charts.extend(matches[:2])  # 前 2 个（通常 s0, s1）

    tasks = []
    for c in test_charts:
        args = list(c["args"])
        # 改 --output 到 REPRO_OUT（镜像生产：output 是虚拟路径，脚本 resolve 后写）
        if "--output" in args:
            i = args.index("--output")
            orig_out = args[i + 1]
            # 保持虚拟形态（/mnt/user-data/outputs/...），但 resolve_sandbox_path 会用 env 解析到真实
            # 生产里 output 是 /mnt/user-data/outputs/...；为不污染，我们让它解析到 REPRO_OUT
            # 方法：临时在 path_env 里把 outputs 指向 REPRO_OUT（但那会影响 inputs 解析）
            # 更简单：直接把 args 里的 output 改成真实 REPRO_OUT 路径（模拟"argv 已预解析"= F1 修复后形态）
            fname = Path(orig_out).name
            args[i + 1] = str(REPRO_OUT / fname)
        tasks.append((c["script"], args, f"{c['id']}_s{c.get('subject_index', 'agg')}"))

    pool = ProcessPoolExecutor(max_workers=4, initializer=worker_init, initargs=(path_env,))
    try:
        futures = {pool.submit(run_chart_task_trace, sc, ar, tid, str(REPRO_OUT)): tid for (sc, ar, tid) in tasks}
        for fut in list(futures):
            tid, rc, err, tb = fut.result(timeout=120)
            status = "OK" if rc == 0 else "FAIL"
            print(f"  [{status}] {tid}: rc={rc} {err[:120] if err else ''}")
            if rc != 0 and tb:
                print(f"        --- traceback ---")
                for line in tb.strip().splitlines()[-8:]:
                    print(f"        {line}")
                print()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    print()
    print("=== REPRO_OUT 落盘 png ===")
    for f in sorted(REPRO_OUT.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")

    # ---- 诊断 4: 对比 — 不改 output（纯虚拟路径，最贴近生产），看是否 PermissionError ----
    print()
    print("=== 诊断 4: output 保持 /mnt/user-data/outputs 虚拟路径（最贴近生产）===")
    REPRO_OUT2 = Path("/tmp/repro-out2-89be7344")
    if REPRO_OUT2.exists():
        shutil.rmtree(REPRO_OUT2)
    REPRO_OUT2.mkdir(parents=True)
    # path_env 把 outputs 指向 REPRO_OUT2（隔离），workspace/uploads 指真实
    path_env2 = dict(path_env)
    path_env2["DEERFLOW_PATH_MNT_USER_DATA_OUTPUTS"] = str(REPRO_OUT2)

    s1_chart = next(c for c in charts if c["id"] == "open_arm_time_ratio_bar" and c.get("subject_index") == 1)
    print(f"  s1 output (virtual): {s1_chart['output']}")
    print(f"  s1 args: {s1_chart['args']}")
    pool2 = ProcessPoolExecutor(max_workers=2, initializer=worker_init, initargs=(path_env2,))
    try:
        fut = pool2.submit(run_chart_task_trace, s1_chart["script"], list(s1_chart["args"]), "oatrb_s1_virt", str(REPRO_OUT2))
        tid, rc, err, tb = fut.result(timeout=120)
        print(f"  [{'OK' if rc==0 else 'FAIL'}] {tid}: rc={rc} {err[:200]}")
        if tb:
            print("  traceback:")
            for line in tb.strip().splitlines()[-12:]:
                print(f"    {line}")
    finally:
        pool2.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()

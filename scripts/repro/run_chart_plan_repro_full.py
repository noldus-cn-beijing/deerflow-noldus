#!/usr/bin/env python3
"""手段 c 决定性测试：完整重放 113 charts 的 run_chart_plan 路径，回答 s0 异常。

把 path_env 的 OUTPUTS 指向一个干净的真实目录（模拟生产 outputs），workspace/uploads
指真实 thread 数据。完整跑 113 task，统计落盘数。这会确定：
  - 到底是 1/113（复现生产）还是 0/113 还是 28+/113
  - box_open_arm 是否总成功（os.makedirs 差异）
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

THREAD_ID = "89be7344-17fc-4fa1-a1ef-ce469d761c2a"
USER_ID = "e281f251-59cc-4dd4-a1d2-66f2f0ebce01"
BACKEND = Path("/home/wangqiuyang/noldus-insight/packages/agent/backend")
WS_REAL = BACKEND / ".deer-flow/users" / USER_ID / "threads" / THREAD_ID / "user-data"
WORKSPACE = WS_REAL / "workspace"
UPLOADS = WS_REAL / "uploads"

# 隔离的 outputs，用真实物理路径（path_env 把 /mnt/user-data/outputs 指这里）
REPRO_OUTPUTS = Path("/tmp/repro-outputs-full")


def build_path_env() -> dict[str, str]:
    mappings = {
        "/mnt/user-data/workspace": str(WORKSPACE),
        "/mnt/user-data/uploads": str(UPLOADS),
        "/mnt/user-data/outputs": str(REPRO_OUTPUTS),
        "/mnt/user-data": str(WS_REAL),
    }
    return {
        "DEERFLOW_PATH_" + c.strip("/").replace("/", "_").replace("-", "_").upper(): l
        for c, l in mappings.items()
    }


def worker_init(path_env):
    if path_env:
        os.environ.update(path_env)
    os.environ.setdefault("MPLBACKEND", "Agg")


def run_task(script, args, tid):
    import io, contextlib
    try:
        mod = importlib.import_module(script)
        # 吞掉 stdout（emit_result 的 [result] 行）
        with contextlib.redirect_stdout(io.StringIO()):
            rc = mod.main(args)
        return (tid, int(rc or 0), "")
    except BaseException as e:
        return (tid, 1, f"{type(e).__name__}: {e}")


def main():
    if REPRO_OUTPUTS.exists():
        shutil.rmtree(REPRO_OUTPUTS)
    REPRO_OUTPUTS.mkdir(parents=True)

    plan = json.loads((WORKSPACE / "plan_charts.json").read_text())
    charts = plan["charts"]
    print(f"Running all {len(charts)} charts via ProcessPoolExecutor (prod-faithful)...")
    print(f"OUTPUTS virtual /mnt/user-data/outputs -> real {REPRO_OUTPUTS}")
    print()

    # 关键：args 原样透传（含 /mnt/user-data/outputs 虚拟 output）—— 完全镜像生产
    tasks = [(c["script"], list(c["args"]), f"{c['id']}_s{c.get('subject_index','agg')}") for c in charts]

    path_env = build_path_env()
    results = {}
    pool = ProcessPoolExecutor(max_workers=8, initializer=worker_init, initargs=(path_env,))
    try:
        futs = {pool.submit(run_task, sc, ar, tid): tid for sc, ar, tid in tasks}
        for fut in futs:
            tid, rc, err = fut.result(timeout=120)
            results[tid] = (rc, err)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    ok = [t for t, (rc, e) in results.items() if rc == 0]
    fail = [(t, e) for t, (rc, e) in results.items() if rc != 0]
    print(f"=== RESULT: {len(ok)}/{len(tasks)} succeeded, {len(fail)} failed ===")
    print()
    print("=== FAILED (first 8, with error) ===")
    for tid, err in fail[:8]:
        print(f"  {tid}: {err[:100]}")
    print()
    print("=== distinct failure error types ===")
    from collections import Counter
    errtypes = Counter(e.split(":")[0] for _, e in fail)
    for et, n in errtypes.most_common():
        print(f"  {n:3d}  {et}")
    print()
    print("=== which chart IDs succeeded? ===")
    ok_ids = Counter(t.rsplit("_s", 1)[0] for t in ok)
    for cid, n in ok_ids.most_common():
        print(f"  {n:3d}  {cid}")
    print()
    print(f"=== PNGs actually on disk in {REPRO_OUTPUTS}: {len(list(REPRO_OUTPUTS.glob('*.png')))} ===")
    # 检查 box_open_arm 是否成功（aggregate，唯一走 _resolve_output_path→makedirs 的）
    print(f"plot_box_open_arm.png exists? {(REPRO_OUTPUTS/'plot_box_open_arm.png').exists()}")
    print(f"plot_open_arm_time_ratio_bar_s0.png exists? {(REPRO_OUTPUTS/'plot_open_arm_time_ratio_bar_s0.png').exists()}")
    print(f"plot_open_arm_time_ratio_bar_s1.png exists? {(REPRO_OUTPUTS/'plot_open_arm_time_ratio_bar_s1.png').exists()}")


if __name__ == "__main__":
    main()

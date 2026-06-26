#!/usr/bin/env python3
"""Standalone tests for the analyze.py perf panel (Panel C).

Run:  python3 scripts/analyze.test.py
Exit 0 = pass. Fabricates a minimal thread workspace + perf.json for each
scenario and asserts the panel output contains the expected verdict tokens.
Keeps the Python perf logic from silently drifting from lib.js (which has its
own node tests for evaluatePerf — same field names + thresholds).
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYZE = HERE / "analyze.py"
assert ANALYZE.exists(), f"analyze.py not found at {ANALYZE}"

WS_USERS_RELPATH = Path("users")  # under tmp root

pass_n, fail_n = 0, 0
def run_case(name, perf_obj, expect_tokens, forbid_tokens=()):
    global pass_n, fail_n
    tmp = Path(tempfile.mkdtemp(prefix="e2e-anatest-"))
    try:
        ws_root = tmp / WS_USERS_RELPATH
        base = ws_root / "u1" / "threads" / "T1" / "user-data"
        (base / "workspace").mkdir(parents=True)
        (base / "outputs").mkdir(parents=True)
        out = tmp / "out"; out.mkdir()
        if perf_obj is not None:
            (out / "perf.json").write_text(json.dumps(perf_obj), encoding="utf-8")
        env = dict(os.environ)
        env.update(E2E_THREAD="T1", E2E_USER_ID="u1", E2E_OUT=str(out), E2E_THREAD_WS_ROOT=str(ws_root))
        res = subprocess.run([sys.executable, str(ANALYZE)], env=env, capture_output=True, text=True)
        text = res.stdout
        # extract Panel C section
        if "Panel C" not in text:
            print(f"  ✗ {name}: Panel C not in output (analyze.py likely FATALed)")
            print(f"    stderr: {res.stderr[-400:]}")
            fail_n += 1; return
        panel = text.split("Panel C", 1)[1]
        ok = all(tok in panel for tok in expect_tokens)
        bad = [t for t in forbid_tokens if t in panel]
        if ok and not bad:
            print(f"  ✓ {name}"); pass_n += 1
        else:
            print(f"  ✗ {name}\n    expected {expect_tokens}\n    forbidden found {bad}\n    panel: {panel[:300]}")
            fail_n += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

GREEN = {"build": "prod", "actions": [
    {"name": "switch_back_thread", "longtask_max_ms": 80, "longtask_total_ms": 200, "interaction_ms": [120, 130]}]}
RED = {"build": "prod", "actions": [
    {"name": "switch_back_thread", "longtask_max_ms": 350, "longtask_total_ms": 900, "interaction_ms": [400]}]}
DEV = {"build": "dev", "actions": [
    {"name": "switch_back_thread", "longtask_max_ms": 80, "longtask_total_ms": 200, "interaction_ms": [120]}]}

run_case("green prod under threshold", GREEN,
         expect_tokens=["✅ 绿", "longtask_max=80ms", "所有 P0 指标在阈值内"],
         forbid_tokens=["退化告警"])
run_case("red prod over threshold (the 切回卡顿 signal)", RED,
         expect_tokens=["🔴 退化告警", "longtask_max=350ms", "性能退化"],
         forbid_tokens=["所有 P0 指标在阈值内"])
run_case("dev build is SKIPPED (never red/green)", DEV,
         expect_tokens=["PERF: skipped", "dev build"],
         forbid_tokens=["退化告警", "所有 P0 指标在阈值内", "✅ 绿"])
run_case("perf.json absent => skipped", None,
         expect_tokens=["PERF: skipped", "perf.json 不存在"],
         forbid_tokens=["退化告警", "✅ 绿"])
run_case("prod but empty actions => skipped", {"build": "prod", "actions": []},
         expect_tokens=["perf.actions 为空"],
         forbid_tokens=["退化告警", "✅ 绿"])

print(f"\n{pass_n} passed, {fail_n} failed")
sys.exit(0 if fail_n == 0 else 1)

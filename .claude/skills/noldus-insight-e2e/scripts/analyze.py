#!/usr/bin/env python3
"""noldus-insight-e2e forensic panel (read-only).

Reads a thread's workspace handoffs / plan / outputs and prints a structured
verdict. Parameterized via env:
  E2E_THREAD          thread id (or read from E2E_OUT/thread.txt)
  E2E_USER_ID         user id (from ensure-login JWT sub)
  E2E_OUT             run output dir (write analyze.txt here too)
  E2E_THREAD_WS_ROOT  optional override of the .deer-flow/users root
  E2E_GATEWAY_LOG     optional gateway log path (to surface the fill race)

Panels:
  A — data-analyst seal (sealed_by=finalize vs degraded preset/in_progress;
      surfaces the concurrent-write race if present)
  B — chart pseudo-completion (chart_files claimed vs png on disk;
      plan_charts output_mode distribution; aggregate-missing)
  other — code-executor / report-writer status + report.md existence
"""
from __future__ import annotations
import json, os, sys, re
from pathlib import Path
from collections import Counter

THREAD = os.environ.get("E2E_THREAD") or ""
OUT = os.environ.get("E2E_OUT", "")
if not THREAD and OUT:
    t = Path(OUT) / "thread.txt"
    if t.exists():
        THREAD = t.read_text().strip()
USER = os.environ.get("E2E_USER_ID", "")
WS_USERS = Path(os.environ.get("E2E_THREAD_WS_ROOT", "") or
                "/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users")
GATEWAY_LOG = os.environ.get("E2E_GATEWAY_LOG",
    "/home/wangqiuyang/noldus-insight/packages/agent/logs/gateway.log")

if not THREAD or not USER:
    print(f"FATAL: need E2E_THREAD ({THREAD!r}) and E2E_USER_ID ({USER!r})", file=sys.stderr)
    sys.exit(1)

BASE = WS_USERS / USER / "threads" / THREAD / "user-data"
WS = BASE / "workspace"
OUTD = BASE / "outputs"
if not BASE.exists():
    print(f"FATAL: thread user-data dir not found: {BASE}", file=sys.stderr)
    sys.exit(1)

lines = []
def p(s=""): lines.append(s)

def load(name):
    fp = WS / name
    if not fp.exists(): return None
    try: return json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e: return {"_parse_error": str(e)}

# snapshot the gateway log ONCE (bounded) for the race / failure signatures
gw_excerpt = []
try:
    gtext = Path(GATEWAY_LOG).read_text(encoding="utf-8", errors="replace") if Path(GATEWAY_LOG).exists() else ""
    for pat in [r"handoff_data_analyst\.json\.tmp.*No such", r"unreadable/invalid JSON",
                r"Tool execution failed.*fill_data_analyst", r"Traceback", r"aggregate 图未全部落盘"]:
        for m in re.finditer(pat, gtext):
            seg = gtext[max(0, m.start()-60): m.end()+20].replace("\n", " ")
            if seg not in gw_excerpt: gw_excerpt.append(seg)
except Exception: pass

p("=" * 70)
p("Panel A — data-analyst seal (#187 fix / degradation)")
p("=" * 70)
da = load("handoff_data_analyst.json")
if da is None:
    p("  ❌ handoff_data_analyst.json 不存在 (data-analyst 未跑/崩溃)")
else:
    kf = da.get("key_findings") or []
    p(f"  status       = {da.get('status')!r}")
    p(f"  sealed_by    = {da.get('sealed_by')!r}   (finalize=封口成功; preset/in_progress=未封口=降级)")
    p(f"  key_findings = {len(kf)} 条")
    for i, k in enumerate(kf[:5]): p(f"    [{i}] {str(k)[:150]}")
    for f in ("outlier_findings", "method_warnings", "recommendations", "errors", "quality_warnings", "parameter_audit_findings"):
        p(f"  {f:24}= {len(da.get(f) or [])} 条")
    if da.get("sealed_by") == "finalize" and da.get("status") in ("completed", "partial", "failed"):
        p("  ✅ #187 修复生效: data-analyst 走到 finalize 封口")
    elif da.get("sealed_by") == "preset" or da.get("status") == "in_progress":
        p("  ❌ 降级: sealed_by=preset / status=in_progress (data-analyst 未 finalize)")
    else:
        p(f"  ⚠️ 待判: sealed_by={da.get('sealed_by')} status={da.get('status')}")
    if gw_excerpt:
        p(f"  ⚠️ gateway 检出并发竞态/异常信号 ({len(gw_excerpt)} 条):")
        for s in gw_excerpt[:3]: p(f"     …{s[:160]}")

p("")
p("=" * 70)
p("Panel B — chart 伪完成 (ETHO-10)")
p("=" * 70)
pngs = sorted(x.name for x in OUTD.glob("*.png")) if OUTD.exists() else []
plot_pngs = sorted(x.name for x in OUTD.glob("plot_*.png")) if OUTD.exists() else []
p(f"  outputs/*.png      = {len(pngs)}")
p(f"  outputs/plot_*.png = {len(plot_pngs)}  (auto-seal 只认此前缀)")
cm = load("handoff_chart_maker.json")
if cm is None:
    p("  handoff_chart_maker.json 不存在")
else:
    cf = cm.get("chart_files") or []
    fc = cm.get("failed_charts") or []
    rc = cm.get("remaining_charts") or []
    p(f"  handoff status={cm.get('status')!r} sealed_by={cm.get('sealed_by')!r}")
    p(f"    (model=自调seal / after_agent_artifacts=auto-seal / executor_artifacts)")
    p(f"  chart_files={len(cf)} failed_charts={len(fc)} remaining_charts={len(rc)}")
    if cf:
        rendered = set(pngs)
        missing = [f.rsplit("/", 1)[-1] for f in cf if f.rsplit("/", 1)[-1] not in rendered]
        p(f"  chart_files 实存 = {len(cf)-len(missing)}/{len(cf)}; 不存在 = {missing[:10]}")
plan = load("plan_charts.json")
if plan is None:
    p("  plan_charts.json 不存在")
else:
    charts = plan.get("charts") or []
    modes = Counter(c.get("output_mode") for c in charts if isinstance(c, dict))
    p(f"  plan charts={len(charts)} output_mode 分布={dict(modes)}")
    p(f"    (aggregate=组间对比 must_have; per_subject=个体图, 2.2 门不对账)")
    p(f"  skipped[]={len(plan.get('skipped') or [])} 条")
    agg = [c.get("output", "").rsplit("/", 1)[-1]
           for c in charts if isinstance(c, dict) and c.get("output_mode") == "aggregate" and isinstance(c.get("output"), str)]
    rendered = set(pngs)
    missing_agg = [a for a in agg if a not in rendered]
    p(f"  aggregate outputs={agg}")
    p(f"  aggregate 未落盘={missing_agg}  (非空且 completed → 2.2 门应抛 ValueError)")
    cf_missing = [f.rsplit('/',1)[-1] for f in cf if f.rsplit('/',1)[-1] not in rendered]
    if cm and cm.get("status") == "completed" and cf_missing:
        p(f"  ⚠️ completed 但 {len(cf_missing)} 个 chart_files 不在磁盘 → 产物真实性漏洞 (ETHO-10)")

p("")
p("=" * 70)
p("Panel other — code-executor / report-writer / report.md")
p("=" * 70)
ce = load("handoff_code_executor.json")
if ce: p(f"  code-executor: status={ce.get('status')} sealed_by={ce.get('sealed_by')} paradigm={ce.get('paradigm')}")
else: p("  code-executor: handoff 不存在")
rw = load("handoff_report_writer.json")
report_md = (OUTD / "report.md").exists() if OUTD.exists() else False
if rw:
    p(f"  report-writer: status={rw.get('status')} sealed_by={rw.get('sealed_by')} report_path={rw.get('report_path')}")
else:
    p("  report-writer: handoff 不存在")
p(f"  outputs/report.md 存在 = {report_md}")
# token-expiry-mid-run signature: no report-writer handoff AND 401 in api log
if not rw:
    api = Path(OUT) / "api-log.json"
    if api.exists():
        try:
            for entry in json.loads(api.read_text()):
                if entry.get("status") in (401, "401"):
                    p("  ⚠️ api 401 + 无 report-writer handoff → 疑似 token 过期, 重跑")
                    break
        except Exception: pass

# ---- Panel C — perf (task B) ------------------------------------------------
# Reads $E2E_OUT/perf.json written by run-e2e.cjs. Thresholds MIRROR lib.js
# defaultThresholds exactly (single source of truth lives in lib.js; if you
# change one, change both). Re-baseline via E2E_PERF_THRESHOLDS_JSON (a JSON
# object overriding any subset) AFTER a clean prod run — never pick from thin
# air (spec B.3). Dev builds are SKIPPED, never red/green (spec B.0).
PERF_THRESH = {"longtask_max_ms": 200, "longtask_total_ms": 800, "interaction_p95_ms": 300}
_thresh_env = os.environ.get("E2E_PERF_THRESHOLDS_JSON", "")
if _thresh_env:
    try:
        PERF_THRESH.update({k: float(v) for k, v in json.loads(_thresh_env).items()})
    except Exception as e:
        p(f"  ⚠️ E2E_PERF_THRESHOLDS_JSON parse failed ({e}), using defaults")

p("")
p("=" * 70)
p("Panel C — 前端性能 (longtask / 交互耗时, 仅 prod 出红绿)")
p("=" * 70)
perf_path = Path(OUT) / "perf.json" if OUT else None
perf = None
if perf_path and perf_path.exists():
    try: perf = json.loads(perf_path.read_text(encoding="utf-8"))
    except Exception as e: p(f"  ⚠️ perf.json 解析失败: {e}")

def _pct(arr, q):
    arr = sorted(x for x in arr if isinstance(x, (int, float)))
    if not arr: return None
    idx = (q / 100.0) * (len(arr) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(arr) - 1)
    if lo == hi: return arr[lo]
    return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)

if perf is None:
    p("  ⏭️ PERF: skipped (perf.json 不存在 — 未跑性能段或 E2E_PERF=0)")
elif str(perf.get("build", "")).lower() != "prod":
    p(f"  ⏭️ PERF: skipped ({perf.get('build', '?')} build — 仅 prod build 可信, 不出红绿)")
    p(f"     (dev = Turbopack + HMR + sourcemap, 数据是噪声; 指定 E2E_PERF_BASE_URL 指向 prod 再跑)")
    if perf.get("actions"):
        p(f"     原始 trace 仍记录了 {len(perf['actions'])} 个动作 (仅参考, 不判定):")
        for a in perf["actions"]:
            p(f"       {a.get('name')}: longtask_max={a.get('longtask_max_ms')} count={a.get('longtask_count')}")
else:
    actions = perf.get("actions") or []
    if not actions:
        p("  ⏭️ PERF: prod build 但无动作采集 (perf.actions 为空)")
    else:
        any_red = False
        for a in actions:
            if a.get("error") or a.get("skipped"):
                p(f"  • {a.get('name')}: 跳过 ({a.get('error') or a.get('skipped')})")
                continue
            checks = []
            lt_max = a.get("longtask_max_ms")
            lt_tot = a.get("longtask_total_ms")
            i_p95 = _pct(a.get("interaction_ms") or [], 95)
            if isinstance(lt_max, (int, float)):
                ok = lt_max <= PERF_THRESH["longtask_max_ms"]; any_red = any_red or not ok
                checks.append(f"longtask_max={lt_max}ms{'≤' if ok else '>'}{int(PERF_THRESH['longtask_max_ms'])}{'✅' if ok else '🔴'}")
            if isinstance(lt_tot, (int, float)):
                ok = lt_tot <= PERF_THRESH["longtask_total_ms"]; any_red = any_red or not ok
                checks.append(f"longtask_total={lt_tot}ms{'≤' if ok else '>'}{int(PERF_THRESH['longtask_total_ms'])}{'✅' if ok else '🔴'}")
            if i_p95 is not None:
                ok = i_p95 <= PERF_THRESH["interaction_p95_ms"]; any_red = any_red or not ok
                checks.append(f"interaction_p95={int(i_p95)}ms{'≤' if ok else '>'}{int(PERF_THRESH['interaction_p95_ms'])}{'✅' if ok else '🔴'}")
            verdict = "🔴 退化告警" if any(c.endswith("🔴") for c in checks) else "✅ 绿"
            p(f"  • {a.get('name')}: {verdict}  " + " | ".join(checks))
        p(f"  阈值: longtask_max<={int(PERF_THRESH['longtask_max_ms'])}ms, "
          f"longtask_total<={int(PERF_THRESH['longtask_total_ms'])}ms, "
          f"interaction_p95<={int(PERF_THRESH['interaction_p95_ms'])}ms")
        if any_red:
            p("  🔴 性能退化: 切回/图廊主线程阻塞超阈值 — 守「代码有修复≠现象消除」, 亲手验证它真能抓回归")
        else:
            p("  ✅ 所有 P0 指标在阈值内")

out = "\n".join(lines)
print(out)
if OUT:
    try: (Path(OUT) / "analyze.txt").write_text(out, encoding="utf-8")
    except Exception: pass

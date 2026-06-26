# Forensic Panel Spec

`scripts/analyze.py` prints a FIXED read-only panel after each e2e run. It does
NOT reproduce subagents (that's a separate manual "手段 c" step — see
`docs/problems/` in the main repo). It only inspects the thread workspace
artifacts + a bounded gateway-log excerpt.

## Panel A — data-analyst seal

Reads `handoff_data_analyst.json`.

- `status` ∈ {`in_progress`, `completed`, `partial`, `failed`}.
  `in_progress` = harness preset template, never finalized → **degraded**.
- `sealed_by` ∈ {`preset`, `finalize`}.
  - `finalize` = data-analyst called `finalize_data_analyst_handoff` → **fix works** (#187).
  - `preset` = the in_progress template was never sealed → **data-analyst degraded/FAILED**.
- `key_findings` count (1-5 bullets). `completed` with empty `key_findings` is
  rejected by the schema validator, so a `completed`+non-empty pair is the
  healthy signature.
- Counts for `outlier_findings` / `method_warnings` / `recommendations` /
  `errors` / `quality_warnings` / `parameter_audit_findings`.
  (`parameter_audit_findings` is deliberately always `[]` — see memory
  `feedback_param_audit_value_vs_distribution_criterion_uncreatable`.)

**Concurrent-write race flag**: if the gateway excerpt matches
`handoff_data_analyst.json.tmp ... No such` or `unreadable/invalid JSON` or
`Tool execution failed ... fill_data_analyst`, the panel surfaces it. This is
the #187 second defect (fill load-modify-write with a fixed `.tmp` path races
when the LLM fires multiple parallel fill tool_calls in one AIMessage). It does
not always cause degradation (SealGate nudges the agent to retry serially), but
it is a reliability hole — see memory
`feedback_fill_data_analyst_concurrent_write_race`.

**Verdict logic**:
- `sealed_by == finalize` and terminal status → ✅ fix works.
- `sealed_by == preset` or `status == in_progress` → ❌ degraded.
- else → ⚠️ manual judgement.

## Panel B — chart pseudo-completion (ETHO-10)

Reads `outputs/*.png` (actual) vs `handoff_chart_maker.json` (claimed) vs
`plan_charts.json` (planned).

- `outputs/*.png` count, `outputs/plot_*.png` count (auto-seal only globs the
  `plot_*` prefix).
- `handoff_chart_maker.json`:
  - `status`, `sealed_by` (`model` = LLM self-sealed; `after_agent_artifacts` =
    SealGate auto-seal at termination; `executor_artifacts` = executor L3 path).
  - `chart_files` (claimed rendered), `failed_charts`, `remaining_charts`
    (chart_budget-truncated per_subject leave-behind fingerprint).
- **Cross-check**: for each `chart_files` entry, does the basename exist in
  `outputs/`? Print `exist/total` + missing[:10]. **completed but chart_files
  not on disk = ETHO-10 product-reality hole** (claimed ≠ truth).
- `plan_charts.json`:
  - `charts` count + `output_mode` distribution (`aggregate` = must-have group
    compare; `per_subject` = per-individual, the 2.2 gate does NOT reconcile
    these).
  - aggregate output basenames; which are NOT on disk (non-empty + completed →
    the 2.2 gate `_reconcile_chart_maker_payload` should have raised).

**Root cause this panel exists to surface** (ETHO-10, verified 2026-06-24):
plan is typically ~1 aggregate + ~N×subjects per_subject. The 2.2 gate only
reconciles aggregate charts, so per_subject charts can be entirely unrendered
and the handoff still legitimately says `completed`. See memory
`feedback_etho10_chart_pseudo_complete_2_2_gate_only_aggregate`.

## Panel other

- `handoff_code_executor.json`: `status`, `sealed_by`, `paradigm`.
- `handoff_report_writer.json`: `status`, `sealed_by`, `report_path`.
- `outputs/report.md` existence (the terminal artifact).
- If `handoff_report_writer.json` absent AND `api-log.json` has a 401 → flag
  "token expired mid-run, rerun" (the <1h ensure-login threshold makes this
  near-impossible, but it's the one silent failure mode worth catching).

## Panel C — frontend perf (longtask / interaction timing)

Reads `<run dir>/perf.json`, written by `run-e2e.cjs` AFTER the pipeline
reaches terminal. The driver runs a deterministic set of known-janky
interactions (spec B.2) and records, per action:

- `longtask_max_ms` / `longtask_total_ms` / `longtask_count` — main-thread
  blocks >50ms, harvested from an injected `PerformanceObserver('longtask')`.
- `interaction_ms[]` — click → render-stable (3 stable rAF-pairs), bounded 5s.

Fixed actions: `switch_back_thread` (navigate away then back to the thread),
`open_gallery` (click the first gallery/artifact affordance — skipped if none).

**Thresholds** (mirror `lib.js defaultThresholds` exactly — single source of
truth; change both or neither):

| check | threshold | meaning |
|---|---|---|
| `longtask_max_ms` | 200 | any single main-thread block during the action |
| `longtask_total_ms` | 800 | sum of all longtasks during the action |
| `interaction_p95_ms` | 300 | click → render-stable, p95 of the samples |

These are the INITIAL prod baseline — they are NOT picked from thin air. To
re-baseline after a clean prod run, set `E2E_PERF_THRESHOLDS_JSON` (a JSON
object overriding any subset, e.g. `'{"longtask_max_ms":150}'`).

**Verdict logic**:
- `perf.json.build != "prod"` → `⏭️ PERF: skipped (dev build)` — raw trace is
  still recorded for reference, but NO red/green. dev = Turbopack + HMR +
  sourcemap = noise. Never treat a green dev run as evidence (spec B.0).
- prod + all checks ≤ threshold → `✅ 绿`.
- prod + any check > threshold → `🔴 退化告警`. This is the signal the panel
  exists to surface: a known-janky interaction regressed. The whole point is
  `代码有修复 ≠ 现象消除` (memory) — a prior dogfood's "切回卡顿" was caught
  only by a throwaway manual CPU-profile script; this panel makes it a
  regression guard. When it goes red, treat it as real and investigate.

**Honest coverage note** (spec B.1: implement-and-label, don't claim what
wasn't measured): only P0 dimensions (longtask + interaction timing) are
captured. FPS / Web Vitals (INP/CLS/LCP) / pixel-diff visual regression are P1/P2
and NOT implemented — do not report them as covered.

## When a panel shows a regression

The skill does NOT auto-run subagent reproduction. If Panel A shows degraded
or Panel B shows claimed≠disk, the operator may request a separate manual
"手段 c" reproduction (independent `SubagentExecutor` run with
`set_current_user` + real workspace + real task prompt, dump `ai_messages` per
turn) — see `/tmp/pw-driver/repro-data-analyst.py` pattern and memory
`feedback_diagnose_subagent_behavior_replay_subagent_not_dump_lead`.

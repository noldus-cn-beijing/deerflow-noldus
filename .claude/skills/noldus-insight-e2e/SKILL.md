---
name: noldus-insight-e2e
description: >
  End-to-end dogfood of the noldus-insight backend against local make dev
  (localhost:2026) using REAL behavioral data. Uploads .xlsx files, drives the
  agent through HITL clarifications (answering from a per-dataset
  e2e-answers.yaml by keyword match; unmatched questions fail-loud instead of
  guessing) to terminal, records SSE/screenshots/clarifications, then runs a
  fixed forensic panel (data-analyst seal, chart pseudo-completion, report
  existence) PLUS a perf panel (longtask / interaction timing — only asserted
  against a PROD build). Use this AFTER iterating on backend/frontend code to
  verify the full pipeline still works on real data, WITHOUT hand-writing the
  dogfood prompt each time. Invoke as /noldus-insight-e2e <data-directory>
  (e.g. a folder of .xlsx under ~/DemoData/real_data/).
argument-hint: <data-directory> [optional: path to e2e-answers.yaml via E2E_ANSWERS]
allowed-tools: [Read, Bash, Write, Glob, Grep]
version: 1.1.0
---

# noldus-insight-e2e

Run a full end-to-end dogfood of the noldus-insight agent against local
`make dev` with the user's REAL behavioral data, then run a fixed forensic
panel. `<skilldir>` below = the directory containing this SKILL.md
(`/home/wangqiuyang/noldus-insight/.claude/skills/noldus-insight-e2e`).

The driver scripts are committed under `<skilldir>/scripts/`. They are
parameterized by env vars set below. **Do not inline the driver logic** — call
the scripts. Background on the forensic panel and monitor patterns lives in
`<skilldir>/references/forensic-panel.md` and `references/monitor-patterns.md`.
A sample answers file lives at `<skilldir>/references/e2e-answers.example.yaml`.

## Hard rules

- **Read-only forensics.** The scripts only upload data + drive the browser +
  inspect artifacts. They never edit backend/frontend source. `analyze.py` is
  read-only.
- **Do not auto-start `make dev`.** If the server is down, tell the user to
  start it and STOP.
- **HITL answers come from `e2e-answers.yaml`, never guessed.** The driver
  answers each clarification by keyword/regex match against the question text
  using a per-dataset `e2e-answers.yaml` (default
  `<data-directory>/e2e-answers.yaml`, override via `E2E_ANSWERS`; format in
  `references/e2e-answers.example.yaml`). An unmatched question **fails loud**
  by default (`on_unmatched: fail`) — it writes
  `unmatched_clarification.json` with the full question + the answers you had,
  and exits non-zero. The driver never fabricates a paradigm-specific answer
  (memory `feedback_oft_single_zone_must_ask_not_guess` — not knowing which
  column is the center zone means ASK, not guess "确认"). To keep the old
  behavior for data where a generic "确认" suffices: omit the answers file
  (degrades to legacy generic 确认) or set `on_unmatched: generic`.
- **Perf is only asserted against a PROD build.** dev build (`make dev`,
  Turbopack + HMR + sourcemaps) perf data is noise — the perf panel prints
  `PERF: skipped (dev build)` and never goes red/green. To get perf assertions,
  point `E2E_PERF_BASE_URL` at a prod-started server (`pnpm build && pnpm
  start`, or the prod compose `:2026`) and set `E2E_PERF_BUILD=prod`. Do NOT
  treat a green dev perf run as evidence of anything.
- **⚠️ `pnpm start` (:3000) is FRONTEND-ONLY — never use it for anything that
  touches `/api/*` (auth, login, thread messages, screenshots of real flows).**
  `pnpm build && pnpm start` boots only the Next.js frontend on `:3000`, with
  NO nginx and NO gateway. nginx is the sole front door that proxies
  `/api/v1/auth/*`, `/api/langgraph/*`, `/api/threads/*` to the gateway:8001
  (see `docker/nginx/nginx.conf`); `:3000` owns only `location /` (the UI).
  So curling/visiting `:3000/api/...` returns **404 by design** — it is NOT a
  build/auth breakage, just the missing front door. It is ONLY valid for
  client-render perf timing (no API). **For login, real-flow screenshots, or
  any `/api/*` assertion, you MUST go through nginx `:2026` (`make dev`) or the
  full prod compose (`docker/docker-compose.yaml`: nginx + frontend + gateway)
  — never bare `:3000`.** (D0 audit 2026-06-30 fell into exactly this trap and
  mis-filed it as a P0 prod-breakage.)
- **Do not auto-run subagent reproduction (手段 c).** If a panel shows a
  regression, offer it as a next step the user can request; do not spawn it.

## Procedure

### Step 0 — parse arg

`$1` is the data directory. If empty, reply "用法：/noldus-insight-e2e
<data-directory>" and stop. Validate with Glob: `Glob "<$1>/*.xlsx"`. If none,
reply "目录里没有 .xlsx: <$1>" and stop. (Files starting with `~$` are Excel
lock files and are filtered by the driver.)

Set `E2E_DATA_DIR=<$1>`.

**Answers file (task A).** If the dataset needs paradigm-specific HITL answers
(OFT "which column is the center zone", EPM "4-zone aggregation", FewZones
column alignment, Treatment grouping), the user should have written an
`e2e-answers.yaml` beside the data. Check `<$1>/e2e-answers.yaml` with Glob;
if absent AND the data needs real answers, tell the user (do NOT let the driver
guess — it will fail-loud on the first unmatched question anyway). The template
is `<skilldir>/references/e2e-answers.example.yaml`. Override the path with
`E2E_ANSWERS=<path>`. With no answers file the driver degrades to the legacy
generic `确认` path.

### Step 1 — probe server

Run: `node <skilldir>/scripts/probe-server.cjs` (exit 0 = up). If it exits
nonzero (server down), reply exactly:

> 本地 make dev 没在 http://localhost:2026 跑。先启动它：
>
> `cd /home/wangqiuyang/noldus-insight/packages/agent && make dev`
>
> 起来后重新运行 `/noldus-insight-e2e <data-directory>`。

Then STOP. Do not start the server yourself.

### Step 2 — playwright deps (one-time)

`if [ ! -d <skilldir>/node_modules/playwright ]; then (cd <skilldir> && npm install); fi`.
If `node_modules` appears but chromium is missing (the cached browser was
cleared), also run `(cd <skilldir> && npx playwright install chromium)`.

### Step 3 — compute run dir + ensure login

Compute a timestamped run dir and create it:
`E2E_OUT=/tmp/noldus-e2e-runs/$(date +%Y%m%d-%H%M%S)`; `mkdir -p $E2E_OUT`.

Set `E2E_STATE=<skilldir>/state.json` (gitignored, per-machine). Run
`node <skilldir>/scripts/ensure-login.cjs`. It regenerates `state.json` only if
the JWT has <1h of life left (so a ~40min run won't 401 mid-run), and prints a
line `E2E_USER_ID=<jwt sub>`. Capture that value into `E2E_USER_ID`.

### Step 4 — launch driver (background)

Export all env vars, then launch the merged driver in the background:

```
E2E_DATA_DIR=<...> E2E_STATE=<skilldir>/state.json E2E_OUT=<run dir> \
E2E_BASE_URL=http://localhost:2026 E2E_USER_ID=<from step 3> \
E2E_ANSWERS="<optional path to e2e-answers.yaml, default <$1>/e2e-answers.yaml>" \
E2E_REQUEST_TEXT="<default or override>" E2E_DEADLINE_MIN=45 \
E2E_PERF_BASE_URL="<prod server URL, or leave unset for dev>" \
E2E_PERF_BUILD="<prod | dev, only if overriding auto-detect>" \
node <skilldir>/scripts/run-e2e.cjs > <run dir>/drive.log 2>&1
```

Use `run_in_background: true`. Note the task id. The driver uploads, sends the
analysis request, drives HITL to terminal (workspace-file-based detection:
`handoff_report_writer.json` AND `report.md` AND no Stop button), and writes
`<run dir>/.terminal` when done (also in its `finally` on any exit).

`E2E_THREAD_WS_ROOT` is auto-resolved by `lib.js` (walks up to
`packages/agent/backend/.deer-flow`); only set it if the auto-resolution fails
(e.g. running from an unrelated cwd — then set `E2E_REPO_ROOT` instead).

### Step 5 — monitor concurrently

Arm a **Monitor** (persistent) against the gateway log for live visibility,
covering BOTH success and failure signatures (silence is not success — see
`references/monitor-patterns.md`):

```
GW=/home/wangqiuyang/noldus-insight/packages/agent/logs/gateway.log
tail -n0 -F "$GW" | grep -E --line-buffered \
  'handoff_[a-z_]+\.json|sealed_by|seal_[a-z_]+_handoff|fill_[a-z_]+|finalize|experiment_summary fact written|report\.md|captured AI message|started background task|completed async execution|reached max_turns|Tool execution failed|Traceback|ValueError|FileNotFoundError|RuntimeError|Killed|RecursionError|TimeoutError|clarification'
```

Simultaneously arm a **bounded Bash waiter** (`run_in_background: true`) for the
single completion notification:

```
until [ -f <run dir>/.terminal ]; do sleep 10; done
```

Surface notable milestone lines to the user as they arrive (subagent handoffs
landing, sealed_by values, any failure signature). Keep going about other work
while waiting — events arrive asynchronously.

### Step 6 — terminal reached

When the waiter fires (or the driver background task exits), **TaskStop** the
Monitor. Read `<run dir>/thread.txt` → set `E2E_THREAD`.

If the driver exited non-zero AND `<run dir>/.terminal` is absent or the drive
log shows FATAL: read the tail of `<run dir>/drive.log`, report the failure with
the last error lines + any failure signatures from the monitor, and stop.

**Unmatched-clarification (exit code 2).** If `<run dir>/unmatched_clarification.json`
exists, the driver hit a HITL question the answers file didn't cover and
fail-louded. Read that file, show the user the full question + which answers
were present, and ask them to add a `match`/`answer` entry to their
`e2e-answers.yaml` (or switch `on_unmatched: generic` if a generic confirm is
acceptable for that data). Do NOT rerun until the answers file is extended.

### Step 7 — forensics

Run the fixed panel:

```
E2E_THREAD=<thread> E2E_USER_ID=<uid> E2E_OUT=<run dir> \
python3 <skilldir>/scripts/analyze.py
```

It prints the panel AND writes `<run dir>/analyze.txt`. Cat the output. The
panel now has 4 sections: A (data-analyst seal), B (chart pseudo-completion),
"other" (code-executor / report-writer / report.md), and **C (perf)** which
auto-reads `<run dir>/perf.json` written by the driver. If `perf.json` says
`build: dev` (or is absent), Panel C prints `PERF: skipped` and never goes
red/green.

Also build a bounded `gateway-excerpt.log` in the run dir — grep the gateway log
for the run window (the durable record; the persistent Monitor was for live
view). Record the run start timestamp (before step 4) and end (now):

```
awk -v s="<start>" -v e="<end>"  ...   # or: grep -E '<patterns>' gateway.log
```

A simple durable approach: copy the lines that match the monitor alternation
into `<run dir>/gateway-excerpt.log`.

### Step 8 — report to the user

Summarize (conclusions, not a log dump):

- **Run meta**: thread id, elapsed (~from drive.log), # of HITL rounds, and
  the answers mode in effect (`prefill` vs `legacy-generic` vs
  `generic-fallback`). Flag any round with kind `prefill(...)` that recurred
  (the prefill answer didn't actually satisfy the agent) or any `generic-fallback`
  / `generic(REPEATED->continue)` (no answers file or `on_unmatched: generic` —
  a real answer may be needed).
- **Panel A** verdict: data-analyst `sealed_by` — `finalize` = pipeline healthy
  (the #187 stepwise-fill path worked); `preset`/`in_progress` = degraded.
- **Panel B** verdict: chart_files claimed vs png on disk; aggregate missing;
  any `completed` with chart_files not on disk (ETHO-10 hole).
- **Panel C** verdict (perf): only meaningful on a PROD build. `skipped (dev
  build)` = rerun against prod before drawing any perf conclusion. On prod:
  longtask_max / longtask_total / interaction_p95 per fixed action, each ✅ or
  🔴 against the threshold. A 🔴 means a known-janky interaction (switch-back
  thread, open gallery) regressed — investigate (the whole point is that "代码
  有修复 ≠ 现象消除"; a green perf run here is real evidence the fix held, a
  red one is real evidence it regressed).
- **report.md** existence.
- Any failure signatures from the gateway excerpt (fill race, Traceback, etc.).
- Path to `<run dir>` for full evidence (clarifications.json, sse-*.txt,
  final.png, analyze.txt).

**Do not auto-run subagent reproduction.** If Panel A degraded or Panel B shows
claimed≠disk, say so and offer: "要我用手段 c 独立复现 subagent 拿逐 turn 吗？"
Let the user decide.

## Paradigm-specific answers → put them in `e2e-answers.yaml`

The preferred path for paradigm-specific HITL answers is the per-dataset
`e2e-answers.yaml` (task A), so the run is unattended and reproducible. Common
answers for the real datasets under `~/DemoData/real_data/` — use these to seed
the answers file, do NOT bake them into the driver:

- **EPM** (`Raw data-EPM-Xuhui-28`): template `B. PlusMaze-FewZones` (data has
  `open`/`closed` zone columns =划了区的变体); grouping by `Treatment`
  (XX/XY/YY/YZ); column semantics `open=开臂归属, closed=闭臂归属`.
- **OFT** (`Raw data-OFT-Xuhui-34`): template `OpenFieldRectangle-AllZones`; the
  center column MUST be confirmed not guessed (memory
  `feedback_oft_single_zone_must_ask_not_guess`) — so the answers file entry for
  the center-zone question must match the ACTUAL column name in that dataset.
- **EZM** (`o迷宫`): `ZeroMaze-AllZones`, open/closed like EPM.
- **TST / FST** (`原始数据-悬尾实验`, `强迫游泳实验`): immobility-based, few/no
  zone clarifications.

See `<skilldir>/references/e2e-answers.example.yaml` for the format. If a run
fail-louds on an unmatched question, read `unmatched_clarification.json`, add
the missing entry, and rerun. Live-typing an answer into the thread mid-run is
still possible (the driver leaves the textarea enabled between rounds) but it
breaks unattended reproducibility — prefer the answers file.

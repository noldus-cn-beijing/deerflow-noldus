---
name: noldus-insight-e2e
description: >
  End-to-end dogfood of the noldus-insight backend against local make dev
  (localhost:2026) using REAL behavioral data. Uploads .xlsx files, drives the
  agent through HITL clarifications to terminal, records SSE/screenshots/
  clarifications, then runs a fixed forensic panel (data-analyst seal status,
  chart pseudo-completion, report existence). Use this AFTER iterating on
  backend code to verify the full pipeline still works on real data, WITHOUT
  hand-writing the dogfood prompt each time. Invoke as
  /noldus-insight-e2e <data-directory> (e.g. a folder of .xlsx under
  ~/DemoData/real_data/).
argument-hint: <data-directory>
allowed-tools: [Read, Bash, Write, Glob, Grep]
version: 1.0.0
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

## Hard rules

- **Read-only forensics.** The scripts only upload data + drive the browser +
  inspect artifacts. They never edit backend source. `analyze.py` is read-only.
- **Do not auto-start `make dev`.** If the server is down, tell the user to
  start it and STOP.
- **Generic HITL answers only.** The driver answers `确认`, escalating to
  `继续推进，按上述确认执行` on a repeated identical question. Do NOT bake
  paradigm-specific answers (FewZones / Treatment / column-semantics) into the
  driver — if a real answer is needed, give it live by typing into the thread
  in a separate step. The driver is paradigm-agnostic on purpose.
- **Do not auto-run subagent reproduction (手段 c).** If a panel shows a
  regression, offer it as a next step the user can request; do not spawn it.

## Procedure

### Step 0 — parse arg

`$1` is the data directory. If empty, reply "用法：/noldus-insight-e2e
<data-directory>" and stop. Validate with Glob: `Glob "<$1>/*.xlsx"`. If none,
reply "目录里没有 .xlsx: <$1>" and stop. (Files starting with `~$` are Excel
lock files and are filtered by the driver.)

Set `E2E_DATA_DIR=<$1>`.

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
E2E_REQUEST_TEXT="<default or override>" E2E_DEADLINE_MIN=45 \
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

### Step 7 — forensics

Run the fixed panel:

```
E2E_THREAD=<thread> E2E_USER_ID=<uid> E2E_OUT=<run dir> \
python3 <skilldir>/scripts/analyze.py
```

It prints the panel AND writes `<run dir>/analyze.txt`. Cat the output.

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

- **Run meta**: thread id, elapsed (~from drive.log), # of HITL rounds.
  Flag any round with kind `generic(REPEATED->continue)` — that means the agent
  looped on a question the generic answer couldn't satisfy; the user may want
  to give a paradigm-specific answer (e.g. FewZones vs AllZones, Treatment
  grouping, open/closed column semantics) and rerun.
- **Panel A** verdict: data-analyst `sealed_by` — `finalize` = pipeline healthy
  (the #187 stepwise-fill path worked); `preset`/`in_progress` = degraded.
- **Panel B** verdict: chart_files claimed vs png on disk; aggregate missing;
  any `completed` with chart_files not on disk (ETHO-10 hole).
- **report.md** existence.
- Any failure signatures from the gateway excerpt (fill race, Traceback, etc.).
- Path to `<run dir>` for full evidence (clarifications.json, sse-*.txt,
  final.png, analyze.txt).

**Do not auto-run subagent reproduction.** If Panel A degraded or Panel B shows
claimed≠disk, say so and offer: "要我用手段 c 独立复现 subagent 拿逐 turn 吗？"
Let the user decide.

## Paradigm-specific answers (only if the user wants them, given LIVE)

If during step 5 the user (watching the monitor) sees the agent stuck on a HITL
question the generic `确认` can't satisfy, they can give a real answer. Common
ones for the real datasets under `~/DemoData/real_data/` (for reference — NOT
baked into the driver):

- **EPM** (`Raw data-EPM-Xuhui-28`): template `B. PlusMaze-FewZones` (data has
  `open`/`closed` zone columns =划了区的变体); grouping by `Treatment`
  (XX/XY/YY/YZ); column semantics `open=开臂归属, closed=闭臂归属`.
- **OFT** (`Raw data-OFT-Xuhui-34`): template `OpenFieldRectangle-AllZones`; the
  center column MUST be confirmed not guessed (memory
  `feedback_oft_single_zone_must_ask_not_guess`).
- **EZM** (`o迷宫`): `ZeroMaze-AllZones`, open/closed like EPM.
- **TST / FST** (`原始数据-悬尾实验`, `强迫游泳实验`): immobility-based, few/no
  zone clarifications.

These are given by the operator live in a separate message, NOT by this skill.

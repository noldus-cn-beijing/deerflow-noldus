// noldus-insight-e2e driver: upload .xlsx → send analysis request → drive the
// agent through HITL clarifications to terminal → record evidence → (prod only)
// run a fixed perf-action script.
//
// Terminal detection is WORKSPACE-FILE based (not UI keywords — those false-
// positive early): handoff_report_writer.json exists AND report.md exists AND
// no Stop button.
//
// HITL answers (task A): the driver reads an e2e-answers.yaml (env
// E2E_ANSWERS, default $E2E_DATA_DIR/e2e-answers.yaml) and answers each
// clarification by KEYWORD/REGEX match against the question text. Unmatched
// questions fail-loud by default (write unmatched_clarification + exit non-zero)
// — the driver never guesses a paradigm-specific answer (memory
// feedback_oft_single_zone_must_ask_not_guess). With NO answers file it
// degrades to the legacy generic "确认" path. on_unmatched: generic restores the
// old fallback for data where a generic confirm is enough.
//
// Perf (task B): only meaningful against a PROD build (dev = Turbopack + HMR +
// sourcemaps = noise). E2E_PERF_BASE_URL points at the prod frontend; if the
// target is dev, perf.json records build:'dev' and the panel SKIPS assertions.
//
// Writes $E2E_OUT/.terminal at terminal break AND in a finally on any exit, so
// the bounded waiter in SKILL.md always resolves. Run in background.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
const { config, resolveChrome, resolveWsRoot, threadUserDataDir,
  loadAnswersFile, matchAnswer } = require('./lib');

const cfg = config();
if (!cfg.DATA_DIR) { console.error('FATAL: E2E_DATA_DIR required'); process.exit(1); }
if (!cfg.OUT) { console.error('FATAL: E2E_OUT required'); process.exit(1); }
if (!cfg.USER_ID) { console.error('FATAL: E2E_USER_ID required (run ensure-login.cjs first)'); process.exit(1); }

fs.mkdirSync(cfg.OUT, { recursive: true });
const TERMINAL_MARKER = path.join(cfg.OUT, '.terminal');
const WS_USERS = resolveWsRoot();
const T0 = Date.now();
const el = () => `${((Date.now() - T0) / 1000).toFixed(1)}s`;

// ---- task A: load answers file ---------------------------------------------
// E2E_ANSWERS overrides; default sits beside the data (one answers file per
// dataset, versionable). Absent file => legacy generic path (answers=null).
const ANSWERS_PATH = process.env.E2E_ANSWERS || path.join(cfg.DATA_DIR, 'e2e-answers.yaml');
const ANSWERS = fs.existsSync(ANSWERS_PATH) ? loadAnswersFile(ANSWERS_PATH) : null;
if (ANSWERS === null) {
  console.log(`[run-e2e] no answers file at ${ANSWERS_PATH} -> legacy generic 确认 path`);
} else if (ANSWERS.error) {
  console.error(`FATAL: answers file ${ANSWERS_PATH} parse error: ${ANSWERS.error}`);
  process.exit(1);
} else {
  console.log(`[run-e2e] answers: ${ANSWERS.answers.length} entries, on_unmatched=${ANSWERS.onUnmatched} (${ANSWERS_PATH})`);
}

// ---- task B: prod/dev detection --------------------------------------------
// Perf is only asserted against a prod build. We detect prod via E2E_PERF_BUILD
// (explicit) or by probing a marker exposed by the prod frontend. If the target
// is the default localhost:2026 (make dev), treat as dev unless overridden.
const PERF_BASE_URL = process.env.E2E_PERF_BASE_URL || cfg.BASE_URL;
function detectBuild() {
  const explicit = process.env.E2E_PERF_BUILD;
  if (explicit) return explicit.toLowerCase();
  // make dev serves on :2026 with Turbopack dev markers; prod compose also on
  // :2026 but built. The honest signal is whether the operator pointed
  // E2E_PERF_BASE_URL at a prod-started server. Default localhost:2026 is dev.
  if (process.env.E2E_PERF_BASE_URL) return 'prod';
  return 'dev';
}
const PERF_BUILD = detectBuild();
const PERF_ENABLED = process.env.E2E_PERF !== '0';
console.log(`[run-e2e] perf: build=${PERF_BUILD} base=${PERF_BASE_URL} enabled=${PERF_ENABLED}`);

// page.goto waitUntil strategy. Default 'networkidle' matches the original
// behavior (works against make dev on :2026). Against a prod build the frontend
// polls /api/models every ~3s, so networkidle never settles and goto times out.
// Set E2E_GOTO_WAIT=domcontentloaded to bypass (the upload/submit steps below
// already wait on concrete elements, so domcontentloaded is sufficient).
const GOTO_WAIT = process.env.E2E_GOTO_WAIT || 'networkidle';

const touchTerminal = (reason) => {
  try { fs.writeFileSync(TERMINAL_MARKER, `${el()} ${reason}\n`); } catch {}
};

// Fail-loud on an unmatched clarification (task A.3). Writes the full question
// + the answers we had, so the operator can extend e2e-answers.yaml.
const failUnmatched = (questionText) => {
  const rec = {
    t: el(),
    question: questionText,
    answers_file: ANSWERS_PATH,
    answers_present: ANSWERS ? ANSWERS.answers.map(a => ({ match: a.match, answer: a.answer })) : [],
    hint: 'add a match/answer entry to e2e-answers.yaml for this question, or set on_unmatched: generic',
  };
  fs.writeFileSync(path.join(cfg.OUT, 'unmatched_clarification.json'), JSON.stringify(rec, null, 2));
  console.error(`[run-e2e] FATAL: unmatched clarification (on_unmatched=fail). question="${String(questionText).slice(0, 200)}..."`);
  console.error(`[run-e2e] wrote ${path.join(cfg.OUT, 'unmatched_clarification.json')}`);
  touchTerminal('unmatched-clarification');
  process.exit(2);
};

(async () => {
  const files = fs.readdirSync(cfg.DATA_DIR).filter(f => f.endsWith('.xlsx') && !f.startsWith('~$')).sort().map(f => path.join(cfg.DATA_DIR, f));
  if (files.length === 0) { console.error(`FATAL: no .xlsx in ${cfg.DATA_DIR}`); process.exit(1); }
  console.log(`[run-e2e] files=${files.length} ws_root=${WS_USERS} deadline=${cfg.DEADLINE_MIN}min`);

  const browser = await chromium.launch({ executablePath: resolveChrome(), headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  try {
    const ctx = await browser.newContext({ storageState: cfg.STATE, viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();

    // CDP: track SSE stream requests for body dump.
    const cdp = await ctx.newCDPSession(page);
    const reqUrl = {};
    const streamReqs = new Set();
    await cdp.send('Network.enable');
    cdp.on('Network.requestWillBeSent', e => { reqUrl[e.requestId] = e.request.url; });
    cdp.on('Network.responseReceived', e => {
      const u = reqUrl[e.requestId] || '';
      if (u.includes('/runs/stream') || u.includes('/runs/join')) streamReqs.add(e.requestId);
    });
    const apiLog = [];
    page.on('request', r => { const u = r.url(); if (u.includes('/api/')) apiLog.push({ t: el(), m: r.method(), u, status: null, body: r.postData() ? r.postData().slice(0, 1500) : null }); });

    // new chat
    await page.goto(`${cfg.BASE_URL}/workspace/chats/new`, { waitUntil: GOTO_WAIT, timeout: 30000 });
    await page.waitForTimeout(2500);
    let thread = '';
    try { thread = (page.url().split('/chats/')[1] || '').split(/[?#]/)[0]; } catch {}

    // upload all .xlsx (wait for chips)
    console.log(`[run-e2e] uploading ${files.length} files`);
    await page.locator('input[type=file]').first().setInputFiles(files);
    let chips = 0;
    for (let i = 0; i < 60; i++) {
      await page.waitForTimeout(1500);
      // Chip "remove" affordance: locale-dependent. English shows `text=Remove`,
      // zh-CN renders `aria-label="移除 <file>"`. Count either, de-duped by the
      // chip container (one per file). `text=Remove` alone stayed at 0 under the
      // zh-CN prod build, so the loop never saw the chips and the run submitted
      // an empty conversation.
      const byText = await page.locator('text=Remove').count().catch(() => 0);
      const byAria = await page.locator('[aria-label*="移除"]').count().catch(() => 0);
      chips = Math.max(byText, byAria);
      if (chips >= files.length) break;
    }
    console.log(`[run-e2e] chips=${chips}/${files.length}`);
    if (chips < files.length) {
      console.error(`FATAL: only ${chips}/${files.length} files attached — refusing to submit an empty conversation.`);
      process.exit(1);
    }

    // send analysis request
    await page.locator('textarea').first().fill(cfg.REQUEST_TEXT);
    await page.waitForTimeout(500);
    await page.locator('button[aria-label="Submit"]').click().catch(async () => { await page.keyboard.press('Enter'); });
    await page.waitForTimeout(3000);
    try { thread = (page.url().split('/chats/')[1] || '').split(/[?#]/)[0] || thread; } catch {}
    fs.writeFileSync(path.join(cfg.OUT, 'thread.txt'), thread || '');
    console.log(`[run-e2e] thread=${thread}`);
    if (!thread) { console.error('FATAL: could not capture thread id from URL'); process.exit(1); }

    const udDir = threadUserDataDir(WS_USERS, cfg.USER_ID, thread);
    const wsDir = path.join(udDir, 'workspace');
    const outDir = path.join(udDir, 'outputs');
    const readWs = () => { try { return fs.readdirSync(wsDir).sort(); } catch { return []; } };
    const readOut = () => { try { return fs.readdirSync(outDir).sort(); } catch { return []; } };

    // ---- drive loop ----
    const deadline = Date.now() + cfg.DEADLINE_MIN * 60 * 1000;
    let lastUiLen = 0, stableCount = 0, lastWsSig = '', answerRound = 0;
    const clarifications = [];
    let lastTailSig = '', repeatedTail = 0;

    while (Date.now() < deadline) {
      await page.waitForTimeout(20000);
      const t = el();
      const hasStop = await page.evaluate(() => !!document.querySelector('button[aria-label="Stop"]')).catch(() => false);
      const uiTxt = await page.evaluate(() => document.body.innerText).catch(() => '');
      const uiLen = uiTxt.length;
      const tail = uiTxt.slice(-2600);
      const ws = readWs();
      const outs = readOut();
      const wsSig = ws.join(',');
      const hasReportHandoff = ws.includes('handoff_report_writer.json');
      const hasReportMd = outs.some(f => /report\.md$/i.test(f));
      const handoffs = ws.filter(f => f.startsWith('handoff_'));
      const pngs = outs.filter(f => f.endsWith('.png')).length;
      console.log(`[run-e2e][${t}] stop=${hasStop} uiLen=${uiLen} streams=${streamReqs.size} handoffs=${handoffs.join('|')} pngs=${pngs} reportMd=${hasReportMd}`);

      // TERMINAL: report handoff + report.md + not running (3-signal AND)
      if (hasReportHandoff && hasReportMd && !hasStop) {
        console.log(`[run-e2e][${t}] TERMINAL: report_writer handoff + report.md + no stop`);
        await page.waitForTimeout(30000); // let the stream flush
        break;
      }

      const isClarif = /确认|请.{0,8}确认|请.{0,8}选择|请.{0,8}回复|请问|⚠️|是.{0,4}还是|选项|回复对应|哪个.{0,4}模板|A\.\s*\*\*/i.test(tail);
      if (!hasStop) {
        if (uiLen === lastUiLen && wsSig === lastWsSig) stableCount++; else { stableCount = 0; lastUiLen = uiLen; lastWsSig = wsSig; }
        if (stableCount >= 3) {
          const tEnabled = await page.locator('textarea').first().isEnabled().catch(() => false);
          if (tEnabled && isClarif) {
            // repeated-question detection (anti-loop guard)
            const tailSig = tail.slice(-1200);
            if (tailSig === lastTailSig) repeatedTail++; else { repeatedTail = 0; lastTailSig = tailSig; }
            answerRound++;

            // ---- task A: resolve the answer ----
            // 1. answers file present: keyword/regex match against the question
            //    text (full tail, not just the regex slice). Hit => use it.
            //    Miss => on_unmatched policy (fail-loud default / generic legacy).
            // 2. no answers file: legacy generic 确认, escalating on repeats.
            let ans, kind;
            if (ANSWERS) {
              const m = matchAnswer(ANSWERS, tailSig);
              if (m) {
                ans = m.answer; kind = `prefill(${m.matchedKey})`;
                // In prefill mode, a repeated IDENTICAL question means the
                // prefill answer did not actually satisfy the agent — fail
                // rather than pretend the run passed (spec A.3 last bullet).
                if (repeatedTail >= 2) {
                  clarifications.push({ round: answerRound, t, kind: 'FAIL(prefill-repeated)', repeated: repeatedTail, tail: tailSig, ans });
                  fs.writeFileSync(path.join(cfg.OUT, 'clarifications.json'), JSON.stringify(clarifications, null, 2));
                  console.error(`[run-e2e][${t}] FATAL: prefill answer "${ans.slice(0, 60)}" did not resolve the question after ${repeatedTail} repeats`);
                  failUnmatched(tailSig);
                }
              } else if (ANSWERS.onUnmatched === 'generic') {
                ans = '确认'; kind = 'generic-fallback';
                if (repeatedTail >= 2) { ans = '继续推进，按上述确认执行，不要重复反问'; kind = 'generic-fallback(REPEATED->continue)'; }
              } else {
                // fail-loud (default): unmatched clarification
                clarifications.push({ round: answerRound, t, kind: 'FAIL(unmatched)', repeated: repeatedTail, tail: tailSig });
                fs.writeFileSync(path.join(cfg.OUT, 'clarifications.json'), JSON.stringify(clarifications, null, 2));
                failUnmatched(tailSig);
              }
            } else {
              // legacy: no answers file
              ans = '确认'; kind = 'generic';
              if (repeatedTail >= 2) { ans = '继续推进，按上述确认执行，不要重复反问'; kind = 'generic(REPEATED->continue)'; }
            }

            clarifications.push({ round: answerRound, t, kind, repeated: repeatedTail, tail: tailSig, ans });
            console.log(`[run-e2e][${t}] CLAR#${answerRound} kind=${kind} rep=${repeatedTail} -> ${ans.slice(0, 80)}`);
            await page.locator('textarea').first().fill(ans);
            await page.waitForTimeout(500);
            await page.locator('button[aria-label="Submit"]').click().catch(async () => { await page.keyboard.press('Enter'); });
            stableCount = 0; lastUiLen = 0; lastWsSig = '';
            await page.waitForTimeout(12000);
            continue;
          } else if (stableCount >= 9) {
            console.log(`[run-e2e][${t}] idle>180s no clarif no stop no ws change => terminal-ish`);
            break;
          }
        }
      } else { stableCount = 0; }
    }

    // ---- task B: perf fixed-action script (prod only) ----
    const perf = await runPerfScript(page, thread).catch(e => {
      console.log(`[run-e2e] perf script skipped/failed: ${(e && e.message) || e}`);
      return { build: PERF_BUILD, error: String((e && e.message) || e), actions: [] };
    });
    fs.writeFileSync(path.join(cfg.OUT, 'perf.json'), JSON.stringify(perf, null, 2));
    console.log(`[run-e2e] perf.json written: build=${perf.build} actions=${(perf.actions || []).length}`);

    // ---- dump evidence ----
    const bodySnaps = {};
    for (const rid of streamReqs) { try { const r = await cdp.send('Network.getResponseBody', { requestId: rid }); bodySnaps[rid] = r.body; } catch {} }
    let idx = 0;
    for (const rid of streamReqs) { fs.writeFileSync(path.join(cfg.OUT, `sse-${idx}.txt`), bodySnaps[rid] || '(empty)'); idx++; }
    fs.writeFileSync(path.join(cfg.OUT, 'api-log.json'), JSON.stringify(apiLog, null, 2));
    fs.writeFileSync(path.join(cfg.OUT, 'clarifications.json'), JSON.stringify(clarifications, null, 2));
    fs.writeFileSync(path.join(cfg.OUT, 'final-body.txt'), await page.evaluate(() => document.body.innerText).catch(() => ''));
    await page.screenshot({ path: path.join(cfg.OUT, 'final.png'), fullPage: true }).catch(() => {});
    console.log(`[run-e2e] END streams=${streamReqs.size} saved=${idx} clarifications=${clarifications.length} thread=${thread}`);
    touchTerminal('end-of-drive');
  } finally {
    await browser.close().catch(() => {});
    touchTerminal('finally');
  }
})().catch(e => {
  console.error('FATAL', e.stack || e);
  touchTerminal('fatal');
  process.exit(1);
});

// ============================================================================
// Task B — fixed-action perf script
// ============================================================================
// Runs AFTER the pipeline reaches terminal. Executes a deterministic set of
// known-janky interactions (spec B.2) and captures, per action:
//   - longtask_max_ms / longtask_total_ms  (main-thread blocks >50ms, via an
//     injected PerformanceObserver('longtask') harvested across the action)
//   - interaction_ms[]                     (click → render-stable samples)
// Produces perf.json. On a dev build we still run (raw trace) but the analyze
// panel will SKIP assertions — perf.build gates that.
async function runPerfScript(page, thread) {
  if (!PERF_ENABLED) return { build: PERF_BUILD, skipped: 'E2E_PERF=0', actions: [] };

  // Inject a longtask buffer once (persists across navigations within the page
  // via addInitScript; we read+reset it around each action).
  await page.addInitScript(() => {
    window.__e2e_longtasks = [];
    if (window.__e2e_lo) return;
    window.__e2e_lo = new PerformanceObserver(list => {
      for (const e of list.getEntries()) window.__e2e_longtasks.push(e.duration);
    });
    try { window.__e2e_lo.observe({ type: 'longtask', buffered: true }); } catch {}
  });

  const chatUrl = `${PERF_BASE_URL}/workspace/chats/${thread}`;
  const actions = [];

  // waitUntilMainQuiet: click→render-stable. Returns ms from call to 3 stable
  // rAF-pairs (bounded to 5s). Negative on failure.
  async function waitUntilMainQuiet() {
    return page.evaluate(async () => {
      const s = performance.now();
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      let stable = 0;
      const dl = performance.now() + 5000;
      while (performance.now() < dl) {
        const a = performance.now();
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        if (performance.now() - a < 32) stable++; else stable = 0;
        if (stable >= 3) break;
        await sleep(20);
      }
      return Math.round(performance.now() - s);
    }).catch(() => -1);
  }
  async function drainLongtasks() {
    const lts = await page.evaluate(() => (window.__e2e_longtasks || []).slice()).catch(() => []);
    const longtask_max_ms = lts.length ? Math.round(Math.max(...lts)) : 0;
    const longtask_total_ms = Math.round(lts.reduce((a, b) => a + b, 0));
    return { longtask_max_ms, longtask_total_ms, longtask_count: lts.length };
  }
  async function resetLongtasks() { await page.evaluate(() => { if (window.__e2e_longtasks) window.__e2e_longtasks.length = 0; }).catch(() => {}); }

  // Action 1: switch_back_thread — navigate away then back, measure jank on return.
  if (thread) {
    try {
      await resetLongtasks();
      await page.goto(`${PERF_BASE_URL}/workspace/chats/new`, { waitUntil: GOTO_WAIT, timeout: 30000 }).catch(() => {});
      await page.waitForTimeout(800);
      const t0 = Date.now();
      await page.goto(chatUrl, { waitUntil: GOTO_WAIT, timeout: 30000 }).catch(() => {});
      const interaction = await waitUntilMainQuiet();
      const lt = await drainLongtasks();
      actions.push({ name: 'switch_back_thread', ...lt, interaction_ms: [interaction], nav_ms: Date.now() - t0 });
      console.log(`[run-e2e][perf] switch_back_thread: longtask_max=${lt.longtask_max_ms}ms total=${lt.longtask_total_ms}ms interaction=${interaction}ms`);
    } catch (e) { actions.push({ name: 'switch_back_thread', error: String(e.message || e) }); }
  }

  // Action 2: open_gallery — click the first artifact/gallery affordance, measure.
  try {
    await resetLongtasks();
    const clicked = await page.evaluate(() => {
      const sels = ['button[aria-label*="gallery" i]', 'button[aria-label*="图廊"]', '[data-testid*="gallery"]', 'button[aria-label*="artifact" i]', '[data-testid*="artifact"]'];
      for (const s of sels) { const el = document.querySelector(s); if (el) { el.click(); return s; } }
      return null;
    }).catch(() => null);
    if (clicked) {
      const interaction = await waitUntilMainQuiet();
      const lt = await drainLongtasks();
      actions.push({ name: 'open_gallery', ...lt, interaction_ms: [interaction], selector: clicked });
      console.log(`[run-e2e][perf] open_gallery(${clicked}): longtask_max=${lt.longtask_max_ms}ms total=${lt.longtask_total_ms}ms interaction=${interaction}ms`);
    } else {
      actions.push({ name: 'open_gallery', skipped: 'no gallery/artifact affordance found', longtask_max_ms: 0, longtask_total_ms: 0, interaction_ms: [] });
      console.log(`[run-e2e][perf] open_gallery: no affordance found (skipped)`);
    }
  } catch (e) { actions.push({ name: 'open_gallery', error: String(e.message || e) }); }

  return { build: PERF_BUILD, base_url: PERF_BASE_URL, thread, actions };
}

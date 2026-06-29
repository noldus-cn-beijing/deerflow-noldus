#!/usr/bin/env node
// switchback-trace.cjs — Step 0 evidence capture for spec
// 2026-06-29-fix-tab-switchback-jank.md.
//
// Reproduces the spec's symptom: while data-analyst streams a LARGE in-flight
// message, simulate the user switching away to another app (VS Code) by
// freezing the page via CDP Page.setWebLifecycleState('frozen'), hold it frozen
// so SSE keeps appending to state while React/scheduler stalls, then
// 'active' to simulate switchback — and measure the main-thread cost of the
// switchback catch-up.
//
// This is EVIDENCE CAPTURE ONLY (read-only dogfood + perf). It writes no
// implementation code. Honours the spec red line: dev build perf is noise —
// it records build and prints a loud warning on dev, but still produces the
// raw arrays so we can sanity-check the flow.
//
// Reuses lib.js (login state, chrome path, ws root) and the upload/drive
// shape from run-e2e.cjs, but diverges at the "drive to data-analyst stream"
// step: instead of driving to terminal, it pauses to inject perf observers,
// freezes mid-stream, then thaws and captures the switchback cost.
//
// Env (in addition to run-e2e's):
//   E2E_OUT                output dir (required)
//   E2E_DATA_DIR           dataset dir (required)
//   E2E_USER_ID            (required; ensure-login.cjs first)
//   E2E_STATE              login storageState (required)
//   E2E_PERF_BUILD         'prod' | 'dev' (required-ish; warn on dev)
//   E2E_FREEZE_MS          how long to keep the page frozen mid-stream (default 25000)
//   E2E_STREAM_DEADLINE_MS overall wait for a large in-flight message (default 180000)

const fs = require('fs');
const path = require('path');
// Resolve playwright + lib.js from the e2e skill dir (where node_modules lives),
// regardless of which cwd this script is invoked from. The skill dir is fixed.
const SKILL_DIR = process.env.E2E_SKILL_DIR || '/home/wangqiuyang/noldus-insight/.claude/skills/noldus-insight-e2e';
const { chromium } = require(path.join(SKILL_DIR, 'node_modules', 'playwright'));
const { config, resolveChrome, resolveWsRoot, threadUserDataDir, loadAnswersFile, matchAnswer } = require(path.join(SKILL_DIR, 'scripts', 'lib'));

const cfg = config();
for (const k of ['E2E_OUT', 'E2E_DATA_DIR', 'E2E_USER_ID']) {
  if (!process.env[k]) { console.error(`FATAL: ${k} required`); process.exit(1); }
}
fs.mkdirSync(cfg.OUT, { recursive: true });

const FREEZE_MS = parseInt(process.env.E2E_FREEZE_MS || '25000', 10);
const STREAM_DEADLINE_MS = parseInt(process.env.E2E_STREAM_DEADLINE_MS || '180000', 10);
const BUILD = (process.env.E2E_PERF_BUILD || 'dev').toLowerCase();
if (BUILD !== 'prod') {
  console.log('[switchback-trace] WARNING: build != prod — perf numbers are NOISE per spec red line. Use for flow sanity only.');
}

const T0 = Date.now();
const el = () => `${((Date.now() - T0) / 1000).toFixed(1)}s`;

(async () => {
  const files = fs.readdirSync(cfg.DATA_DIR).filter(f => f.endsWith('.xlsx') && !f.startsWith('~$')).sort().map(f => path.join(cfg.DATA_DIR, f));
  if (!files.length) { console.error(`FATAL: no .xlsx in ${cfg.DATA_DIR}`); process.exit(1); }

  const browser = await chromium.launch({ executablePath: resolveChrome(), headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  try {
    const ctx = await browser.newContext({ storageState: cfg.STATE, viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    const cdp = await ctx.newCDPSession(page);

    // Inject longtask + paint holder BEFORE first navigation (addInitScript so it
    // survives client-side route changes). We also expose a function to read the
    // in-flight message size from the DOM so we can correlate cost with size.
    await page.addInitScript(() => {
      window.__sb_longtasks = []; // {duration, startTime}
      window.__sb_paints = [];    // {startTime} firstPaint-ish markers via longtask-adjacent rAF
      try {
        const lo = new PerformanceObserver(list => {
          for (const e of list.getEntries()) window.__sb_longtasks.push({ duration: e.duration, startTime: e.startTime });
        });
        lo.observe({ type: 'longtask', buffered: true });
      } catch (e) { window.__sb_lo_err = String(e); }
      // helper: estimate the in-flight streaming message size from the DOM.
      // The streaming assistant message is the last message bubble; we sum the
      // textContent length of its markdown container as a proxy for bytes.
      window.__sb_inflight_size = function () {
        // Try several known container shapes (deerflow/noldus message list).
        const sels = ['[data-streaming]', '[data-inflight]', '[data-testid="message-streaming"]'];
        for (const s of sels) { const el = document.querySelector(s); if (el) return el.textContent.length; }
        // Fallback: last prose article/bubble in the message scroll area.
        const msgs = document.querySelectorAll('[data-message-role="assistant"], article, .prose');
        if (msgs.length) return msgs[msgs.length - 1].textContent.length;
        return document.body.innerText.length;
      };
    });

    // ---- upload + send (mirrors run-e2e.cjs) ----
    await page.goto(`${cfg.BASE_URL}/workspace/chats/new`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2500);
    await page.locator('input[type=file]').first().setInputFiles(files);
    for (let i = 0; i < 60; i++) {
      await page.waitForTimeout(1500);
      const chips = await page.locator('text=Remove').count().catch(() => 0);
      if (chips >= files.length) break;
    }
    await page.locator('textarea').first().fill(cfg.REQUEST_TEXT);
    await page.waitForTimeout(500);
    await page.locator('button[aria-label="Submit"]').click().catch(async () => { await page.keyboard.press('Enter'); });
    await page.waitForTimeout(3000);
    let thread = (page.url().split('/chats/')[1] || '').split(/[?#]/)[0] || '';
    fs.writeFileSync(path.join(cfg.OUT, 'thread.txt'), thread);
    console.log(`[switchback-trace][${el()}] thread=${thread}`);

    // ---- HITL answer file (perf repro: drive clarifications so the pipeline
    // reaches a large data-analyst stream). Generic-confirm fallback so the run
    // does not stall on an uncovered question. ----
    const ANSWERS_PATH = process.env.E2E_ANSWERS || path.join(cfg.DATA_DIR, 'e2e-answers.yaml');
    const ANSWERS = fs.existsSync(ANSWERS_PATH) ? loadAnswersFile(ANSWERS_PATH) : null;
    const clarifications = [];
    async function maybeAnswerClarification() {
      // Detect an awaiting-clarification state: textarea enabled + no Stop +
      // a clarification-ish marker in the tail. Answer from the file or generic.
      const hasStop = await page.evaluate(() => !!document.querySelector('button[aria-label="Stop"]')).catch(() => false);
      if (hasStop) return false;
      const tail = await page.evaluate(() => document.body.innerText.slice(-2600)).catch(() => '');
      const isClarif = /确认|请.{0,8}确认|请.{0,8}选择|请.{0,8}回复|请问|⚠️|是.{0,4}还是|选项|回复对应|哪个.{0,4}模板|A\.\s*\*\*/i.test(tail);
      if (!isClarif) return false;
      const tEnabled = await page.locator('textarea').first().isEnabled().catch(() => false);
      if (!tEnabled) return false;
      let ans = '确认，按上述执行';
      let kind = 'generic';
      if (ANSWERS) {
        const m = matchAnswer(ANSWERS, tail.slice(-1200));
        if (m) { ans = m.answer; kind = `prefill(${m.matchedKey})`; }
        else if (ANSWERS.onUnmatched === 'generic') { ans = '确认'; kind = 'generic-fallback'; }
      }
      clarifications.push({ t: el(), kind, tail: tail.slice(-600), ans: ans.slice(0, 120) });
      console.log(`[switchback-trace][${el()}] CLAR kind=${kind} -> ${ans.slice(0, 60)}`);
      await page.locator('textarea').first().fill(ans);
      await page.waitForTimeout(400);
      await page.locator('button[aria-label="Submit"]').click().catch(async () => { await page.keyboard.press('Enter'); });
      await page.waitForTimeout(8000);
      return true;
    }

    // ---- wait for a LARGE in-flight message ----
    // Poll the in-flight size; once it crosses a threshold (data-analyst prose),
    // we have our "big streaming message" moment. Capture its growth. Answer any
    // HITL clarifications that block progress along the way.
    const sizeSamples = [];
    const streamStart = Date.now();
    let reached = false;
    const BIG_THRESHOLD = 4000; // chars — data-analyst multi-paragraph territory
    let lastSize = 0, stableCount = 0;
    while (Date.now() - streamStart < STREAM_DEADLINE_MS) {
      await page.waitForTimeout(4000);
      const sz = await page.evaluate(() => window.__sb_inflight_size && window.__sb_inflight_size()).catch(() => 0);
      const hasStop = await page.evaluate(() => !!document.querySelector('button[aria-label="Stop"]')).catch(() => false);
      sizeSamples.push({ t: el(), size: sz, stop: hasStop });
      console.log(`[switchback-trace][${el()}] inflight=${sz} stop=${hasStop}`);
      if (sz >= BIG_THRESHOLD) { reached = true; break; }
      // stalled (no growth, not streaming, no big message yet) → try answering a clarification
      if (sz === lastSize) stableCount++; else { stableCount = 0; lastSize = sz; }
      if (stableCount >= 2) { const answered = await maybeAnswerClarification(); if (answered) { stableCount = 0; lastSize = 0; } }
    }
    if (!reached) {
      console.log(`[switchback-trace][${el()}] WARN: never reached inflight>=${BIG_THRESHOLD} while streaming. Last size=${sizeSamples.length ? sizeSamples[sizeSamples.length - 1].size : 0}. Continuing with whatever we have.`);
    }
    fs.writeFileSync(path.join(cfg.OUT, 'inflight-growth.json'), JSON.stringify(sizeSamples, null, 2));
    fs.writeFileSync(path.join(cfg.OUT, 'clarifications.json'), JSON.stringify(clarifications, null, 2));

    // ---- baseline the longtask buffer, then FREEZE mid-stream ----
    const preFreezeLt = await page.evaluate(() => (window.__sb_longtasks || []).slice()).catch(() => []);
    const preFreezeSize = await page.evaluate(() => window.__sb_inflight_size()).catch(() => 0);
    console.log(`[switchback-trace][${el()}] PRE-FREEZE inflight=${preFreezeSize} longtasks=${preFreezeLt.length}`);

    // Freeze = simulate user switched to another app. Page.setWebLifecycleState
    // 'frozen' induces the rAF/timer throttling the spec's root cause hinges on.
    try {
      await cdp.send('Page.enable');
      await cdp.send('Page.setWebLifecycleState', { state: 'frozen' });
      console.log(`[switchback-trace][${el()}] FROZEN for ${FREEZE_MS}ms`);
    } catch (e) {
      console.log(`[switchback-trace][${el()}] setWebLifecycleState(frozen) failed: ${e.message} — continuing (real tab may not freeze in headless)`);
    }
    await page.waitForTimeout(FREEZE_MS);

    // record how much the in-flight GREW during freeze (proves SSE kept appending)
    const frozenGrewBy = await page.evaluate((before) => {
      const now = window.__sb_inflight_size ? window.__sb_inflight_size() : 0;
      return { before, now, delta: now - before };
    }, preFreezeSize).catch(() => ({ before: preFreezeSize, now: 0, delta: 0 }));

    // ---- THAW = switchback. Reset longtask buffer the instant we unfreeze so
    // the captured longtasks are purely the switchback catch-up cost. ----
    await page.evaluate(() => { if (window.__sb_longtasks) window.__sb_longtasks.length = 0; }).catch(() => {});
    const thawStart = Date.now();
    const thawPerfStart = await page.evaluate(() => performance.now()).catch(() => 0);
    try {
      await cdp.send('Page.setWebLifecycleState', { state: 'active' });
      console.log(`[switchback-trace][${el()}] THAWED — capturing switchback cost`);
    } catch (e) { console.log(`[switchback-trace][${el()}] setWebLifecycleState(active) failed: ${e.message}`); }

    // Let the switchback catch-up play out, then drain.
    await page.waitForTimeout(6000);
    const switchbackLt = await page.evaluate((perfStart) => {
      return {
        longtasks: (window.__sb_longtasks || []).slice(),
        elapsed: performance.now() - perfStart,
      };
    }, thawPerfStart).catch(() => ({ longtasks: [], elapsed: 0 }));
    const thawElapsedMs = Date.now() - thawStart;

    // ---- also capture a navigation-based switchback (the #223 scenario) for
    // contrast — navigate away then back, measure that jank too. This lets us
    // tell the two mechanisms apart (mount cost vs in-flight reparsal). ----
    await page.evaluate(() => { if (window.__sb_longtasks) window.__sb_longtasks.length = 0; }).catch(() => {});
    const navChatUrl = `${cfg.BASE_URL}/workspace/chats/${thread}`;
    await page.goto(`${cfg.BASE_URL}/workspace/chats/new`, { waitUntil: 'networkidle', timeout: 30000 }).catch(() => {});
    await page.waitForTimeout(800);
    const navT0 = Date.now();
    await page.goto(navChatUrl, { waitUntil: 'networkidle', timeout: 30000 }).catch(() => {});
    await page.waitForTimeout(4000);
    const navSwitchbackLt = await page.evaluate(() => (window.__sb_longtasks || []).slice()).catch(() => []);
    const navElapsedMs = Date.now() - navT0;

    // ---- write report ----
    function summarize(label, lts, elapsedMs) {
      const arr = (lts || []).map(x => typeof x === 'number' ? x : x.duration);
      const max = arr.length ? Math.round(Math.max(...arr)) : 0;
      const total = Math.round(arr.reduce((a, b) => a + b, 0));
      const over200 = arr.filter(x => x > 200).length;
      const over500 = arr.filter(x => x > 500).length;
      return { label, count: arr.length, longtask_max_ms: max, longtask_total_ms: total, over_200ms: over200, over_500ms: over500, window_ms: Math.round(elapsedMs) };
    }

    const report = {
      build: BUILD,
      thread,
      dataset: cfg.DATA_DIR,
      generated_at: el(),
      pre_freeze_inflight_size: preFreezeSize,
      frozen_grew_by: frozenGrewBy.delta,
      freeze_ms: FREEZE_MS,
      inflight_growth_samples: sizeSamples.length,
      switchback_after_freeze: summarize('switchback_after_freeze', switchbackLt.longtasks, switchbackLt.elapsed || thawElapsedMs),
      nav_switchback_contrast: summarize('nav_switchback_contrast', navSwitchbackLt, navElapsedMs),
      // raw longtask arrays (with startTime) for attribution:
      //   a single huge longtask ~= in-flight reparsal (root cause 2/3)
      //   many medium longtasks spread out ~= backlog flush (root cause 1)
      switchback_raw_longtasks: switchbackLt.longtasks,
      nav_raw_longtasks: navSwitchbackLt,
      notes: [
        'longtask entries >50ms are main-thread blocks.',
        'longtask_max_ms is the single biggest block during the switchback window.',
        'If switchback_after_freeze.longtask_max_ms >> nav_switchback_contrast.longtask_max_ms AND it correlates with a large frozen_grew_by, that supports root cause 2/3 (in-flight reparsal).',
        'If longtasks are many+medium+spread, that supports root cause 1 (backlog flush).',
        'dev build numbers are NOISE — only trust prod.',
      ],
    };
    fs.writeFileSync(path.join(cfg.OUT, 'switchback-trace.json'), JSON.stringify(report, null, 2));
    console.log(`[switchback-trace][${el()}] === SUMMARY ===`);
    console.log(JSON.stringify({
      pre_freeze_inflight_size: report.pre_freeze_inflight_size,
      frozen_grew_by: report.frozen_grew_by,
      switchback_after_freeze: report.switchback_after_freeze,
      nav_switchback_contrast: report.nav_switchback_contrast,
    }, null, 2));
    console.log(`[switchback-trace][${el()}] wrote ${path.join(cfg.OUT, 'switchback-trace.json')}`);
  } finally {
    await browser.close().catch(() => {});
  }
})().catch(e => { console.error('FATAL', e.stack || e); process.exit(1); });

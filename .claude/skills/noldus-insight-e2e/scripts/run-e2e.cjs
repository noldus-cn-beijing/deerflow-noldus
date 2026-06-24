// noldus-insight-e2e driver (merged): upload .xlsx → send analysis request →
// drive the agent through HITL clarifications to terminal → record evidence.
//
// Terminal detection is WORKSPACE-FILE based (not UI keywords — those false-
// positive early): handoff_report_writer.json exists AND report.md exists AND
// no Stop button. HITL answers are GENERIC ("确认", escalating to "继续推进"
// on a repeated identical question) — paradigm-specific answers are given live
// by the operator if needed; this script does not own them.
//
// Writes $E2E_OUT/.terminal at terminal break AND in a finally on any exit, so
// the bounded waiter in SKILL.md always resolves. Run in background.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
const { config, resolveChrome, resolveWsRoot, threadUserDataDir } = require('./lib');

const cfg = config();
if (!cfg.DATA_DIR) { console.error('FATAL: E2E_DATA_DIR required'); process.exit(1); }
if (!cfg.OUT) { console.error('FATAL: E2E_OUT required'); process.exit(1); }
if (!cfg.USER_ID) { console.error('FATAL: E2E_USER_ID required (run ensure-login.cjs first)'); process.exit(1); }

fs.mkdirSync(cfg.OUT, { recursive: true });
const TERMINAL_MARKER = path.join(cfg.OUT, '.terminal');
const WS_USERS = resolveWsRoot();
const T0 = Date.now();
const el = () => `${((Date.now() - T0) / 1000).toFixed(1)}s`;

const touchTerminal = (reason) => {
  try { fs.writeFileSync(TERMINAL_MARKER, `${el()} ${reason}\n`); } catch {}
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
    await page.goto(`${cfg.BASE_URL}/workspace/chats/new`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2500);
    let thread = '';
    try { thread = (page.url().split('/chats/')[1] || '').split(/[?#]/)[0]; } catch {}

    // upload all .xlsx (wait for chips)
    console.log(`[run-e2e] uploading ${files.length} files`);
    await page.locator('input[type=file]').first().setInputFiles(files);
    let chips = 0;
    for (let i = 0; i < 60; i++) {
      await page.waitForTimeout(1500);
      chips = await page.locator('text=Remove').count().catch(() => 0);
      if (chips >= files.length) break;
    }
    console.log(`[run-e2e] chips=${chips}/${files.length}`);
    if (chips < files.length) console.log(`[run-e2e] WARN: not all files chipped`);

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
            let ans = '确认'; let kind = 'generic';
            if (repeatedTail >= 2) { ans = '继续推进，按上述确认执行，不要重复反问'; kind = 'generic(REPEATED->continue)'; }
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

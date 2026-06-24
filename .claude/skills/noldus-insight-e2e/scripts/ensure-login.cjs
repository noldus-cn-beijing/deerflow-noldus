// Ensure a fresh-enough Playwright storageState (state.json) exists.
// - If state.json missing OR access_token JWT has <1h of life left → regenerate by
//   logging in via /login (email/password → storageState).
// - Threshold is <1h (not exact expiry) so a ~40min e2e run starting on a stale
//   token won't 401 mid-run. TTL is ~7d, so regeneration is rare.
// Prints `E2E_USER_ID=<jwt sub>` (derived from the token, never hardcoded).
const fs = require('fs');
const { chromium } = require('playwright');
const { config, resolveChrome, readStateClaims } = require('./lib');

const REGEN_THRESHOLD_MS = 60 * 60 * 1000; // 1h

function freshEnough(statePath) {
  if (!fs.existsSync(statePath)) return false;
  const claims = readStateClaims(statePath);
  if (!claims || !claims.exp) return false;
  const remaining = claims.exp * 1000 - Date.now();
  return remaining > REGEN_THRESHOLD_MS;
}

async function regenerate(cfg) {
  const chrome = resolveChrome();
  const browser = await chromium.launch({ executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  try {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    await page.goto(`${cfg.BASE_URL}/login`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.fill('input[type="email"], input[name="email"]', cfg.LOGIN_EMAIL).catch(async () => {
      // fall back to placeholder-based selector if no type=email input
      await page.fill('input[placeholder*="邮"], input[type="text"]', cfg.LOGIN_EMAIL);
    });
    await page.fill('input[type="password"]', cfg.LOGIN_PASSWORD);
    await Promise.all([
      page.waitForLoadState('networkidle', { timeout: 30000 }).catch(() => {}),
      page.click('button:has-text("登录")'),
    ]);
    await page.waitForTimeout(2500);
    fs.mkdirSync(require('path').dirname(cfg.STATE), { recursive: true });
    await ctx.storageState({ path: cfg.STATE });
    const cookies = await ctx.cookies();
    const names = cookies.map(c => c.name).join(',');
    console.error(`[ensure-login] regenerated state; cookies=${names}; url=${page.url()}`);
  } finally {
    await browser.close();
  }
}

(async () => {
  const cfg = config();
  if (freshEnough(cfg.STATE)) {
    const claims = readStateClaims(cfg.STATE);
    const remainingMin = Math.round((claims.exp * 1000 - Date.now()) / 60000);
    console.error(`[ensure-login] state fresh (token ${remainingMin}min left), skipping login`);
  } else {
    console.error(`[ensure-login] state missing/stale (<1h), regenerating via /login`);
    await regenerate(cfg);
  }
  const claims = readStateClaims(cfg.STATE);
  if (!claims || !claims.sub) {
    console.error('FATAL: no access_token sub in state.json after login');
    process.exit(1);
  }
  // machine-readable line for the SKILL.md runbook to capture
  console.log(`E2E_USER_ID=${claims.sub}`);
})().catch(e => { console.error('FATAL', e.stack || e); process.exit(1); });

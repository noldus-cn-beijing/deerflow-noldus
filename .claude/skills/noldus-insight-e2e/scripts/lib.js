// Shared helpers for noldus-insight-e2e skill scripts.
// Pure CommonJS (.cjs/.js) — no build step. All config via env vars so scripts
// stay standalone-testable: `E2E_DATA_DIR=... node scripts/run-e2e.cjs`.
const fs = require('fs');
const path = require('path');
const os = require('os');

// ---- env with sane defaults -------------------------------------------------
function env(key, def) {
  const v = process.env[key];
  return v === undefined || v === '' ? def : v;
}

const SKILL_DIR = env('E2E_SKILL_DIR', path.resolve(__dirname, '..'));

const config = () => ({
  DATA_DIR: env('E2E_DATA_DIR', ''),
  OUT: env('E2E_OUT', ''),
  STATE: env('E2E_STATE', path.join(SKILL_DIR, 'state.json')),
  BASE_URL: env('E2E_BASE_URL', 'http://localhost:2026'),
  LOGIN_EMAIL: env('E2E_LOGIN_EMAIL', 'qiuyang.wang@noldus.com'),
  LOGIN_PASSWORD: env('E2E_LOGIN_PASSWORD', '19961031'),
  REQUEST_TEXT: env('E2E_REQUEST_TEXT',
    '请分析这批行为学数据，给出专业的行为学判读：混杂因素排查、组间效应量、离群值，并出图和报告。'),
  DEADLINE_MIN: parseInt(env('E2E_DEADLINE_MIN', '45'), 10),
  USER_ID: env('E2E_USER_ID', ''),
  THREAD: env('E2E_THREAD', ''),
  THREAD_WS_ROOT: env('E2E_THREAD_WS_ROOT', ''),
  REPO_ROOT: env('E2E_REPO_ROOT', ''),
});

// ---- chromium executable path: pinned cache + glob fallback -----------------
// Cache layout: ~/.cache/ms-playwright/chromium-<ver>/chrome-linux64/chrome.
// Pin the known-good version; if missing, glob the highest chromium-* version.
function resolveChrome() {
  const override = process.env.E2E_CHROME;
  if (override && fs.existsSync(override)) return override;
  const cacheDir = path.join(os.homedir(), '.cache', 'ms-playwright');
  const pinned = path.join(cacheDir, 'chromium-1217', 'chrome-linux64', 'chrome');
  if (fs.existsSync(pinned)) return pinned;
  try {
    const dirs = fs.readdirSync(cacheDir)
      .filter(d => /^chromium-\d+$/.test(d))
      .sort((a, b) => parseInt(b.split('-')[1], 10) - parseInt(a.split('-')[1], 10));
    for (const d of dirs) {
      const p = path.join(cacheDir, d, 'chrome-linux64', 'chrome');
      if (fs.existsSync(p)) return p;
    }
  } catch { /* cache missing — caller will error */ }
  return pinned; // last resort; launch will fail loudly with actionable msg
}

// ---- JWT decode (no verify — we only read exp/sub from a token we just minted)
function decodeJwtExp(token) {
  try {
    const parts = token.split('.');
    if (parts.length < 2) return null;
    let p = parts[1];
    p = p + '='.repeat((4 - (p.length % 4)) % 4);
    const json = JSON.parse(Buffer.from(p, 'base64').toString('utf-8'));
    return { exp: json.exp || 0, sub: json.sub || '' };
  } catch { return null; }
}

// Read access_token cookie from a Playwright storageState JSON; decode JWT.
function readStateClaims(statePath) {
  try {
    const st = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
    const cookies = st.cookies || [];
    const at = cookies.find(c => c.name === 'access_token');
    if (!at) return null;
    return decodeJwtExp(at.value);
  } catch { return null; }
}

// ---- thread workspace root resolution --------------------------------------
// Walk up from cwd to find packages/agent/backend/.deer-flow anchor, OR honor
// E2E_REPO_ROOT override. Makes the skill portable across worktrees.
function resolveWsRoot() {
  const override = process.env.E2E_THREAD_WS_ROOT;
  if (override) return override;
  let base = process.env.E2E_REPO_ROOT || process.cwd();
  // If repo root given, anchor under it; otherwise walk up looking for the marker.
  const marker = path.join('packages', 'agent', 'backend', '.deer-flow');
  let dir = base;
  for (let i = 0; i < 12; i++) {
    if (fs.existsSync(path.join(dir, marker))) {
      return path.join(dir, 'packages', 'agent', 'backend', '.deer-flow', 'users');
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  // fallback: assume the known absolute layout
  return path.join('/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users');
}

// Per-thread user-data dir.
function threadUserdataDir(wsRootUsers, userId, threadId) {
  return path.join(wsRootUsers, userId, 'threads', threadId, 'user-data');
}

module.exports = {
  env, config, SKILL_DIR,
  resolveChrome, decodeJwtExp, readStateClaims,
  resolveWsRoot, threadUserDataDir: threadUserdataDir,
};

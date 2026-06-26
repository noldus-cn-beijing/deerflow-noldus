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

// ============================================================================
// Task A — HITL answer prefill (e2e-answers.yaml)
// ============================================================================
// A restricted-subset YAML parser for e2e-answers.yaml. We deliberately do NOT
// pull in js-yaml (keeps the skill dependency-free / standalone-testable). The
// supported grammar is exactly:
//   answers:
//     - match: ["kw1", "kw2", "regex.{0,4}"]   # any-keyword OR match
//       answer: "free text"
//   on_unmatched: fail | generic
// Lines outside this shape are ignored. Parse failures (bad on_unmatched value,
// entry missing match/answer) surface as { error } so the driver can fail-loud
// instead of silently degrading to "no answers" (守 feedback_oft_single_zone_
// must_ask_not_guess — never silently guess).
function parseAnswersYaml(text) {
  const out = { answers: [], onUnmatched: 'fail', error: null };
  if (!text || !text.trim()) return out;
  const lines = text.split(/\r?\n/);
  let i = 0;
  for (; i < lines.length; i++) { if (/^\s*answers\s*:\s*$/.test(lines[i])) break; }
  if (i >= lines.length) {
    for (const ln of lines) {
      const m = ln.match(/^\s*on_unmatched\s*:\s*(\S+)\s*$/);
      if (m) { const v = m[1]; if (v === 'fail' || v === 'generic') out.onUnmatched = v; else out.error = `invalid on_unmatched: ${v}`; }
    }
    return out;
  }
  i++;
  for (; i < lines.length; i++) {
    const ln = lines[i];
    const onMatch = ln.match(/^\s*on_unmatched\s*:\s*(\S+)\s*$/);
    if (onMatch) {
      const v = onMatch[1];
      if (v === 'fail' || v === 'generic') out.onUnmatched = v;
      else out.error = `invalid on_unmatched value: ${v} (must be fail|generic)`;
      continue;
    }
    const itemStart = ln.match(/^\s*-\s+(match|answer)\s*:\s*(.*)$/);
    if (!itemStart) {
      const cont = ln.match(/^\s+(match|answer)\s*:\s*(.*)$/);
      if (cont && out.answers.length) { _fillEntry(out.answers[out.answers.length - 1], cont[1], cont[2]); }
      continue;
    }
    const entry = { match: null, answer: null };
    out.answers.push(entry);
    _fillEntry(entry, itemStart[1], itemStart[2]);
  }
  for (const e of out.answers) {
    if (!Array.isArray(e.match) || e.match.length === 0) { out.error = out.error || 'answers entry missing `match` array'; }
    if (typeof e.answer !== 'string' || e.answer.length === 0) { out.error = out.error || 'answers entry missing `answer` string'; }
  }
  if (out.error) out.answers = out.answers.filter(e => Array.isArray(e.match) && e.match.length && typeof e.answer === 'string' && e.answer.length);
  return out;
}

// Fill one match/answer field from a raw RHS. Inline-array `["a","b"]` or bare.
// Array parsing splits on commas NOT inside quotes, so a regex quantifier like
// `{0,4}` (which contains a comma) inside a quoted keyword survives intact.
function _fillEntry(entry, field, raw) {
  raw = raw.trim();
  if (raw.startsWith('[')) {
    entry.match = _splitArray(_extractBracket(raw));
  } else if (field === 'match') {
    entry.match = raw ? [_unquote(_stripComment(raw))] : [];
  } else {
    entry.answer = _unquote(_stripComment(raw));
  }
}
// Extract the content between the first `[` and its matching `]` (respecting
// quotes), ignoring anything after the closing bracket (trailing `# comment`).
function _extractBracket(raw) {
  let depth = 0, q = null, end = -1;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (q) { if (ch === q) q = null; continue; }
    if (ch === '"' || ch === "'") { q = ch; continue; }
    if (ch === '[') depth++;
    else if (ch === ']') { depth--; if (depth === 0) { end = i; break; } }
  }
  if (end < 0) end = raw.length; // unbalanced — take everything
  return raw.slice(1, end);
}
// Strip a trailing `# comment` that is NOT inside a quote.
function _stripComment(raw) {
  let q = null;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i];
    if (q) { if (ch === q) q = null; continue; }
    if (ch === '"' || ch === "'") { q = ch; continue; }
    if (ch === '#') return raw.slice(0, i).trim();
  }
  return raw.trim();
}
// Split a YAML inline array body on commas, respecting quotes.
function _splitArray(inner) {
  const out = [];
  let cur = '', q = null;
  for (let i = 0; i < inner.length; i++) {
    const ch = inner[i];
    if (q) {
      cur += ch;
      if (ch === q) q = null;
    } else if (ch === '"' || ch === "'") {
      q = ch; cur += ch;
    } else if (ch === ',') {
      out.push(_unquote(cur.trim())); cur = '';
    } else { cur += ch; }
  }
  if (cur.trim()) out.push(_unquote(cur.trim()));
  return out.filter(s => s.length > 0);
}
function _unquote(s) {
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) return s.slice(1, -1);
  return s;
}

// Match a question against the answers list. Returns {answer, matchedKey} for
// the FIRST entry whose any keyword matches, or null. Keywords are matched as
// regex against the question substring (so "哪.{0,4}列" works). A malformed
// regex in one keyword is contained (skipped) so a single typo can't kill all
// matching.
function matchAnswer(parsed, questionText) {
  if (!parsed || !parsed.answers || !parsed.answers.length) return null;
  const q = questionText || '';
  for (const entry of parsed.answers) {
    if (!Array.isArray(entry.match)) continue;
    for (const kw of entry.match) {
      let re;
      try { re = new RegExp(kw); } catch { continue; }
      if (re.test(q)) return { answer: entry.answer, matchedKey: kw };
    }
  }
  return null;
}

// Load + parse the answers file. Returns null when absent (distinct from a
// parse error) so the driver can take the legacy generic path.
function loadAnswersFile(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  return parseAnswersYaml(fs.readFileSync(filePath, 'utf-8'));
}

// ============================================================================
// Task B — perf thresholds (single source of truth; analyze.py parses perf.json
// with the SAME field names + thresholds — keep them in sync).
// ============================================================================
// Thresholds are FIRST-RUN calibrated against a prod baseline (spec B.3: not
// picked from thin air). Defaults are the initial prod baseline; overridable via
// E2E_PERF_THRESHOLDS_JSON for re-baselining without a code change.
const defaultThresholds = {
  longtask_max_ms: 200,      // any single main-thread block > this = regression warn
  longtask_total_ms: 800,    // sum of longtasks during one fixed action
  interaction_p95_ms: 300,   // click → render-stable p95
};

// Evaluate a perf.json payload. Returns:
//   {status:'green'|'red', checks:[{name,action,value,threshold,unit,pass}]}
//   {status:'skipped', reason, checks:[]}  — dev build OR no perf captured.
// Never returns red/green on a dev build (spec B.0: dev data is noise).
function evaluatePerf(perf, thresholds) {
  const T = thresholds || defaultThresholds;
  if (!perf) return { status: 'skipped', reason: 'no perf.json captured', checks: [] };
  if (perf.build && String(perf.build).toLowerCase() !== 'prod') {
    return { status: 'skipped', reason: `PERF: skipped (${perf.build} build — only prod is trustworthy)`, checks: [] };
  }
  const actions = Array.isArray(perf.actions) ? perf.actions : [];
  if (!actions.length) return { status: 'skipped', reason: 'no perf actions captured', checks: [] };
  const checks = [];
  for (const a of actions) {
    if (typeof a.longtask_max_ms === 'number') {
      checks.push({ name: 'longtask_max', action: a.name, value: a.longtask_max_ms, threshold: T.longtask_max_ms, unit: 'ms', pass: a.longtask_max_ms <= T.longtask_max_ms });
    }
    if (typeof a.longtask_total_ms === 'number') {
      checks.push({ name: 'longtask_total', action: a.name, value: a.longtask_total_ms, threshold: T.longtask_total_ms, unit: 'ms', pass: a.longtask_total_ms <= T.longtask_total_ms });
    }
    if (Array.isArray(a.interaction_ms) && a.interaction_ms.length) {
      const p95 = _pct(a.interaction_ms, 95);
      checks.push({ name: 'interaction_p95', action: a.name, value: Math.round(p95), threshold: T.interaction_p95_ms, unit: 'ms', pass: p95 <= T.interaction_p95_ms });
    }
  }
  const status = checks.length && checks.every(c => c.pass) ? 'green' : 'red';
  return { status, checks };
}

// percentile (linear interpolation).
function _pct(arr, p) {
  const s = arr.slice().sort((a, b) => a - b);
  if (!s.length) return 0;
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

module.exports = {
  env, config, SKILL_DIR,
  resolveChrome, decodeJwtExp, readStateClaims,
  resolveWsRoot, threadUserDataDir: threadUserdataDir,
  // task A
  parseAnswersYaml, matchAnswer, loadAnswersFile,
  // task B
  defaultThresholds, evaluatePerf,
};

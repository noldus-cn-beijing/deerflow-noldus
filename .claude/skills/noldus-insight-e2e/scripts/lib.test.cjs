// Unit tests for noldus-insight-e2e shared helpers (lib.js).
// Deterministic — no Playwright, no server, no network. Run with:
//   node scripts/lib.test.cjs
// Exit 0 = all pass. Pure-logic coverage for the HITL answer prefill (task A)
// and the perf-threshold evaluation (task B). The full live dogfood (prod-build
// perf capture, real-data HITL) is a manual acceptance step, not a unit test.
const assert = require('assert');
const {
  parseAnswersYaml, matchAnswer, defaultThresholds, evaluatePerf,
} = require('./lib');

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log(`  ✓ ${name}`); }
  catch (e) { fail++; console.error(`  ✗ ${name}\n    ${e.stack || e.message}`); }
}

// ============================================================================
// Task A — answers YAML parsing + keyword/regex matching
// ============================================================================

test('parseAnswersYaml: basic shape — answers list + on_unmatched default fail', () => {
  const y = `
answers:
  - match: ["模板", "PlusMaze"]
    answer: "EPM 高架十字迷宫"
  - match: ["中心区", "哪.{0,4}列"]
    answer: "中心区=zone_center"
on_unmatched: fail
`;
  const p = parseAnswersYaml(y);
  assert.strictEqual(p.onUnmatched, 'fail');
  assert.strictEqual(p.answers.length, 2);
  assert.deepStrictEqual(p.answers[0].match, ['模板', 'PlusMaze']);
  assert.strictEqual(p.answers[0].answer, 'EPM 高架十字迷宫');
});

test('parseAnswersYaml: on_unmatched defaults to fail (spec A.3 default)', () => {
  const p = parseAnswersYaml('answers:\n  - match: ["x"]\n    answer: "y"\n');
  assert.strictEqual(p.onUnmatched, 'fail');
});

test('parseAnswersYaml: accepts generic (fallback to legacy 确认 behavior)', () => {
  const p = parseAnswersYaml('answers:\n  - match: ["x"]\n    answer: "y"\non_unmatched: generic\n');
  assert.strictEqual(p.onUnmatched, 'generic');
});

test('parseAnswersYaml: rejects invalid on_unmatched value (loud, not silent fallback)', () => {
  const p = parseAnswersYaml('answers: []\non_unmatched: guess\n');
  assert.ok(p.error, 'unknown on_unmatched should surface a parse error');
});

test('parseAnswersYaml: rejects entry missing match or answer', () => {
  const bad = parseAnswersYaml('answers:\n  - match: ["x"]\n'); // no answer
  assert.ok(bad.error, 'entry without answer must error (avoid silent no-op match)');
  const bad2 = parseAnswersYaml('answers:\n  - answer: "y"\n'); // no match
  assert.ok(bad2.error, 'entry without match must error');
});

test('parseAnswersYaml: empty / no-answers file yields answers=[] onUnmatched=fail', () => {
  // No answers file at all is the legacy path — represented as answers=[]. The
  // driver decides (absent file => generic legacy; present-but-empty is fine too).
  const p = parseAnswersYaml('');
  assert.deepStrictEqual(p.answers, []);
  assert.strictEqual(p.onUnmatched, 'fail');
});

test('matchAnswer: hits on any keyword in the match array (OR, not AND)', () => {
  const p = parseAnswersYaml(`
answers:
  - match: ["模板", "PlusMaze", "范式"]
    answer: "EPM"
`);
  assert.ok(matchAnswer(p, '请确认这是哪个 PlusMaze 模板？'));
  assert.ok(matchAnswer(p, '请问数据属于什么范式？'));
  assert.ok(matchAnswer(p, '请选择模板'));
});

test('matchAnswer: supports regex inside match keywords (哪.{0,4}列)', () => {
  const p = parseAnswersYaml(`
answers:
  - match: ["哪.{0,4}列", "分析区"]
    answer: "中心区=zone_center"
`);
  assert.ok(matchAnswer(p, '请问中心是哪一列代表中心区？'));
  assert.ok(matchAnswer(p, '边缘对应哪个分析区列？'));
  // no-op keyword that doesn't appear
  assert.ok(!matchAnswer(p, '请确认数据无误'));
});

test('matchAnswer: returns the FIRST matching entry (order matters, spec A.2)', () => {
  const p = parseAnswersYaml(`
answers:
  - match: ["通用"]
    answer: "FIRST"
  - match: ["通用"]
    answer: "SECOND"
`);
  const m = matchAnswer(p, '一个通用问题');
  assert.strictEqual(m.answer, 'FIRST');
});

test('matchAnswer: returns null on no match (driver then decides fail vs generic)', () => {
  const p = parseAnswersYaml(`answers:\n  - match: ["中心区"]\n    answer: "x"`);
  assert.strictEqual(matchAnswer(p, '一个完全无关的问题'), null);
});

test('matchAnswer: question text is matched as substring (anchored regex), not whole-string', () => {
  // A keyword must appear anywhere in the question. "Treatment" as a literal.
  const p = parseAnswersYaml(`answers:\n  - match: ["Treatment"]\n    answer: "按 Treatment 分组"`);
  assert.ok(matchAnswer(p, '这批数据要按 Treatment 列分组吗？'));
});

test('matchAnswer: malformed regex in a keyword is contained — that entry skipped, others still work', () => {
  // A bad regex in ONE keyword must not crash matching of OTHERS (defensive).
  const p = parseAnswersYaml(`
answers:
  - match: ["([unclosed", "ok-key"]
    answer: "A"
  - match: ["good"]
    answer: "B"
`);
  const m = matchAnswer(p, 'this has good keyword');
  assert.ok(m && m.answer === 'B', 'second entry must still match despite first entry bad regex');
});

test('parseAnswersYaml: comma inside a regex quantifier ({0,4}) survives array split', () => {
  // Regression pin: a naive comma-split would turn ["哪.{0,4}列","分析区"] into
  // 3 broken tokens. The quantifier comma must be respected.
  const p = parseAnswersYaml(`
answers:
  - match: ["哪.{0,4}列", "分析区"]
    answer: "center=zone_center"
`);
  assert.strictEqual(p.answers[0].match.length, 2, 'two keywords, not split on the quantifier comma');
  assert.strictEqual(p.answers[0].match[0], '哪.{0,4}列');
  assert.strictEqual(p.answers[0].match[1], '分析区');
  assert.ok(matchAnswer(p, '请问中心是哪一列代表？'), 'regex with quantifier comma actually matches');
});

test('parseAnswersYaml: trailing inline comment (# ...) after a value is stripped', () => {
  const p = parseAnswersYaml(`
answers:
  - match: ["Treatment"]   # grouping column
    answer: "按 Treatment 分组"
`);
  assert.strictEqual(p.answers[0].answer, '按 Treatment 分组');
  assert.deepStrictEqual(p.answers[0].match, ['Treatment']);
});

// ============================================================================
// Task B — perf threshold evaluation
// ============================================================================

test('defaultThresholds: defines P0 checks (longtask max + total block)', () => {
  assert.ok(defaultThresholds.longtask_max_ms > 0, 'longtask_max_ms threshold must be set');
  assert.ok(defaultThresholds.longtask_total_ms > 0);
  assert.ok(defaultThresholds.interaction_p95_ms > 0);
});

test('evaluatePerf: green when all P0 metrics under threshold', () => {
  const perf = {
    build: 'prod',
    actions: [
      { name: 'switch_back_thread', longtask_max_ms: 80, longtask_total_ms: 200, interaction_ms: [120, 130] },
      { name: 'open_gallery', longtask_max_ms: 60, longtask_total_ms: 150, interaction_ms: [90] },
    ],
  };
  const v = evaluatePerf(perf, defaultThresholds);
  assert.strictEqual(v.status, 'green');
  assert.ok(v.checks.length >= 2);
  assert.ok(v.checks.every(c => c.pass));
});

test('evaluatePerf: red when a longtask exceeds threshold (the 切回卡顿 signal)', () => {
  const perf = {
    build: 'prod',
    actions: [
      { name: 'switch_back_thread', longtask_max_ms: 350, longtask_total_ms: 900, interaction_ms: [400] },
    ],
  };
  const v = evaluatePerf(perf, defaultThresholds);
  assert.strictEqual(v.status, 'red');
  assert.ok(v.checks.some(c => !c.pass && /longtask_max/.test(c.name)));
});

test('evaluatePerf: skipped (not red/green) when build != prod (spec B.0 — dev data is noise)', () => {
  const perf = { build: 'dev', actions: [] };
  const v = evaluatePerf(perf, defaultThresholds);
  assert.strictEqual(v.status, 'skipped');
  assert.ok(/dev/i.test(v.reason || ''));
});

test('evaluatePerf: skipped when perf.json absent / no actions captured', () => {
  const v = evaluatePerf(null, defaultThresholds);
  assert.strictEqual(v.status, 'skipped');
  assert.ok(/no perf/i.test(v.reason || '') || /absent/i.test(v.reason || ''));
});

test('evaluatePerf: interaction p95 over threshold also turns red', () => {
  const perf = {
    build: 'prod',
    actions: [
      { name: 'switch_back_thread', longtask_max_ms: 50, longtask_total_ms: 80, interaction_ms: [600, 650, 700] },
    ],
  };
  const v = evaluatePerf(perf, defaultThresholds);
  assert.strictEqual(v.status, 'red');
});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);

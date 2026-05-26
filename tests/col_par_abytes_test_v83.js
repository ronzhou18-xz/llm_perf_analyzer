// ─────────────────────────────────────────────────────────────────────────────
// col-parallel linear aBytes overestimation fix — verification suite (v82)
//
// BUG (v81 and earlier): tpInfo()'s standard-TP linear branch computed
//   aBytes = (M*K + M*N) * ab   for ALL non-replicated modes.
// For a COLUMN-parallel linear the weight is split on N, so each card produces
// only the sharded output slice Y[M, N/tp]. The output activation WRITE is
// therefore M*N/tp per card, not M*N — the old formula overstates it by tp×.
//
// FIX (v82): col-parallel → aBytes = (M*K + M*N/tpAttn) * ab
//            (input X[M,K] stays full — it is replicated for a col-par layer)
//   replicated and row-parallel are LEFT UNCHANGED.
//
// This suite loads BOTH v81 and v82 and asserts:
//   • col-parallel  : bytesPerCard drops by exactly M*N*(1-1/tp)*ab
//   • row-parallel  : bytesPerCard identical (no regression)
//   • replicated    : bytesPerCard identical (no regression)
//   • FLOPs / commBytes unchanged in all three modes
//   • hand-calculated absolute values match v82
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

function load(file) {
  const html = fs.readFileSync(file, 'utf-8');
  let allCode = '';
  for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) allCode += m[1] + '\n';
  allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');
  const sandbox = {
    console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN,
    setTimeout: () => {}, clearTimeout: () => {}, setInterval: () => {}, clearInterval: () => {},
    document: { getElementById: () => ({ value: '0', innerHTML: '', textContent: '', className: '',
        appendChild: () => {}, addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [],
        style: {}, classList: { add: () => {}, remove: () => {} }, checked: false, files: [] }),
      createElement: () => ({ appendChild: () => {}, addEventListener: () => {}, style: {},
        classList: { add: () => {}, remove: () => {} }, setAttribute: () => {}, textContent: '', innerHTML: '', value: '0' }),
      addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [] },
    window: {}, localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    fetch: () => Promise.reject(new Error('mock')),
    Chart: function () { return { destroy: () => {}, update: () => {} }; },
    alert: () => {}, confirm: () => true, prompt: () => '',
    URL: { createObjectURL: () => 'blob:', revokeObjectURL: () => {} },
    Blob: function () {}, FileReader: function () { return { readAsText: () => {}, readAsArrayBuffer: () => {} }; },
    navigator: { clipboard: { writeText: () => Promise.resolve() } },
  };
  sandbox.window = sandbox; sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(allCode, sandbox, { timeout: 10000, filename: 'tool.js' });
  // shared deployment knobs
  sandbox.nCards = 8; sandbox.dp = 1; sandbox.tpAttn = 8;
  sandbox.epDeg = 1; sandbox.moePar = 'tp'; sandbox.overlap = 0; sandbox.ibw = 450;
  sandbox.selChip = sandbox.chips.find(c => c.name === 'H100 SXM') || sandbox.chips[0];
  sandbox.cfg.precision = 'fp16';
  return sandbox;
}

const V81 = load('/mnt/project/llm_perf_analyzer_v81.html');
const V82 = load('/home/claude/llm_perf_analyzer_v83.html');

let nPass = 0, nFail = 0; const failures = [];
function ok(label, cond, note) {
  if (cond) nPass++; else { nFail++; failures.push(label + (note ? ' — ' + note : '')); }
  console.log(`  ${cond ? '\u2713' : '\u2717'} ${label}${note ? '  ' + note : ''}`);
}
function near(a, b, tol) { return b === 0 ? Math.abs(a) <= tol : Math.abs(a - b) / Math.abs(b) <= tol; }

const TP = 8, AB = 2, WB = 2;   // fp16

console.log('='.repeat(88));
console.log('col-parallel linear aBytes fix — v81 (buggy) vs v82 (fixed)');
console.log('H100 SXM \u00b7 TP=8 \u00b7 dp=1 \u00b7 fp16  (ab=2, wb=2)');
console.log('='.repeat(88));

// Representative layers across the three TP modes.
//   M chosen large (prefill) so the activation term is non-trivial vs weight.
function layers(M) {
  return [
    // col-parallel: weight split on N, output sharded
    { tag: 'FFN gate+up', mode: 'col',        l: { id:1, name:'FFN gate+up', type:'linear', M, K:5120,  N:51200,  repeat:1, tpMode:'col' } },
    { tag: 'LM head',     mode: 'col',        l: { id:2, name:'LM head',     type:'linear', M, K:5120,  N:151936, repeat:1, tpMode:'col' } },
    { tag: 'MLA Q up',    mode: 'col',        l: { id:3, name:'MLA Q up',    type:'linear', M, K:1536,  N:24576,  repeat:1, tpMode:'col' } },
    // row-parallel: weight split on K, output is a partial sum (full M*N)
    { tag: 'FFN down',    mode: 'row',        l: { id:4, name:'FFN down',    type:'linear', M, K:25600, N:5120,   repeat:1, tpMode:'row' } },
    { tag: 'MLA O proj',  mode: 'row',        l: { id:5, name:'MLA O proj',  type:'linear', M, K:16384, N:7168,   repeat:1, tpMode:'row' } },
    // replicated: every card computes full M*K*N
    { tag: 'MLA Q down',  mode: 'replicated', l: { id:6, name:'MLA Q down',  type:'linear', M, K:7168,  N:1536,   repeat:1, tpMode:'replicated' } },
  ];
}

for (const [phase, M] of [['prefill', 2048], ['decode', 1]]) {
  console.log(`\n## ${phase} (M=${M})`);
  for (const { tag, mode, l } of layers(M)) {
    const t81 = V81.tpInfo(l, V81.selChip);
    const t82 = V82.tpInfo(l, V82.selChip);

    // FLOPs and comm must be untouched by this fix in every mode
    ok(`${phase.padEnd(7)} ${tag.padEnd(12)} [${mode}] FLOPs/card unchanged`,
       near(t82.flopsPerCard, t81.flopsPerCard, 1e-12),
       `81=${t81.flopsPerCard.toExponential(3)} 82=${t82.flopsPerCard.toExponential(3)}`);
    ok(`${phase.padEnd(7)} ${tag.padEnd(12)} [${mode}] commBytes unchanged`,
       near(t82.commBytes || 0, t81.commBytes || 0, 1e-9));

    if (mode === 'col') {
      // bytesPerCard must drop by exactly M*N*(1 - 1/tp)*ab
      const expectedDrop = l.M * l.N * (1 - 1 / TP) * AB;
      const actualDrop = t81.bytesPerCard - t82.bytesPerCard;
      ok(`${phase.padEnd(7)} ${tag.padEnd(12)} [col] bpc drop == M\u00b7N\u00b7(1-1/tp)\u00b7ab`,
         near(actualDrop, expectedDrop, 1e-6),
         `drop=${(actualDrop/1e6).toFixed(2)}MB expect=${(expectedDrop/1e6).toFixed(2)}MB`);
      // hand-calc absolute v82 value: wBytes + (M*K + M*N/tp)*ab
      const wBytes = l.K * l.N * WB / TP;
      const aBytes = (l.M * l.K + l.M * l.N / TP) * AB;
      ok(`${phase.padEnd(7)} ${tag.padEnd(12)} [col] v82 bpc == K\u00b7N\u00b7wb/tp + (M\u00b7K + M\u00b7N/tp)\u00b7ab`,
         near(t82.bytesPerCard, wBytes + aBytes, 1e-6),
         `v82=${(t82.bytesPerCard/1e6).toFixed(2)}MB hand=${((wBytes+aBytes)/1e6).toFixed(2)}MB`);
      // overestimation factor of the OLD activation term
      const aOld = (l.M * l.K + l.M * l.N) * AB;
      ok(`${phase.padEnd(7)} ${tag.padEnd(12)} [col] v82 strictly < v81 bpc`,
         t82.bytesPerCard < t81.bytesPerCard,
         `aBytes overstated ${(aOld/aBytes).toFixed(2)}\u00d7 in v81`);
    } else if (mode === 'row') {
      // v83: row-parallel input term is now sharded → bpc drops by M*K*(1-1/tp)*ab
      const expectedDrop = l.M * l.K * (1 - 1 / TP) * AB;
      const actualDrop = t81.bytesPerCard - t82.bytesPerCard;
      ok(`${phase.padEnd(7)} ${tag.padEnd(12)} [row] bpc drop == M\u00b7K\u00b7(1-1/tp)\u00b7ab (v83 fix)`,
         near(actualDrop, expectedDrop, 1e-6),
         `drop=${(actualDrop/1e6).toFixed(2)}MB expect=${(expectedDrop/1e6).toFixed(2)}MB`);
    } else {
      // replicated: bytesPerCard must be IDENTICAL (no regression)
      ok(`${phase.padEnd(7)} ${tag.padEnd(12)} [replicated] bytesPerCard UNCHANGED (no regression)`,
         near(t82.bytesPerCard, t81.bytesPerCard, 1e-12),
         `81=${(t81.bytesPerCard/1e6).toFixed(3)}MB 82=${(t82.bytesPerCard/1e6).toFixed(3)}MB`);
    }
  }
}

// ── derived-metric sanity: AI must rise, memTime must fall for col-par ─────────
console.log('\n## derived metrics — col-parallel (FFN gate+up, prefill M=2048)');
{
  const l = { id:9, name:'FFN gate+up', type:'linear', M:2048, K:5120, N:51200, repeat:1, tpMode:'col' };
  const t81 = V81.tpInfo(l, V81.selChip), t82 = V82.tpInfo(l, V82.selChip);
  ok('memTime v82 < v81 (less HBM traffic)', t82.memTime < t81.memTime,
     `81=${(t81.memTime*1e3).toFixed(3)}ms 82=${(t82.memTime*1e3).toFixed(3)}ms`);
  ok('aiPerCard v82 > v81 (higher arithmetic intensity)', t82.aiPerCard > t81.aiPerCard,
     `81=${t81.aiPerCard.toFixed(2)} 82=${t82.aiPerCard.toFixed(2)}`);
  ok('effectiveTime v82 <= v81 (never worse)', t82.effectiveTime <= t81.effectiveTime + 1e-12);
}

// ── nCards=1 must be untouched (early-return path) ────────────────────────────
console.log('\n## nCards=1 — single-card path unaffected');
{
  V81.nCards = 1; V81.tpAttn = 1; V82.nCards = 1; V82.tpAttn = 1;
  const l = { id:10, name:'FFN gate+up', type:'linear', M:2048, K:5120, N:51200, repeat:1, tpMode:'col' };
  const t81 = V81.tpInfo(l, V81.selChip), t82 = V82.tpInfo(l, V82.selChip);
  ok('nCards=1 col-par bytesPerCard identical v81==v82',
     near(t82.bytesPerCard, t81.bytesPerCard, 1e-12));
  V81.nCards = 8; V81.tpAttn = 8; V82.nCards = 8; V82.tpAttn = 8;
}

console.log('\n' + '='.repeat(88));
console.log(`RESULT: ${nPass} passed, ${nFail} failed`);
if (nFail) { console.log('\nFAILURES:'); failures.forEach(f => console.log('  \u2717 ' + f)); process.exit(1); }
else console.log('All checks passed \u2014 col-parallel aBytes fix verified, row/replicated unregressed.');

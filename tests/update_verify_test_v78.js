// ─────────────────────────────────────────────────────────────────────────────
// v76 update verification
//   (1) per-rank "(per rank)" qualifier moved from the batch <label> to a
//       .fhint UNDER the input  → cfgBatchHint/-D toggle, label stays plain.
//   (2) switching parallelism keeps GLOBAL batch fixed; per-rank batch follows
//       (onTPChange: dp change → newPerRank = ceil(prevGlobalBatch / dp)).
//   (3) tpInfo()/tpInfoTotal() expose commBytes on EVERY return path; the
//       decode-step table renders a Comm/call column.
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || '/home/claude/llm_perf_analyzer_v78.html';
const html = fs.readFileSync(FILE, 'utf-8');
let allCode = '';
for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) allCode += m[1] + '\n';
allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');

// ── stateful mock DOM ───────────────────────────────────────────────────────
const elements = {};
function makeSelect() {
  return {
    _options: [], _value: '1',
    get value() { return this._value; },
    set value(v) { this._value = String(v); },
    set innerHTML(h) { if (h === '') this._options.length = 0; },
    get innerHTML() { return ''; },
    appendChild(opt) { this._options.push(opt); if (this._options.length === 1) this._value = String(opt.value); },
    querySelector(sel) { const m = sel.match(/value="([^"]+)"/); if (!m) return null;
      return this._options.find(o => String(o.value) === m[1]) || null; },
    querySelectorAll() { return []; },
    addEventListener() {}, style: {}, classList: { add() {}, remove() {} }, disabled: false,
  };
}
function makeEl(id) {
  if (id === 'dpSel' || id === 'epDegSel') return makeSelect();
  return { id, value: '', textContent: '', innerHTML: '', className: '', checked: false, files: [],
    style: {}, classList: { add() {}, remove() {} }, disabled: false,
    appendChild() {}, addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; } };
}
const documentMock = {
  getElementById: (id) => { if (!elements[id]) elements[id] = makeEl(id); return elements[id]; },
  createElement: (tag) => tag === 'option'
    ? { value: '', textContent: '', _tag: 'option' }
    : { appendChild() {}, addEventListener() {}, style: {}, classList: { add() {}, remove() {} },
        setAttribute() {}, textContent: '', innerHTML: '', value: '0' },
  addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [],
};
const sandbox = {
  console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN,
  setTimeout: () => {}, clearTimeout: () => {}, setInterval: () => {}, clearInterval: () => {},
  document: documentMock, window: {}, localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  fetch: () => Promise.reject(new Error('mock')),
  Chart: function () { return { destroy: () => {}, update: () => {} }; },
  alert: () => {}, confirm: () => true, prompt: () => '',
  URL: { createObjectURL: () => 'blob:', revokeObjectURL: () => {} },
  Blob: function () {}, FileReader: function () { return { readAsText: () => {}, readAsArrayBuffer: () => {} }; },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
try { vm.runInContext(allCode, sandbox, { timeout: 10000, filename: 'tool.js' }); }
catch (e) { console.log('Load error:', e.message); process.exit(1); }

// stub out heavy re-render fns so onTPChange can run; keep state/calc fns real
['renderTab', 'renderLayerList', 'renderChipList', 'updateCfgInfo', 'renderArchInfo',
 'loadHfEp', 'refreshMoEParAvailability'].forEach(fn => { if (typeof sandbox[fn] === 'function') sandbox[fn] = () => {}; });

let nPass = 0, nFail = 0;
const failures = [];
function ok(label, cond, note) {
  if (cond) nPass++; else { nFail++; failures.push(label + (note ? ' — ' + note : '')); }
  console.log(`  ${cond ? '✓' : '✗'} ${label}${note ? '  ' + note : ''}`);
}
function near(a, b, tol) { return b === 0 ? Math.abs(a) <= tol : Math.abs(a - b) / Math.abs(b) <= tol; }

console.log('═'.repeat(86));
console.log('v76 update verification');
console.log(`File: ${FILE}`);
console.log('═'.repeat(86));

// ════════════════════════════════════════════════════════════════════════════
// (3) commBytes exposed on every tpInfo return path + consistent with commTime
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## (3) tpInfo/tpInfoTotal expose commBytes — bytes ↔ time consistency');
sandbox.ibw = 450;
sandbox.overlap = 0;
sandbox.selChip = sandbox.chips.find(c => c.name === 'H100 SXM') || sandbox.chips[0];
sandbox.cfg.precision = 'fp16';
const IBW = 450e9;
const M_HID = 6144, M_FFN = 2048, M_NE = 256, M_TOPK = 8, M_NS = 1;

function mkMoE(M) { return { id: 1, name: 'MoE', type: 'moe', M, hid: M_HID, ffnDim: M_FFN,
  topK: M_TOPK, nExperts: M_NE, nShared: M_NS, repeat: 3 }; }
function mkGU(M) { return { id: 2, name: 'gate_up', type: 'linear', moeGroup: 'gate_up', M, K: M_HID, N: 2 * M_FFN,
  moeExperts: M_NE, moeTopK: M_TOPK, moeFfnDim: M_FFN, moeHid: M_HID, repeat: 3 }; }
function mkDN(M) { return { id: 3, name: 'down', type: 'linear', moeGroup: 'down', M, K: M_FFN, N: M_HID,
  moeExperts: M_NE, moeTopK: M_TOPK, moeFfnDim: M_FFN, moeHid: M_HID, repeat: 3 }; }
function mkRowLin(M) { return { id: 4, name: 'Attn O proj (×64)', type: 'linear', M, K: 8192, N: 8192, repeat: 5 }; }
function mkEmb(M) { return { id: 5, name: 'embed_tokens', type: 'embedding', M, K: 152064, N: 6144, repeat: 1 }; }
function mkAttn(M) { return { id: 6, name: 'attn', type: 'attention', B: M, S: 1, H: 64, nKV: 8, D: 128,
  kvCache: 4096, fa2: true, repeat: 4 }; }

const CONFIGS = [
  { nCards: 1,  dp: 1,  epDeg: 1,  moePar: 'tp',     label: 'single card' },
  { nCards: 8,  dp: 1,  epDeg: 1,  moePar: 'tp',     label: 'TP8 (MoE TP)' },
  { nCards: 8,  dp: 1,  epDeg: 8,  moePar: 'ep',     label: 'TP1×EP8' },
  { nCards: 16, dp: 16, epDeg: 16, moePar: 'ep',     label: 'DP16-EP16' },
  { nCards: 32, dp: 8,  epDeg: 4,  moePar: 'hybrid', label: 'DP8·EP4×eTp8' },
];
for (const cf of CONFIGS) {
  sandbox.nCards = cf.nCards; sandbox.dp = cf.dp; sandbox.tpAttn = Math.max(1, cf.nCards / cf.dp);
  sandbox.epDeg = cf.epDeg; sandbox.moePar = cf.moePar;
  let allHaveField = true, allConsistent = true, totalScales = true;
  for (const mk of [mkMoE, mkGU, mkDN, mkRowLin, mkEmb, mkAttn]) {
    for (const M of [1, 4096]) {
      const layer = mk(M);
      const one = sandbox.tpInfo(layer, sandbox.selChip);
      const tot = sandbox.tpInfoTotal(layer, sandbox.selChip);
      if (typeof one.commBytes !== 'number' || isNaN(one.commBytes)) allHaveField = false;
      if (typeof tot.commBytes !== 'number' || isNaN(tot.commBytes)) allHaveField = false;
      // commBytes / (ibw*1e9) must equal commTime  (overlap=0 isolates raw comm)
      if (!near(one.commBytes / IBW, one.commTime, 1e-6)) allConsistent = false;
      // tpInfoTotal.commBytes == tpInfo.commBytes * repeat
      const r = mk === mkEmb ? 1 : layer.repeat;
      if (!near(tot.commBytes, one.commBytes * r, 1e-9)) totalScales = false;
    }
  }
  ok(`${cf.label.padEnd(18)} commBytes present on all return paths`, allHaveField);
  ok(`${cf.label.padEnd(18)} commBytes/(ibw·1e9) == commTime`, allConsistent);
  ok(`${cf.label.padEnd(18)} tpInfoTotal.commBytes == ×repeat`, totalScales);
}

// row-parallel linear AllReduce volume must equal the known closed form
console.log('\n## (3) Comm/call magnitude spot-check (row-par linear AllReduce)');
{
  sandbox.nCards = 8; sandbox.dp = 1; sandbox.tpAttn = 8; sandbox.epDeg = 1; sandbox.moePar = 'tp';
  const M = 4096, N = 8192, ab = 2, tpAttn = 8;
  const one = sandbox.tpInfo(mkRowLin(M), sandbox.selChip);
  const ref = 2 * ((tpAttn - 1) / tpAttn) * M * N * ab;
  ok('TP8 "Attn O proj" AllReduce bytes == 2·(tp-1)/tp·M·N·ab', near(one.commBytes, ref, 1e-6),
     `got=${(one.commBytes / 1e6).toFixed(2)}MB ref=${(ref / 1e6).toFixed(2)}MB`);
  // col-parallel linear (no AllReduce) → commBytes 0
  const colLin = { id: 9, name: 'Attn Q proj (×64)', type: 'linear', M, K: 8192, N: 16384, repeat: 1 };
  const colOne = sandbox.tpInfo(colLin, sandbox.selChip);
  ok('TP8 col-par linear → commBytes 0', colOne.commBytes === 0, `got=${colOne.commBytes}`);
}

// ════════════════════════════════════════════════════════════════════════════
// (2) switching parallelism keeps GLOBAL batch fixed; per-rank follows
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## (2) onTPChange — global batch invariant across DP switches');

// helper: drive onTPChange the way the UI would
function driveTP(exp, dpVal, ovVal) {
  elements['tpS'] = elements['tpS'] || makeEl('tpS');
  elements['ovS'] = elements['ovS'] || makeEl('ovS');
  elements['tpS'].value = String(exp);
  elements['ovS'].value = String(ovVal == null ? 0 : ovVal);
  if (dpVal != null) { const ds = sandbox.document.getElementById('dpSel'); ds.value = String(dpVal); }
  sandbox.onTPChange();
}
function gb() { return sandbox.cfg.batch * sandbox.dp; }

// give the model a couple of layers so applyConfigToLayers has something to touch
sandbox.layers = [ mkAttn(1), { id: 20, name: 'q_proj', type: 'linear', M: 1, K: 6144, N: 8192, repeat: 4 } ];
sandbox.mode = 'decode';

// Case A: nCards=8, establish dp=1 with per-rank batch 8 → global 8
sandbox.cfg.batch = 8;
sandbox.document.getElementById('cfgBatch').value = '8';
sandbox.document.getElementById('cfgBatchD').value = '8';
driveTP(3, 1, 0);                       // 2^3 = 8 cards, dp=1
const gbA = gb();
ok('A: dp=1 baseline — global batch = 8', gbA === 8, `global=${gbA} (batch=${sandbox.cfg.batch}×dp=${sandbox.dp})`);

// Case B: switch dp 1→2 — global must STAY 8, per-rank must FOLLOW to 4
driveTP(3, 2, 0);
ok('B: dp 1→2 — global batch unchanged (8)', gb() === 8, `global=${gb()}`);
ok('B: dp 1→2 — per-rank batch followed to 4', sandbox.cfg.batch === 4, `batch=${sandbox.cfg.batch}`);
ok('B: dp 1→2 — cfgBatch input mirrors per-rank', elements['cfgBatch'].value == 4 && elements['cfgBatchD'].value == 4,
   `cfgBatch=${elements['cfgBatch'].value} cfgBatchD=${elements['cfgBatchD'].value}`);
ok('B: dp 1→2 — layer.B pushed via applyConfigToLayers', sandbox.layers[0].B === 4, `layer.B=${sandbox.layers[0].B}`);

// Case C: switch dp 2→4 — global stays 8, per-rank → 2
driveTP(3, 4, 0);
ok('C: dp 2→4 — global batch unchanged (8)', gb() === 8, `global=${gb()}`);
ok('C: dp 2→4 — per-rank batch followed to 2', sandbox.cfg.batch === 2, `batch=${sandbox.cfg.batch}`);

// Case D: switch dp 4→8 — global stays 8, per-rank → 1
driveTP(3, 8, 0);
ok('D: dp 4→8 — global batch unchanged (8)', gb() === 8, `global=${gb()}`);
ok('D: dp 4→8 — per-rank batch followed to 1', sandbox.cfg.batch === 1, `batch=${sandbox.cfg.batch}`);

// Case E: collapse dp 8→1 — global stays 8, per-rank jumps back to 8
driveTP(3, 1, 0);
ok('E: dp 8→1 — global batch unchanged (8)', gb() === 8, `global=${gb()}`);
ok('E: dp 8→1 — per-rank batch back to 8', sandbox.cfg.batch === 8, `batch=${sandbox.cfg.batch}`);

// Case F: overlap-only change must NOT touch batch (dp unchanged)
const batchBeforeF = sandbox.cfg.batch;
driveTP(3, 1, 0.5);
ok('F: overlap-only change leaves batch untouched', sandbox.cfg.batch === batchBeforeF,
   `batch=${sandbox.cfg.batch} (was ${batchBeforeF})`);

// Case G: non-divisible global — dp 1→4 with per-rank 6 (global 6) → ceil(6/4)=2, global 'computes as' 8
sandbox.cfg.batch = 6; elements['cfgBatch'].value = '6'; elements['cfgBatchD'].value = '6';
driveTP(3, 1, 0);                       // re-establish dp=1, global=6
driveTP(3, 4, 0);                       // dp 1→4
ok('G: non-divisible — per-rank = ceil(6/4) = 2', sandbox.cfg.batch === 2, `batch=${sandbox.cfg.batch}`);
ok('G: non-divisible — original global (6) stashed in Global field',
   elements['cfgGlobalBatch'].value == 6, `cfgGlobalBatch=${elements['cfgGlobalBatch'].value}`);

// ════════════════════════════════════════════════════════════════════════════
// (1) per-rank qualifier lives in a hint UNDER the input, label stays plain
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## (1) batch per-rank qualifier — hint under input, not in label');

// dp>1 → hint visible, label plain "Batch size"
driveTP(3, 2, 0);
const lbl = elements['cfgBatchLbl'], lblD = elements['cfgBatchLblD'];
const hint = elements['cfgBatchHint'], hintD = elements['cfgBatchHintD'];
ok('dp>1 — label is plain "Batch size" (no clipped suffix)',
   lbl.textContent === 'Batch size' && lblD.textContent === 'Batch size',
   `lbl="${lbl.textContent}"`);
ok('dp>1 — per-rank hint shown under input', hint.style.display === '' && hintD.style.display === '',
   `display="${hint.style.display}"`);

// dp=1 → hint hidden
driveTP(3, 1, 0);
ok('dp=1 — per-rank hint hidden', hint.style.display === 'none' && hintD.style.display === 'none',
   `display="${hint.style.display}"`);
ok('dp=1 — label still plain "Batch size"',
   lbl.textContent === 'Batch size' && lblD.textContent === 'Batch size');

// hint element actually exists in the HTML with the expected copy
const htmlHasHint = /id="cfgBatchHint"[^>]*>per DP rank/.test(html) && /id="cfgBatchHintD"[^>]*>per DP rank/.test(html);
ok('HTML — cfgBatchHint/-D fhint elements present with per-rank copy', htmlHasHint);
// the old label-suffix code path must be gone
const noOldSuffix = !/Batch size \(per rank\)/.test(html);
ok('HTML — old "Batch size (per rank)" label suffix removed', noOldSuffix);

// decode table must carry the new Comm/call header
const hasCommCol = /<th>Comm\/call<\/th>/.test(html);
ok('HTML — decode step table has Comm/call column header', hasCommCol);

console.log('\n' + '═'.repeat(86));
console.log(`Summary: ${nPass} pass, ${nFail} fail (of ${nPass + nFail} total)`);
if (failures.length) { console.log('\nFailures:'); failures.forEach(f => console.log('  • ' + f)); }
console.log('═'.repeat(86));
process.exit(nFail ? 1 : 0);

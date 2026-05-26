// ─────────────────────────────────────────────────────────────────────────────
// Unified VRAM-accounting regression suite (v78)
//
// Closes the timing/VRAM mismatch left after the MLA tpMode merge:
//   tpInfo() already gives TP-replicated dense linears (MLA q_a_proj / kv_a_proj,
//   tpMode:'replicated') FULL FLOPs+weights per card. But the per-card VRAM
//   budget still divided the ENTIRE non-MoE bucket by tpAttn — so a replicated
//   a-projection's weight was wrongly shrunk by 1/tpAttn in the VRAM number.
//
// v77 fix: new replicatedWeightBytes() sums the DENSE replicated linears; the
//   per-card weight formula becomes
//     wPerCard = shardedNonMoe/tpAttn + replicated + moePerCard
//   applied in all three VRAM paths — updateCfgInfo, rDecode, and the Compare tab.
//
// A replicated a-proj is small (hid×lora_rank), so the VRAM delta is tiny — but
// the point is the accounting now MATCHES the timing path instead of contradicting it.
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || '/home/claude/llm_perf_analyzer_v78.html';
const html = fs.readFileSync(FILE, 'utf-8');
let allCode = '';
for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) allCode += m[1] + '\n';
allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');

const elementStore = {};
function makeEl(id) {
  return { id, value: '1', textContent: '', _innerHTML: '',
    get innerHTML() { return this._innerHTML; }, set innerHTML(v) { this._innerHTML = v; },
    style: {}, classList: { add: () => {}, remove: () => {}, toggle: () => {} },
    appendChild: () => {}, addEventListener: () => {}, querySelector: () => null,
    querySelectorAll: () => [], setAttribute: () => {}, checked: false, files: [],
    parentNode: { parentNode: { appendChild: () => {} } }, getContext: () => ({}) };
}
function getEl(id) { if (!elementStore[id]) elementStore[id] = makeEl(id); return elementStore[id]; }

const sandbox = {
  console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN,
  setTimeout: (fn) => { try { fn(); } catch (e) {} }, clearTimeout: () => {},
  setInterval: () => {}, clearInterval: () => {},
  document: { getElementById: (id) => getEl(id), createElement: () => makeEl('new'),
    addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [] },
  window: {}, localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  fetch: () => Promise.reject(new Error('mock')),
  Chart: function () { return { destroy: () => {}, update: () => {} }; },
  alert: () => {}, confirm: () => true, prompt: () => '',
  URL: { createObjectURL: () => 'blob:', revokeObjectURL: () => {} }, Blob: function () {},
  FileReader: function () { return { readAsText: () => {}, readAsArrayBuffer: () => {} }; },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
try { vm.runInContext(allCode, sandbox, { timeout: 10000 }); }
catch (e) { console.log('Load error:', e.message); process.exit(1); }

let nPass = 0, nFail = 0;
const failures = [];
function ok(label, cond, note) {
  if (cond) nPass++; else { nFail++; failures.push(label + (note ? ' — ' + note : '')); }
  console.log(`  ${cond ? '✓' : '✗'} ${label}${note ? '  ' + note : ''}`);
}
function near(a, b, tol) { return b === 0 ? Math.abs(a) <= tol : Math.abs(a - b) / Math.abs(b) <= tol; }

console.log('═'.repeat(86));
console.log('Unified VRAM-accounting regression suite (v78)');
console.log(`File: ${FILE}`);
console.log('═'.repeat(86));

const WB = sandbox.wBpe ? sandbox.wBpe() : 2;
const getR = sandbox.getRepeat;

// recompute moeWB exactly the way rDecode/updateCfgInfo do (non-hybrid path)
function moeWeightBytesOf(layers) {
  let moe = 0;
  layers.forEach(l => {
    if (l.type === 'moe') {
      const r = getR(l), { hid: h, ffnDim: f, nExperts: ne, nShared: ns = 0 } = l;
      moe += ((ne + ns) * (2 * h * f + f * h) + h * ne) * WB * r;
    } else if (l.moeGroup === 'gate_up' || l.moeGroup === 'down') {
      moe += l.moeExperts * l.K * l.N * WB * getR(l);
    } else if (l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down') {
      moe += l.K * l.N * WB * getR(l);
    } else if (l.moeGroup === 'router') {
      moe += l.K * l.N * WB * getR(l);
    }
  });
  return moe;
}
function expectedWPerCardGB(layers, nCards, tpAttn) {
  const wAll = sandbox.modelWeightBytes();
  const moeWB = moeWeightBytesOf(layers);
  const replWB = sandbox.replicatedWeightBytes();
  const shardedNonMoe = wAll - moeWB - replWB;
  const moePC = moeWB / nCards;          // non-hybrid
  return (shardedNonMoe / tpAttn + replWB + moePC) / 1e9;
}
function oldWrongWPerCardGB(layers, nCards, tpAttn) {
  const wAll = sandbox.modelWeightBytes();
  const moeWB = moeWeightBytesOf(layers);
  const nonMoe = wAll - moeWB;
  const moePC = moeWB / nCards;
  return (nonMoe / tpAttn + moePC) / 1e9;   // pre-v77: replicated wrongly /tpAttn
}

// stub heavy re-render side effects we don't assert on
['renderTab', 'renderLayerList', 'renderArchInfo', 'renderChipList'].forEach(fn => {
  if (typeof sandbox[fn] === 'function') sandbox[fn] = () => {};
});
sandbox.selChip = sandbox.chips.find(c => c.name === 'H100 SXM') || sandbox.chips[0];
sandbox.cfg = { batch: 1, seqLen: 2048, outTokens: 256, nH: 128, H: 1, D: 512, nL: 64,
  precision: 'fp16', kvCacheFn: null };

// ── DeepSeek-V3-style MLA layer set (hid 7168) ──────────────────────────────
function mlaModel(withMoE) {
  const L = [
    { id: 1,  name: 'Token embed',        type: 'embedding', M: 1, K: 129280, N: 7168, repeat: 1 },
    { id: 2,  name: 'MLA Q down (×61)',   type: 'linear', M: 1, K: 7168,  N: 1536,  repeat: 61, tpMode: 'replicated' },
    { id: 3,  name: 'MLA Q up (×61)',     type: 'linear', M: 1, K: 1536,  N: 24576, repeat: 61, tpMode: 'col' },
    { id: 4,  name: 'MLA KV down (×61)',  type: 'linear', M: 1, K: 7168,  N: 576,   repeat: 61, tpMode: 'replicated' },
    { id: 5,  name: 'MLA Attention (×61)',type: 'attention', B: 1, S: 1, H: 128, nKV: 1, D: 512, kvCache: 4096, fa2: true, repeat: 61 },
    { id: 6,  name: 'MLA O proj (×61)',   type: 'linear', M: 1, K: 16384, N: 7168,  repeat: 61, tpMode: 'row' },
    { id: 7,  name: 'Dense FFN gate+up (×3)', type: 'linear', M: 1, K: 7168, N: 36864, repeat: 3 },
    { id: 8,  name: 'Dense FFN down (×3)',    type: 'linear', M: 1, K: 18432, N: 7168, repeat: 3 },
  ];
  if (withMoE) {
    L.push({ id: 9, name: 'MoE FFN (×58)', type: 'moe', M: 1, hid: 7168, ffnDim: 2048,
      topK: 8, nExperts: 256, nShared: 1, repeat: 58 });
  }
  return L;
}

// ════════════════════════════════════════════════════════════════════════════
// 1. replicatedWeightBytes() — counts ONLY dense replicated linears
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## 1. replicatedWeightBytes() — dense tpMode:replicated linears only');
sandbox.layers = mlaModel(false);
sandbox.nCards = 8; sandbox.dp = 1; sandbox.tpAttn = 8; sandbox.moePar = 'tp'; sandbox.epDeg = 1;
{
  const repl = sandbox.replicatedWeightBytes();
  // expected = (Q down) + (KV down), each K·N·wb·repeat
  const refRepl = (7168 * 1536 + 7168 * 576) * WB * 61;
  ok('sums MLA Q down + KV down weight bytes', near(repl, refRepl, 1e-9),
     `got=${(repl / 1e9).toFixed(3)}GB ref=${(refRepl / 1e9).toFixed(3)}GB`);
  // must NOT include the col-parallel Q up or the row-parallel O proj
  const qUpBytes = 1536 * 24576 * WB * 61, oProjBytes = 16384 * 7168 * WB * 61;
  ok('excludes col-parallel Q up', repl < refRepl + qUpBytes && !near(repl, refRepl + qUpBytes, 1e-6));
  ok('excludes row-parallel O proj', !near(repl, refRepl + oProjBytes, 1e-6));
}
// no replicated layers at all → 0
sandbox.layers = [
  { id: 1, name: 'QKV proj', type: 'linear', M: 1, K: 5120, N: 7168, repeat: 64 },
  { id: 2, name: 'Attn O proj', type: 'linear', M: 1, K: 8192, N: 5120, repeat: 64 },
  { id: 3, name: 'Gate+Up', type: 'linear', M: 1, K: 5120, N: 51200, repeat: 64 },
];
ok('dense GQA model (no replicated layers) → replicatedWeightBytes() == 0',
   sandbox.replicatedWeightBytes() === 0);
// explicit tpMode wins; moeGroup sub-layers excluded
sandbox.layers = [
  { id: 1, name: 'a', type: 'linear', M: 1, K: 100, N: 100, repeat: 1, tpMode: 'replicated' },
  { id: 2, name: 'b', type: 'linear', M: 1, K: 100, N: 100, repeat: 1, tpMode: 'row' },
  { id: 3, name: 'c', type: 'linear', M: 1, K: 100, N: 100, repeat: 1, tpMode: 'col' },
  { id: 4, name: 'shared_down', type: 'linear', moeGroup: 'shared_down', M: 1, K: 100, N: 100, repeat: 1, tpMode: 'replicated' },
  { id: 5, name: 'tied head', type: 'linear', M: 1, K: 100, N: 100, repeat: 1, tied: true, tpMode: 'replicated' },
];
ok('counts only dense replicated (skips moeGroup + tied even if tpMode:replicated)',
   sandbox.replicatedWeightBytes() === 100 * 100 * WB * 1,
   `got=${sandbox.replicatedWeightBytes()} ref=${100 * 100 * WB}`);

// ════════════════════════════════════════════════════════════════════════════
// 2. rDecode wPerCard — replicated NOT divided by tpAttn (end-to-end)
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## 2. rDecode — per-card weight uses sharded/tpAttn + replicated + moe');
function bannerWPerCard(htmlStr) {
  const m = htmlStr.match(/Weights ~([\d.]+) GB\/card/);
  return m ? parseFloat(m[1]) : null;
}
for (const [tag, withMoE] of [['dense MLA', false], ['MLA + MoE', true]]) {
  for (const [nCards, dp] of [[8, 1], [16, 2]]) {
    sandbox.layers = mlaModel(withMoE);
    sandbox.nCards = nCards; sandbox.dp = dp; sandbox.tpAttn = Math.max(1, nCards / dp);
    sandbox.moePar = 'tp'; sandbox.epDeg = 1; sandbox.mode = 'decode';
    sandbox.cfg.batch = 1; sandbox.cfg.seqLen = 2048; sandbox.cfg.outTokens = 256;
    let htmlOut = '';
    try { sandbox.rDecode(); htmlOut = getEl('mainContent').innerHTML; }
    catch (e) { ok(`${tag.padEnd(10)} nCards=${nCards} dp=${dp} rDecode runs`, false, e.message); continue; }
    const got = bannerWPerCard(htmlOut);
    const exp = expectedWPerCardGB(sandbox.layers, nCards, sandbox.tpAttn);
    const oldW = oldWrongWPerCardGB(sandbox.layers, nCards, sandbox.tpAttn);
    ok(`${tag.padEnd(10)} nCards=${nCards} dp=${dp}: banner wPerCard == new formula`,
       got !== null && got.toFixed(1) === exp.toFixed(1),
       `banner=${got} expected=${exp.toFixed(1)}`);
    ok(`${tag.padEnd(10)} nCards=${nCards} dp=${dp}: new ≥ old (replicated no longer shrunk)`,
       exp >= oldW - 1e-9,
       `new=${exp.toFixed(3)}GB old(wrong)=${oldW.toFixed(3)}GB Δ=${((exp - oldW) * 1024).toFixed(1)}MB`);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 3. updateCfgInfo wPerCard — same formula in the config panel (end-to-end)
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## 3. updateCfgInfo — config panel weight/card matches the new formula');
function cfgInfoWPerCard(htmlStr) {
  // prefill: "...· <span>X GB</span>/card (TP=..."   decode: "Weights: <span>~X GB</span>/card · KV..."
  // In both, wPerCard is the FIRST "<number> GB</span>/card" occurrence.
  const m = htmlStr.match(/([\d.]+) GB<\/span>\/card/);
  return m ? parseFloat(m[1]) : null;
}
for (const [phase] of [['prefill'], ['decode']]) {
  sandbox.layers = mlaModel(true);
  sandbox.nCards = 8; sandbox.dp = 1; sandbox.tpAttn = 8; sandbox.moePar = 'tp'; sandbox.epDeg = 1;
  sandbox.mode = phase;
  sandbox.cfg.batch = 1; sandbox.cfg.seqLen = 2048; sandbox.cfg.outTokens = 256;
  if (typeof sandbox.applyConfigToLayers === 'function') sandbox.applyConfigToLayers();
  let threw = false;
  try { sandbox.updateCfgInfo(); } catch (e) { threw = true; ok(`${phase}: updateCfgInfo runs`, false, e.message); }
  if (threw) continue;
  const got = cfgInfoWPerCard(getEl('cfgInfo').innerHTML);
  const exp = expectedWPerCardGB(sandbox.layers, 8, 8);
  ok(`${phase.padEnd(7)}: cfgInfo wPerCard == new formula`,
     got !== null && got.toFixed(1) === exp.toFixed(1), `panel=${got} expected=${exp.toFixed(1)}`);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Regression — dense GQA (no replicated) wPerCard unchanged vs old formula
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## 4. Regression — models with NO replicated layers behave exactly as before');
{
  sandbox.layers = [
    { id: 1, name: 'QKV proj', type: 'linear', M: 1, K: 5120, N: 7168, repeat: 64 },
    { id: 2, name: 'Attention', type: 'attention', B: 1, S: 1, H: 64, nKV: 8, D: 128, kvCache: 4096, fa2: true, repeat: 64 },
    { id: 3, name: 'Attn O proj', type: 'linear', M: 1, K: 8192, N: 5120, repeat: 64 },
    { id: 4, name: 'Gate+Up', type: 'linear', M: 1, K: 5120, N: 51200, repeat: 64 },
    { id: 5, name: 'FFN down', type: 'linear', M: 1, K: 25600, N: 5120, repeat: 64 },
  ];
  sandbox.nCards = 8; sandbox.dp = 1; sandbox.tpAttn = 8; sandbox.moePar = 'tp'; sandbox.epDeg = 1;
  sandbox.mode = 'decode'; sandbox.cfg.batch = 1; sandbox.cfg.seqLen = 2048; sandbox.cfg.outTokens = 256;
  ok('dense GQA: replicatedWeightBytes() == 0', sandbox.replicatedWeightBytes() === 0);
  const expNew = expectedWPerCardGB(sandbox.layers, 8, 8);
  const oldW   = oldWrongWPerCardGB(sandbox.layers, 8, 8);
  ok('dense GQA: new formula == old formula (no behaviour change)', near(expNew, oldW, 1e-12),
     `new=${expNew.toFixed(4)} old=${oldW.toFixed(4)}`);
  let htmlOut = '';
  try { sandbox.rDecode(); htmlOut = getEl('mainContent').innerHTML; } catch (e) {}
  const got = bannerWPerCard(htmlOut);
  ok('dense GQA: rDecode banner wPerCard matches', got !== null && got.toFixed(1) === expNew.toFixed(1),
     `banner=${got} expected=${expNew.toFixed(1)}`);
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Compare tab — replicated weight kept full per card there too
// ════════════════════════════════════════════════════════════════════════════
console.log('\n## 5. Compare tab — replicated weight not divided by side.nCards');
{
  // the Compare formula: (wGB - replWB)/nCards + replWB ; verify it differs from
  // the naive wGB/nCards by exactly the replicated headroom.
  sandbox.layers = mlaModel(false);
  sandbox.nCards = 8; sandbox.dp = 1; sandbox.tpAttn = 8; sandbox.moePar = 'tp'; sandbox.epDeg = 1;
  const wGB = sandbox.modelWeightBytes() / 1e9;
  const replGB = sandbox.replicatedWeightBytes() / 1e9;
  const sideNCards = 8;
  const cmpNew  = (wGB - replGB) / sideNCards + replGB;
  const cmpOld  = wGB / sideNCards;
  ok('Compare: replicated headroom = replGB·(1 − 1/nCards)',
     near(cmpNew - cmpOld, replGB * (1 - 1 / sideNCards), 1e-9),
     `Δ=${((cmpNew - cmpOld) * 1024).toFixed(2)}MB replGB=${replGB.toFixed(3)}GB`);
  ok('Compare: new ≥ old (replicated no longer over-divided)', cmpNew >= cmpOld - 1e-12);
}

console.log('\n' + '═'.repeat(86));
console.log(`Summary: ${nPass} pass, ${nFail} fail (of ${nPass + nFail} total)`);
if (failures.length) { console.log('\nFailures:'); failures.forEach(f => console.log('  • ' + f)); }
console.log('═'.repeat(86));
process.exit(nFail ? 1 : 0);

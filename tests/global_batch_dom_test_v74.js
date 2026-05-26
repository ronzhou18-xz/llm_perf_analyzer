// ─────────────────────────────────────────────────────────────────────────────
// DOM behavior tests for Global Batch ↔ Per-rank Batch linkage (v69+).
//
// Tests the UI handlers in isolation by simulating DOM inputs.
// Covers:
//   - onPerRankBatchChange: typing per-rank updates global = per-rank × dp
//   - onGlobalBatchChange: typing global → per-rank = ceil(global / dp), global
//     displays user's typed value, note shows "→ computes as N (ceil)" if not divisible
//   - syncGlobalBatchDisplay: row hidden when dp=1, shown when dp>1
//   - Mode switching keeps both fields in sync
//   - Barrel effect: when global is not divisible by dp, computation uses ceil'd value
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || '/home/claude/llm_perf_analyzer_v69.html';
const html = fs.readFileSync(FILE, 'utf-8');
const scriptMatches = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
let allCode = '';
for (const m of scriptMatches) allCode += m[1] + '\n';
allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');

// ── Build a stateful DOM mock ────────────────────────────────────────────────
function makeFakeElement(id, initialValue='1') {
  return {
    id,
    value: initialValue,
    textContent: '',
    innerHTML: '',
    style: {},
    classList: { add: ()=>{}, remove: ()=>{}, toggle: ()=>{} },
    appendChild: ()=>{},
    addEventListener: ()=>{},
    querySelector: ()=>null,
    querySelectorAll: ()=>[],
    setAttribute: ()=>{},
    checked: false,
    files: [],
  };
}

const inputs = {};
function getOrMake(id) {
  if (!inputs[id]) inputs[id] = makeFakeElement(id);
  return inputs[id];
}

const sandbox = {
  console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN,
  setTimeout: ()=>{}, clearTimeout: ()=>{},
  setInterval: ()=>{}, clearInterval: ()=>{},
  document: {
    getElementById: (id) => getOrMake(id),
    createElement: () => makeFakeElement('new'),
    addEventListener: ()=>{}, querySelector: ()=>null, querySelectorAll: ()=>[],
  },
  window: {}, localStorage: { getItem: ()=>null, setItem: ()=>{}, removeItem: ()=>{} },
  fetch: ()=>Promise.reject(), Chart: function(){return {destroy:()=>{}, update:()=>{}};},
  alert: ()=>{}, confirm: ()=>true, prompt: ()=>'',
  URL: { createObjectURL: ()=>'blob:', revokeObjectURL: ()=>{} }, Blob: function(){},
  FileReader: function(){return {readAsText:()=>{}, readAsArrayBuffer:()=>{}};},
  navigator: { clipboard: { writeText: ()=>Promise.resolve() } },
};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
try { vm.runInContext(allCode, sandbox, { timeout: 10000 }); }
catch (e) { console.log('Load error:', e.message); process.exit(1); }

// Pre-seed inputs that the handlers expect to exist
['cfgBatch','cfgBatchD','cfgSeq','cfgSeqD','cfgOut',
 'cfgGlobalBatch','cfgGlobalBatchD','cfgGlobalBatchField','cfgGlobalBatchFieldD',
 'cfgGlobalBatchNote','cfgGlobalBatchNoteD','cfgBatchLbl','cfgBatchLblD',
 'cfgGlobalBatchLbl','cfgGlobalBatchLblD',
 'cfgInfo','tpS','tpV','ovS','ovV','dpSel','dpHint','epDegSel','moeParHint',
 'archInfo','layerList','mainContent','prefillBtn','decodeBtn','configHdr',
 'prefillFields','decodeFields','modeTag','seqLabel'].forEach(getOrMake);
inputs.cfgSeq.value = '2048'; inputs.cfgSeqD.value = '2048'; inputs.cfgOut.value = '256';

let nPass = 0, nFail = 0;
function expect(label, cond, detail) {
  if (cond) { nPass++; console.log(`  ✓ ${label}`); }
  else      { nFail++; console.log(`  ✗ ${label}  (${detail||''})`); }
}

console.log('═'.repeat(86));
console.log('Global Batch ↔ Per-rank Batch DOM linkage tests');
console.log('═'.repeat(86));

// Bypass renderTab/renderLayerList side-effects by stubbing them out
sandbox.renderTab = () => {};
sandbox.renderLayerList = () => {};
sandbox.renderArchInfo = () => {};
sandbox.applyConfigToLayers = () => {};
sandbox.updateCfgInfo = () => {};

// Test 1: dp=1, global batch field SHOWN (v74+), value = per-rank batch (pure TP)
console.log('\n[Test 1] dp=1: Global batch field shown, value == per-rank batch');
sandbox.dp = 1; sandbox.mode = 'prefill';
sandbox.cfg = {batch:1, seqLen:2048, outTokens:256, nH:64, H:8, D:128, nL:64, precision:'fp16'};
inputs.cfgBatch.value = '1';
sandbox.syncGlobalBatchDisplay();
expect('Global field shown when dp=1 (v74: always visible)',
  inputs.cfgGlobalBatchField.style.display !== 'none',
  `display=${inputs.cfgGlobalBatchField.style.display}`);
expect('Global value == per-rank batch when dp=1 (= 1)',
  parseInt(inputs.cfgGlobalBatch.value) === 1,
  `got=${inputs.cfgGlobalBatch.value}`);
expect('No ceil note when dp=1',
  inputs.cfgGlobalBatchNote.textContent === '',
  `note="${inputs.cfgGlobalBatchNote.textContent}"`);

// Test 2: dp=4, global field shown, value = batch * dp
console.log('\n[Test 2] dp=4, batch=8: Global field shown with value 32');
sandbox.dp = 4;
sandbox.cfg.batch = 8;
inputs.cfgBatch.value = '8';
sandbox.syncGlobalBatchDisplay();
expect('Global field shown when dp>1',
  inputs.cfgGlobalBatchField.style.display !== 'none',
  `display=${inputs.cfgGlobalBatchField.style.display}`);
expect('Global value auto-populates to batch × dp = 32',
  parseInt(inputs.cfgGlobalBatch.value) === 32,
  `got=${inputs.cfgGlobalBatch.value}`);
expect('No "ceil note" when divisible',
  inputs.cfgGlobalBatchNote.textContent === '',
  `note="${inputs.cfgGlobalBatchNote.textContent}"`);

// Test 3: User types per-rank=16, global auto-updates
console.log('\n[Test 3] User edits per-rank from 8 → 16: global updates to 64');
inputs.cfgBatch.value = '16';
sandbox.onPerRankBatchChange();
expect('cfg.batch = 16 after per-rank edit', sandbox.cfg.batch === 16, `batch=${sandbox.cfg.batch}`);
expect('Global field auto-updates to 64',
  parseInt(inputs.cfgGlobalBatch.value) === 64,
  `got=${inputs.cfgGlobalBatch.value}`);
expect('Per-rank mirrored to decode field',
  parseInt(inputs.cfgBatchD.value) === 16, `decode=${inputs.cfgBatchD.value}`);

// Test 4: User types global=32 (divisible by dp=4) → per-rank=8
console.log('\n[Test 4] User edits global 64 → 32 (divisible): per-rank = 8');
inputs.cfgGlobalBatch.value = '32';
sandbox.onGlobalBatchChange();
expect('cfg.batch = 8 after global=32, dp=4', sandbox.cfg.batch === 8, `batch=${sandbox.cfg.batch}`);
expect('Per-rank field shows 8',
  inputs.cfgBatch.value === '8' || parseInt(inputs.cfgBatch.value) === 8,
  `got=${inputs.cfgBatch.value}`);
expect('No "ceil note" when divisible',
  inputs.cfgGlobalBatchNote.textContent === '',
  `note="${inputs.cfgGlobalBatchNote.textContent}"`);

// Test 5: User types global=15 (NOT divisible by dp=4) → per-rank=ceil(15/4)=4
//         Display: global shows 15, per-rank shows 4, note shows "→ computes as 16 (ceil)"
console.log('\n[Test 5] User edits global 32 → 15 (NOT divisible by dp=4): per-rank = ceil(15/4) = 4');
inputs.cfgGlobalBatch.value = '15';
sandbox.onGlobalBatchChange();
expect('cfg.batch = ceil(15/4) = 4', sandbox.cfg.batch === 4, `batch=${sandbox.cfg.batch}`);
expect('Per-rank field shows 4 (ceil)', parseInt(inputs.cfgBatch.value) === 4, `got=${inputs.cfgBatch.value}`);
expect('Global field keeps user-typed value 15',
  parseInt(inputs.cfgGlobalBatch.value) === 15, `got=${inputs.cfgGlobalBatch.value}`);
expect('Ceil note displayed: → computes as 16 (ceil)',
  /computes as 16/.test(inputs.cfgGlobalBatchNote.textContent),
  `note="${inputs.cfgGlobalBatchNote.textContent}"`);

// Test 6: Barrel effect — verify computation uses ceil'd value (per-rank × dp = 16, not 15)
console.log('\n[Test 6] Barrel effect: actual workload uses per-rank × dp = 16 (not 15)');
expect('cfg.batch × dp = 16 (effective global after ceil)',
  sandbox.cfg.batch * sandbox.dp === 16,
  `got=${sandbox.cfg.batch * sandbox.dp}`);

// Test 7: dp change triggers re-display
console.log('\n[Test 7] DP changes to 8: per-rank stays at 4, global displays as 32');
sandbox.dp = 8;
sandbox.syncGlobalBatchDisplay();
expect('Global re-derived from cfg.batch=4 × new dp=8 = 32',
  parseInt(inputs.cfgGlobalBatch.value) === 32,
  `got=${inputs.cfgGlobalBatch.value}`);

// Test 8: per-rank=1 edge case
console.log('\n[Test 8] Edge case: per-rank=1, dp=32 → global=32');
sandbox.dp = 32; sandbox.cfg.batch = 1;
inputs.cfgBatch.value = '1';
sandbox.syncGlobalBatchDisplay();
expect('global=32 when batch=1, dp=32',
  parseInt(inputs.cfgGlobalBatch.value) === 32,
  `got=${inputs.cfgGlobalBatch.value}`);

// Test 9: global=1 edge → per-rank=1
console.log('\n[Test 9] Edge case: user types global=1, dp=32 → per-rank=ceil(1/32)=1');
inputs.cfgGlobalBatch.value = '1';
sandbox.onGlobalBatchChange();
expect('per-rank=ceil(1/32)=1', sandbox.cfg.batch === 1, `batch=${sandbox.cfg.batch}`);

// Test 10: dp returns to 1 → row STAYS visible (v74+), global == per-rank, cfg.batch unchanged
console.log('\n[Test 10] DP back to 1: row stays visible, global == per-rank, cfg.batch preserved');
sandbox.dp = 1; sandbox.cfg.batch = 8;
inputs.cfgBatch.value = '8';
sandbox.syncGlobalBatchDisplay();
expect('Global field still visible when dp=1 (v74: always visible)',
  inputs.cfgGlobalBatchField.style.display !== 'none',
  `display=${inputs.cfgGlobalBatchField.style.display}`);
expect('Global value == per-rank batch (8) when dp=1',
  parseInt(inputs.cfgGlobalBatch.value) === 8,
  `got=${inputs.cfgGlobalBatch.value}`);
expect('cfg.batch preserved at 8', sandbox.cfg.batch === 8, `batch=${sandbox.cfg.batch}`);

// Test 11: dp=1, user edits global field directly → mirrors 1:1 to per-rank
console.log('\n[Test 11] dp=1: user types global=24 → per-rank = 24 (1:1, pure TP)');
inputs.cfgGlobalBatch.value = '24';
sandbox.onGlobalBatchChange();
expect('cfg.batch = 24 after global edit at dp=1', sandbox.cfg.batch === 24, `batch=${sandbox.cfg.batch}`);
expect('Per-rank field shows 24', parseInt(inputs.cfgBatch.value) === 24, `got=${inputs.cfgBatch.value}`);
expect('Global field shows 24 (no ceil at dp=1)',
  parseInt(inputs.cfgGlobalBatch.value) === 24, `got=${inputs.cfgGlobalBatch.value}`);
expect('No ceil note at dp=1', inputs.cfgGlobalBatchNote.textContent === '',
  `note="${inputs.cfgGlobalBatchNote.textContent}"`);

// Test 12: dp=1, user edits per-rank → global mirrors 1:1
console.log('\n[Test 12] dp=1: user types per-rank=12 → global = 12 (1:1, pure TP)');
inputs.cfgBatch.value = '12';
sandbox.onPerRankBatchChange();
expect('cfg.batch = 12 after per-rank edit at dp=1', sandbox.cfg.batch === 12, `batch=${sandbox.cfg.batch}`);
expect('Global field shows 12', parseInt(inputs.cfgGlobalBatch.value) === 12, `got=${inputs.cfgGlobalBatch.value}`);

console.log('\n' + '═'.repeat(86));
console.log(`Summary: ${nPass} pass, ${nFail} fail (of ${nPass+nFail} total)`);
console.log('═'.repeat(86));
process.exit(nFail ? 1 : 0);

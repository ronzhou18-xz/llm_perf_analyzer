// ─────────────────────────────────────────────────────────────────────────────
// Communication-path regression suite (v75)
//
// Guards the MoE all-to-all / AllReduce volume estimates in tpInfo(), which had
// NO test coverage before v75. The v74→v75 fix: a2a/AR volumes are PER-CARD
// (base = Meff/nCards), not global Meff — the old `Meff` base overstated EP/DP
// comm by ×nCards (×16 for DP16-EP16, ×32 for DP32-EP32).
//
// Reference model (vLLM/SGLang DeepEP-style large EP, dp = ep = nCards):
//   Each card owns M local tokens + nE/nCards experts.
//   Dispatch: card sends its own M·topK routings out, (nCards-1)/nCards leave.
//   Combine : symmetric return.
//   Per-card a2a (dispatch+combine) = 2·M·topK·hid·ab·(nCards-1)/nCards
//   ── base scales with M (per-rank tokens) = Meff/nCards, NOT global Meff.
//
// Hybrid (EP=epDeg × expert_TP=expertTp): a2a runs over epDeg groups; an extra
// intra-group ring-AllReduce rides on the down projection. We check the a2a
// component against the per-card reference and allow the AR term as headroom.
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || '/home/claude/llm_perf_analyzer_v75.html';
const html = fs.readFileSync(FILE, 'utf-8');
let allCode = '';
for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) allCode += m[1] + '\n';
allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');

const sandbox = {
  console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN,
  setTimeout: () => {}, clearTimeout: () => {}, setInterval: () => {}, clearInterval: () => {},
  document: { getElementById: () => ({ value:'0',innerHTML:'',textContent:'',className:'',
      appendChild:()=>{},addEventListener:()=>{},querySelector:()=>null,querySelectorAll:()=>[],
      style:{},classList:{add:()=>{},remove:()=>{}},checked:false,files:[] }),
    createElement: () => ({ appendChild:()=>{},addEventListener:()=>{},style:{},
      classList:{add:()=>{},remove:()=>{}},setAttribute:()=>{},textContent:'',innerHTML:'',value:'0' }),
    addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [] },
  window: {}, localStorage: { getItem:()=>null,setItem:()=>{},removeItem:()=>{} },
  fetch: () => Promise.reject(new Error('mock')),
  Chart: function(){ return {destroy:()=>{},update:()=>{}}; },
  alert: ()=>{}, confirm: ()=>true, prompt: ()=>'',
  URL: { createObjectURL:()=>'blob:',revokeObjectURL:()=>{} },
  Blob: function(){}, FileReader: function(){ return {readAsText:()=>{},readAsArrayBuffer:()=>{}}; },
  navigator: { clipboard: { writeText:()=>Promise.resolve() } },
};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
try { vm.runInContext(allCode, sandbox, { timeout: 10000, filename: 'tool.js' }); }
catch (e) { console.log('Load error:', e.message); process.exit(1); }

// ── Models (hid, moe_ffn, nE, topK, nShared) ────────────────────────────────
const MODELS = {
  'GLM-5.1':     { hid: 6144, ffn: 2048, nE: 256, topK: 8, nShared: 1 },
  'DeepSeek-V3': { hid: 7168, ffn: 2048, nE: 256, topK: 8, nShared: 1 },
  'Qwen3-235B':  { hid: 4096, ffn: 1536, nE: 128, topK: 8, nShared: 0 },
};

const wb = 2, ab = 2, IBW = 450e9;

function setup(nCards, dp, epDeg, moePar) {
  sandbox.nCards = nCards;
  sandbox.dp = dp;
  sandbox.tpAttn = Math.max(1, nCards / dp);
  sandbox.epDeg = epDeg;
  sandbox.moePar = moePar;
  sandbox.overlap = 0;                       // isolate raw commTime
  sandbox.ibw = 450;
  sandbox.selChip = sandbox.chips.find(c => c.name === 'H100 SXM') || sandbox.chips[0];
  sandbox.cfg.precision = 'fp16';
  sandbox.cfg.batch = 1;
}

// Per-card a2a reference (dispatch+combine) for pure EP / DP+EP.
function refEPa2a(m, M, nCards) {
  return 2 * M * m.topK * m.hid * ab * (nCards - 1) / nCards;
}
// Per-card a2a reference for hybrid (a2a runs over epDeg groups only).
function refHybridA2a(m, M, dp, nCards, epDeg) {
  const Meff = M * dp;
  return 2 * (Meff / nCards) * m.topK * m.hid * ab * (epDeg - 1) / epDeg;
}

let nPass = 0, nFail = 0;
const failures = [];
function check(label, got, want, tol, extraNote) {
  const err = want === 0 ? Math.abs(got) : Math.abs(got - want) / want;
  const ok = err <= tol;
  if (ok) { nPass++; } else { nFail++; failures.push(`${label}: got=${got.toExponential(3)} want=${want.toExponential(3)} err=${(err*100).toFixed(1)}%`); }
  console.log(`  ${ok ? '✓' : '✗'} ${label.padEnd(54)} err=${(err*100).toFixed(2)}%${extraNote ? '  ' + extraNote : ''}`);
  return ok;
}

console.log('═'.repeat(86));
console.log('Communication-path regression suite (v75)');
console.log(`File: ${FILE}`);
console.log('H100 SXM · NVLink 450 GB/s uni · overlap=0 · fp16');
console.log('═'.repeat(86));

// ── 1. Pure EP / DP+EP: monolithic + expanded sub-layers ────────────────────
const EP_CASES = [
  { nCards:8,  dp:8,  label:'DP8-EP8'   },
  { nCards:16, dp:16, label:'DP16-EP16' },
  { nCards:32, dp:32, label:'DP32-EP32' },
  { nCards:16, dp:1,  label:'TP1-EP16 (dp=1)' },
];
const PHASES = [ ['decode',1], ['prefill',4096] ];

for (const mName of Object.keys(MODELS)) {
  const m = MODELS[mName];
  console.log(`\n## ${mName} — pure EP a2a (per-card == DeepEP reference)`);
  for (const c of EP_CASES) {
    for (const [phase, M] of PHASES) {
      setup(c.nCards, c.dp, c.nCards, 'ep');
      // monolithic
      const mono = sandbox.tpInfo(
        { id:1, name:'MoE', type:'moe', M, hid:m.hid, ffnDim:m.ffn,
          topK:m.topK, nExperts:m.nE, nShared:m.nShared, repeat:1 }, sandbox.selChip);
      // expanded gate_up (dispatch) + down (combine)
      const gu = sandbox.tpInfo(
        { id:2, name:'gate_up', type:'linear', moeGroup:'gate_up', M, K:m.hid, N:2*m.ffn,
          moeExperts:m.nE, moeTopK:m.topK, moeFfnDim:m.ffn, moeHid:m.hid, repeat:1 }, sandbox.selChip);
      const dn = sandbox.tpInfo(
        { id:3, name:'down', type:'linear', moeGroup:'down', M, K:m.ffn, N:m.hid,
          moeExperts:m.nE, moeTopK:m.topK, moeFfnDim:m.ffn, moeHid:m.hid, repeat:1 }, sandbox.selChip);
      // reference uses per-card token count M (= Meff/nCards since dp∈{1,nCards})
      const Mpc = (c.dp * M) / c.nCards;
      const refBytes = refEPa2a(m, Mpc, c.nCards);
      const refT = refBytes / IBW;
      check(`${c.label.padEnd(17)} ${phase.padEnd(7)} monolithic`, mono.commTime, refT, 0.001);
      check(`${c.label.padEnd(17)} ${phase.padEnd(7)} gate_up+down`, gu.commTime + dn.commTime, refT, 0.001);
    }
  }
}

// ── 2. Hybrid: a2a component must match per-card reference (AR = headroom) ───
const HY_CASES = [
  { nCards:32, dp:8,  epDeg:4,  label:'DP8 ·EP4×eTp8'  },
  { nCards:16, dp:4,  epDeg:4,  label:'DP4 ·EP4×eTp4'  },
  { nCards:32, dp:1,  epDeg:8,  label:'TP4×EP8 (dp=1)' },
];
for (const mName of Object.keys(MODELS)) {
  const m = MODELS[mName];
  console.log(`\n## ${mName} — hybrid: a2a ≥ ref, total = a2a + intra-group AR (bounded)`);
  for (const c of HY_CASES) {
    for (const [phase, M] of PHASES) {
      setup(c.nCards, c.dp, c.epDeg, 'hybrid');
      const expertTp = c.nCards / c.epDeg;
      const mono = sandbox.tpInfo(
        { id:1, name:'MoE', type:'moe', M, hid:m.hid, ffnDim:m.ffn,
          topK:m.topK, nExperts:m.nE, nShared:m.nShared, repeat:1 }, sandbox.selChip);
      const refA2aT = refHybridA2a(m, M, c.dp, c.nCards, c.epDeg) / IBW;
      // expected AR ceiling (per-card, token-granularity): 2·(eTp-1)/eTp·(Meff/nCards)·hid·ab
      const Meff = M * c.dp;
      const refArT = expertTp > 1
        ? (2 * ((expertTp-1)/expertTp) * (Meff/c.nCards) * m.hid * ab) / IBW : 0;
      const refTotal = refA2aT + refArT;
      // total comm should equal a2a + AR within 1%
      check(`${c.label.padEnd(17)} ${phase.padEnd(7)} a2a+AR total`, mono.commTime, refTotal, 0.01,
            `(a2a ref ${(refA2aT*1e6).toFixed(2)}µs + AR ref ${(refArT*1e6).toFixed(2)}µs)`);
      // and must NOT be the old ×nCards-inflated value
      const oldInflated = 2 * Meff * m.topK * m.hid * ab * (c.epDeg-1)/c.epDeg / IBW;
      check(`${c.label.padEnd(17)} ${phase.padEnd(7)} not ×nCards inflated`,
            mono.commTime < oldInflated * 0.5 ? 0 : 1, 0, 0);
    }
  }
}

// ── 3. Scale-invariance: per-card comm must be ~flat across DP=EP=8/16/32 ────
// (global work grows with nCards, but per-card sourced tokens stay = M).
console.log('\n## Scale-invariance — per-card MoE comm flat as DP=EP scales 8→16→32');
for (const mName of Object.keys(MODELS)) {
  const m = MODELS[mName];
  const ts = [8,16,32].map(n => {
    setup(n, n, n, 'ep');
    return sandbox.tpInfo({ id:1, name:'MoE', type:'moe', M:1, hid:m.hid, ffnDim:m.ffn,
      topK:m.topK, nExperts:m.nE, nShared:m.nShared, repeat:1 }, sandbox.selChip).commTime;
  });
  // ts should be within a few % of each other (only the (N-1)/N ring factor drifts)
  const spread = (Math.max(...ts) - Math.min(...ts)) / Math.min(...ts);
  check(`${mName.padEnd(14)} decode per-card comm spread 8/16/32`, spread, 0, 0.20,
        `[${ts.map(t=>(t*1e6).toFixed(2)+'µs').join(', ')}]`);
}

console.log('\n' + '═'.repeat(86));
console.log(`Summary: ${nPass} pass, ${nFail} fail (of ${nPass+nFail} total)`);
if (failures.length) { console.log('\nFailures:'); failures.forEach(f => console.log('  • ' + f)); }
console.log('═'.repeat(86));
process.exit(nFail ? 1 : 0);

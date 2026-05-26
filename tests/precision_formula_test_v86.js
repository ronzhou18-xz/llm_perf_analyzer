// ─────────────────────────────────────────────────────────────────────────────
// v86 precision-formula consistency test
//
// Bug fixed in v86: in the Bottleneck table, switching precision (W16A16→W8A8)
// changed the numeric result of a cell but NOT the formula text underneath it,
// because bnDerive()'s attention/MoE byExpr branches printed pure symbols
// (`·ab`, `·kvb`, "Σ expert weight stream …") and the MoE sub-layer elided the
// activation term with `…`, instead of substituting live wBpe()/aBpe()/kvBpe().
//
// This suite asserts, for every layer type, the invariant:
//   if the cell RESULT changes between two precisions, the formula TEXT must
//   change too — and conversely a precision-independent result keeps stable text.
//
// Usage:  node precision_formula_test_v86.js
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

function load(file) {
  const html = fs.readFileSync(file, 'utf-8');
  let code = '';
  for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) code += m[1] + '\n';
  code = code.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');
  const el = (id) => {
    const e = { value:'0', innerHTML:'', textContent:'', className:'', checked:false,
      disabled:false, files:[], style:{}, dataset:{}, options:[],
      appendChild:()=>{}, addEventListener:()=>{}, removeChild:()=>{},
      querySelector:()=>null, querySelectorAll:()=>[],
      classList:{add:()=>{},remove:()=>{},toggle:()=>{},contains:()=>false},
      setAttribute:()=>{}, getAttribute:()=>null, getContext:()=>({}) };
    e.parentNode = { parentNode:{appendChild:()=>{}}, appendChild:()=>{} };
    return e;
  };
  const sb = {
    console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN, isFinite,
    setTimeout:()=>{}, clearTimeout:()=>{}, setInterval:()=>{}, clearInterval:()=>{}, requestAnimationFrame:()=>{},
    document:{ getElementById:()=>el(), createElement:()=>el(), addEventListener:()=>{},
      querySelector:()=>null, querySelectorAll:()=>[] },
    window:{}, localStorage:{getItem:()=>null,setItem:()=>{},removeItem:()=>{}},
    fetch:()=>Promise.reject(new Error('mock')),
    Chart:function(){return{destroy:()=>{},update:()=>{}};},
    alert:()=>{}, confirm:()=>true, prompt:()=>'',
    URL:{createObjectURL:()=>'blob:',revokeObjectURL:()=>{}},
    Blob:function(){}, FileReader:function(){return{};},
    navigator:{clipboard:{writeText:()=>Promise.resolve()}},
  };
  sb.window = sb; sb.globalThis = sb;
  vm.createContext(sb);
  vm.runInContext(code, sb, { timeout:10000, filename:'tool.js' });
  Object.assign(sb, { nCards:8, dp:1, tpAttn:8, epDeg:1, moePar:'tp',
    mode:'prefill', overlap:0, ibw:450 });
  sb.selChip = sb.chips.find(c => c.name === 'H200 SXM') || sb.chips[0];
  return sb;
}

const V = load('/home/claude/llm_perf_analyzer_v86.html');
const strip = (h) => h.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();

let nPass = 0, nFail = 0; const failures = [];
function ok(label, cond, note) {
  if (cond) nPass++; else { nFail++; failures.push(label + (note ? ' — ' + note : '')); }
  console.log(`  ${cond ? '\u2713' : '\u2717'} ${label}${note ? '  ' + note : ''}`);
}

console.log('='.repeat(92));
console.log('v86 — Bottleneck per-cell formula tracks precision (W16A16 / W8A8 / W8A8+KV8)');
console.log('='.repeat(92));

// Each entry: a layer + which cells SHOULD react to a given precision switch.
const cases = [
  { tag:'linear (dense)',
    layer:{id:1,name:'FFN down',type:'linear',M:2048,K:25600,N:5120,repeat:32,tpMode:'row'} },
  { tag:'linear (embedding)',
    layer:{id:2,name:'Token embed',type:'embedding',M:2048,K:151936,N:5120,repeat:1} },
  { tag:'attention (FA2 MHA)',
    layer:{id:3,name:'Attention',type:'attention',B:1,S:4096,H:32,nKV:8,D:128,kvCache:0,fa2:true,repeat:32} },
  { tag:'attention (no FA2)',
    layer:{id:4,name:'Attention',type:'attention',B:1,S:4096,H:32,nKV:8,D:128,kvCache:0,fa2:false,repeat:32} },
  { tag:'linear_attn (δ-net)',
    layer:{id:5,name:'GatedDeltaNet',type:'linear_attn',M:2048,nQK:16,nV:32,D:128,K:6144,repeat:24} },
  { tag:'moe (monolithic)',
    layer:{id:6,name:'MoE FFN',type:'moe',M:2048,hid:7168,ffnDim:2048,topK:8,nExperts:256,nShared:1,repeat:58} },
  { tag:'moe sub gate_up',
    layer:{id:7,name:'MoE gate_up',type:'linear',moeGroup:'gate_up',moeTopK:8,moeExperts:256,M:2048,K:7168,N:4096,repeat:58} },
  { tag:'moe sub down',
    layer:{id:8,name:'MoE down',type:'linear',moeGroup:'down',moeTopK:8,moeExperts:256,M:2048,K:2048,N:7168,repeat:58} },
];

function snapshot(l, prec) {
  V.cfg.precision = prec;
  const base = V.calcL(l);
  const ti   = V.tpInfo(l, V.selChip);
  const der  = V.bnDerive(l, V.selChip, ti, base);
  return {
    bytes: base.bytes, flops: base.flops,
    computeT: ti.computeTime, memT: ti.memTime,
    flText: strip(der.flExpr), byText: strip(der.byExpr),
    ctText: strip(der.ctExpr), mtText: strip(der.mtExpr),
  };
}

// ── core invariant: result changes  <=>  formula text changes ────────────────
function checkPair(tag, l, pA, pB) {
  const a = snapshot(l, pA), b = snapshot(l, pB);
  const pairLbl = `${tag} [${pA}→${pB}]`;
  // bytes  <-> byText
  const byResChg = Math.abs(a.bytes - b.bytes) > 1e-6;
  const byTxtChg = a.byText !== b.byText;
  ok(`${pairLbl} bytes: result/formula move together`,
     byResChg === byTxtChg,
     byResChg ? (byTxtChg ? 'both changed' : 'RESULT CHANGED, FORMULA FROZEN')
              : (byTxtChg ? 'formula changed but result stable' : 'both stable'));
  // memTime <-> mtText  (mem time follows bytesPerCard)
  const mtResChg = Math.abs(a.memT - b.memT) / Math.max(b.memT,1e-30) > 1e-9;
  const mtTxtChg = a.mtText !== b.mtText;
  ok(`${pairLbl} memTime: result/formula move together`,
     mtResChg === mtTxtChg,
     mtResChg ? (mtTxtChg ? 'both changed' : 'RESULT CHANGED, FORMULA FROZEN')
              : 'both stable');
  // computeTime <-> ctText  (compute time follows effective TFLOPS)
  const ctResChg = Math.abs(a.computeT - b.computeT) / Math.max(b.computeT,1e-30) > 1e-9;
  const ctTxtChg = a.ctText !== b.ctText;
  ok(`${pairLbl} computeTime: result/formula move together`,
     ctResChg === ctTxtChg,
     ctResChg ? (ctTxtChg ? 'both changed' : 'RESULT CHANGED, FORMULA FROZEN')
              : 'both stable');
}

console.log('\n## Invariant: a changed cell result must come with a changed formula text');
for (const c of cases) {
  console.log(`\n[${c.tag}]`);
  checkPair(c.tag, c.layer, 'fp16', 'fp8');     // W16A16 → W8A8
  checkPair(c.tag, c.layer, 'fp8',  'fp8kv');   // W8A8   → W8A8+KV8
}

// ── explicit regressions for the exact bug ───────────────────────────────────
console.log('\n## Targeted regression — formerly-frozen formula strings');
{
  // MoE monolithic: weight bytes halve under W8A8 → byText must show wb=1
  const moe = cases.find(c => c.tag === 'moe (monolithic)').layer;
  const m16 = snapshot(moe, 'fp16'), m8 = snapshot(moe, 'fp8');
  ok('MoE byExpr is no longer generic (contains substituted numbers)',
     /\d/.test(m8.byText) && !/Σ expert weight stream/.test(m8.byText));
  ok('MoE byExpr text differs fp16 vs fp8', m16.byText !== m8.byText);
  ok('MoE bytes actually drop under W8A8', m8.bytes < m16.bytes,
     `${(m16.bytes/1e6).toFixed(0)}MB → ${(m8.bytes/1e6).toFixed(0)}MB`);

  // Attention no-FA2: KV bytes depend on kvb → frozen only across fp8→fp8kv
  const attn = cases.find(c => c.tag === 'attention (no FA2)').layer;
  const a8 = snapshot(attn, 'fp8'), akv = snapshot(attn, 'fp8kv');
  ok('Attention byExpr shows numeric KV term (not bare ·kvb symbol)',
     /\d·\d/.test(a8.byText.replace(/ /g,'')) || /·1\b|·2\b/.test(a8.byText));
  ok('Attention byExpr text differs fp8 vs fp8kv (kvb 2→1)', a8.byText !== akv.byText);

  // MoE sub gate_up: activation term must be spelled out, no ellipsis
  const gu = cases.find(c => c.tag === 'moe sub gate_up').layer;
  const g8 = snapshot(gu, 'fp8');
  ok('MoE sub gate_up byExpr has no elided "…" activation term',
     !g8.byText.includes('…'));

  // FLOPs are precision-independent — flText must stay stable (sanity)
  for (const c of cases) {
    const f16 = snapshot(c.layer, 'fp16'), f8 = snapshot(c.layer, 'fp8');
    ok(`${c.tag}: FLOPs formula stable across precision (FLOPs are precision-free)`,
       f16.flText === f8.flText && Math.abs(f16.flops - f8.flops) < 1e-3);
  }
}

console.log('\n' + '='.repeat(92));
console.log(`RESULT: ${nPass} passed, ${nFail} failed`);
if (nFail) { console.log('\nFailures:'); failures.forEach(f => console.log('  - ' + f)); }
console.log('='.repeat(92));
process.exit(nFail ? 1 : 0);

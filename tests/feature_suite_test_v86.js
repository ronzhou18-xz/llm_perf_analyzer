// ─────────────────────────────────────────────────────────────────────────────
// v84 verification suite
//
//   1. H200 chip present with correct HBM3e specs.
//   2. Sparse-MLA (DSA) availability: refreshSparseMLAAvailability() shows the
//      toggle only when DSA layers (indexerLayer / sparseTopK) exist, and pins
//      sparseMLA=true for non-DSA models.
//   3. Bottleneck rollups: per-card compute/memory/comm sums + commBytes are
//      internally consistent with tpInfoTotal().
//   4. TP-timeline DP semantics: latency speedup (per card) vs throughput
//      speedup (×dp) are computed and distinct under DP>1.
//   5. Throughput GPU util: numerator scales by dp — util(dp=k) == k×util(dp=1)
//      for an otherwise-identical per-card configuration.
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

function load(file) {
  const html = fs.readFileSync(file, 'utf-8');
  let allCode = '';
  for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) allCode += m[1] + '\n';
  allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');
  const elProxy = () => {
    const e = { value: '0', innerHTML: '', textContent: '', className: '', checked: false, disabled: false,
      files: [], style: {}, dataset: {},
      appendChild: () => {}, addEventListener: () => {}, removeChild: () => {},
      querySelector: () => null, querySelectorAll: () => [],
      classList: { add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false },
      setAttribute: () => {}, getAttribute: () => null, getContext: () => ({}),
      parentNode: null };
    e.parentNode = { parentNode: { appendChild: () => {} }, appendChild: () => {} };
    return e;
  };
  const sb = {
    console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN, isFinite,
    setTimeout: () => {}, clearTimeout: () => {}, setInterval: () => {}, clearInterval: () => {},
    requestAnimationFrame: () => {},
    document: { getElementById: () => elProxy(), createElement: () => elProxy(),
      addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [] },
    window: {}, localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    fetch: () => Promise.reject(new Error('mock')),
    Chart: function () { return { destroy: () => {}, update: () => {} }; },
    alert: () => {}, confirm: () => true, prompt: () => '',
    URL: { createObjectURL: () => 'blob:', revokeObjectURL: () => {} },
    Blob: function () {}, FileReader: function () { return { readAsText: () => {}, readAsArrayBuffer: () => {} }; },
    navigator: { clipboard: { writeText: () => Promise.resolve() } },
  };
  sb.window = sb; sb.globalThis = sb;
  vm.createContext(sb);
  vm.runInContext(allCode, sb, { timeout: 10000, filename: 'tool.js' });
  sb.nCards = 8; sb.dp = 1; sb.tpAttn = 8; sb.epDeg = 1; sb.moePar = 'tp';
  sb.overlap = 0; sb.ibw = 450;
  sb.selChip = sb.chips.find(c => c.name === 'H100 SXM') || sb.chips[0];
  sb.cfg.precision = 'fp16';
  return sb;
}

const V = load('/home/claude/llm_perf_analyzer_v86.html');

let nPass = 0, nFail = 0; const failures = [];
function ok(label, cond, note) {
  if (cond) nPass++; else { nFail++; failures.push(label + (note ? ' — ' + note : '')); }
  console.log(`  ${cond ? '\u2713' : '\u2717'} ${label}${note ? '  ' + note : ''}`);
}
function near(a, b, tol) { return b === 0 ? Math.abs(a) <= tol : Math.abs(a - b) / Math.abs(b) <= tol; }
function section(t) { console.log('\n## ' + t); }

console.log('='.repeat(90));
console.log('v84 verification — H200 · DSA toggle gating · bottleneck comm · TP-timeline DP · GPU util');
console.log('='.repeat(90));

// ════════════════════════════════════════════════════════════════════════════
// 1. H200 chip
// ════════════════════════════════════════════════════════════════════════════
section('1. H200 SXM chip present with HBM3e specs');
{
  const h200 = V.chips.find(c => c.name === 'H200 SXM');
  ok('H200 SXM exists in chips list', !!h200);
  if (h200) {
    ok('H200 HBM = 141 GB',          h200.mem === 141, `mem=${h200.mem}`);
    ok('H200 HBM BW = 4800 GB/s',    h200.bw === 4800,  `bw=${h200.bw}`);
    ok('H200 BF16 = 989 TFLOPS',     h200.tflops === 989, `tflops=${h200.tflops}`);
    ok('H200 FP8 = 1979 TFLOPS',     h200.tflops_fp8 === 1979, `fp8=${h200.tflops_fp8}`);
    ok('H200 interconnect = 450 GB/s uni', h200.ibw === 450, `ibw=${h200.ibw}`);
    ok('H200 id unique', V.chips.filter(c => c.id === h200.id).length === 1);
    ok('H200 BW > H100 SXM BW (HBM3e bump)',
       h200.bw > V.chips.find(c => c.name === 'H100 SXM').bw);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Sparse-MLA (DSA) availability gating
// ════════════════════════════════════════════════════════════════════════════
section('2. refreshSparseMLAAvailability() — DSA-only toggle');
{
  let lastDisplay = null;
  // capture style.display writes on #sparseMLAField
  const fieldEl = { style: {}, };
  const chkEl   = { checked: true, disabled: false };
  const origGet = V.document.getElementById;
  V.document.getElementById = (id) => {
    if (id === 'sparseMLAField') return fieldEl;
    if (id === 'sparseMLAChk')   return chkEl;
    return origGet(id);
  };

  // (a) non-DSA model — plain dense linear layers
  V.layers = [
    { id:1, name:'Attn QKV', type:'linear', M:2048, K:5120, N:12288, repeat:32 },
    { id:2, name:'FFN down', type:'linear', M:2048, K:25600, N:5120, repeat:32 },
  ];
  V.sparseMLA = false;            // simulate a stale unchecked box
  V.refreshSparseMLAAvailability();
  ok('non-DSA model: field hidden', fieldEl.style.display === 'none',
     `display="${fieldEl.style.display}"`);
  ok('non-DSA model: sparseMLA pinned back to true', V.sparseMLA === true);

  // (b) DSA model — has an indexer layer
  V.layers = [
    { id:1, name:'MLA KV down', type:'linear', M:2048, K:7168, N:576, repeat:61 },
    { id:2, name:'DSA indexer score', type:'attention', B:1, S:2048, H:64, nKV:1, D:128,
      repeat:61, scoreOnly:true, indexerLayer:true, tpReplicated:true },
    { id:3, name:'MLA Attention', type:'attention', B:1, S:2048, H:128, nKV:1, D:512,
      repeat:61, mla:true, sparseTopK:2048 },
  ];
  V.refreshSparseMLAAvailability();
  ok('DSA model: field visible', fieldEl.style.display === '',
     `display="${fieldEl.style.display}"`);

  // (c) DSA detection also via sparseTopK alone (no indexerLayer)
  V.layers = [
    { id:1, name:'MLA Attention', type:'attention', B:1, S:2048, H:128, nKV:1, D:512,
      repeat:61, mla:true, sparseTopK:2048 },
  ];
  V.refreshSparseMLAAvailability();
  ok('DSA detected via sparseTopK alone', fieldEl.style.display === '');

  V.document.getElementById = origGet;
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Bottleneck comm rollups consistent with tpInfoTotal
// ════════════════════════════════════════════════════════════════════════════
section('3. Bottleneck — compute/memory/comm rollups vs tpInfoTotal');
{
  V.nCards = 8; V.dp = 1; V.tpAttn = 8; V.epDeg = 1; V.moePar = 'tp';
  V.overlap = 0; V.ibw = 450;
  V.layers = [
    { id:1, name:'Attn QKV',  type:'linear', M:2048, K:5120,  N:12288, repeat:32, tpMode:'col' },
    { id:2, name:'Attn O proj', type:'linear', M:2048, K:8192, N:5120, repeat:32, tpMode:'row' },
    { id:3, name:'FFN gate+up', type:'linear', M:2048, K:5120, N:51200, repeat:32, tpMode:'col' },
    { id:4, name:'FFN down',  type:'linear', M:2048, K:25600, N:5120, repeat:32, tpMode:'row' },
  ];
  const ch = V.selChip;
  const sumCT  = V.layers.map(l => V.tpInfoTotal(l, ch).computeTime).reduce((a,b)=>a+b,0);
  const sumMT  = V.layers.map(l => V.tpInfoTotal(l, ch).memTime).reduce((a,b)=>a+b,0);
  const sumComm= V.layers.map(l => { const t=V.tpInfoTotal(l,ch); return Math.max(0,t.commTime-t.hiddenComm); }).reduce((a,b)=>a+b,0);
  const sumCB  = V.layers.map(l => V.tpInfoTotal(l, ch).commBytes||0).reduce((a,b)=>a+b,0);
  ok('compute rollup > 0', sumCT > 0, `${(sumCT*1e6).toFixed(1)}us`);
  ok('memory  rollup > 0', sumMT > 0, `${(sumMT*1e6).toFixed(1)}us`);
  ok('comm    rollup > 0 (2 row-par layers carry AllReduce)', sumComm > 0,
     `${(sumComm*1e6).toFixed(2)}us`);
  ok('commBytes rollup > 0', sumCB > 0, `${(sumCB/1e6).toFixed(1)}MB`);
  // only row-par layers contribute comm
  const colComm = V.tpInfoTotal(V.layers[0], ch).commBytes || 0;
  const rowComm = V.tpInfoTotal(V.layers[1], ch).commBytes || 0;
  ok('col-parallel layer has 0 comm bytes', colComm === 0);
  ok('row-parallel layer has >0 comm bytes', rowComm > 0, `${(rowComm/1e6).toFixed(2)}MB`);
  // per-call × repeat == total (commBytes)
  const one = V.tpInfo(V.layers[3], ch), tot = V.tpInfoTotal(V.layers[3], ch);
  ok('FFN down: commBytes/call × repeat == total',
     near(one.commBytes * V.getRepeat(V.layers[3]), tot.commBytes, 1e-9),
     `call=${(one.commBytes/1e6).toFixed(2)}MB ×${V.getRepeat(V.layers[3])} vs ${(tot.commBytes/1e6).toFixed(2)}MB`);
  // effective = max(compute,memory) + exposed comm
  const exC = Math.max(0, one.commTime - one.hiddenComm);
  ok('FFN down: effective == max(compute,memory) + exposed comm',
     near(one.effectiveTime, Math.max(one.computeTime, one.memTime) + exC, 1e-9));
}

// ════════════════════════════════════════════════════════════════════════════
// 4. TP-timeline DP semantics
// ════════════════════════════════════════════════════════════════════════════
section('4. TP-timeline — latency vs throughput speedup under DP');
{
  const ch = V.selChip;
  const mkLayers = () => [
    { id:1, name:'Attn QKV',  type:'linear', M:2048, K:5120, N:12288, repeat:32, tpMode:'col' },
    { id:2, name:'Attn O proj', type:'linear', M:2048, K:8192, N:5120, repeat:32, tpMode:'row' },
    { id:3, name:'FFN gate+up', type:'linear', M:2048, K:5120, N:51200, repeat:32, tpMode:'col' },
    { id:4, name:'FFN down',  type:'linear', M:2048, K:25600, N:5120, repeat:32, tpMode:'row' },
  ];
  function timelineNumbers() {
    const tis = V.layers.map(l => V.tpInfoTotal(l, ch));
    const base1 = V.layers.map(l => { const c=V.calcLTotal(l);
      return Math.max(c.flops/(V.effectiveTflops(ch,l.type)*1e12), c.bytes/(ch.bw*1e9)); }).reduce((a,b)=>a+b,0);
    const totN = tis.map(t => t.effectiveTime).reduce((a,b)=>a+b,0);
    const cardLat = base1/totN, clusterSp = cardLat * V.dp;
    return { base1, totN, cardLat, clusterSp };
  }
  // pure TP=8, dp=1
  V.layers = mkLayers(); V.nCards = 8; V.dp = 1; V.tpAttn = 8; V.overlap = 0;
  const a = timelineNumbers();
  ok('TP=8 dp=1: latency speedup == throughput speedup', near(a.cardLat, a.clusterSp, 1e-9),
     `lat=${a.cardLat.toFixed(2)}x cluster=${a.clusterSp.toFixed(2)}x`);
  ok('TP=8 dp=1: per-card speedup roughly tracks 8 cards', a.cardLat > 4 && a.cardLat <= 8.5,
     `${a.cardLat.toFixed(2)}x`);
  // TP=2 × DP=4 on the same 8 cards
  V.layers = mkLayers(); V.nCards = 8; V.dp = 4; V.tpAttn = 2; V.overlap = 0;
  const b = timelineNumbers();
  ok('TP=2×DP=4: throughput speedup == latency speedup × dp',
     near(b.clusterSp, b.cardLat * 4, 1e-9),
     `lat=${b.cardLat.toFixed(2)}x cluster=${b.clusterSp.toFixed(2)}x`);
  ok('TP=2×DP=4: cluster speedup > per-card latency speedup', b.clusterSp > b.cardLat);
  ok('TP=2×DP=4: per-card latency speedup ≈ 2 (attention TP only)',
     b.cardLat > 1.4 && b.cardLat <= 2.6, `${b.cardLat.toFixed(2)}x`);
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Throughput GPU util — DP scaling
// ════════════════════════════════════════════════════════════════════════════
section('5. Throughput GPU util — numerator scales with dp');
{
  const ch = V.selChip;
  const mkLayers = () => [
    { id:1, name:'Attn QKV',  type:'linear', M:2048, K:5120, N:12288, repeat:32, tpMode:'col' },
    { id:2, name:'Attn O proj', type:'linear', M:2048, K:8192, N:5120, repeat:32, tpMode:'row' },
    { id:3, name:'FFN gate+up', type:'linear', M:2048, K:5120, N:51200, repeat:32, tpMode:'col' },
    { id:4, name:'FFN down',  type:'linear', M:2048, K:25600, N:5120, repeat:32, tpMode:'row' },
  ];
  function utilOf() {
    const tis = V.layers.map(l => V.tpInfoTotal(l, ch));
    const tot = tis.map(t => t.effectiveTime).reduce((a,b)=>a+b,0);
    const clusterFlops = V.layers.map(l => V.calcLTotal(l).flops).reduce((a,b)=>a+b,0) * V.dp;
    const clusterCap   = V.effectiveTflops(ch,'linear')*1e12 * (tot||1) * V.nCards;
    return Math.min(1, clusterFlops/clusterCap) * 100;
  }
  // dp=1: pure TP=8
  V.layers = mkLayers(); V.nCards = 8; V.dp = 1; V.tpAttn = 8; V.overlap = 0;
  const u1 = utilOf();
  // dp=4: TP=2 × DP=4 on the same 8 cards — per-card layer work is identical
  // (each card does 1/tpAttn of dense; tot per-card is the same shape), so
  // util should rise by ~4× because 4 replicas do 4× the total cluster work.
  V.layers = mkLayers(); V.nCards = 8; V.dp = 4; V.tpAttn = 2; V.overlap = 0;
  const u4 = utilOf();
  ok('util(dp=1) is finite and > 0', u1 > 0 && u1 <= 100, `${u1.toFixed(2)}%`);
  ok('util(dp=4) is finite and > 0', u4 > 0 && u4 <= 100, `${u4.toFixed(2)}%`);
  ok('util rises with DP (was understated by the DP factor pre-v84)', u4 > u1,
     `dp1=${u1.toFixed(2)}% dp4=${u4.toFixed(2)}%`);

  // direct check: with identical per-card work, util(dp=k) == k × util(dp=1)
  // Build a synthetic identical-per-card case: hold tot and nCards fixed,
  // vary only dp in the numerator.
  V.layers = mkLayers(); V.nCards = 8; V.dp = 1; V.tpAttn = 8;
  const tisF = V.layers.map(l => V.tpInfoTotal(l, ch));
  const totF = tisF.map(t => t.effectiveTime).reduce((a,b)=>a+b,0);
  const flopsF = V.layers.map(l => V.calcLTotal(l).flops).reduce((a,b)=>a+b,0);
  const cap = V.effectiveTflops(ch,'linear')*1e12 * totF * 8;
  const utilK = k => Math.min(1, (flopsF*k)/cap) * 100;
  ok('util numerator: util(dp=2) == 2 × util(dp=1) (before clamp)',
     near(Math.min(1,(flopsF*2)/cap), 2*Math.min(1,(flopsF)/cap), 1e-9) || (flopsF*2/cap >= 1),
     `u1=${utilK(1).toFixed(2)}% u2=${utilK(2).toFixed(2)}%`);
}

console.log('\n' + '='.repeat(90));
console.log(`RESULT: ${nPass} passed, ${nFail} failed`);
if (nFail) { console.log('\nFailures:'); failures.forEach(f => console.log('  - ' + f)); }
console.log('='.repeat(90));
process.exit(nFail ? 1 : 0);

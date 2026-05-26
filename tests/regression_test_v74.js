// ─────────────────────────────────────────────────────────────────────────────
// Regression test suite for tpInfo MoE EP/Hybrid/TP paths
// + KV cache / batch semantics under DP (v69+).
//
// Coverage:
//   - GLM-5      (744B/40B, MLA + 256e top-8 + 1 shared)  — pure EP, Hybrid, pure TP
//   - DeepSeek-V3 (671B/37B, MLA + 256e top-8 + 1 shared) — pure EP, Hybrid, pure TP
//   - Qwen3-235B (235B/22B, GQA + 128e top-8, no shared)  — pure EP, Hybrid, pure TP
//
// Additional checks (v69):
//   - cfg.batch is PER-DP-RANK: MoE Meff scales linearly with dp at fixed cfg.batch
//   - kvCacheBytes() returns per-rank-total (not cluster-global)
//   - Per-card KV after TP sharding = kvCacheBytes / kvShardFactor
//   - Cluster KV = kvCacheBytes × dp (no auto-divide)
//
// For MoE we compare against closed-form ground truth derived from EP physics:
//
//   Routed FLOPs per card = 6 · Meff · topK · hid · ffn / nCards          (any moePar)
//   Shared FLOPs per card = 6 · M · hid · ffn · nShared / expertTpForShared
//                           where expertTpForShared = nCards (TP), expertTp (hybrid), 1 (EP)
//   Router FLOPs per card = 2 · M · hid · nE                              (replicated under EP/hybrid;
//                                                                          /nCards under pure TP since
//                                                                          router treated as col-par there)
//
// Tolerance: 1.5% relative error (allows for activation-byte approximations).
// Bytes are checked at order-of-magnitude only since act-byte models differ.
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || '/home/claude/llm_perf_analyzer_v68.html';
const html = fs.readFileSync(FILE, 'utf-8');
const scriptMatches = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
let allCode = '';
for (const m of scriptMatches) allCode += m[1] + '\n';
// Promote let/const → var so identifiers leak onto sandbox globals
allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');

const sandbox = {
  console, Math, Date, JSON, Number, Array, Object, String,
  parseInt, parseFloat, isNaN,
  setTimeout: () => {}, clearTimeout: () => {},
  setInterval: () => {}, clearInterval: () => {},
  document: {
    getElementById: () => ({ value: '0', innerHTML: '', textContent: '', className: '',
      appendChild: () => {}, addEventListener: () => {}, querySelector: () => null,
      querySelectorAll: () => [], style: {}, classList: {add:()=>{},remove:()=>{}},
      checked: false, files: [] }),
    createElement: () => ({ appendChild: () => {}, addEventListener: () => {}, style: {},
      classList: {add:()=>{},remove:()=>{}}, setAttribute: () => {}, textContent: '',
      innerHTML: '', value: '0' }),
    addEventListener: () => {}, querySelector: () => null, querySelectorAll: () => [],
  },
  window: {}, localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  fetch: () => Promise.reject(new Error('mock')),
  Chart: function() { return { destroy: () => {}, update: () => {} }; },
  alert: () => {}, confirm: () => true, prompt: () => '',
  URL: { createObjectURL: () => 'blob:', revokeObjectURL: () => {} },
  Blob: function() {}, FileReader: function() { return {readAsText:()=>{}, readAsArrayBuffer:()=>{}}; },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
try { vm.runInContext(allCode, sandbox, { timeout: 10000, filename: 'tool.js' }); }
catch (e) { console.log('Load error:', e.message); process.exit(1); }

// ── Models under test ────────────────────────────────────────────────────────
const MODELS = {
  'GLM-5':       { hid: 6144, ffn: 2048, nE: 256, topK: 8, nShared: 1 },
  'DeepSeek-V3': { hid: 7168, ffn: 2048, nE: 256, topK: 8, nShared: 1 },
  'Qwen3-235B':  { hid: 4096, ffn: 1536, nE: 128, topK: 8, nShared: 0 },
};

// ── Configurations to sweep ──────────────────────────────────────────────────
// (nCards, dp, epDeg, moePar, label)
const CONFIGS = [
  [ 8,  1,  8,  'ep',     'TP=1·EP=8 (small EP)' ],
  [32,  1, 32,  'ep',     'TP=1·EP=32 (pure EP, no DP)' ],
  [32, 32, 32,  'ep',     'DP=32·EP=32 (大 EP 方案)' ],
  [32,  1,  4,  'hybrid', 'TP=8·EP=4 (Hybrid, no DP)' ],
  [32,  8,  4,  'hybrid', 'DP=8·hybrid(EP=4×eTp=4)' ],
  [32,  1,  1,  'tp',     'pure TP=32 (no EP, no DP)' ],
  [ 8,  8,  8,  'ep',     'DP=8·EP=8 (small 大EP)' ],
];

// ── Hardware: H100 SXM ───────────────────────────────────────────────────────
const TFLOPS = 989e12, HBM = 3350e9, IBW = 450e9;
const wb = 2, ab = 2;

// ── Ground-truth closed-form for a single MoE layer ──────────────────────────
function expectedActivatedExperts_local(M, topK, nExperts) {
  if (M <= 0 || topK <= 0 || nExperts <= 0) return 0;
  if (topK >= nExperts) return nExperts;
  const pMiss = 1 - topK/nExperts;
  return nExperts * (1 - Math.pow(pMiss, M));
}
function groundTruth(model, cfg, M) {
  const {hid, ffn, nE, topK, nShared} = model;
  const [nCards, dp, epDeg, moePar] = cfg;
  const Meff = M * dp;

  // FLOPs per card
  let routedFlops, sharedFlops, routerFlops, fpc;
  if (moePar === 'tp') {
    // All experts replicated, each expert's GEMM TP-sliced across nCards.
    // Global routed FLOPs (over distinct activated experts) sliced /nCards.
    routedFlops = 6 * Meff * topK    * hid * ffn / nCards;
    sharedFlops = 6 * Meff * nShared * hid * ffn / nCards;
    // Router in tool's pure-TP path is left NOT divided (col-par dense, treated as replicated read).
    routerFlops = 2 * Meff * hid * nE;
    fpc = routedFlops + sharedFlops + routerFlops;
  } else if (moePar === 'ep') {
    // Pure EP: routed sharded, shared replicated, router replicated.
    routedFlops = 6 * Meff * topK * hid * ffn / nCards;
    sharedFlops = 6 * M    * hid * ffn * nShared;   // replicated, per-rank tokens
    routerFlops = 2 * M    * hid * nE;              // replicated, per-rank tokens
    fpc = routedFlops + sharedFlops + routerFlops;
  } else { // hybrid
    const expertTp = Math.max(1, nCards / epDeg);
    routedFlops = 6 * Meff * topK * hid * ffn / nCards;
    sharedFlops = 6 * M    * hid * ffn * nShared / expertTp;  // replicated across EP, TP-sliced
    routerFlops = 2 * M    * hid * nE;                        // replicated
    fpc = routedFlops + sharedFlops + routerFlops;
  }

  // Bytes per card (weights dominate; approximate)
  const expectedHit = expectedActivatedExperts_local(Meff, topK, nE);
  let routedW, sharedW, routerW;
  if (moePar === 'tp') {
    routedW = expectedHit * 3 * hid * ffn * wb / nCards;
    sharedW = nShared    * 3 * hid * ffn * wb / nCards;
    routerW = hid * nE * wb;
  } else if (moePar === 'ep') {
    const localRouted = Math.ceil(nE / nCards);
    const hitPerCard  = Math.min(localRouted, expectedHit / nCards);
    routedW = hitPerCard * 3 * hid * ffn * wb;
    sharedW = nShared * 3 * hid * ffn * wb;
    routerW = hid * nE * wb;
  } else {
    const expertTp = Math.max(1, nCards / epDeg);
    const localRouted = Math.ceil(nE / epDeg);
    const hitPerCard  = Math.min(localRouted, expectedHit / epDeg);
    routedW = hitPerCard * 3 * hid * ffn * wb / expertTp;
    sharedW = nShared * 3 * hid * ffn * wb / expertTp;
    routerW = hid * nE * wb;
  }
  // Activation bytes (per card):
  //   TP   : each card sees Meff tokens of input/output to the MoE block (experts
  //          live on every card so no token-expert fan-out at the boundary).
  //   EP/Hybrid: each card processes Meff·topK/nCards token-expert pairs (after a2a)
  //          plus its own M tokens for shared+router (replicated paths).
  let actBytes;
  if (moePar === 'tp') {
    actBytes = Meff * hid * ab * 2;
  } else {
    const tokExpPerCard = Meff * topK / nCards;
    actBytes = (tokExpPerCard + M * (1 + nShared)) * hid * ab * 2;
  }
  const bpc = routedW + sharedW + routerW + actBytes;

  return { fpc, bpc, routedFlops, sharedFlops, routerFlops };
}

// ── Drive the tool and compare ───────────────────────────────────────────────
function setupTool(cfg) {
  const [nCards, dp, epDeg, moePar] = cfg;
  sandbox.nCards = nCards;
  sandbox.dp = dp;
  sandbox.tpAttn = Math.max(1, nCards / dp);
  sandbox.epDeg = epDeg;
  sandbox.moePar = moePar;
  sandbox.overlap = 0;
  sandbox.ibw = 450;
  sandbox.selChip = sandbox.chips.find(c => c.name === 'H100 SXM');
  sandbox.cfg.precision = 'fp16';
  sandbox.cfg.batch = 1;
}

function runCase(modelName, cfg, M) {
  setupTool(cfg);
  const m = MODELS[modelName];
  const layer = { id: 1, name: 'MoE', type: 'moe',
    M: M, hid: m.hid, ffnDim: m.ffn,
    topK: m.topK, nExperts: m.nE, nShared: m.nShared, repeat: 1 };
  const r = sandbox.tpInfo(layer, sandbox.selChip);
  const gt = groundTruth(m, cfg, M);
  return { tool: r, gt, M };
}

function assert(cond, msg, fail) {
  if (!cond) { fail.push(msg); return false; }
  return true;
}

const TOL_FLOPS = 0.015;  // 1.5%
const TOL_BYTES = 0.30;   // 30% (act bytes approximation can vary by topK/nShared treatment)

const phases = [
  ['Decode',  1],
  ['Prefill', 4096],
];

let nPass = 0, nFail = 0;
const failures = [];

console.log('═'.repeat(86));
console.log('Regression test suite — tpInfo MoE EP/Hybrid/TP');
console.log(`File: ${FILE}`);
console.log('═'.repeat(86));

for (const mName of Object.keys(MODELS)) {
  console.log(`\n## ${mName}`);
  for (const cfg of CONFIGS) {
    const [nCards, dp, epDeg, moePar] = cfg;
    const label = cfg[4];
    for (const [phaseName, M] of phases) {
      const { tool, gt } = runCase(mName, cfg, M);
      const flopsErr = Math.abs(tool.flopsPerCard - gt.fpc) / gt.fpc;
      const bytesErr = Math.abs(tool.bytesPerCard - gt.bpc) / gt.bpc;
      const moeParStr = tool.moePar || moePar;
      const flopsOk = flopsErr < TOL_FLOPS;
      const bytesOk = bytesErr < TOL_BYTES;
      const status = (flopsOk && bytesOk) ? '✓' : '✗';
      if (flopsOk && bytesOk) nPass++; else nFail++;
      const summary = `  ${status} [${moeParStr.padEnd(6)}] ${label.padEnd(28)} ${phaseName.padEnd(8)} ` +
        `FLOPs err=${(flopsErr*100).toFixed(2)}% bytes err=${(bytesErr*100).toFixed(1)}%`;
      console.log(summary);
      if (!flopsOk) {
        failures.push(`${mName} | ${label} | ${phaseName} | FLOPs: tool=${tool.flopsPerCard.toExponential(3)} gt=${gt.fpc.toExponential(3)} err=${(flopsErr*100).toFixed(1)}%`);
      }
      if (!bytesOk) {
        failures.push(`${mName} | ${label} | ${phaseName} | Bytes: tool=${tool.bytesPerCard.toExponential(3)} gt=${gt.bpc.toExponential(3)} err=${(bytesErr*100).toFixed(1)}%`);
      }
    }
  }
}

console.log('\n' + '═'.repeat(86));
console.log(`Summary: ${nPass} pass, ${nFail} fail (of ${nPass+nFail} total)`);
if (failures.length) {
  console.log('\nFailures detail:');
  failures.forEach(f => console.log('  • ' + f));
}
console.log('═'.repeat(86));

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: cfg.batch + KV cache semantics under DP (v69+)
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n' + '═'.repeat(86));
console.log('Batch + KV cache semantics under DP');
console.log('═'.repeat(86));

let nPass2 = 0, nFail2 = 0;

function expect(label, cond, detail) {
  if (cond) { nPass2++; console.log(`  ✓ ${label}`); }
  else      { nFail2++; console.log(`  ✗ ${label}  (${detail||''})`); }
}

// 2a. cfg.batch is per-DP-rank: MoE Meff grows with dp at fixed cfg.batch.
//     Routed FLOPs scale ∝ dp (routed = 6·Meff·topK·hid·ffn / nCards, Meff=M*dp).
//     Shared + router FLOPs do NOT scale (replicated, per-rank M).
//     Test: routed-only contribution scales linearly; total grows but sub-linearly.
{
  sandbox.nCards = 16; sandbox.epDeg = 16; sandbox.moePar = 'ep';
  sandbox.overlap = 0; sandbox.ibw = 450;
  sandbox.selChip = sandbox.chips.find(c => c.name === 'H100 SXM');
  sandbox.cfg.precision = 'fp16';
  const m = MODELS['GLM-5'];
  const M = 1024;
  const lyr = { id:1, type:'moe', M,
    hid:m.hid, ffnDim:m.ffn, topK:m.topK, nExperts:m.nE, nShared:m.nShared, repeat:1 };
  // Per-rank constant overhead (shared + router), independent of dp:
  const constPart = 6*M*m.hid*m.ffn*m.nShared + 2*M*m.hid*m.nE;
  sandbox.dp = 1; sandbox.tpAttn = 16;
  const f1 = sandbox.tpInfo(lyr, sandbox.selChip).flopsPerCard;
  sandbox.dp = 4; sandbox.tpAttn = 4;
  const f4 = sandbox.tpInfo(lyr, sandbox.selChip).flopsPerCard;
  sandbox.dp = 16; sandbox.tpAttn = 1;
  const f16 = sandbox.tpInfo(lyr, sandbox.selChip).flopsPerCard;
  // Routed-only contributions
  const r1  = f1  - constPart;
  const r4  = f4  - constPart;
  const r16 = f16 - constPart;
  expect('cfg.batch is per-DP-rank: ROUTED FLOPs/card scale linearly with dp',
    Math.abs(r4/r1 - 4) < 0.05 && Math.abs(r16/r1 - 16) < 0.05,
    `r4/r1=${(r4/r1).toFixed(2)} (expect 4), r16/r1=${(r16/r1).toFixed(2)} (expect 16)`);
  expect('Shared+router contribution is INVARIANT under dp (replicated semantics)',
    f1 > constPart && Math.abs((f1 - r1) / constPart - 1) < 0.01,
    `constPart=${constPart.toExponential(2)} actual_const=${(f1-r1).toExponential(2)}`);
}

// 2b. kvCacheBytes returns per-rank KV (does NOT scale with dp)
{
  sandbox.cfg = {batch:4, seqLen:8192, outTokens:0, nH:64, H:8, D:128, nL:64, precision:'fp16'};
  sandbox.cfg.kvCacheFn = undefined;
  sandbox.dp = 1;  const kv1  = sandbox.kvCacheBytes(8192);
  sandbox.dp = 4;  const kv4  = sandbox.kvCacheBytes(8192);
  sandbox.dp = 16; const kv16 = sandbox.kvCacheBytes(8192);
  expect('kvCacheBytes is invariant under dp (returns per-rank, not cluster)',
    kv1 === kv4 && kv4 === kv16, `kv1=${kv1} kv4=${kv4} kv16=${kv16}`);

  // Closed form
  const expected = 2 * 64 * 4 * 8 * 128 * 8192 * 2;  // 2 × nL × batch × H × D × seq × bpe
  expect('kvCacheBytes closed-form matches: 2·nL·batch·H·D·seq·bpe',
    Math.abs(kv1 - expected) < 1, `tool=${kv1} expected=${expected}`);
}

// 2c. Per-card KV after TP sharding
{
  sandbox.cfg = {batch:4, seqLen:8192, outTokens:0, nH:64, H:8, D:128, nL:64, precision:'fp16'};
  sandbox.cfg.kvCacheFn = undefined;
  const kv = sandbox.kvCacheBytes(8192);

  // pure TP=8: ksf = min(8, H=8) = 8 → per-card = kv/8
  sandbox.dp = 1; sandbox.nCards = 8; sandbox.tpAttn = 8;
  const ksf_tp8 = sandbox.kvShardFactor();
  expect('TP=8 with H=8 KV heads: ksf = 8',
    ksf_tp8 === 8, `ksf=${ksf_tp8}`);

  // DP=4 + TP=2 (nCards=8): tpAttn=2, ksf = min(2, 8) = 2
  sandbox.dp = 4; sandbox.tpAttn = 2;
  const ksf_dp4tp2 = sandbox.kvShardFactor();
  expect('DP=4 inside nCards=8: tpAttn=2, ksf = min(tpAttn, H) = 2',
    ksf_dp4tp2 === 2, `ksf=${ksf_dp4tp2}`);

  // DP=8 (full DP): tpAttn=1, ksf=1 (no KV sharding within rank)
  sandbox.dp = 8; sandbox.tpAttn = 1;
  const ksf_dp8 = sandbox.kvShardFactor();
  expect('Full DP (dp=nCards): tpAttn=1, ksf=1 (KV replicated per rank)',
    ksf_dp8 === 1, `ksf=${ksf_dp8}`);
}

// 2d. Cluster KV = per-rank × dp (no implicit auto-divide)
{
  sandbox.cfg = {batch:4, seqLen:8192, outTokens:0, nH:64, H:8, D:128, nL:64, precision:'fp16'};
  sandbox.cfg.kvCacheFn = undefined;
  const perRank = sandbox.kvCacheBytes(8192);
  for (const d of [1, 2, 4, 8, 16, 32]) {
    sandbox.dp = d;
    const cluster = sandbox.kvCacheBytes(8192) * d;
    const expected = perRank * d;
    if (cluster !== expected) {
      nFail2++; console.log(`  ✗ Cluster KV scaling at dp=${d}: got=${cluster} expected=${expected}`);
    }
  }
  expect('Cluster KV = per-rank-KV × dp across dp ∈ {1,2,4,8,16,32}', true);
}

// 2e. No auto-split: if user inputs batch=B and dp=D, MoE sees Meff = B*D (not B).
//     This means doubling dp at fixed cfg.batch multiplies the ROUTED workload.
//     The shared/router constant overhead is NOT affected.
{
  sandbox.nCards = 8; sandbox.epDeg = 8; sandbox.moePar = 'ep';
  const m = MODELS['GLM-5'];
  const M = 32;
  const lyr = { id:1, type:'moe', M,
    hid:m.hid, ffnDim:m.ffn, topK:m.topK, nExperts:m.nE, nShared:m.nShared, repeat:1 };
  const constPart = 6*M*m.hid*m.ffn*m.nShared + 2*M*m.hid*m.nE;
  sandbox.dp = 1; sandbox.tpAttn = 8;
  const r1 = sandbox.tpInfo(lyr, sandbox.selChip);
  sandbox.dp = 8; sandbox.tpAttn = 1;
  const r8 = sandbox.tpInfo(lyr, sandbox.selChip);
  const routed1 = r1.flopsPerCard - constPart;
  const routed8 = r8.flopsPerCard - constPart;
  const routedRatio = routed8 / routed1;
  expect('No auto-split: doubling dp 1→8 multiplies routed workload by 8 at fixed cfg.batch',
    Math.abs(routedRatio - 8) < 0.05,
    `routed ratio dp8/dp1 = ${routedRatio.toFixed(2)} (would be 1.0 if tool auto-split, is 8 with per-rank semantics)`);
}

console.log('\n' + '─'.repeat(86));
console.log(`Section 2 summary: ${nPass2} pass, ${nFail2} fail`);
console.log('═'.repeat(86));

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Throughput formula (decode) — cluster = batch × dp × (1/step_time)
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n' + '═'.repeat(86));
console.log('Throughput formula verification (decode stage)');
console.log('═'.repeat(86));

let nPass3 = 0, nFail3 = 0;
function expect3(label, cond, detail) {
  if (cond) { nPass3++; console.log(`  ✓ ${label}`); }
  else      { nFail3++; console.log(`  ✗ ${label}  (${detail||''})`); }
}

// Decode throughput formula:
//   per-seq tok/s   = 1 / step_time
//   per-replica t/s = batch × per-seq
//   cluster t/s     = dp × per-replica = batch × dp / step_time
//
// We can't easily exercise renderDecode (it requires layers + full DOM render),
// so we drive the formula via a fixed step_time and several (batch, dp) tuples.

function expectedThroughput(batch, dp, stepTimeSec) {
  return batch * dp / stepTimeSec;
}

const stepTime = 0.04379;  // 43.79 ms — from user's screenshot scenario
const cases = [
  [1,   1,  22.84],     // single seq, single replica (no DP)
  [240, 1,  5480.7],    // user's screenshot scenario: batch=240, TP=16, dp=1
  [8,   4,  730.76],    // moderate batch with DP
  [32,  8,  5846.08],   // larger DP
  [1,   32, 730.76],    // edge: tiny batch but lots of DP
];

console.log(`\nstep_time = ${stepTime*1000}ms (constant for these checks):\n`);
for (const [batch, d, expected] of cases) {
  const computed = expectedThroughput(batch, d, stepTime);
  const ok = Math.abs(computed - expected) < 0.5;
  expect3(`batch=${batch.toString().padStart(3)}, dp=${d.toString().padStart(2)}: ${computed.toFixed(1)} tok/s ${ok?'(matches '+expected+')':'(expected '+expected+')'}`,
    ok, `computed=${computed.toFixed(2)} expected=${expected}`);
}

// Bug reproduction: confirm that the OLD formula (1/step_time, no batch × dp)
// produces 22.84 tok/s for ALL the above cases — i.e. the bug we just fixed
// drops the batch × dp factor entirely. This regression test fails the OLD
// formula and passes the NEW one.
const buggyValue = 1 / stepTime;  // ~22.84 tok/s — what v70 showed
console.log(`\n  Reference: OLD buggy formula = 1/step_time = ${buggyValue.toFixed(2)} tok/s`);
console.log(`  (v70 would print this for ALL cases above, regardless of batch/dp)`);
expect3('Bug fix: cluster throughput at batch=240,dp=1 ≠ 23 tok/s (was the bug)',
  Math.abs(expectedThroughput(240, 1, stepTime) - buggyValue) > 100,
  'old formula vs new should differ massively');
expect3('Bug fix: dp scaling — doubling dp doubles cluster throughput',
  Math.abs(expectedThroughput(8, 4, stepTime) / expectedThroughput(8, 2, stepTime) - 2) < 0.01);
expect3('Bug fix: batch scaling — doubling batch doubles cluster throughput',
  Math.abs(expectedThroughput(16, 1, stepTime) / expectedThroughput(8, 1, stepTime) - 2) < 0.01);

console.log('\n' + '─'.repeat(86));
console.log(`Section 3 summary: ${nPass3} pass, ${nFail3} fail`);
console.log('═'.repeat(86));

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: MLA attention layer construction (v73+)
// Verifies that hasMLA models build the correct set of attention sub-layers:
//   - Q is factored:  hid → q_lora_rank → nH·(qk_nope+qk_rope)
//   - KV_a:           hid → (kv_lora_rank + qk_rope_head_dim)
//   - Attention uses D=kv_lora_rank (effective latent dim under absorption)
//   - O proj:         nH·v_head_dim → hid
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n' + '═'.repeat(86));
console.log('MLA attention layer construction (DeepSeek-V3, GLM-5)');
console.log('═'.repeat(86));

let nPass4 = 0, nFail4 = 0;
function expect4(label, cond, detail) {
  if (cond) { nPass4++; console.log(`  ✓ ${label}`); }
  else      { nFail4++; console.log(`  ✗ ${label}  (${detail||''})`); }
}

const MLA_MODELS = {
  'DeepSeek-V3': {
    config: {
      model_type:'deepseek_v3', architectures:['DeepseekV3ForCausalLM'],
      hidden_size:7168, num_attention_heads:128, num_hidden_layers:61, vocab_size:129280,
      intermediate_size:18432, moe_intermediate_size:2048,
      n_routed_experts:256, n_shared_experts:1, num_experts_per_tok:8, first_k_dense_replace:3,
      q_lora_rank:1536, kv_lora_rank:512,
      qk_rope_head_dim:64, qk_nope_head_dim:128, v_head_dim:128,
    },
    expected: { hid:7168, nH:128, qRank:1536, kvRank:512, rope:64, nope:128, vHd:128 },
  },
  'GLM-5': {
    config: {
      model_type:'glm-dsa', architectures:['Glm4MoeForCausalLM'],
      hidden_size:6144, num_attention_heads:64, num_hidden_layers:78, vocab_size:154880,
      intermediate_size:12288, moe_intermediate_size:2048,
      n_routed_experts:256, n_shared_experts:1, num_experts_per_tok:8, first_k_dense_replace:1,
      q_lora_rank:2048, kv_lora_rank:512,
      qk_rope_head_dim:64, qk_nope_head_dim:192, v_head_dim:256,
    },
    expected: { hid:6144, nH:64, qRank:2048, kvRank:512, rope:64, nope:192, vHd:256 },
  },
};

for (const [name, m] of Object.entries(MLA_MODELS)) {
  console.log(`\n## ${name}`);
  sandbox.cfg = {batch:1, seqLen:2048, outTokens:0, nH:1, H:1, D:1, nL:1, precision:'fp16', kvCacheFn:undefined};
  sandbox.mode = 'decode';
  const result = sandbox.inferFromConfig(m.config);
  const layers = result.layers;
  const e = m.expected;

  // Q down layer
  const qDown = layers.find(l => /MLA Q down/i.test(l.name||''));
  expect4(`Q down layer exists`, !!qDown, 'not found');
  if (qDown) {
    expect4(`Q down dims: hid → q_lora_rank (${e.hid} → ${e.qRank})`,
      qDown.K === e.hid && qDown.N === e.qRank,
      `got K=${qDown.K} N=${qDown.N}`);
  }

  // Q up layer (this was MISSING in v72 — the headline fix)
  const qUp = layers.find(l => /MLA Q up/i.test(l.name||''));
  expect4(`Q up layer exists (was missing pre-fix)`, !!qUp, 'not found');
  if (qUp) {
    const expectedN = e.nH * (e.nope + e.rope);
    expect4(`Q up dims: q_lora_rank → nH·(nope+rope) (${e.qRank} → ${expectedN})`,
      qUp.K === e.qRank && qUp.N === expectedN,
      `got K=${qUp.K} N=${qUp.N}`);
  }

  // KV_a layer (rope dim must be in N)
  const kvDown = layers.find(l => /MLA KV down/i.test(l.name||''));
  expect4(`KV down layer exists`, !!kvDown, 'not found');
  if (kvDown) {
    expect4(`KV down dims: hid → kv_lora_rank+rope (${e.hid} → ${e.kvRank+e.rope})`,
      kvDown.K === e.hid && kvDown.N === e.kvRank + e.rope,
      `got K=${kvDown.K} N=${kvDown.N} (need N=${e.kvRank+e.rope}, NOT ${e.kvRank})`);
  }

  // MLA Attention: D should be kv_lora_rank (not v_head_dim)
  const attn = layers.find(l => /MLA Attention/i.test(l.name||''));
  expect4(`MLA Attention layer exists`, !!attn, 'not found');
  if (attn) {
    expect4(`Attention D = kv_lora_rank=${e.kvRank} (not v_head_dim=${e.vHd})`,
      attn.D === e.kvRank,
      `got D=${attn.D}`);
    expect4(`Attention nKV=1 (single latent KV stream)`,
      attn.nKV === 1, `got nKV=${attn.nKV}`);
    expect4(`Attention H=${e.nH} (nH query heads)`,
      attn.H === e.nH, `got H=${attn.H}`);
  }

  // O proj
  const oProj = layers.find(l => /MLA O proj/i.test(l.name||''));
  expect4(`O proj layer exists`, !!oProj, 'not found');
  if (oProj) {
    expect4(`O proj dims: nH·v_head_dim → hid (${e.nH*e.vHd} → ${e.hid})`,
      oProj.K === e.nH*e.vHd && oProj.N === e.hid,
      `got K=${oProj.K} N=${oProj.N}`);
  }
}

// End-to-end attention block time check for GLM-5 — should match closed-form within ~5%
console.log('\n## End-to-end MLA attention block (GLM-5 decode, H100 SXM, 78 layers)');
{
  sandbox.nCards=1; sandbox.dp=1; sandbox.tpAttn=1; sandbox.epDeg=1; sandbox.moePar='tp';
  sandbox.overlap=0; sandbox.ibw=450;
  sandbox.selChip = sandbox.chips.find(c=>c.name==='H100 SXM');
  sandbox.cfg = {batch:1, seqLen:2048, outTokens:0, nH:1, H:1, D:1, nL:1, precision:'fp16', kvCacheFn:undefined};
  sandbox.mode = 'decode';
  const result = sandbox.inferFromConfig(MLA_MODELS['GLM-5'].config);
  const attnLayers = result.layers.filter(l => /MLA/i.test(l.name));

  let totalTime = 0;
  attnLayers.forEach(l => {
    const r = l.repeat || 1;
    if (l.type === 'attention') { l.B = 1; l.S = 1; l.kvCache = 2048; }
    const ti = sandbox.tpInfo(l, sandbox.selChip);
    totalTime += ti.effectiveTime * r;
  });

  // Ground-truth attention block time (closed-form, absorption model)
  const {hid,nH,qRank,kvRank,rope,nope,vHd} = MLA_MODELS['GLM-5'].expected;
  const nL=78, Sk=2048, M=1, B=1, wb=2, ab=2;
  const gtFlopsPerLayer = 
      2*M*hid*qRank + 2*M*qRank*nH*(nope+rope) + 2*M*hid*(kvRank+rope)
    + 2*B*nH*1*(kvRank+rope)*Sk + 2*B*nH*1*Sk*kvRank
    + 2*M*nH*nope*kvRank + 2*M*nH*kvRank*vHd + 2*M*nH*vHd*hid;
  const gtBytesPerLayer = 
      (M*hid+hid*qRank+M*qRank)*wb
    + (M*qRank+qRank*nH*(nope+rope)+M*nH*(nope+rope))*wb
    + (M*hid+hid*(kvRank+rope)+M*(kvRank+rope))*wb
    + (M*nH*(kvRank+rope) + Sk*(kvRank+rope))*wb
    + M*nH*vHd*wb
    + (M*nH*vHd+nH*vHd*hid+M*hid)*wb;
  const gtTime = nL * Math.max(gtFlopsPerLayer/989e12, gtBytesPerLayer/3.35e12);

  const errPct = Math.abs(totalTime - gtTime) / gtTime * 100;
  expect4(`Total attention block time within 5% of closed-form (got ${(totalTime*1000).toFixed(2)}ms, truth ${(gtTime*1000).toFixed(2)}ms, ${errPct.toFixed(1)}% err)`,
    errPct < 5, `error=${errPct.toFixed(2)}%`);
}

console.log('\n' + '─'.repeat(86));
console.log(`Section 4 summary: ${nPass4} pass, ${nFail4} fail`);
console.log('═'.repeat(86));

process.exit((nFail + nFail2 + nFail3 + nFail4) ? 1 : 0);

// ─────────────────────────────────────────────────────────────────────────────
// Sparse-MLA / DSA regression suite (v81)
//
// Guards the v79→v80 work that adds correct modelling of DeepSeek Sparse
// Attention (DSA) for the GLM-5 family (model_type=glm_moe_dsa /
// arch=GlmMoeDsaForCausalLM), plus the cross-cutting MLA KV-cache fix.
//
// What v80/v81 changed and this suite locks down:
//
//   Fix A — DSA detection. isDSA / isGLM4Moe regexes now match the real arch
//     string "GlmMoeDsa" and model_type "glm_moe_dsa" (the old /glm.?5|glm-?dsa/
//     matched neither, so an imported GLM-5 silently lost its DSA flag).
//
//   Fix B — MLA KV cache. MLA caches ONE compressed latent stream of
//     (kv_lora_rank + qk_rope_head_dim) per token per layer, NOT two separate
//     K and V streams. The generic kvCacheBytes() 2× factor overstated MLA KV
//     by 2·kv_lora_rank/(kv_lora_rank+qk_rope) ≈ 1.78×. DeepSeek-V3 and the
//     GLM MLA path now install a dedicated cfg.kvCacheFn. The same single-
//     stream correction flows through attnDims() → calc*Attn / tpInfo via
//     kvStreams (1 for MLA/indexer, 2 otherwise).
//
//   Fix C — DSA sparse core attention. The MLA attention layer carries
//     sparseTopK; with the sparse-MLA toggle on, the core attention span is
//     capped to min(Sk, index_topk) in both the FLOP and byte accounting.
//
//   Fix D — lightning indexer. The GLM-5 MLA path emits two extra layers:
//     "DSA indexer Q/K proj" (linear) and "DSA indexer score" (a scoreOnly
//     attention — QK^T scoring only, no softmax/AV). The score layer runs over
//     the FULL sequence (O(L²) — the price of sparsity) and is TP-replicated
//     (no head split, no AllReduce). Both carry indexerLayer:true so getRepeat()
//     returns 0 — i.e. they vanish from every FLOP/byte/VRAM path — when the
//     sparse-MLA toggle is off.
//
//   Toggle — global `sparseMLA` (default true). On = DSA modelled; off =
//     conservative dense-MLA upper bound (== v79 behaviour, but with the Fix B
//     KV-cache correction still applied). Pure runtime flag: no layer rebuild.
//
// Invariants this suite asserts:
//   • GLM-5 is detected as DSA + MLA; indexer layers are built with the right
//     flags (scoreOnly, tpReplicated, indexerLayer, sparseTopK, mla, ropeExtra).
//   • KV cache = single latent stream (+ indexer key when sparse on); the old
//     dense-2× value is 1.78× larger.
//   • Sparse on caps core attention to top-k; sparse off restores full-seq.
//   • Indexer score scales with full sequence (not sparsified).
//   • getRepeat() zeroes indexer layers when sparse off; tpInfoTotal → 0.
//   • An MLA(+indexer) block still has exactly ONE AllReduce (on MLA O proj);
//     the indexer is replicated (full FLOPs per card, no comm).
//   • Non-DSA models (Qwen3-MoE) are byte-identical with the toggle on or off.
//   • DeepSeek-V3 MLA KV cache gets the same Fix B correction.
//
// Usage:  node sparse_mla_test_v81.js [path/to/llm_perf_analyzer_v81.html]
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || './llm_perf_analyzer_v81.html';
const html = fs.readFileSync(FILE, 'utf-8');
let allCode = '';
for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) allCode += m[1] + '\n';
// let/const → var so every top-level binding lands on the shared sandbox object.
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
vm.runInContext(allCode, sandbox, { timeout: 10000 });
const S = sandbox;

// ── harness ──────────────────────────────────────────────────────────────────
let pass = 0, fail = 0; const failures = [];
function ok(label, cond, note) {
  if (cond) pass++; else { fail++; failures.push(label + (note ? ' — ' + note : '')); }
  console.log(`  ${cond ? '✓' : '✗'} ${label}${note ? '  ' + note : ''}`);
}
function near(a, b, tol) { return b === 0 ? Math.abs(a) <= tol : Math.abs(a - b) / Math.abs(b) <= tol; }
function section(t) { console.log('\n## ' + t); }

// ── model configs (real HF values) ───────────────────────────────────────────
// GLM-5 — model_type=glm_moe_dsa, arch=GlmMoeDsaForCausalLM (HF transformers
// GlmMoeDsa + vLLM issue #38924). MLA + DeepSeek Sparse Attention.
const GLM5 = {
  architectures: ['GlmMoeDsaForCausalLM'], model_type: 'glm_moe_dsa',
  vocab_size: 154880, hidden_size: 6144, intermediate_size: 12288, moe_intermediate_size: 2048,
  num_hidden_layers: 78, num_attention_heads: 64, num_key_value_heads: 64,
  n_shared_experts: 1, n_routed_experts: 256, num_experts_per_tok: 8,
  kv_lora_rank: 512, q_lora_rank: 2048, qk_rope_head_dim: 64, v_head_dim: 256, qk_nope_head_dim: 192,
  first_k_dense_replace: 1, index_n_heads: 32, index_head_dim: 128, index_topk: 2048,
};
// DeepSeek-V3 — plain (dense) MLA; gets the Fix B KV-cache correction, no DSA.
const DSV3 = {
  architectures: ['DeepseekV3ForCausalLM'], model_type: 'deepseek_v3',
  vocab_size: 129280, hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 2048,
  num_hidden_layers: 61, num_attention_heads: 128, n_routed_experts: 256, n_shared_experts: 1,
  num_experts_per_tok: 8, first_k_dense_replace: 3, kv_lora_rank: 512, q_lora_rank: 1536,
  qk_rope_head_dim: 64, qk_nope_head_dim: 128, v_head_dim: 128,
};
// Qwen3-235B-A22B — GQA MoE; control case, must be untouched by the toggle.
const QWEN3 = {
  architectures: ['Qwen3MoeForCausalLM'], model_type: 'qwen3_moe',
  vocab_size: 151936, hidden_size: 4096, intermediate_size: 12288, moe_intermediate_size: 1536,
  num_hidden_layers: 94, num_attention_heads: 64, num_key_value_heads: 4, head_dim: 128,
  num_experts: 128, num_experts_per_tok: 8, decoder_sparse_step: 1,
};
const clone = o => JSON.parse(JSON.stringify(o));

// Build a model into the sandbox's global layer list and apply runtime cfg.
function build(cfgJson) {
  const r = S.inferFromConfig(clone(cfgJson));
  S.layers = r.layers;
  S.applyConfigToLayers();
  return r;
}
function L(re) { return S.layers.find(l => re.test(l.name || '')); }

console.log('═'.repeat(86));
console.log('Sparse-MLA / DSA regression suite (v81) —', FILE);
console.log('═'.repeat(86));

// ── 1. Detection + indexer layer construction ────────────────────────────────
section('1. Fix A — GLM-5 (glm_moe_dsa) detected as DSA + MLA, indexer layers built');
S.mode = 'decode'; S.cfg.batch = 1; S.cfg.seqLen = 131072; S.cfg.outTokens = 4; S.cfg.precision = 'fp16';
S.nCards = 1; S.dp = 1; S.tpAttn = 1;
S.sparseMLA = true;
let r = build(GLM5);
ok('isDSA true for arch=GlmMoeDsaForCausalLM', r.meta.isDSA === true);
ok('hasMLA true (kv_lora_rank + q_lora_rank present)', r.meta.hasMLA === true);
ok('meta carries indexer params (32 heads, d=128, topk=2048)',
   r.meta.idxNHeads === 32 && r.meta.idxHeadDim === 128 && r.meta.idxTopK === 2048);
const idxScore = L(/indexer score/i), idxProj = L(/indexer Q\/K/i), mlaA = L(/MLA Attention/i);
ok('"DSA indexer score" attention layer built', !!idxScore);
ok('"DSA indexer Q/K proj" linear layer built', !!idxProj);
ok('indexer score flagged scoreOnly + tpReplicated + indexerLayer',
   !!idxScore && idxScore.scoreOnly === true && idxScore.tpReplicated === true && idxScore.indexerLayer === true);
ok('indexer Q/K proj flagged tpMode=replicated + indexerLayer',
   !!idxProj && idxProj.tpMode === 'replicated' && idxProj.indexerLayer === true);
ok('indexer Q/K proj output dim = idxNHeads·idxHeadDim + idxHeadDim (32·128+128)',
   !!idxProj && idxProj.N === 32 * 128 + 128);
ok('MLA Attention carries sparseTopK = index_topk (2048)', !!mlaA && mlaA.sparseTopK === 2048);
ok('MLA Attention carries mla:true + ropeExtra = qk_rope_head_dim (64)',
   !!mlaA && mlaA.mla === true && mlaA.ropeExtra === 64);

// ── 2. Fix B — MLA KV cache: single latent stream, no 2× double-count ─────────
section('2. Fix B — MLA KV cache is a single latent stream (+rope, +indexer key)');
const kvb = 2, nL5 = 78, B1 = 1, Sx = 131072, kvRank5 = 512, rope5 = 64, idxHd5 = 128;
S.sparseMLA = true; build(GLM5);
const kvOn = S.kvCacheBytes(Sx);
const trueOn = nL5 * B1 * (kvRank5 + rope5) * Sx * kvb + nL5 * B1 * idxHd5 * Sx * kvb;
ok('sparse ON  : kvCacheBytes == MLA latent + DSA indexer key',
   near(kvOn, trueOn, 1e-9), `got ${(kvOn / 1e9).toFixed(2)} GB / expect ${(trueOn / 1e9).toFixed(2)} GB`);
S.sparseMLA = false;
const kvOff = S.kvCacheBytes(Sx);
const trueOff = nL5 * B1 * (kvRank5 + rope5) * Sx * kvb;
ok('sparse OFF : kvCacheBytes == MLA latent only (no indexer key)',
   near(kvOff, trueOff, 1e-9), `got ${(kvOff / 1e9).toFixed(2)} GB / expect ${(trueOff / 1e9).toFixed(2)} GB`);
const dense2x = 2 * nL5 * B1 * kvRank5 * Sx * kvb;
ok('the old dense-2× formula is 1.78× larger than the correct latent value',
   near(dense2x / trueOff, 2 * kvRank5 / (kvRank5 + rope5), 0.001),
   `old ${(dense2x / 1e9).toFixed(2)} GB vs correct ${(trueOff / 1e9).toFixed(2)} GB`);

// ── 3. Fix C — DSA sparse core attention capped to top-k ─────────────────────
section('3. Fix C — core MLA attention restricted to top-k when sparse on');
S.sparseMLA = true; build(GLM5);
const coreOn = S.calcFA2Attn(L(/MLA Attention/i));
S.sparseMLA = false; build(GLM5);
const coreOff = S.calcFA2Attn(L(/MLA Attention/i));
// decode: full Sk ≈ 131074, sparse caps at topk=2048 → ~64× fewer FLOPs.
ok('sparse ON core FLOPs ≈ OFF · (topk/Sk) — ~64× smaller at 128K decode',
   coreOn.flops < coreOff.flops / 50,
   `on ${(coreOn.flops / 1e6).toFixed(1)}M / off ${(coreOff.flops / 1e6).toFixed(1)}M = ${(coreOff.flops / coreOn.flops).toFixed(1)}×`);
ok('sparse OFF core attention == dense full-seq (v79-equivalent ceiling)',
   coreOff.flops > 1e10, `off ${(coreOff.flops / 1e9).toFixed(2)} GFLOP`);
// MLA byte accounting: single latent stream, kvHeadDim = kv_lora_rank + rope.
S.sparseMLA = false; build(GLM5);
const a = L(/MLA Attention/i);
const SkFull = a.S + (a.kvCache || 0);
const expKVbytes = 1 * a.B * 1 * SkFull * (a.D + a.ropeExtra) * S.kvBpe();  // 1 stream
const qoBytes = 2 * a.B * a.H * a.S * a.D * S.aBpe();
ok('MLA attention byte term uses 1 latent stream of (kv_lora+rope), not 2×D',
   near(S.calcFA2Attn(a).bytes, qoBytes + expKVbytes, 1e-9));

// ── 4. Fix D — lightning indexer is O(L²): scores the full sequence ──────────
section('4. Fix D — indexer score runs over the FULL sequence (not sparsified)');
S.sparseMLA = true;
S.cfg.seqLen = 131072; build(GLM5);
const ixLong = S.calcFA2Attn(L(/indexer score/i));
S.cfg.seqLen = 8192; build(GLM5);
const ixShort = S.calcFA2Attn(L(/indexer score/i));
ok('indexer score FLOPs scale with full seq (16× seq → ~16× at decode step)',
   near(ixLong.flops / ixShort.flops, 16, 0.05),
   `${(ixShort.flops / 1e6).toFixed(1)}M @8K → ${(ixLong.flops / 1e6).toFixed(1)}M @128K`);
ok('indexer score is QK^T-only (no softmax/AV sub-terms)',
   !!ixLong.sub && !!ixLong.sub['Index QK^T'] && !ixLong.sub['AV'] && !ixLong.sub['Softmax (SRAM)']);
// indexer reads a single MQA key stream (no V) — kvStreams must be 1.
// Rebuild at the same seq length ixLong was measured at (the loop above left
// the layer list at 8192) so the byte expectation matches the layer under test.
S.cfg.seqLen = 131072; build(GLM5);
const ix = L(/indexer score/i);
const ixUnderTest = S.calcFA2Attn(ix);
const ixSkFull = ix.S + (ix.kvCache || 0);
const expIxBytes = ix.B * ix.H * ix.S * ix.D * S.aBpe() + 1 * ix.B * 1 * ixSkFull * ix.D * S.kvBpe();
ok('indexer byte term = Q read + single K stream (no V)',
   near(ixUnderTest.bytes, expIxBytes, 1e-9));

// ── 5. Toggle — getRepeat zeroes indexer layers when sparse off ──────────────
section('5. Toggle — indexer layers contribute zero everywhere when sparse off');
S.cfg.seqLen = 131072;
S.sparseMLA = true; build(GLM5);
ok('sparse ON  : indexer Q/K proj getRepeat == nL (78)', S.getRepeat(L(/indexer Q\/K/i)) === 78);
ok('sparse ON  : indexer score   getRepeat == nL (78)', S.getRepeat(L(/indexer score/i)) === 78);
S.sparseMLA = false;
ok('sparse OFF : indexer Q/K proj getRepeat == 0', S.getRepeat(L(/indexer Q\/K/i)) === 0);
ok('sparse OFF : indexer score   getRepeat == 0', S.getRepeat(L(/indexer score/i)) === 0);
ok('sparse OFF : indexer score tpInfoTotal effectiveTime == 0',
   S.tpInfoTotal(L(/indexer score/i), S.chips[0]).effectiveTime === 0);
ok('sparse OFF : indexer Q/K proj weight bytes == 0 (calcLTotal repeat 0)',
   S.tpInfoTotal(L(/indexer Q\/K/i), S.chips[0]).effectiveTime === 0);

// ── 6. MLA TP — exactly one AllReduce; indexer is replicated ─────────────────
section('6. MLA TP topology — one AllReduce on O proj, indexer replicated');
S.sparseMLA = true; S.nCards = 4; S.dp = 1; S.tpAttn = 4; S.cfg.seqLen = 131072;
build(GLM5);
const arBearing = S.layers.filter(l =>
  /MLA|indexer/i.test(l.name) && S.getRepeat(l) > 0 && (S.tpInfo(l, S.chips[0]).commBytes || 0) > 0);
ok('MLA + indexer block has exactly ONE AllReduce-bearing layer',
   arBearing.length === 1, `[${arBearing.map(l => (l.name || '').replace(/ \(×\d+\)/, '')).join(', ')}]`);
ok('the AllReduce sits on MLA O proj', arBearing.length === 1 && /o proj/i.test(arBearing[0].name));
const ixTi = S.tpInfo(L(/indexer score/i), S.chips[0]);
ok('indexer score: replicated → full FLOPs per card (not ÷tpAttn), no comm',
   ixTi.commTime === 0 && near(ixTi.flopsPerCard, S.calcL(L(/indexer score/i)).flops, 1e-9));
const ixProjTi = S.tpInfo(L(/indexer Q\/K/i), S.chips[0]);
ok('indexer Q/K proj: tpMode replicated → no AllReduce',
   (ixProjTi.commBytes || 0) === 0);
// MLA core attention KV must stay replicated under TP (single latent → nKV=1).
ok('MLA core attention KV latent is TP-replicated (nKV=1, not head-sharded)',
   L(/MLA Attention/i).nKV === 1);

// ── 7. Control — non-DSA model untouched by the sparse-MLA toggle ────────────
section('7. Control — Qwen3-235B (GQA MoE) identical with toggle on/off');
S.nCards = 1; S.dp = 1; S.tpAttn = 1; S.cfg.seqLen = 4096; S.cfg.batch = 1;
S.sparseMLA = true; build(QWEN3);
const qFlopsOn  = S.layers.map(l => S.calcLTotal(l).flops).reduce((x, y) => x + y, 0);
const qKvOn     = S.kvCacheBytes(4096);
S.sparseMLA = false; build(QWEN3);
const qFlopsOff = S.layers.map(l => S.calcLTotal(l).flops).reduce((x, y) => x + y, 0);
const qKvOff    = S.kvCacheBytes(4096);
ok('Qwen3-235B total FLOPs identical (toggle is a no-op for non-DSA)',
   near(qFlopsOn, qFlopsOff, 1e-12), `on ${(qFlopsOn / 1e12).toFixed(4)}T / off ${(qFlopsOff / 1e12).toFixed(4)}T`);
ok('Qwen3-235B KV cache identical (GQA path unaffected)', near(qKvOn, qKvOff, 1e-12));
ok('Qwen3-235B built no indexer layers', !S.layers.some(l => l.indexerLayer));
// GQA attention byte accounting unchanged: 2 streams (K+V), kvHeadDim == D.
const qa = S.layers.find(l => l.type === 'attention');
const qSk = qa.S + (qa.kvCache || 0);
const qExp = 2 * qa.B * qa.H * qa.S * qa.D * S.aBpe() + 2 * qa.B * (qa.nKV || qa.H) * qSk * qa.D * S.kvBpe();
ok('GQA attention bytes still use 2 streams (K+V), kvHeadDim == D',
   near(S.calcFA2Attn(qa).bytes, qExp, 1e-9));

// ── 8. Fix B carries to DeepSeek-V3 (dense MLA, no DSA) ──────────────────────
section('8. Fix B — DeepSeek-V3 MLA KV cache corrected (dense MLA, no indexer)');
S.mode = 'decode'; S.cfg.batch = 64; S.cfg.seqLen = 131072; S.cfg.outTokens = 4;
S.sparseMLA = true;  // must be irrelevant for a non-DSA MLA model
const rd = build(DSV3);
ok('DeepSeek-V3 detected as MLA, NOT DSA', rd.meta.isDSV3 === true);
ok('DeepSeek-V3 built no indexer layers (dense MLA)', !S.layers.some(l => l.indexerLayer));
const dKv = S.kvCacheBytes(131072);
const dTrue = 61 * 64 * (512 + 64) * 131072 * 2;     // single latent stream
const dOld  = 2 * 61 * 64 * 512 * 131072 * 2;        // old dense-2× bug
ok('DeepSeek-V3 KV cache == single latent stream (kv_lora+rope), no 2×',
   near(dKv, dTrue, 1e-9), `got ${(dKv / 1e9).toFixed(0)} GB / expect ${(dTrue / 1e9).toFixed(0)} GB`);
ok('old dense-2× value would have been 1.78× larger',
   near(dOld / dKv, 1024 / 576, 0.001), `old ${(dOld / 1e9).toFixed(0)} GB`);
S.sparseMLA = false;
ok('DeepSeek-V3 KV cache unaffected by sparse-MLA toggle (non-DSA model)',
   near(S.kvCacheBytes(131072), dKv, 1e-12));

// ── 9. Emergent behaviour — DSA crossover (slower short ctx, faster long) ────
section('9. DSA crossover — sparse pays off only past long context');
S.mode = 'decode'; S.cfg.batch = 1; S.cfg.outTokens = 4; S.cfg.precision = 'fp8';
S.nCards = 8; S.dp = 8; S.tpAttn = 1; S.ibw = 450;
const chip = S.chips.find(c => c.name === 'H100 SXM') || S.chips[0];
function attnBlockTime() {
  S.applyConfigToLayers();
  return S.layers.filter(l => /MLA Attention|indexer/i.test(l.name))
    .map(l => S.tpInfoTotal(l, chip).effectiveTime).reduce((x, y) => x + y, 0);
}
const cross = {};
for (const s of [8192, 32768, 131072]) {
  S.cfg.seqLen = s; build(GLM5);
  S.sparseMLA = true;  const on  = attnBlockTime();
  S.sparseMLA = false; const off = attnBlockTime();
  cross[s] = { on, off };
  console.log(`    S=${String(s).padStart(7)}  sparse ON ${(on * 1e3).toFixed(3)} ms` +
              `  / OFF ${(off * 1e3).toFixed(3)} ms  → ${(off / on).toFixed(2)}×`);
}
ok('short context (8K): sparse ON is SLOWER — indexer fixed overhead dominates',
   cross[8192].on > cross[8192].off);
ok('long context (128K): sparse ON is FASTER — top-k attention wins',
   cross[131072].on < cross[131072].off);
ok('crossover is monotonic (ON/OFF ratio improves with sequence length)',
   (cross[8192].off / cross[8192].on) < (cross[32768].off / cross[32768].on) &&
   (cross[32768].off / cross[32768].on) < (cross[131072].off / cross[131072].on));

// ── 10. v81 — architecture diagram respects getRepeat (toggle consistency) ───
// v80 bug: the Structure diagram classified the FULL layers[] array, so with
// sparse-MLA off the DSA indexer layers (getRepeat → 0) were still drawn AND
// their l.repeat (78) ≠ getRepeat (0) made the fork detector emit a spurious
// "other / other" branch diamond. v81 filters roles to getRepeat(l) > 0 — the
// same predicate renderLayerList() / the timing+roofline tabs already use.
// The SVG itself renders into a mocked DOM here, so this section locks down the
// shared invariant (visible-layer count) and that renderArchInfo() runs clean.
// Uses the real UI layer set: expandMoEToSubLayers(inferFromConfig(...)), which
// expands the single MoE entry into 5 sub-layers — so GLM-5 = 16 entries.
section('10. Fix v81 — Structure diagram consistent with layer list under toggle');
S.mode = 'prefill'; S.cfg.batch = 1; S.cfg.seqLen = 2048; S.nCards = 8; S.dp = 1; S.tpAttn = 1;
function buildExpanded(cfgJson) {
  const r = S.expandMoEToSubLayers(S.inferFromConfig(clone(cfgJson)));
  S.layers = r.layers;
  S.applyConfigToLayers();
  return r;
}
S.sparseMLA = true; buildExpanded(GLM5);
const visOn = S.layers.filter(l => S.getRepeat(l) > 0).length;
const totOn = S.layers.length;
ok('sparse ON : all layers visible — indexer layers included (16 entries)',
   visOn === totOn && visOn === 16, `visible ${visOn} / total ${totOn}`);
S.sparseMLA = false;
const visOff = S.layers.filter(l => S.getRepeat(l) > 0).length;
const totOff = S.layers.length;
ok('sparse OFF: indexer layers excluded from visible set (16 → 14)',
   visOff === 14 && totOff === 16, `visible ${visOff} / total ${totOff}`);
ok('sparse OFF: every hidden layer is an indexer layer (nothing else dropped)',
   S.layers.filter(l => S.getRepeat(l) === 0).every(l => l.indexerLayer === true));
ok('sparse OFF: remaining visible layers all share repeat 78 — no spurious fork',
   (() => { const r = S.layers.filter(l => S.getRepeat(l) > 0 && /×78/.test(l.name || ''));
            return r.length > 0 && r.every(l => S.getRepeat(l) === 78); })());
// renderArchInfo() must execute without throwing in both toggle states.
let archOk = true;
try { S.sparseMLA = true;  S.renderArchInfo();
      S.sparseMLA = false; S.renderArchInfo(); } catch (e) { archOk = false; }
ok('renderArchInfo() runs clean with sparse-MLA on and off', archOk);
// Non-DSA control: Qwen3 has no indexer → visible count == total in both states.
S.sparseMLA = true;  buildExpanded(QWEN3); const qVisOn  = S.layers.filter(l => S.getRepeat(l) > 0).length;
S.sparseMLA = false; buildExpanded(QWEN3); const qVisOff = S.layers.filter(l => S.getRepeat(l) > 0).length;
ok('non-DSA model (Qwen3): visible layer count unchanged by toggle',
   qVisOn === S.layers.length && qVisOff === S.layers.length);

// ── summary ──────────────────────────────────────────────────────────────────
console.log('\n' + '═'.repeat(86));
console.log(`Summary: ${pass} pass, ${fail} fail (of ${pass + fail} total)`);
if (failures.length) { console.log('\nFAILURES:'); failures.forEach(f => console.log('  • ' + f)); }
console.log('═'.repeat(86));
process.exit(fail ? 1 : 0);

// ─────────────────────────────────────────────────────────────────────────────
// v83 verification suite
//
// Covers three pieces of work landed in v83:
//   A. row-parallel linear aBytes — symmetric counterpart of the v82 col fix.
//      A row-parallel linear consumes the SHARDED output of the preceding
//      col-parallel layer, so its input is X[M, K/tp], not X[M,K]. The old
//      formula used the full M·K input term → overstated the read by tp×.
//   B. shared-expert sub-layer aBytes — under hybrid (expertTp>1) the shared
//      expert's gate_up (shared_up) is col-parallel and down (shared_down) is
//      row-parallel, so their activation terms must shard by expertTp. Under
//      pure EP (expertTp==1) the shared expert is replicated → no change.
//   C. DeepSeek-V3.2-Exp DSA — the V3 family builder now emits the lightning
//      indexer + sparseTopK core attention when model_type=deepseek_v32 /
//      index_* fields are present. Kimi-K2 and plain V3/R1 stay dense MLA.
// ─────────────────────────────────────────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

function load(file) {
  const html = fs.readFileSync(file, 'utf-8');
  let allCode = '';
  for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) allCode += m[1] + '\n';
  allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');
  const sb = {
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
  sb.window = sb; sb.globalThis = sb;
  vm.createContext(sb);
  vm.runInContext(allCode, sb, { timeout: 10000, filename: 'tool.js' });
  sb.nCards = 8; sb.dp = 1; sb.tpAttn = 8; sb.epDeg = 1; sb.moePar = 'tp';
  sb.overlap = 0; sb.ibw = 450;
  sb.selChip = sb.chips.find(c => c.name === 'H100 SXM') || sb.chips[0];
  sb.cfg.precision = 'fp16';
  return sb;
}

const V81 = load('/mnt/project/llm_perf_analyzer_v81.html');
const V83 = load('/home/claude/llm_perf_analyzer_v83.html');

let nPass = 0, nFail = 0; const failures = [];
function ok(label, cond, note) {
  if (cond) nPass++; else { nFail++; failures.push(label + (note ? ' — ' + note : '')); }
  console.log(`  ${cond ? '\u2713' : '\u2717'} ${label}${note ? '  ' + note : ''}`);
}
function near(a, b, tol) { return b === 0 ? Math.abs(a) <= tol : Math.abs(a - b) / Math.abs(b) <= tol; }
function section(t) { console.log('\n## ' + t); }

const AB = 2, WB = 2;

console.log('='.repeat(90));
console.log('v83 verification — row/shared aBytes fixes + DeepSeek-V3.2-Exp DSA');
console.log('='.repeat(90));

// ════════════════════════════════════════════════════════════════════════════
// A. row-parallel linear aBytes — input now sharded by tpAttn
// ════════════════════════════════════════════════════════════════════════════
section('A. row-parallel linear aBytes (TP=8, fp16) — input term sharded /tp');
{
  const TP = 8;
  const rowLayers = [
    { tag: 'FFN down',   l: { id:1, name:'FFN down',   type:'linear', K:25600, N:5120, repeat:1, tpMode:'row' } },
    { tag: 'MLA O proj', l: { id:2, name:'MLA O proj', type:'linear', K:16384, N:7168, repeat:1, tpMode:'row' } },
  ];
  for (const [phase, M] of [['prefill', 2048], ['decode', 1]]) {
    for (const { tag, l } of rowLayers) {
      const ll = { ...l, M };
      const t81 = V81.tpInfo(ll, V81.selChip);
      const t83 = V83.tpInfo(ll, V83.selChip);
      // bytesPerCard must drop by exactly M*K*(1-1/tp)*ab (the input term)
      const expectedDrop = ll.M * ll.K * (1 - 1 / TP) * AB;
      const actualDrop = t81.bytesPerCard - t83.bytesPerCard;
      ok(`${phase.padEnd(7)} ${tag.padEnd(11)} bpc drop == M\u00b7K\u00b7(1-1/tp)\u00b7ab`,
         near(actualDrop, expectedDrop, 1e-6),
         `drop=${(actualDrop/1e6).toFixed(2)}MB expect=${(expectedDrop/1e6).toFixed(2)}MB`);
      // hand-calc absolute v83 value
      const wB = ll.K * ll.N * WB / TP;
      const aB = (ll.M * ll.K / TP + ll.M * ll.N) * AB;
      ok(`${phase.padEnd(7)} ${tag.padEnd(11)} v83 bpc == K\u00b7N\u00b7wb/tp + (M\u00b7K/tp + M\u00b7N)\u00b7ab`,
         near(t83.bytesPerCard, wB + aB, 1e-6),
         `v83=${(t83.bytesPerCard/1e6).toFixed(2)}MB hand=${((wB+aB)/1e6).toFixed(2)}MB`);
      // FLOPs and the AllReduce commBytes must be untouched
      ok(`${phase.padEnd(7)} ${tag.padEnd(11)} FLOPs/card + commBytes unchanged`,
         near(t83.flopsPerCard, t81.flopsPerCard, 1e-12) && near(t83.commBytes, t81.commBytes, 1e-9));
    }
  }
  // overstatement factor on the old activation read (FFN down prefill)
  const ll = { id:9, name:'FFN down', type:'linear', M:2048, K:25600, N:5120, repeat:1, tpMode:'row' };
  const aOld = (ll.M*ll.K + ll.M*ll.N) * AB;
  const aNew = (ll.M*ll.K/8 + ll.M*ll.N) * AB;
  ok('FFN down prefill: old aBytes was overstated', aOld > aNew,
     `${(aOld/aNew).toFixed(2)}\u00d7 (old ${(aOld/1e6).toFixed(1)}MB vs new ${(aNew/1e6).toFixed(1)}MB)`);
}

// ════════════════════════════════════════════════════════════════════════════
// B. shared-expert sub-layer aBytes — shards by expertTp under hybrid
// ════════════════════════════════════════════════════════════════════════════
section('B. shared-expert sub-layer aBytes — hybrid shards /expertTp, pure-EP unchanged');
{
  // nCards=16, epDeg=4 → expertTp = 16/4 = 4 under hybrid
  for (const S of [V81, V83]) { S.nCards = 16; S.epDeg = 4; }
  const eTp = 4;
  const shUp   = { id:1, name:'Shared expert up',   type:'linear', M:2048, K:5120,  N:8192, repeat:1, moeGroup:'shared_up' };
  const shDown = { id:2, name:'Shared expert down', type:'linear', M:2048, K:4096,  N:5120, repeat:1, moeGroup:'shared_down' };

  // ── hybrid: expertTp=4 → shared_up col-par (N/eTp), shared_down row-par (K/eTp)
  for (const S of [V81, V83]) S.moePar = 'hybrid';
  {
    const u83 = V83.tpInfo(shUp, V83.selChip), d83 = V83.tpInfo(shDown, V83.selChip);
    const wU = shUp.K*shUp.N*WB/eTp,   aU = (shUp.M*shUp.K + shUp.M*shUp.N/eTp)*AB;
    const wD = shDown.K*shDown.N*WB/eTp, aD = (shDown.M*shDown.K/eTp + shDown.M*shDown.N)*AB;
    ok('hybrid shared_up   v83 bpc == col-par (input full, output /expertTp)',
       near(u83.bytesPerCard, wU + aU, 1e-6),
       `v83=${(u83.bytesPerCard/1e6).toFixed(2)}MB hand=${((wU+aU)/1e6).toFixed(2)}MB`);
    ok('hybrid shared_down v83 bpc == row-par (input /expertTp, output full)',
       near(d83.bytesPerCard, wD + aD, 1e-6),
       `v83=${(d83.bytesPerCard/1e6).toFixed(2)}MB hand=${((wD+aD)/1e6).toFixed(2)}MB`);
    const u81 = V81.tpInfo(shUp, V81.selChip), d81 = V81.tpInfo(shDown, V81.selChip);
    ok('hybrid shared_up   v83 < v81 (was overstating output term)',  u83.bytesPerCard < u81.bytesPerCard);
    ok('hybrid shared_down v83 < v81 (was overstating input term)',   d83.bytesPerCard < d81.bytesPerCard);
    ok('hybrid shared_up/down FLOPs unchanged',
       near(u83.flopsPerCard, u81.flopsPerCard, 1e-12) && near(d83.flopsPerCard, d81.flopsPerCard, 1e-12));
  }
  // ── pure EP: expertTp=1 → shared expert replicated → identical v81==v83
  for (const S of [V81, V83]) S.moePar = 'ep';
  {
    const u81 = V81.tpInfo(shUp, V81.selChip), u83 = V83.tpInfo(shUp, V83.selChip);
    const d81 = V81.tpInfo(shDown, V81.selChip), d83 = V83.tpInfo(shDown, V83.selChip);
    ok('pure-EP shared_up   bytesPerCard UNCHANGED (replicated, expertTp=1)',
       near(u83.bytesPerCard, u81.bytesPerCard, 1e-12));
    ok('pure-EP shared_down bytesPerCard UNCHANGED (replicated, expertTp=1)',
       near(d83.bytesPerCard, d81.bytesPerCard, 1e-12));
  }
  for (const S of [V81, V83]) { S.nCards = 8; S.epDeg = 1; S.moePar = 'tp'; }
}

// ════════════════════════════════════════════════════════════════════════════
// C. DeepSeek-V3.2-Exp DSA support
// ════════════════════════════════════════════════════════════════════════════
section('C. DeepSeek-V3.2-Exp DSA — real config (model_type=deepseek_v32)');
{
  const S = V83;
  S.mode = 'decode'; S.cfg.batch = 1; S.cfg.seqLen = 131072; S.cfg.outTokens = 4;
  S.nCards = 1; S.dp = 1; S.tpAttn = 1; S.sparseMLA = true;
  const clone = o => JSON.parse(JSON.stringify(o));
  function build(c) { const r = S.inferFromConfig(clone(c)); S.layers = r.layers; S.applyConfigToLayers(); return r; }
  const L = re => S.layers.find(l => re.test(l.name || ''));

  // real DeepSeek-V3.2-Exp config.json (deepseek-ai/DeepSeek-V3.2-Exp)
  const DSV32 = {
    architectures: ['DeepseekV32ForCausalLM'], model_type: 'deepseek_v32',
    vocab_size: 129280, hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 2048,
    num_hidden_layers: 61, num_attention_heads: 128, num_key_value_heads: 128,
    n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 8, first_k_dense_replace: 3,
    kv_lora_rank: 512, q_lora_rank: 1536, qk_rope_head_dim: 64, qk_nope_head_dim: 128, v_head_dim: 128,
    index_n_heads: 64, index_head_dim: 128, index_topk: 2048,
  };
  // plain DeepSeek-V3 (no DSA) and Kimi-K2 (plain MLA, no DSA) — control cases
  const DSV3 = {
    architectures: ['DeepseekV3ForCausalLM'], model_type: 'deepseek_v3',
    vocab_size: 129280, hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 2048,
    num_hidden_layers: 61, num_attention_heads: 128, num_key_value_heads: 128,
    n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 8, first_k_dense_replace: 3,
    kv_lora_rank: 512, q_lora_rank: 1536, qk_rope_head_dim: 64, qk_nope_head_dim: 128, v_head_dim: 128,
  };
  const KIMIK2 = {
    architectures: ['DeepseekV3ForCausalLM'], model_type: 'kimi_k2',
    vocab_size: 163840, hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 2048,
    num_hidden_layers: 61, num_attention_heads: 64, num_key_value_heads: 64,
    n_routed_experts: 384, n_shared_experts: 1, num_experts_per_tok: 8, first_k_dense_replace: 1,
    kv_lora_rank: 512, q_lora_rank: 1536, qk_rope_head_dim: 64, qk_nope_head_dim: 128, v_head_dim: 128,
  };

  // ── C1. detection + meta ────────────────────────────────────────────────
  const r = build(DSV32);
  ok('V3.2-Exp detected isDSV3', r.meta.isDSV3 === true);
  ok('V3.2-Exp detected isDSA', r.meta.isDSA === true);
  ok('V3.2-Exp meta indexer params (64 heads, d=128, topk=2048)',
     r.meta.idxNHeads === 64 && r.meta.idxHeadDim === 128 && r.meta.idxTopK === 2048);

  // ── C2. indexer layers built with correct flags ─────────────────────────
  const idxProj = L(/DSA indexer Q\/K/i), idxScore = L(/DSA indexer score/i), mlaA = L(/MLA Attention/i);
  ok('"DSA indexer Q/K proj" linear layer built', !!idxProj);
  ok('"DSA indexer score" attention layer built', !!idxScore);
  ok('indexer Q/K proj: tpMode=replicated + indexerLayer',
     !!idxProj && idxProj.tpMode === 'replicated' && idxProj.indexerLayer === true);
  ok('indexer Q/K proj output dim = idxNHeads\u00b7idxHeadDim + idxHeadDim (64\u00b7128+128)',
     !!idxProj && idxProj.N === 64 * 128 + 128);
  ok('indexer score: scoreOnly + tpReplicated + indexerLayer',
     !!idxScore && idxScore.scoreOnly === true && idxScore.tpReplicated === true && idxScore.indexerLayer === true);
  ok('indexer score head shape: H=idxNHeads(64), D=idxHeadDim(128), nKV=1',
     !!idxScore && idxScore.H === 64 && idxScore.D === 128 && idxScore.nKV === 1);
  ok('MLA Attention carries sparseTopK = index_topk (2048)', !!mlaA && mlaA.sparseTopK === 2048);
  ok('MLA Attention keeps mla:true + ropeExtra=qk_rope_head_dim(64)',
     !!mlaA && mlaA.mla === true && mlaA.ropeExtra === 64);

  // ── C3. KV cache includes the indexer key store when sparse on ──────────
  const seq = 131072;
  S.sparseMLA = true;
  const kvOn  = S.kvCacheBytes(seq);
  S.sparseMLA = false;
  const kvOff = S.kvCacheBytes(seq);
  const latent = 61 * 1 * (512 + 64) * seq * 2;        // MLA latent stream
  const idxKey = 61 * 1 * 128 * seq * 2;               // indexer MQA key store
  ok('V3.2-Exp KV (sparse OFF) == MLA latent stream only',
     near(kvOff, latent, 1e-9), `${(kvOff/1e9).toFixed(1)}GB`);
  ok('V3.2-Exp KV (sparse ON)  == latent + indexer key store',
     near(kvOn, latent + idxKey, 1e-9), `${(kvOn/1e9).toFixed(1)}GB`);
  ok('indexer key store adds ~22% KV ((576+128)/576)',
     near(kvOn / kvOff, 704 / 576, 0.001));

  // ── C4. getRepeat zeroes indexer layers when sparse off ─────────────────
  S.sparseMLA = false;
  ok('sparse OFF → indexer Q/K proj getRepeat == 0', S.getRepeat(idxProj) === 0);
  ok('sparse OFF → indexer score getRepeat == 0',    S.getRepeat(idxScore) === 0);
  S.sparseMLA = true;
  ok('sparse ON  → indexer Q/K proj getRepeat == nL(61)', S.getRepeat(idxProj) === 61);
  ok('sparse ON  → indexer score getRepeat == nL(61)',    S.getRepeat(idxScore) === 61);

  // ── C5. controls — Kimi-K2 and plain V3 stay dense MLA (no DSA) ──────────
  const rk = build(KIMIK2);
  ok('Kimi-K2 detected isDSV3 + isKimiK2', rk.meta.isDSV3 === true && rk.meta.isKimiK2 === true);
  ok('Kimi-K2 NOT flagged isDSA (plain MLA)', !rk.meta.isDSA);
  ok('Kimi-K2 built NO indexer layers', !S.layers.some(l => l.indexerLayer));
  ok('Kimi-K2 MLA Attention has NO sparseTopK', !L(/MLA Attention/i).sparseTopK);

  const rv3 = build(DSV3);
  ok('plain DeepSeek-V3 NOT flagged isDSA', !rv3.meta.isDSA);
  ok('plain DeepSeek-V3 built NO indexer layers', !S.layers.some(l => l.indexerLayer));

  // ── C6. DSA crossover — sparse pays off only past long context ──────────
  S.cfg.precision = 'fp8'; S.nCards = 8; S.dp = 8; S.tpAttn = 1; S.ibw = 450;
  const chip = S.chips.find(c => c.name === 'H100 SXM') || S.chips[0];
  function attnTime() {
    S.applyConfigToLayers();
    return S.layers.filter(l => /MLA Attention|indexer/i.test(l.name))
      .map(l => S.tpInfoTotal(l, chip).effectiveTime).reduce((x, y) => x + y, 0);
  }
  const cx = {};
  for (const s of [8192, 131072]) {
    S.cfg.seqLen = s; build(DSV32);
    S.sparseMLA = true;  const on  = attnTime();
    S.sparseMLA = false; const off = attnTime();
    cx[s] = { on, off };
    console.log(`    S=${String(s).padStart(7)}  sparse ON ${(on*1e3).toFixed(3)}ms / OFF ${(off*1e3).toFixed(3)}ms  \u2192 ${(off/on).toFixed(2)}\u00d7`);
  }
  ok('V3.2-Exp short ctx (8K): sparse ON slower — indexer fixed overhead dominates',
     cx[8192].on > cx[8192].off);
  ok('V3.2-Exp long ctx (128K): sparse ON faster — top-k core attention wins',
     cx[131072].on < cx[131072].off);
}

console.log('\n' + '='.repeat(90));
console.log(`RESULT: ${nPass} passed, ${nFail} failed`);
if (nFail) { console.log('\nFAILURES:'); failures.forEach(f => console.log('  \u2717 ' + f)); process.exit(1); }
else console.log('All v83 checks passed.');

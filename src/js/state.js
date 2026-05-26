// ── State ─────────────────────────────────────────────────────────────────────
// ibw = interconnect bandwidth per card, **unidirectional** GB/s.
// Spec sheets usually quote bidirectional aggregate (2× ibw).
//   BW1100 : 448 GB/s bi → 224 uni
//   A100   : 600 GB/s bi (NVLink 3) → 300 uni
//   H100   : 900 GB/s bi (NVLink 4) → 450 uni
//   H200   : 900 GB/s bi (NVLink 4, same as H100) → 450 uni
//   H20    : 900 GB/s bi (same NVLink 4 as H100) → 450 uni
//   4090   : PCIe 4.0 x16 ≈ 32 GB/s bi → 16 uni (no NVLink)
// H200 = H100 GPU (same Hopper compute: 989/1979 TFLOPS BF16/FP8) paired with
//        141 GB HBM3e at 4.8 TB/s — a bandwidth+capacity bump over H100 SXM.
let chips = [
  {id:4, name:'BW1100',    bw:1800, tflops:340,  tflops_fp8:680,  mem:144, ibw:224},
  {id:1, name:'A100 SXM',  bw:2000, tflops:312,  tflops_fp8:624,  mem:80,  ibw:300},
  {id:2, name:'H100 SXM',  bw:3350, tflops:989,  tflops_fp8:1979, mem:80,  ibw:450},
  {id:6, name:'H200 SXM',  bw:4800, tflops:989,  tflops_fp8:1979, mem:141, ibw:450},
  {id:3, name:'H20',       bw:4000, tflops:148,  tflops_fp8:296,  mem:96,  ibw:450},
  {id:5, name:'RTX 4090',  bw:1008, tflops:330,  tflops_fp8:660,  mem:24,  ibw:16 },
];
// Default: Qwen3-32B (dense)
// hidden=5120, nL=64, nH=64, nKV=8, hd=128, ffn=25600, vocab=151936, tie_embed=false
let layers = [
  {id:1, name:'Token embed',         type:'embedding', M:2048, K:151936, N:5120,   repeat:1},
  {id:2, name:'Attn QKV (×64)',       type:'linear',    M:2048, K:5120,   N:12288,  repeat:64},  // 64Q+8KV+8KV×128=12288
  {id:3, name:'Attention (×64)',      type:'attention', B:1, S:2048, H:64, nKV:8, D:128, kvCache:0, fa2:true, repeat:64},
  {id:4, name:'Attn O proj (×64)',    type:'linear',    M:2048, K:8192,   N:5120,   repeat:64},
  {id:5, name:'FFN gate+up (×64)',    type:'linear',    M:2048, K:5120,   N:51200,  repeat:64},
  {id:6, name:'FFN down (×64)',       type:'linear',    M:2048, K:25600,  N:5120,   repeat:64},
  {id:7, name:'LM head',             type:'linear',    M:2048, K:5120,   N:151936, repeat:1},
];
let currentModelName = 'Qwen3-32B';  // shown in layers section header
let selChip   = chips[0];  // BW1100 is default
let activeTab = 'roofline';
let charts    = {};
let editId    = null;
let nCards        = 1;
let overlap   = 0;
let ibw       = 224;  // synced from selChip.ibw (BW1100 default: 224 GB/s unidirectional)
let mode      = 'prefill';
// v80: sparse-MLA toggle. When ON (default) DSA models (GLM-5 / glm_moe_dsa)
// model the lightning indexer + top-k restricted core attention. When OFF the
// core attention runs over the full sequence and the indexer layers are
// dropped (repeat 0) — i.e. a conservative "dense MLA" compute upper bound.
// It governs COMPUTE only; MLA latent KV-cache VRAM is unaffected by it.
let sparseMLA = true;
// cfg mirrors Qwen3-32B defaults
// H = nKV heads (for KV cache bytes), nH = query heads (for attention FLOPs)
let cfg = {batch:1, seqLen:2048, outTokens:256, nH:64, H:8, D:128, nL:64, precision:'fp16'};

// ── Precision helpers ────────────────────────────────────────────────────────
// FP8 inference has two distinct flavours with very different perf characteristics:
//
//   W8A16: weights stored as FP8 (1B), dequantized to BF16 in SRAM before GEMM.
//     → Weight HBM reads halved (bandwidth saving), but compute still uses
//       FP16 tensor cores → SAME TFLOPS as BF16. No change to activations or KV.
//     → Main benefit: VRAM ÷2 for weights, and memory-bound layers run ~2× faster
//       (half the bytes to stream), but compute-bound layers see NO speedup.
//
//   W8A8: both weights AND activation inputs quantized to FP8 for GEMM.
//     → Uses FP8 tensor cores → 2× TFLOPS vs FP16.
//     → Weight HBM reads halved (same as W8A16).
//     → Activation bytes in HBM remain BF16 (quant/dequant happens on-chip).
//     → Benefits BOTH memory-bound (bandwidth) AND compute-bound (TFLOPS) layers.
//     → KV cache can optionally also be FP8 for additional VRAM savings.
//
// Attention layers (QK^T, softmax, AV) always operate on BF16 activations and
// have no weight matrices, so they always run at FP16 TFLOPS regardless of mode.
//
// cfg.precision values:
//   'fp16'    — BF16/FP16 baseline (default)
//   'fp8w'    — W8A16: weight 1B, GEMM at FP16 TFLOPS
//   'fp8'     — W8A8:  weight 1B, GEMM at FP8 TFLOPS (2×)
//   'fp8kv'   — W8A8 + FP8 KV cache
function wBpe()  { return cfg.precision === 'fp16' ? 2 : 1; }   // weight bytes per element
function aBpe()  { return 2; }                                    // activation bytes (always BF16 in HBM)
function kvBpe() { return cfg.precision === 'fp8kv' ? 1 : 2; }   // KV cache bytes per element
// MoE EP dispatch all-to-all byte width. DeepEP-style implementations send the
// dispatch a2a in FP8 (token hidden vectors are quantised before scatter) but
// keep the COMBINE a2a in BF16 — combine does weighted summation of expert
// outputs, where FP8 accumulation loses too much precision. So under W8A8 modes
// dispatch is 1B/elem while combine stays at aBpe()=2B/elem. Under BF16/W8A16
// both legs are 2B (symmetric). This only affects a2a comm volume, not HBM
// activation bytes (which are always BF16, see aBpe).
function dispatchBpe() { return (cfg.precision === 'fp8' || cfg.precision === 'fp8kv') ? 1 : 2; }
function precisionLabel() {
  if (cfg.precision === 'fp8w')  return 'W8A16';
  if (cfg.precision === 'fp8')   return 'W8A8';
  if (cfg.precision === 'fp8kv') return 'W8A8+KV8';
  return 'BF16';
}
// Effective TFLOPS: only W8A8 modes use FP8 tensor cores, and only for GEMM layers.
// W8A16 keeps FP16 TFLOPS (dequant in SRAM, GEMM on BF16).
// Attention layers always use FP16 TFLOPS (no weight GEMM — activation-only ops).
function effectiveTflops(chip, layerType) {
  if ((cfg.precision === 'fp8' || cfg.precision === 'fp8kv')
      && layerType !== 'attention' && layerType !== 'linear_attn') {
    return chip.tflops_fp8 || chip.tflops * 2;
  }
  return chip.tflops;
}

// ── Repeat multiplier ─────────────────────────────────────────────────────────
// Each layer has a `repeat` field (default 1). Total time = per-layer time × repeat.
// Returns 0 for repeat=0 so zero-repeat entries contribute nothing to timing or counts.
function getRepeat(l) {
  // v80: DSA lightning-indexer layers exist only when sparse-MLA modelling is
  // enabled. With it off they contribute zero FLOPs / bytes / VRAM everywhere
  // (renderLayerList already hides repeat==0 rows).
  if (l.indexerLayer && !sparseMLA) return 0;
  const r = parseInt(l.repeat); return (isNaN(r) || r < 0) ? 1 : r;
}

// ── Mode & config ─────────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.getElementById('prefillBtn').className = 'mode-btn'+(m==='prefill'?' active':'');
  document.getElementById('decodeBtn').className  = 'mode-btn'+(m==='decode' ?' active':'');
  document.getElementById('configHdr').textContent = m==='prefill' ? 'Prefill config' : 'Decode config';
  document.getElementById('prefillFields').style.display = m==='prefill' ? '' : 'none';
  document.getElementById('decodeFields').style.display  = m==='decode'  ? '' : 'none';
  // Sync values between prefill and decode inputs when switching
  if (m==='decode') {
    document.getElementById('cfgBatchD').value = document.getElementById('cfgBatch').value;
    document.getElementById('cfgSeqD').value   = document.getElementById('cfgSeq').value;
    document.getElementById('cfgGlobalBatchD').value = document.getElementById('cfgGlobalBatch').value;
  } else {
    document.getElementById('cfgBatch').value = document.getElementById('cfgBatchD').value;
    document.getElementById('cfgSeq').value   = document.getElementById('cfgSeqD').value;
    document.getElementById('cfgGlobalBatch').value = document.getElementById('cfgGlobalBatchD').value;
  }
  document.getElementById('modeTag').textContent   = m==='prefill' ? 'PREFILL' : 'DECODE';
  document.getElementById('modeTag').style.background = m==='prefill' ? '#EEEDFE' : '#E1F5EE';
  document.getElementById('modeTag').style.color   = m==='prefill' ? '#3C3489' : '#085041';
  onCfgChange();
}

function onCfgChange() {
  if (mode === 'decode') {
    cfg.batch   = Math.max(1, parseInt(document.getElementById('cfgBatchD').value)||1);
    cfg.seqLen  = Math.max(1, parseInt(document.getElementById('cfgSeqD').value)||2048);
  } else {
    cfg.batch   = Math.max(1, parseInt(document.getElementById('cfgBatch').value)||1);
    cfg.seqLen  = Math.max(1, parseInt(document.getElementById('cfgSeq').value)||2048);
  }
  cfg.outTokens = Math.max(1, parseInt(document.getElementById('cfgOut').value)||256);
  // nH, H, D, nL are model-architecture params — not read from runtime config UI
  syncGlobalBatchDisplay();
  applyConfigToLayers();
  updateCfgInfo();
  renderArchInfo();
  renderLayerList();
  refreshMoePeakToggle();
  renderTab();
}

// Handler: user edited the PER-RANK batch field directly.
//   - Read per-rank value, update cfg.batch.
//   - Refresh the Global batch field display = batch × dp (overwrites any stale
//     user-typed global value).
//   - Trigger recalc.
function onPerRankBatchChange() {
  // Mirror to whichever batch field corresponds to the OTHER mode for consistency,
  // then fall through to onCfgChange which reads from the active mode's input.
  const activeId = mode === 'decode' ? 'cfgBatchD' : 'cfgBatch';
  const otherId  = mode === 'decode' ? 'cfgBatch'  : 'cfgBatchD';
  const v = Math.max(1, parseInt(document.getElementById(activeId).value)||1);
  document.getElementById(otherId).value = v;
  // Pre-clear typed global so syncGlobalBatchDisplay rewrites it from cfg.batch*dp
  // (otherwise a stale typed value could trigger the "ceil note" branch).
  const gP = document.getElementById('cfgGlobalBatch');
  const gD = document.getElementById('cfgGlobalBatchD');
  if (gP) gP.value = '';
  if (gD) gD.value = '';
  onCfgChange();
}

// Handler: user edited the GLOBAL batch field directly.
//   - Read global, divide by dp with ceiling (barrel-effect: slowest rank dominates).
//   - Write per-rank = ceil(global / dp) back to the per-rank field.
//   - cfg.batch picks up the per-rank value via onCfgChange.
//   - Note: actual computation uses per-rank × dp, which may exceed the user's
//     typed global when global is not divisible by dp. A "Note" badge shows
//     the rounding so the user understands the gap.
function onGlobalBatchChange() {
  const gActiveId = mode === 'decode' ? 'cfgGlobalBatchD' : 'cfgGlobalBatch';
  const gOtherId  = mode === 'decode' ? 'cfgGlobalBatch'  : 'cfgGlobalBatchD';
  const prActiveId = mode === 'decode' ? 'cfgBatchD' : 'cfgBatch';
  const prOtherId  = mode === 'decode' ? 'cfgBatch'  : 'cfgBatchD';
  const g = Math.max(1, parseInt(document.getElementById(gActiveId).value)||1);
  // Use dp from the parallelism state (may be 1 if DP not configured)
  const D = Math.max(1, (typeof dp === 'number' && dp >= 1) ? dp : 1);
  const perRank = Math.ceil(g / D);
  document.getElementById(prActiveId).value = perRank;
  document.getElementById(prOtherId).value  = perRank;
  // Mirror the user's typed global to the other-mode field (consistency)
  document.getElementById(gOtherId).value = g;
  onCfgChange();  // will re-sync the global display to perRank*D and update note
}

// Refresh the Global-batch UI for the current cfg.batch / dp state.
// The Global-batch field is ALWAYS shown so the semantics stay consistent:
//   - dp>1: global batch = per-rank batch × dp (work split across DP replicas)
//   - dp=1: pure TP, no DP → global batch == per-rank batch (every card sees the
//           same batch; "global" and "per-rank" are numerically identical).
// Throughput everywhere is computed against the GLOBAL batch (= cfg.batch × dp),
// so surfacing it unconditionally keeps the UI honest for the dp=1 case too.
// When the user types a global value that isn't divisible by dp, per-rank rounds
// UP (barrel effect: slowest rank dominates step time) and we surface the gap as
// a "→ computes as N (ceil)" hint under the field.
function syncGlobalBatchDisplay() {
  const D = Math.max(1, (typeof dp === 'number' && dp >= 1) ? dp : 1);
  const effectiveGlobal = cfg.batch * D;
  ['', 'D'].forEach(suf => {
    const field = document.getElementById('cfgGlobalBatchField' + suf);
    const inp   = document.getElementById('cfgGlobalBatch'      + suf);
    const note  = document.getElementById('cfgGlobalBatchNote'  + suf);
    if (!field || !inp) return;
    field.style.display = '';   // always visible — consistent semantics for dp=1 and dp>1
    if (D <= 1) {
      // Pure TP: global batch == per-rank batch. Keep the field in sync and
      // editable; editing it just mirrors straight to per-rank (1:1).
      inp.value = effectiveGlobal;
      if (note) note.textContent = '';
      return;
    }
    // Read the user's last-typed global (may differ from effectiveGlobal due to ceil)
    const typed = parseInt(inp.value) || 0;
    if (typed > 0 && typed < effectiveGlobal && Math.ceil(typed / D) === cfg.batch) {
      // User typed a non-divisible global; ceil rounded up — keep their value
      // displayed and surface the rounding so they see the barrel effect.
      if (note) note.textContent = `→ computes as ${effectiveGlobal} (ceil)`;
    } else {
      inp.value = effectiveGlobal;
      if (note) note.textContent = '';
    }
  });
}

function renderArchInfo() {
  const el = document.getElementById('archInfo');
  if (!el) return;
  const gqa = cfg.H < cfg.nH ? ` <span style="color:#888">(GQA ${cfg.nH}:${cfg.H})</span>` : '';
  el.innerHTML = `<span>${cfg.nL}</span> layers · <span>${cfg.nH}</span> Q-heads${gqa} · KV heads <span>${cfg.H}</span> · head dim <span>${cfg.D}</span>`;
}

// Enable/disable MoE parallelism selector based on whether the current model
// actually has any MoE layers. Non-MoE models don't need this knob.
function refreshMoEParAvailability() {
  const hasMoE = layers.some(l => l.moeGroup || l.type === 'moe');
  const sel   = document.getElementById('epDegSel');
  const field = document.getElementById('moeParField');
  const label = document.getElementById('moeParLabel');
  const hint  = document.getElementById('moeParHint');
  if (!field) return;
  if (sel) sel.disabled = !hasMoE;
  if (hasMoE) {
    field.style.opacity = '1';
    if (sel) { sel.style.cursor = 'pointer'; sel.style.background = '#f9f9f8'; }
    if (label) label.textContent = 'MoE EP degree';
  } else {
    field.style.opacity = '0.45';
    if (sel) { sel.style.cursor = 'not-allowed'; sel.style.background = '#f0f0ee'; }
    if (label) label.textContent = 'MoE EP degree (N/A — no MoE layers)';
    if (hint) hint.textContent = 'Load an MoE model to enable';
    // Reset to pure TP
    if (sel) sel.value = '1';
    epDeg = 1;
    moePar = 'tp';
  }
}

function openArchEdit() {
  document.getElementById('ov').style.display='flex';
  const mb=document.getElementById('mb'); mb.className='modal';
  mb.innerHTML=`<h3>Model architecture</h3>
    <p style="font-size:12px;color:#888;margin-bottom:12px">These are fixed by the model. Loading a HuggingFace config will overwrite them automatically.</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div class="f"><label>Num layers (nL)</label><input id="aNL" type="number" value="${cfg.nL}"/></div>
      <div class="f"><label>Query heads (nH)</label><input id="aNH" type="number" value="${cfg.nH}"/></div>
      <div class="f"><label>KV heads (nKV)</label><input id="aKH" type="number" value="${cfg.H}"/></div>
      <div class="f"><label>Head dim (D)</label><input id="aDD" type="number" value="${cfg.D}"/></div>
    </div>
    <div class="mbtns"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn p" onclick="saveArchEdit()">Save</button></div>`;
}

function saveArchEdit() {
  cfg.nL = Math.max(1, parseInt(document.getElementById('aNL').value)||cfg.nL);
  cfg.nH = Math.max(1, parseInt(document.getElementById('aNH').value)||cfg.nH);
  cfg.H  = Math.max(1, parseInt(document.getElementById('aKH').value)||cfg.H);
  cfg.D  = Math.max(1, parseInt(document.getElementById('aDD').value)||cfg.D);
  // Manual arch edit: user explicitly wants nH/nKV/D applied to attention layers.
  // (applyConfigToLayers does NOT do this anymore, to protect MLA / hybrid models
  //  from being clobbered on every batch/seq change.)
  layers.forEach(l => {
    if (l.type === 'attention') { l.H = cfg.nH; l.nKV = cfg.H; l.D = cfg.D; }
  });
  closeModal();
  applyConfigToLayers();
  updateCfgInfo();
  renderArchInfo();
  renderLayerList();
  renderTab();
}

function applyConfigToLayers() {
  // Updates only dynamic fields (batch/seq/mode). Does NOT overwrite per-layer
  // H/D on attention layers — those are fixed at import time per architecture.
  // E.g. DeepSeek MLA uses v_head_dim for attention; Qwen3.5 full-attn uses nH
  // query heads; clobbering these on every cfg change would destroy the model.
  // Per-layer `swaWindow` (set on import for V4-Flash SWA-only layers) caps S
  // and kvCache to the local window — without this hook the values would be
  // restored to full seqLen on every cfg edit and the SWA-only fix would be lost.
  const M = mode==='prefill' ? cfg.batch * cfg.seqLen : cfg.batch;
  layers.forEach(l => {
    if (l.type==='linear' || l.type==='embedding') l.M = M;
    if (l.type==='linear_attn') l.M = M;
    if (l.type==='moe' || l.moeGroup) l.M = M;
    if (l.type==='attention') {
      l.B = cfg.batch;
      const sw = l.swaWindow;  // optional sliding-window cap (e.g. V4 SWA-only)
      if (mode==='prefill') {
        l.S = sw ? Math.min(cfg.seqLen, sw) : cfg.seqLen;
        l.kvCache = 0;
      } else {
        l.S = 1;
        const fullKv = cfg.seqLen + Math.floor(cfg.outTokens/2);
        l.kvCache = sw ? Math.min(sw, fullKv) : fullKv;
      }
    }
  });
}

// ── VRAM budget ───────────────────────────────────────────────────────────────
// Weight bytes (fp16 = 2 bytes per param):
//   Linear layer:    weight matrix = K × N, so bytes = K × N × 2  (×repeat)
//   Embedding layer: lookup table  = K entries × hidden, but stored as K × 1 in our model
//                    so bytes = K × 2  (just the embedding table rows, repeat=1)
//   Attention layer: NO weight params — QKV proj and O proj are modelled as separate
//                    linear layers in the layer list. The attention layer only represents
//                    the QK^T / softmax / AV compute, which is parameter-free.

let moePar = 'tp';  // derived from epDeg: 'tp' (ep=1), 'ep' (ep=nCards), 'hybrid' (1<ep<nCards).
let epDeg  = 1;    // EP degree. 1 = pure TP, nCards = pure EP, between = hybrid.
let dp     = 1;    // DP degree. 1 = no DP. dp>1 → attention replicated, MoE effective M *= dp.
let tpAttn = 1;    // TP degree for attention = nCards / dp. When dp=nCards, tpAttn=1 (fully replicated).

// v88: Peak/imbalance modeling toggle for MoE routed experts.
// Default (false): per-card avg under load-balanced averaging — each EP group sees
//   distinct/epDeg active experts (fractional), uniformly TP-sliced across expertTp cards.
// On (true): per-card peak (worst card's load) — the busiest EP group sees
//   ceil(distinct/epDeg) active experts; the multiplier vs avg is
//   peakMul = ceil(distinct/epDeg) × epDeg / distinct.
// Only routed gate_up/down (weight, FLOPs, activation bytes per token-expert pair)
// scale by peakMul. Shared experts (replicated, uniformly TP-sliced) and router
// (replicated) are not affected. Communication volumes also unchanged.
// Most pronounced in sparse decode under pure EP (e.g., DSV3 decode M=1, EP=64:
//   peakMul = 8, so the 8 hot cards see ~88 MB/card vs 11 MB/card avg).
let moePeakMode = false;

// peakMul vs avg for routed MoE work; returns 1 when not in peak mode, when EP is
// off (epDeg≤1), or when distinct is degenerate.
function moePeakMul(distinct, epDeg_) {
  if (!moePeakMode) return 1;
  if (!(epDeg_ > 1) || !(distinct > 0)) return 1;
  return Math.ceil(distinct / epDeg_) * epDeg_ / distinct;
}

function onTPChange() {
  // v76: capture the DP degree as it stood BEFORE this change so we can keep
  // the GLOBAL batch invariant across parallelism switches (per-rank follows).
  const prevDp = (typeof dp === 'number' && dp >= 1) ? dp : 1;
  const exp = parseInt(document.getElementById('tpS').value);
  nCards      = 1 << exp;  // 2^exp
  overlap = parseFloat(document.getElementById('ovS').value);

  // ── DP degree dropdown (1..nCards) ────────────────────────────────────────
  const dpSel  = document.getElementById('dpSel');
  const dpHint = document.getElementById('dpHint');
  if (dpSel) {
    const prevDP = parseInt(dpSel.value) || 1;
    dpSel.innerHTML = '';
    for (let d = 1; d <= nCards; d *= 2) {
      if (nCards % d === 0) {
        const opt = document.createElement('option');
        opt.value = d;
        const tpa = nCards / d;
        if (d === 1) opt.textContent = `1 — no DP (Attn TP=${nCards})`;
        else if (tpa === 1) opt.textContent = `${d} — full DP (Attn replicated)`;
        else opt.textContent = `${d} — DP=${d}, Attn TP=${tpa}`;
        dpSel.appendChild(opt);
      }
    }
    if (dpSel.querySelector(`option[value="${prevDP}"]`)) dpSel.value = prevDP;
    dp = parseInt(dpSel.value) || 1;
  } else { dp = 1; }
  tpAttn = Math.max(1, nCards / dp);

  // ── v76: keep GLOBAL batch fixed across parallelism switches ──────────────
  // When the DP degree changes, the user's intent is that the deployment-wide
  // (global) batch stays the same — the per-rank batch is what should follow.
  // Global batch = per-rank batch × DP, so:
  //   prevGlobalBatch = cfg.batch × prevDp   (global as it stood before)
  //   newPerRank      = ceil(prevGlobalBatch / dp)   (barrel effect on non-divisible)
  // We write the recomputed per-rank into both batch fields and stash the
  // original global into the Global-batch fields so syncGlobalBatchDisplay
  // surfaces a "ceil" note when the split isn't exact. When dp is unchanged
  // (e.g. only the overlap slider moved) this is a no-op.
  if (dp !== prevDp) {
    const prevGlobalBatch = Math.max(1, cfg.batch * prevDp);
    const newPerRank = Math.max(1, Math.ceil(prevGlobalBatch / dp));
    cfg.batch = newPerRank;
    const bP = document.getElementById('cfgBatch');
    const bD = document.getElementById('cfgBatchD');
    if (bP) bP.value = newPerRank;
    if (bD) bD.value = newPerRank;
    const gP = document.getElementById('cfgGlobalBatch');
    const gD = document.getElementById('cfgGlobalBatchD');
    if (gP) gP.value = prevGlobalBatch;
    if (gD) gD.value = prevGlobalBatch;
    // cfg.batch changed → push the new per-rank batch into the layer list so
    // every downstream calc (tpInfo/calcL) sees the updated M/B fields.
    applyConfigToLayers();
  }

  if (dpHint) {
    if (nCards < 2) dpHint.textContent = '';
    else if (dp <= 1) dpHint.textContent = 'All cards do tensor-parallel for attention + dense';
    else dpHint.textContent = `Attn TP=${tpAttn} per replica · ${dp} DP replicas · Global batch = batch × ${dp} · MoE M_eff = M×${dp}`;
  }

  // ── v76: surface per-rank semantics as a hint UNDER the batch input ──────
  // Pre-v76 the per-rank qualifier was appended onto the label text itself,
  // but the label has ellipsis overflow and clipped the parenthetical away.
  // Now the label stays a plain "Batch size" and the per-rank qualifier is
  // shown as a .fhint below the input, where nothing covers it.
  const batchLbl  = document.getElementById('cfgBatchLbl');
  const batchLblD = document.getElementById('cfgBatchLblD');
  if (batchLbl)  batchLbl.textContent  = 'Batch size';
  if (batchLblD) batchLblD.textContent = 'Batch size';
  const batchHint  = document.getElementById('cfgBatchHint');
  const batchHintD = document.getElementById('cfgBatchHintD');
  const showBatchHint = dp > 1;
  if (batchHint)  batchHint.style.display  = showBatchHint ? '' : 'none';
  if (batchHintD) batchHintD.style.display = showBatchHint ? '' : 'none';

  // ── EP degree dropdown (1..nCards) ────────────────────────────────────────
  const epSel  = document.getElementById('epDegSel');
  const hint   = document.getElementById('moeParHint');
  if (epSel && nCards >= 2) {
    const prevVal = parseInt(epSel.value) || 1;
    epSel.innerHTML = '';
    for (let ep = 1; ep <= nCards; ep *= 2) {
      if (nCards % ep === 0) {
        const opt = document.createElement('option');
        opt.value = ep;
        const eTp = nCards / ep;
        if (ep === 1) opt.textContent = `1 — pure TP (all experts TP-sliced)`;
        else if (ep === nCards) opt.textContent = `${ep} — pure EP (experts distributed)`;
        else opt.textContent = `${ep} — EP=${ep} × expert_TP=${eTp}`;
        epSel.appendChild(opt);
      }
    }
    if (epSel.querySelector(`option[value="${prevVal}"]`)) epSel.value = prevVal;
    epDeg = parseInt(epSel.value) || 1;
  } else {
    if (epSel) { epSel.innerHTML = '<option value="1">1 — single card</option>'; }
    epDeg = 1;
  }

  // Derive moePar
  if (epDeg <= 1)       moePar = 'tp';
  else if (epDeg >= nCards) moePar = 'ep';
  else                  moePar = 'hybrid';

  // MoE hint
  if (hint) {
    if (nCards < 2) {
      hint.textContent = 'Set total cards ≥ 2 to configure parallelism';
    } else {
      const eTp = Math.max(1, nCards / epDeg);
      const dpTag = dp > 1 ? ` · M_eff=M×${dp}` : '';
      if (moePar === 'tp')
        hint.textContent = `MoE: all experts TP-sliced · AllReduce${dpTag}`;
      else if (moePar === 'ep')
        hint.textContent = `MoE: ${epDeg}-way EP · AllToAll${dpTag}`;
      else
        hint.textContent = `MoE: EP=${epDeg} × eTp=${eTp} · A2A+AR${dpTag}`;
    }
  }

  document.getElementById('tpV').textContent = nCards;
  document.getElementById('ovV').textContent = overlap.toFixed(2);
  // Refresh global-batch widget visibility & values whenever dp changes.
  syncGlobalBatchDisplay();
  refreshMoePeakToggle();
  updateCfgInfo();
  renderLayerList();
  renderTab();
}

// v88: Show the MoE peak/imbalance toggle only when (a) the current model has
// MoE layers, and (b) EP > 1 (under pure TP the imbalance multiplier is always
// 1). Also recompute the inline hint that previews the peak multiplier so the
// user can see, at a glance, how much "peak" diverges from "avg" for the
// current parallelism + decode/prefill regime.
function refreshMoePeakToggle() {
  const row  = document.getElementById('moePeakRow');
  const cb   = document.getElementById('moePeakSel');
  const hint = document.getElementById('moePeakHint');
  const tag  = document.getElementById('moePeakTag');
  if (!row || !cb) return;
  const hasMoE = (typeof layers !== 'undefined') && Array.isArray(layers)
                 && layers.some(l => l && (l.moeGroup || l.type === 'moe'));
  const epOn   = epDeg > 1;
  if (!hasMoE || !epOn) {
    row.style.display = 'none';
    if (tag) tag.style.display = 'none';
    return;
  }
  row.style.display = 'flex';
  cb.checked = !!moePeakMode;
  if (tag) tag.style.display = moePeakMode ? '' : 'none';
  // Preview: compute peakMul for the largest routed MoE in the model at the
  // current effective M (mode-dependent: decode → M=1, prefill → cfg.seqLen).
  if (hint) {
    let preview = null;
    try {
      const moeLayer = layers.find(l => l && (l.moeGroup === 'gate_up' || l.type === 'moe'));
      if (moeLayer) {
        const nE   = moeLayer.moeExperts || moeLayer.nExperts || 1;
        const topK = moeLayer.moeTopK || moeLayer.topK || 1;
        const M    = (mode === 'decode') ? 1 : (cfg.seqLen || 1);
        const Meff = M * dp;
        const dist = expectedActivatedExperts(Meff, topK, nE);
        const mul  = (epDeg > 1 && dist > 0)
          ? Math.ceil(dist / epDeg) * epDeg / dist : 1;
        preview = mul;
      }
    } catch (e) { /* silent — UI hint only */ }
    if (preview && preview > 1.01) {
      hint.textContent = `(${preview.toFixed(1)}× vs avg @ ${mode})`;
      hint.style.color = '#854F0B';
    } else if (preview != null) {
      hint.textContent = `(=avg @ ${mode}, saturated)`;
      hint.style.color = '#888';
    } else {
      hint.textContent = '';
    }
  }
}

function onMoePeakChange() {
  const cb = document.getElementById('moePeakSel');
  moePeakMode = !!(cb && cb.checked);
  refreshMoePeakToggle();
  renderTab();
}

// Pull interconnect bandwidth from the currently selected chip and refresh the
// read-only display. Call after selChip changes or chip is edited/deleted.
function syncIbwFromChip() {
  ibw = (selChip && selChip.ibw) ? selChip.ibw : 224;
  const c = document.getElementById('ibwChipLbl');
  const u = document.getElementById('ibwUniLbl');
  const b = document.getElementById('ibwBiLbl');
  if (c) c.textContent = selChip ? selChip.name : '—';
  if (u) u.textContent = ibw;
  if (b) b.textContent = ibw * 2;
}

function onPrecisionChange() {
  cfg.precision = document.getElementById('precisionSel').value;
  updateCfgInfo();
  renderLayerList();
  renderTab();
}

// v80: Sparse-MLA (DSA) toggle. Governs whether DSA models are modelled with the
// lightning indexer + top-k restricted core attention (ON) or as a dense-MLA
// upper bound (OFF). Pure runtime flag — getRepeat()/attnDims() read it live, so
// no layer rebuild is needed; just recompute and re-render.
function onSparseMLAChange() {
  const chk = document.getElementById('sparseMLAChk');
  sparseMLA = chk ? chk.checked : true;
  updateCfgInfo();
  renderLayerList();
  renderTab();
}

// v84: A DSA-capable model carries lightning-indexer layers (l.indexerLayer) or
// a top-k-restricted core attention (l.sparseTopK). The Sparse MLA toggle is only
// meaningful for those — for every other model it is hidden, and sparseMLA is
// pinned ON (the no-op default) so a previously-unchecked box can't silently
// affect a freshly loaded non-DSA model. Called from renderLayerList() so it
// re-evaluates on every model load / layer edit.
function refreshSparseMLAAvailability() {
  const field = document.getElementById('sparseMLAField');
  if (!field) return;
  const isDSA = layers.some(l => l.indexerLayer || l.sparseTopK);
  if (isDSA) {
    field.style.display = '';
    const chk = document.getElementById('sparseMLAChk');
    if (chk) { chk.checked = sparseMLA; chk.disabled = false; }
  } else {
    field.style.display = 'none';
    sparseMLA = true;   // pin to the no-op default for non-DSA models
    const chk = document.getElementById('sparseMLAChk');
    if (chk) chk.checked = true;
  }
}

// ── TP / EP model ─────────────────────────────────────────────────────────────
// Communication model (Megatron-LM convention):
//
//   Standard TP for dense blocks:
//     QKV / gate / up / router / lm_head → column-parallel (weight split on N).
//       Output is sharded along feature dim; next op takes sharded input. NO COMM.
//     O proj / FFN down / shared_down   → row-parallel (weight split on K).
//       Output is full-sized but partial; needs AllReduce to sum across cards.
//     Attention / linear_attn op itself → NO COMM of its own (reduce happens
//       on the following O proj, which is row-parallel).
//     Embedding (vocab-parallel)        → per-card partial output; AllReduce
//       to gather-sum the full embedding.
//
//   MoE under Expert Parallelism (EP):
//     Routed experts sharded across cards; shared experts REPLICATED on every
//     card (no sharding, no comm).
//     Each transformer block carries ONE dispatch + ONE combine All-to-All,
//     both on hidden-sized data expanded by topK (each token is sent to topK
//     experts → topK copies).
//     gate_up sub-layer carries dispatch (1 a2a); down carries combine (1 a2a).
//     v78 NOTE: dispatch and combine are NOT symmetric in byte width. DeepEP-
//     style EP sends dispatch in FP8 (dispatchBpe) but combine in BF16 (aBpe) —
//     combine does weighted summation, FP8 accumulation is too lossy. Under
//     BF16/W8A16 both are 2B (symmetric); under W8A8 dispatch=1B, combine=2B,
//     so the per-block a2a total is (dispatchBpe+aBpe) not 2·aBpe.
//     Router: col-par dense linear, no comm of its own.
//
//   MoE under Tensor Parallelism (TP, force):
//     All experts replicated; each expert's GEMM is sharded.
//     gate_up col-par → no comm; down row-par → AllReduce on M*hid at end.
//
//   Ring AllReduce time = 2·(N-1)/N · data / BW_unidirectional
//   Ring All-to-All  time ≈ (N-1)/N  · data / BW_unidirectional  (per a2a call)
//   `ibw` is unidirectional BW (see chip.ibw, synced via syncIbwFromChip).
//   v75 NOTE: `data` in BOTH formulas is the PER-CARD volume. For MoE a2a that
//   is the card's own sourced tokens (Meff/nCards · topK · hid), NOT the global
//   Meff — each card only pushes its own DP-rank share onto the fabric.

// ── MoE EP token-count convention (the "M-convention") ───────────────────────
// In DP×EP deployments the local input batch M and the post-AllToAll batch
// can differ by a factor of dp, so picking the right "M" for FLOPs and bytes
// is load-bearing. There are TWO physically distinct conventions, and the EP
// path uses BOTH — different sub-modules of the MoE block follow different
// ones, and the choice is dictated by deployment, not by us.
//
//   Meff = M × dp        — global tokens entering the MoE across all DP ranks
//
//   (A) Sharded (post-a2a) convention — used for ROUTED experts.
//       After dispatch, each card holds nExperts/nCards experts and processes
//       Meff·topK/nCards token-expert pairs. FLOPs, weight HBM reads, and
//       activation bytes are ALL per-card = global/nCards. The a2a comm volume
//       is also per-card (Meff/nCards · topK · hid), since each card only
//       pushes its own DP-rank share onto the fabric.
//
//   (B) Replicated (pre-a2a) convention — used for SHARED experts and ROUTER.
//       Both live on every DP rank with full weight. Each card runs its OWN
//       M tokens through them (not Meff). No comm involved. Their FLOPs and
//       bytes scale with the per-rank M; cluster-wide there is dp-fold
//       redundancy in the work, which is the actual cost paid by the
//       DeepSeek-V3/DeepEP serving stack.
//
// Invariant the convention guarantees:
//   Holding Meff fixed and reshuffling (M, dp), routed per-card FLOPs and
//   bytes stay CONSTANT, while shared+router per-card scale ∝ M (∝ 1/dp).
//   Diagnostic: 0.00% drift on the routed gate_up sub-layer across (dp, M) =
//   {(1,8192), (2,4096), (4,2048), (8,1024)}.
//
// History: this used to be inconsistent before v75 — routed FLOPs assumed (A)
// while activation bytes assumed (B), producing dp-dependent AI drift in
// DP×EP configs. v75/v78/v87/v88 progressively converged routed weight bytes,
// activation bytes, comm volume, and (under peak mode) peakMul scaling onto a
// single sharded view. Shared and router were already replicated; the v87
// regression test pinned their per-rank M semantics in place.
//
// When extending the EP path, the rule is:
//   - Anything that flows through dispatch/combine → use Meff/nCards (A)
//   - Anything that lives per-DP-rank with full weight → use M (B)

// Classify a linear layer as row-parallel (needs AllReduce on its output)
// vs column-parallel (no comm) by name. Defaults to col-par when unknown.
//
// NOTE on shared_down: It IS row-parallel, so this function returns true.
// However, under MoE-TP mode, its AllReduce is DEFERRED and FUSED with the
// routed MoE down's AllReduce (local partial += shared_partial, then one
// AllReduce over the combined [M,H] buffer). This matches vLLM's
// RowParallelLinear(reduce_results=False) pattern in DeepSeekV2MoE. The
// fusion is handled in tpInfo's standard-TP comm block, not here.
function isRowParLinearByName(l) {
  if (l.moeGroup === 'shared_down') return true;
  const name = (l.name||'').toLowerCase();
  // Matches: "o proj", "o_proj", "oproj", " down", "_down", ".down", "down proj",
  //          "down_proj", "mlp.down_proj", "ffn down", "moe down", etc.
  // v77: two name-side fixes, same root cause (a real row-parallel layer was
  // silently falling through to col-parallel and losing its AllReduce):
  //   • o-proj clause `o\s*proj` → `o[\s_]*proj` — underscore spelling
  //     ("...self_attn.o_proj") now matches, not just the space spelling.
  //   • down prefix `(^|\s|_)` → `(^|[\s_.])` — safetensors layer names use a
  //     dot as the path separator ("...mlp.down_proj"), so ".down" must count
  //     as a word boundary just like " down" / "_down".
  // Does NOT match: "gate_proj", "q_proj", "k_proj", "v_proj", "up_proj"
  //   (char before "proj" is q/k/v/p/e — never "o"); "downstream"/"downsample"
  //   (the trailing-boundary group `(\s|_|$|\b)` still rejects those).
  return /(^|[\s_.])down(\s|_|$|\b)|o[\s_]*proj/i.test(name);
}

// Determine the TP sharding mode of a linear layer:
//   'replicated' — computed IN FULL on every card; no weight/FLOP split, NO AllReduce.
//                  MLA's q_a_proj / kv_a_proj use this: their output feeds a
//                  layernorm over the full latent dim, so DeepSeek/vLLM keep them
//                  replicated rather than pay a reduce. (Layer sets tpMode:'replicated'.)
//   'row'        — weight split on K; output is a partial sum → needs AllReduce.
//                  O proj, FFN down, MoE down.
//   'col'        — weight split on N; output is sharded cleanly → no AllReduce. (default)
// An explicit l.tpMode always wins over the name-based heuristic. This prevents
// false positives like "MLA Q down" / "MLA KV down" — they contain "down" but are
// actually replicated a-projections, NOT row-parallel down-projections. Without
// this, each got a spurious AllReduce (v73 counted 3 AR per MLA block; only 1 is real).
function tpModeOf(l) {
  if (l.tpMode === 'replicated' || l.tpMode === 'row' || l.tpMode === 'col') return l.tpMode;
  if (l.type === 'linear' && isRowParLinearByName(l)) return 'row';
  return 'col';
}

function tpInfo(l, chip) {
  const base = calcL(l);
  const eTF = effectiveTflops(chip, l.type) * 1e12;
  const wb = wBpe(), ab = aBpe();
  if (nCards===1) {
    const t = Math.max(base.flops/eTF, base.bytes/(chip.bw*1e9));
    return {computeTime:base.flops/eTF, memTime:base.bytes/(chip.bw*1e9),
            commTime:0, hiddenComm:0, commBytes:0, effectiveTime:t,
            flopsPerCard:base.flops, bytesPerCard:base.bytes, aiPerCard:base.ai};
  }

  // ── MoE monolithic: EP or TP ──────────────────────────────────────────────
  if (l.type === 'moe') {
    const {M, hid, ffnDim, topK, nExperts, nShared=0} = l;
    // MoE always spans all nCards cards (TP via AllReduce, EP via AllToAll).
    // With dp>1, all DP replicas' tokens enter the shared MoE computation:
    //   TP: every card reads expert slices for ALL tokens' activated experts → Meff = M×dp
    //   EP: tokens dispatched across all cards via AllToAll → Meff = M×dp
    const Meff = M * dp;

    if (moePar === 'tp') {
      // All experts replicated on every card; each expert's GEMM sharded.
      // gate_up col-par + down row-par → ONE AllReduce per MoE block (on Meff*hid).
      const distinct = expectedActivatedExperts(Meff, topK, nExperts);
      const expertWPerCard = (distinct + nShared) * (2*hid*ffnDim + ffnDim*hid) * wb / nCards;
      const routerWPerCard = hid * nExperts * wb;             // replicated
      const wBytesPerCard  = expertWPerCard + routerWPerCard;
      const actBytesPerCard = Meff * hid * ab * 2;
      const bpc = wBytesPerCard + actBytesPerCard;
      const expertFlops = 6 * Meff * hid * ffnDim * (topK + nShared);
      const routerFlops = 2 * Meff * hid * nExperts;
      const fpc = expertFlops / nCards + routerFlops;
      const cT = fpc / eTF;
      const mT = bpc / (chip.bw*1e9);
      const arBytes = 2 * ((nCards-1)/nCards) * Meff * hid * ab;
      const commT = arBytes / (ibw*1e9);
      const busy  = Math.max(cT, mT);
      const hidComm = Math.min(commT, busy) * overlap;
      return {computeTime:cT, memTime:mT, commTime:commT, hiddenComm:hidComm,
              commBytes:arBytes,
              effectiveTime:busy+commT-hidComm,
              flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc, moePar:'tp'};
    }

    if (moePar === 'hybrid') {
      // ── Hybrid TP+EP ─────────────────────────────────────────────────────
      // Attention/dense use TP=nCards (full tensor parallel).
      // MoE uses EP=epDeg groups, each with expert_TP = nCards/epDeg intra-group TP.
      //   - Routed experts: distributed across EP groups, TP-sharded within group.
      //     Global routed FLOPs = 6·Meff·topK·hid·ffn → per-card = global/nCards
      //     (since epDeg×expertTp = nCards). Equivalently: each EP group sees
      //     Meff·topK/epDeg activations split across expertTp cards.
      //   - Shared experts: REPLICATED across EP groups, TP-sharded within group.
      //     Each card runs its OWN DP rank's M tokens (NOT Meff) through 1/expertTp
      //     of the shared FFN.
      //   - Router: REPLICATED on every card. Each card runs its own M tokens
      //     (NOT Meff) through full hid→nExperts gate.
      // Communication: AllToAll between EP groups + AllReduce within expert_TP.
      // v88: when moePeakMode is on, scale routed weight bytes, FLOPs, and
      //   token-expert activation bytes by peakMul (busiest EP group). Shared,
      //   router, and comm volumes are not affected (they are uniform).
      const expertTp    = Math.max(1, nCards / epDeg);
      const localRouted = Math.ceil(nExperts / epDeg);
      const expectedHit = expectedActivatedExperts(Meff, topK, nExperts);
      const hitPerCard  = Math.min(localRouted, expectedHit / epDeg);
      const peakMul     = moePeakMul(expectedHit, epDeg);
      const expertW1    = (2*hid*ffnDim + ffnDim*hid) * wb;
      const routedWBytes = hitPerCard * expertW1 / expertTp * peakMul;
      const sharedWBytes = nShared   * expertW1 / expertTp;
      const routerWPerCard = hid * nExperts * wb;
      const wBytesPerCard = routedWBytes + sharedWBytes + routerWPerCard;
      const tokExpPerCard   = (Meff * topK / nCards) * peakMul;             // a2a-distributed pairs per card (×peakMul under imbalance)
      const actBytesPerCard = (tokExpPerCard + M * (1 + nShared)) * hid * ab * 2;
      const bpc = wBytesPerCard + actBytesPerCard;
      const routedFlops  = 6 * Meff * topK * hid * ffnDim / nCards * peakMul; // global / nCards × peakMul
      const sharedFlops  = 6 * M    * hid * ffnDim * nShared / expertTp;     // per-rank M, TP-sliced (was Meff — bug fix)
      const routerFlops  = 2 * M    * hid * nExperts;                        // per-rank M, replicated (was Meff — bug fix)
      const fpc = routedFlops + sharedFlops + routerFlops;
      const cT = fpc / eTF;
      const mT = bpc / (chip.bw*1e9);
      // v75: a2a/AR volumes are PER-CARD. Each card only sources its own
      // Meff/nCards token-equivalents onto the fabric (not the global Meff).
      // Was `Meff` → overstated comm by ×nCards under DP+EP. Matches the
      // per-card convention already used for FLOPs (/nCards) and act bytes.
      // v78: dispatch (dispatchBpe) + combine (aBpe) — asymmetric under W8A8.
      const a2aBytes = (Meff/nCards) * topK * hid * (dispatchBpe() + ab) * (epDeg-1) / epDeg;
      const arBytes  = expertTp > 1 ? 2 * ((expertTp-1)/expertTp) * (Meff/nCards) * hid * ab : 0;
      const commT = (a2aBytes + arBytes) / (ibw*1e9);
      const busy  = Math.max(cT, mT);
      const hidComm = Math.min(commT, busy) * overlap;
      return {computeTime:cT, memTime:mT, commTime:commT, hiddenComm:hidComm,
              commBytes:a2aBytes+arBytes,
              effectiveTime:busy+commT-hidComm,
              flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc, moePar:'hybrid'};
    }

    // ── Expert Parallelism: routed sharded, shared REPLICATED, router REPLICATED ──
    // Per-card workload model (Meff = M*dp = global tokens entering MoE via a2a):
    //   - Routed: global FLOPs = 6·Meff·topK·hid·ffn, distributed evenly across
    //     nCards cards via a2a → per-card = global/nCards. Equivalently by expert
    //     view: localRouted experts × (Meff·topK/nExperts tokens/expert) × 6·hid·ffn.
    //   - Shared: REPLICATED → each card runs its OWN DP rank's M tokens (NOT Meff)
    //     through the full shared FFN. No a2a involvement.
    //   - Router: REPLICATED (needs full hid×nExperts gate to pick topK) → each card
    //     runs its own M tokens through full router. Weight is hid*nExperts*wb per
    //     card (NOT divided by nCards).
    //   - Activation bytes: each card handles ≈ Meff·topK/nCards routed token-expert
    //     pairs (after a2a), plus its own M tokens through shared + router.
    // v88: when moePeakMode is on, scale routed weight bytes, FLOPs, and the
    //   routed activation pairs by peakMul (busiest card). Shared, router, and
    //   comm volumes are unchanged.
    const localRouted  = Math.ceil(nExperts / nCards);
    const expectedHit  = expectedActivatedExperts(Meff, topK, nExperts);
    const hitPerCard   = Math.min(localRouted, expectedHit / nCards);
    const peakMul      = moePeakMul(expectedHit, nCards);                  // epDeg = nCards in pure EP
    const routedWBytes = hitPerCard * (2*hid*ffnDim + ffnDim*hid) * wb * peakMul;
    const sharedWBytes = nShared    * (2*hid*ffnDim + ffnDim*hid) * wb;   // replicated, full weight on every card
    const routerWBytes = hid * nExperts * wb;                              // replicated (was /nCards — bug fix)
    const wBytesPerCard = routedWBytes + sharedWBytes + routerWBytes;
    const tokExpPerCard   = (Meff * topK / nCards) * peakMul;              // a2a-distributed token-expert pairs per card (×peakMul under imbalance)
    const actBytesPerCard = (tokExpPerCard + M * (1 + nShared)) * hid * ab * 2;
    const bpc = wBytesPerCard + actBytesPerCard;
    const routedFlops = 6 * Meff * topK * hid * ffnDim / nCards * peakMul; // global / nCards × peakMul
    const sharedFlops = 6 * M    * hid * ffnDim * nShared;                 // per-rank M (was Meff — bug fix)
    const routerFlops = 2 * M    * hid * nExperts;                         // per-rank M, replicated (was Meff/nCards — bug fix)
    const fpc = routedFlops + sharedFlops + routerFlops;
    const cT = fpc / eTF;
    const mT = bpc / (chip.bw*1e9);
    // v75: per-card a2a — each card sources Meff/nCards token-equivalents,
    // not the global Meff. Was `Meff` → ×nCards comm overstatement under DP+EP.
    // v78: dispatch (dispatchBpe) + combine (aBpe). Symmetric 2·ab under BF16/
    // W8A16; under W8A8 dispatch is FP8 → (1+2)·base instead of (2+2)·base.
    const a2aBytes = (Meff/nCards) * topK * hid * (dispatchBpe() + ab) * (nCards-1) / nCards;
    const commT = a2aBytes / (ibw*1e9);
    const busy  = Math.max(cT, mT);
    const hidComm = Math.min(commT, busy) * overlap;
    return {computeTime:cT, memTime:mT, commTime:commT, hiddenComm:hidComm,
            commBytes:a2aBytes,
            effectiveTime:busy+commT-hidComm,
            flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc, moePar:'ep'};
  }

  // ── Expanded MoE sub-layers ─────────────────────────────────────────────
  // Shared expert under EP / hybrid: replicated across EP groups.
  //   EP:     full weight on every card, full FLOPs (no TP within group).
  //   Hybrid: replicated across EP groups, TP-sharded within expert_TP group.
  if ((l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down') && (moePar === 'ep' || moePar === 'hybrid')) {
    const expertTp = moePar === 'hybrid' ? Math.max(1, nCards / epDeg) : 1;
    const fpc = base.flops / expertTp;
    const wBytes = l.K * l.N * wb / expertTp;
    // v83: activation traffic shards with expertTp exactly like a standard-TP
    // linear — shared_up is col-parallel (output Y[M, N/expertTp] sharded),
    // shared_down is row-parallel (input X[M, K/expertTp] sharded). Under pure
    // EP expertTp==1 and the shared expert is fully replicated, so both terms
    // collapse back to full M×K + M×N (no change for the EP path).
    const aBytes = (l.moeGroup === 'shared_up')
      ? (l.M*l.K + l.M*l.N/expertTp) * ab
      : (l.M*l.K/expertTp + l.M*l.N) * ab;
    const bpc = wBytes + aBytes;
    const cT = fpc / eTF;
    const mT = bpc / (chip.bw*1e9);
    // shared_down AllReduce within expertTp is deferred/fused with routed down AR
    // (same pattern as pure TP shared_down fusion), so commT=0 here.
    const parLabel = moePar === 'hybrid' ? 'hybrid-shared' : 'ep-shared';
    return {computeTime:cT, memTime:mT, commTime:0, hiddenComm:0, commBytes:0,
            effectiveTime:Math.max(cT,mT),
            flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc, moePar:parLabel};
  }

  // MoE routed sub-layers: gate_up carries dispatch, down carries combine
  if (l.moeGroup === 'gate_up' || l.moeGroup === 'down') {
    const {M, K, N, moeExperts:nE, moeTopK:topK} = l;

    if (moePar === 'tp') {
      // Pure TP MoE: every card holds ALL nE experts' weights, each expert sharded 1/nCards
      // along the ffnDim axis. gate_up is col-par (no comm), down is row-par (AllReduce).
      // HBM weight read accounts for SPARSITY — only ACTIVATED experts' shards are loaded.
      //   distinct = E[# distinct experts hit by M tokens routing top-K of nE]
      //   decode (M=1): distinct = topK      (massive sparsity, only topK/nE of weights touched)
      //   prefill (M big): distinct → nE     (saturates, all experts hit)
      // This matches the monolithic MoE-TP branch (line ~758) which uses the same `distinct`.
      const distinct = expectedActivatedExperts(M, topK, nE);
      const fpc = base.flops / nCards;
      const wPerCard = (distinct * K * N * wb) / nCards;
      // Activation bytes differ for gate_up vs down due to GroupedGEMM topology:
      //   gate_up: input [M, hid] read once, output [M, topK, 2*ffnDim] (each token
      //            sent to topK experts) → M*K + M*N*topK
      //   down:    input [M, topK, ffnDim] (topK expert outputs combine), output [M, hid]
      //            → M*K*topK + M*N
      const aPerCard = l.moeGroup === 'gate_up'
        ? (M*K + M*N*topK) * ab
        : (M*K*topK + M*N) * ab;
      const bpc = wPerCard + aPerCard;
      const cT = fpc / eTF;
      const mT = bpc / (chip.bw*1e9);
      const commBytes = l.moeGroup === 'down'
        ? 2 * ((nCards-1)/nCards) * M * N * ab
        : 0;
      const commT = commBytes / (ibw*1e9);
      const busy = Math.max(cT, mT);
      const hidComm = Math.min(commT, busy) * overlap;
      return {computeTime:cT, memTime:mT, commTime:commT, hiddenComm:hidComm,
              commBytes:commBytes,
              effectiveTime:busy+commT-hidComm,
              flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc, moePar:'tp'};
    }

    if (moePar === 'hybrid') {
      const Meff = M * dp;
      const expertTp    = Math.max(1, nCards / epDeg);
      const localRouted = Math.ceil(nE / epDeg);
      const distinct    = expectedActivatedExperts(Meff, topK, nE);
      const hitPerCard  = Math.min(localRouted, distinct / epDeg);
      const peakMul     = moePeakMul(distinct, epDeg);
      const fpc = (base.flops * dp / nCards) * peakMul;
      const wPerCard = (hitPerCard * K * N * wb) / expertTp * peakMul;
      // v87: per-card activation bytes after a2a — each card processes
      // tokExpPerCard = Meff*topK/nCards token-expert pairs (NOT global Meff),
      // and each pair reads K-dim input + writes N-dim output (symmetric across
      // gate_up and down). Was `(Meff*K + Meff*N*topK)` style — missing the
      // /nCards divisor + spurious extra ×topK on the output dim, causing
      // ~28× overcount in prefill, plus a ~250MB inconsistency vs the monolithic
      // EP path (line 1299-1300) which uses tokExpPerCard correctly.
      // v88: scale by peakMul when peak/imbalance mode is on (busiest card
      // processes peakMul × the avg token-expert pair count).
      const tokExpPerCard = (Meff * topK / nCards) * peakMul;
      const aPerCard = tokExpPerCard * (K + N) * ab;
      const bpc = wPerCard + aPerCard;
      const cT = fpc / eTF;
      const mT = bpc / (chip.bw*1e9);
      const hiddenDim = l.moeGroup === 'gate_up' ? K : N;
      // v75: per-card a2a/AR — base is Meff/nCards (per-card sourced tokens),
      // not global Meff. Was `Meff` → ×nCards comm overstatement under DP+EP.
      // v78: gate_up carries dispatch (FP8-capable → dispatchBpe); down carries
      // combine (weighted sum → BF16, aBpe). arBytes is a real partial-sum
      // AllReduce → stays BF16.
      const a2aAb = l.moeGroup === 'gate_up' ? dispatchBpe() : ab;
      const a2aBytes = (Meff/nCards) * topK * hiddenDim * a2aAb * (epDeg-1) / epDeg;
      const arBytes  = (l.moeGroup === 'down' && expertTp > 1)
        ? 2 * ((expertTp-1)/expertTp) * (Meff/nCards) * N * ab : 0;
      const commT = (a2aBytes + arBytes) / (ibw*1e9);
      const busy = Math.max(cT, mT);
      const hidComm = Math.min(commT, busy) * overlap;
      return {computeTime:cT, memTime:mT, commTime:commT, hiddenComm:hidComm,
              commBytes:a2aBytes+arBytes,
              effectiveTime:busy+commT-hidComm,
              flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc, moePar:'hybrid'};
    }

    // EP: routed sharded with dp pooling.
    const Meff = M * dp;
    const distinct = expectedActivatedExperts(Meff, topK, nE);
    const peakMul  = moePeakMul(distinct, nCards);                       // epDeg = nCards in pure EP
    const fpc = (base.flops * dp / nCards) * peakMul;
    const wPerCard = ((distinct * K * N * wb) / nCards) * peakMul;
    // v87: per-card activation bytes after a2a — each card processes
    // tokExpPerCard = Meff*topK/nCards token-expert pairs (NOT global Meff),
    // and each pair reads K-dim input + writes N-dim output (symmetric across
    // gate_up and down). See hybrid path above for full rationale; this is the
    // pure-EP twin of the same fix.
    // v88: scale by peakMul under peak/imbalance mode.
    const tokExpPerCard = (Meff * topK / nCards) * peakMul;
    const aPerCard = tokExpPerCard * (K + N) * ab;
    const bpc = wPerCard + aPerCard;
    const cT = fpc / eTF;
    const mT = bpc / (chip.bw*1e9);
    const hiddenDim = l.moeGroup === 'gate_up' ? K : N;
    // v75: per-card a2a — base is Meff/nCards (per-card sourced tokens), not
    // global Meff. Was `Meff` → ×nCards comm overstatement under DP+EP.
    // v78: gate_up carries dispatch (FP8-capable → dispatchBpe); down carries
    // combine (weighted sum → must stay BF16, aBpe).
    const a2aAb = l.moeGroup === 'gate_up' ? dispatchBpe() : ab;
    const a2aBytes = (Meff/nCards) * topK * hiddenDim * a2aAb * (nCards-1) / nCards;
    const commT = a2aBytes / (ibw*1e9);
    const busy = Math.max(cT, mT);
    const hidComm = Math.min(commT, busy) * overlap;
    return {computeTime:cT, memTime:mT, commTime:commT, hiddenComm:hidComm,
            commBytes:a2aBytes,
            effectiveTime:busy+commT-hidComm,
            flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc, moePar:'ep'};
  }

  // ── Expanded MoE router sub-layer ────────────────────────────────────────
  // Under TP / hybrid: router is REPLICATED (every card needs full gate for top-K).
  // Under EP: router is col-par sharded → falls through to generic linear (/nCards).
  if (l.moeGroup === 'router' && (moePar === 'tp' || moePar === 'hybrid')) {
    const fpc = base.flops;                   // replicated: full FLOPs
    const wBytes = l.K * l.N * wb;            // replicated: full weight
    const aBytes = (l.M*l.K + l.M*l.N) * ab;
    const bpc = wBytes + aBytes;
    const cT = fpc / eTF;
    const mT = bpc / (chip.bw*1e9);
    return {computeTime:cT, memTime:mT, commTime:0, hiddenComm:0, commBytes:0,
            effectiveTime:Math.max(cT,mT),
            flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc,
            moePar: moePar === 'hybrid' ? 'hybrid-router' : 'tp-router'};
  }

  // ── Standard TP: linear / embedding / attention (uses tpAttn for DP support) ─
  // When dp>1, non-MoE layers use tpAttn=nCards/dp (attention TP within each DP group).
  // If tpAttn===1 (fully replicated attention), return base values with no comm.
  if (tpAttn <= 1) {
    const t = Math.max(base.flops/eTF, base.bytes/(chip.bw*1e9));
    return {computeTime:base.flops/eTF, memTime:base.bytes/(chip.bw*1e9),
            commTime:0, hiddenComm:0, commBytes:0, effectiveTime:t,
            flopsPerCard:base.flops, bytesPerCard:base.bytes, aiPerCard:base.ai};
  }
  // Linear layers carry a TP mode (explicit l.tpMode or name-inferred via tpModeOf):
  //   replicated → full FLOPs/weights on every card, NO AllReduce (e.g. MLA a-projs)
  //   row/col    → weight split by tpAttn; row adds an AllReduce on its output
  // replicated does NOT save compute — it saves the reduce; every card recomputes
  // the full M×K×N (the a-projections are cheap, so this is the right trade).
  const linTpMode = (l.type === 'linear') ? tpModeOf(l) : null;
  // v80: a DSA lightning-indexer attention layer is TP-replicated — the indexer
  // sums per-head relevance scores then top-k selects, which doesn't head-split
  // cleanly across TP, so production replicates it (full compute every card,
  // no AllReduce). Like a 'replicated' linear, it saves the reduce, not compute.
  const attnReplicated = (l.type === 'attention' && l.tpReplicated);
  const fpc = (linTpMode === 'replicated' || attnReplicated) ? base.flops : base.flops / tpAttn;
  let wBytes, aBytes;
  if (l.type==='linear') {
    // Activation HBM traffic is PER-CARD. Input X[M,K] and output Y[M,N] each
    // shard — or not — depending on the TP mode of the linear:
    //   replicated → X full,  Y full          (every card recomputes M×K×N)
    //   col-par    → X full,  Y SHARDED [M,N/tp]  (weight split on N → each
    //                card produces only its own N-slice; input is replicated)
    //   row-par    → X sharded [M,K/tp], Y full [M,N] partial sum
    // v82 FIX (col): col-parallel previously used the FULL M×N output term,
    // overstating its activation write by exactly tpAttn× (e.g. FFN gate+up
    // N=ffn*2, LM head N=vocab, MLA Q-up). Each card only writes the sharded
    // [M, N/tpAttn] slice, so the output term is divided by tpAttn; the input
    // term M×K stays full — a col-parallel layer reads the replicated input.
    // v83 FIX (row): symmetric bug — row-parallel consumes the SHARDED output
    // of the preceding col-parallel layer, so its input is X[M, K/tpAttn], not
    // X[M,K]. The old full M×K input term overstated the read by tpAttn×. The
    // output M×N stays full: it is a per-card partial sum an AllReduce combines.
    if (linTpMode === 'replicated') {
      wBytes = l.K*l.N*wb;
      aBytes = (l.M*l.K + l.M*l.N)*ab;
    } else if (linTpMode === 'col') {
      wBytes = l.K*l.N*wb/tpAttn;
      aBytes = (l.M*l.K + l.M*l.N/tpAttn)*ab;
    } else {                                   // 'row'
      wBytes = l.K*l.N*wb/tpAttn;
      aBytes = (l.M*l.K/tpAttn + l.M*l.N)*ab;
    }
  }
  else if (l.type==='embedding') {
    const N_eff = (l.N && l.N > 1) ? l.N : 4096;
    wBytes = l.M * N_eff * wb / tpAttn;
    aBytes = l.M * N_eff * ab + l.M * 4;
  }
  else if (l.type==='attention') {
    // v80: use attnDims so MLA (single latent stream, kvHeadDim=D+rope) and DSA
    // (Sk capped to top-k) are accounted identically to the calc*Attn path.
    const nKV_l = l.nKV || l.H;
    const {B,H,D} = l;
    const {Sq,Sk,kvHeadDim,kvStreams} = attnDims(l);
    const kvb = kvBpe();
    if (attnReplicated) {
      // Indexer: full Q read + full index-key latent read on every card.
      wBytes = B*H*Sq*D*ab + kvStreams*B*nKV_l*Sk*kvHeadDim*kvb;
      if (!l.scoreOnly) wBytes += B*H*Sq*D*ab;                       // O write
    } else {
      const kvShards = Math.min(tpAttn, nKV_l);   // MLA: nKV=1 → 1 → KV replicated
      wBytes = B*H*Sq*D*ab/tpAttn                                    // Q (head-sharded)
             + kvStreams*B*nKV_l*Sk*kvHeadDim*kvb/kvShards;          // K(+V) latent
      if (!l.scoreOnly) wBytes += B*H*Sq*D*ab/tpAttn;                // O write
      if (!l.fa2 && !l.scoreOnly) wBytes += 2*B*H*Sq*Sk*ab/tpAttn;   // score matrix (non-FA2)
    }
    aBytes = 0;
  }
  else                      { wBytes = base.bytes/tpAttn; aBytes = 0; }
  const bpc = wBytes+aBytes;
  const cT  = fpc/eTF;
  const mT  = bpc/(chip.bw*1e9);

  let commT = 0, commBytes = 0;
  const isSharedDownFusedAR = (l.moeGroup === 'shared_down' && typeof moePar !== 'undefined' && (moePar === 'tp' || moePar === 'hybrid'));
  // Only ROW-parallel linears need an AllReduce. 'replicated' and 'col' do not —
  // a replicated layer already has the full output on every card, a col-parallel
  // layer's output is cleanly sharded and the reduce is deferred to the next row layer.
  if (l.type === 'linear' && linTpMode === 'row' && !isSharedDownFusedAR) {
    const actOut = l.M * l.N * ab;
    commBytes = 2 * ((tpAttn-1)/tpAttn) * actOut;
    commT = commBytes / (ibw*1e9);
  } else if (l.type === 'embedding') {
    const actOut = l.M * Math.max(l.N,1) * ab;
    commBytes = 2 * ((tpAttn-1)/tpAttn) * actOut;
    commT = commBytes / (ibw*1e9);
  }

  const busy = Math.max(cT, mT);
  const hid  = Math.min(commT, busy) * overlap;
  return {computeTime:cT, memTime:mT, commTime:commT, hiddenComm:hid, commBytes:commBytes,
          effectiveTime:busy+commT-hid, flopsPerCard:fpc, bytesPerCard:bpc, aiPerCard:fpc/bpc};
}

// tpInfo accounting for repeat — returns zero times for repeat=0 entries
function tpInfoTotal(l, chip) {
  const r = getRepeat(l);
  if (r === 0) return {effectiveTime:0, computeTime:0, memTime:0, commTime:0, hiddenComm:0, commBytes:0,
                       flopsPerCard:0, bytesPerCard:0, aiPerCard:0};
  const ti = tpInfo(l, chip);
  return {...ti,
    effectiveTime: ti.effectiveTime * r,
    computeTime:   ti.computeTime * r,
    memTime:       ti.memTime * r,
    commTime:      ti.commTime * r,
    hiddenComm:    ti.hiddenComm * r,
    commBytes:     (ti.commBytes || 0) * r,
  };
}

// tpShardShape: describe the per-card M/K/N (or heads/experts) AFTER TP sharding.
// Returned string is rendered as a supplementary line in the layer list so users
// can see exactly what matmul shape each card sees. Returns null when nCards<=1 or
// when sharding doesn't change any visible dimension (caller should skip the line).
//
// Sharding conventions (aligned with how tpInfo accounts for weight/FLOPs):
//   - Column-parallel linear (weight split on N):  "Attn QKV", "FFN gate+up",
//     "Shared expert up", "MoE Router", "LM head", "MoE gate+up" under TP mode.
//   - Row-parallel linear (weight split on K):     "Attn O proj", "FFN down",
//     "DeltaNet O proj", "Shared expert down", "MoE down" under TP mode.
//   - Attention / GatedDeltaNet: heads are split, hidden dim H shards by nCards.
//   - MoE gate_up/down under EP: experts are distributed across cards (not K/N).
//   - Embedding: vocab-parallel, K (vocab) is split.
function tpShardShape(l) {
  if (nCards <= 1) return null;
  const name = (l.name||'').toLowerCase();

  // Column-parallel vs row-parallel classification — an explicit l.tpMode
  // (set on MLA a-proj layers) overrides the name heuristic, keeping this in
  // lockstep with tpModeOf/tpInfo. Row-parallel output projections reduce on
  // K; column-parallel input/gate projections split on N. "down" matrices in
  // FFN/MoE are row-parallel too.
  const linMode = (l.type === 'linear') ? tpModeOf(l) : null;
  const isRowParLinear = linMode === 'row';
  const isColParLinear = linMode === 'col' &&
    /qkv|q\s*proj|k\s*proj|v\s*proj|gate|up|router|lm\s*head/i.test(l.name||'');

  // ── Attention (multi-head) — heads split across TP cards ────────────────
  // GQA: Q heads (nH) shard cleanly along nCards; KV heads (nKV) cap at min(nKV, nCards).
  // When nCards > nKV, KV heads must be replicated (each card still holds at least 1 KV group).
  if (l.type === 'attention') {
    // v80: a TP-replicated DSA indexer runs in full on every card (no head split).
    if (l.tpReplicated) {
      return `TP×${tpAttn} (replicated, no AR): H=${l.H} · D=${l.D} · S=${l.S} — full on every card`;
    }
    const hPerCard = l.H / nCards;
    const hIsInt = Number.isInteger(hPerCard);
    const nKV = l.nKV || l.H;
    const kvPerCard = Math.max(1, nKV / nCards);
    const kvIsInt = Number.isInteger(kvPerCard) || nCards >= nKV;
    const kvReplicated = nCards > nKV;
    const kvLabel = kvReplicated
      ? ` · KV=${nKV} (replicated, nCards>nKV=${nKV})`
      : ` · KV=${kvIsInt ? kvPerCard : kvPerCard.toFixed(2)}/${nKV}`;
    return `TP×${tpAttn}: H=${hIsInt ? hPerCard : hPerCard.toFixed(2)} (of ${l.H})${kvLabel} · D=${l.D} · S=${l.S}${hIsInt?'':' ⚠ H not divisible'}`;
  }

  // ── Linear attention (GatedDeltaNet) — key/value heads split ─────────────
  if (l.type === 'linear_attn') {
    const nQKpc = l.nQK / nCards, nVpc = l.nV / nCards;
    const ok = Number.isInteger(nQKpc) && Number.isInteger(nVpc);
    return `TP×${tpAttn}: nQK=${ok?nQKpc:nQKpc.toFixed(2)} (of ${l.nQK}) · nV=${ok?nVpc:nVpc.toFixed(2)} (of ${l.nV}) · D=${l.D}${ok?'':' ⚠ heads not divisible'}`;
  }

  // ── Legacy monolithic MoE (type='moe') — EP / TP / Hybrid sharding ──────
  if (l.type === 'moe') {
    const nE = l.nExperts || 0;
    if (moePar === 'tp') {
      return `MoE-TP×${nCards}: all ${nE} experts (each 1/${nCards} sliced) · top${l.topK} · M=${l.M} K=${l.hid} N=${l.ffnDim}`;
    }
    if (moePar === 'hybrid') {
      const eTp = Math.max(1, nCards / epDeg);
      const pc = Math.ceil(nE / epDeg);
      return `Hybrid EP=${epDeg}×eTp=${eTp}: ${pc} experts/group (1/${eTp} sliced) · top${l.topK} · M=${l.M}`;
    }
    const pc = Math.ceil(nE / nCards);
    return `EP×${nCards}: ${pc} experts/card (of ${nE}) · top${l.topK} · M=${l.M} K=${l.hid} N=${l.ffnDim}`;
  }

  // ── Expanded MoE sub-layers (router / gate_up / down / shared_*) ────────
  if (l.moeGroup) {
    const {M, K, N, moeExperts:nE, moeTopK:topK} = l;
    if (l.moeGroup === 'router') {
      if (moePar === 'tp' || moePar === 'hybrid') {
        return `Router REPLICATED: M=${M} K=${K} N=${N} (full on every card)`;
      }
      // EP: col-par
      const Npc = N / nCards;
      return `TP×${tpAttn}: M=${M} K=${K} N=${Number.isInteger(Npc)?Npc:Npc.toFixed(2)} (of ${N})`;
    }
    if (l.moeGroup === 'gate_up' || l.moeGroup === 'down') {
      if (moePar === 'tp') {
        // MoE-TP: all experts resident on every card; each expert's matmul split.
        if (l.moeGroup === 'gate_up') {
          const Npc = N / nCards;
          return `MoE-TP×${nCards}: all ${nE} experts · top${topK} · M=${M} K=${K} N=${Number.isInteger(Npc)?Npc:Npc.toFixed(2)} (of ${N})`;
        } else {
          const Kpc = K / nCards;
          return `MoE-TP×${nCards}: all ${nE} experts · top${topK} · M=${M} K=${Number.isInteger(Kpc)?Kpc:Kpc.toFixed(2)} (of ${K}) N=${N}`;
        }
      }
      if (moePar === 'hybrid') {
        const eTp = Math.max(1, nCards / epDeg);
        const pc = Math.ceil((nE||0) / epDeg);
        const dim = l.moeGroup === 'gate_up' ? `N=${N/eTp}(/${eTp})` : `K=${K/eTp}(/${eTp})`;
        return `Hybrid EP=${epDeg}×eTp=${eTp}: ${pc} experts/group · top${topK} · ${dim}`;
      }
      // EP (default): experts distributed, per-card K×N shape unchanged.
      const expPerCard = Math.ceil((nE||0) / nCards);
      return `EP×${nCards}: ${expPerCard} experts/card (of ${nE}) · top${topK} · M=${M} K=${K} N=${N}`;
    }
    if (l.moeGroup === 'shared_up') {
      if (moePar === 'ep') {
        return `EP×${nCards}: shared expert REPLICATED on every card · M=${M} K=${K} N=${N} (no shard, no comm)`;
      }
      if (moePar === 'hybrid') {
        const eTp = Math.max(1, nCards / epDeg);
        const Npc = N / eTp;
        return `Hybrid: shared REPLICATED across EP=${epDeg} · eTp=${eTp} col-par · N=${Number.isInteger(Npc)?Npc:Npc.toFixed(0)}(/${eTp})`;
      }
      const Npc = N / nCards;
      return `TP×${tpAttn} (col-par, no AR): M=${M} K=${K} N=${Number.isInteger(Npc)?Npc:Npc.toFixed(2)} (of ${N})`;
    }
    if (l.moeGroup === 'shared_down') {
      if (moePar === 'ep') {
        return `EP×${nCards}: shared expert REPLICATED on every card · M=${M} K=${K} N=${N} (no shard, no comm)`;
      }
      if (moePar === 'hybrid') {
        const eTp = Math.max(1, nCards / epDeg);
        const Kpc = K / eTp;
        return `Hybrid: shared REPLICATED across EP=${epDeg} · eTp=${eTp} row-par (AR fused) · K=${Number.isInteger(Kpc)?Kpc:Kpc.toFixed(0)}(/${eTp})`;
      }
      const Kpc = K / nCards;
      return `TP×${tpAttn} (row-par, AR fused w/ routed down): M=${M} K=${Number.isInteger(Kpc)?Kpc:Kpc.toFixed(2)} (of ${K}) N=${N}`;
    }
  }

  // ── Embedding: vocab-parallel (K is vocab) ──────────────────────────────
  if (l.type === 'embedding') {
    const Kpc = l.K / nCards;
    return `TP×${tpAttn}: vocab=${Number.isInteger(Kpc)?Kpc.toLocaleString():Kpc.toFixed(0)} (of ${l.K.toLocaleString()}) · N=${l.N}`;
  }

  // ── Plain dense linear — decide row- vs col-parallel by tpMode ──────────
  if (l.type === 'linear') {
    if (linMode === 'replicated') {
      // Replicated: every card computes the full M×K×N — no split, no AllReduce.
      return `TP×${tpAttn} (replicated, no AR): M=${l.M} K=${l.K} N=${l.N} — full on every card`;
    }
    if (isRowParLinear) {
      const Kpc = l.K / nCards;
      return `TP×${tpAttn} (row-par, +AR): M=${l.M} K=${Number.isInteger(Kpc)?Kpc:Kpc.toFixed(2)} (of ${l.K}) N=${l.N}`;
    }
    if (isColParLinear) {
      const Npc = l.N / nCards;
      return `TP×${tpAttn} (col-par, no AR): M=${l.M} K=${l.K} N=${Number.isInteger(Npc)?Npc:Npc.toFixed(2)} (of ${l.N})`;
    }
    // Unknown linear — default to col-parallel (matches tpInfo's weight/nCards assumption).
    const Npc = l.N / nCards;
    return `TP×${tpAttn} (col-par*, no AR): M=${l.M} K=${l.K} N=${Number.isInteger(Npc)?Npc:Npc.toFixed(2)} (of ${l.N})`;
  }

  return null;
}

// ── Calc ──────────────────────────────────────────────────────────────────────
// Attention HBM byte accounting for GQA:
//   Q is [B, Sq, nH, D] — bytes = B·nH·Sq·D · ab   (full Q heads)
//   K is [B, Sk, nKV, D] — bytes = B·nKV·Sk·D · kvb  (GQA-shared, broadcast in SRAM, NOT in HBM)
//   V is [B, Sk, nKV, D] — bytes = B·nKV·Sk·D · kvb
//   O is [B, Sq, nH, D] — bytes = B·nH·Sq·D · ab   (full Q heads)
// FLOPs use full nH (each Q head produces its own scores against its KV-group's K/V).
// l.H is nH (Q heads); l.nKV is the KV-head count (defaults to nH for plain MHA).
//
// v80 — attnDims() centralises three attention-shape corrections so the FLOP /
// byte / TP paths all agree:
//   • mla         : the K/V cache is a SINGLE shared compressed latent stream,
//                   not two independent K and V streams → kvStreams = 1 (vs 2).
//                   A scoreOnly indexer likewise reads one K stream (no V).
//   • ropeExtra   : MLA also caches a decoupled RoPE key (qk_rope_head_dim) on
//                   top of the kv_lora_rank latent → kvHeadDim = D + ropeExtra.
//                   The QK^T score uses kvHeadDim; AV uses D only (RoPE part
//                   carries no value).
//   • sparseTopK  : DSA (DeepSeek Sparse Attention, GLM-5 / glm_moe_dsa) core
//                   attention is restricted to the top-k selected tokens, so
//                   the effective Sk is min(Sk, topk) when sparseMLA is on.
// For every non-MLA / non-DSA layer ropeExtra/mla/sparseTopK are absent, so
// kvStreams=2, kvHeadDim=D, Sk=full — i.e. behaviour is byte-identical to v79.

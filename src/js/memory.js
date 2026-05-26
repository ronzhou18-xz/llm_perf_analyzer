function modelWeightBytes() {
  let total = 0;
  const wb = wBpe();
  layers.forEach(l => {
    const r = getRepeat(l);
    if (l.type === 'linear') {
      // Tied LM head shares its weight matrix with the token embedding table —
      // count those weights only once (under the embedding layer) to avoid
      // double-counting model params. The runtime FLOPs/bytes still get computed
      // because the matmul still happens at inference time.
      if (l.tied) return;
      if (l.moeGroup === 'gate_up' || l.moeGroup === 'down') {
        total += l.moeExperts * l.K * l.N * wb * r;
      } else {
        total += l.K * l.N * wb * r;
      }
    }
    else if (l.type === 'embedding') {
      const hidDim = (l.N && l.N > 1) ? l.N : 4096;
      total += l.K * hidDim * wb * r;
    }
    else if (l.type === 'linear_attn') { /* no extra params */ }
    else if (l.type === 'moe') {
      const {hid, ffnDim, nExperts, nShared=0} = l;
      // Expert weights: SwiGLU = gate(hid→ffn) + up(hid→ffn) + down(ffn→hid)
      // = 2*hid*ffn (gate+up combined) + ffn*hid (down) = 3*hid*ffn per expert.
      total += (nExperts + nShared) * (2*hid*ffnDim + ffnDim*hid) * wb * r;
      // Router weights: hid → nExperts, replicated/sharded with the rest of the
      // model under TP. Including this for parity with the expanded sub-layer
      // path (which adds router via moeGroup='router').
      total += hid * nExperts * wb * r;
    }
  });
  return total;
}

// v78: Sum of weight bytes for DENSE linear layers that are TP-REPLICATED
// (tpMode:'replicated'). Unlike row/col-parallel linears — sharded 1/tpAttn —
// these sit in FULL on every card, so the per-card VRAM budget must NOT divide
// them by tpAttn. MLA's q_a_proj / kv_a_proj are the canonical case: their
// output feeds a layernorm over the full latent dim, so DeepSeek/vLLM keep them
// replicated. This unifies VRAM accounting with tpInfo's timing path, which
// already gives replicated linears full FLOPs/weights per card (linTpMode).
// Only DENSE linears (no moeGroup) are counted — routed/shared/router MoE
// sub-layers are handled entirely by the MoE-weight path.
function replicatedWeightBytes() {
  let total = 0;
  const wb = wBpe();
  layers.forEach(l => {
    if (l.type !== 'linear' || l.tied || l.moeGroup) return;
    if (tpModeOf(l) === 'replicated') total += l.K * l.N * wb * getRepeat(l);
  });
  return total;
}

function kvCacheBytes(seqLen_) {
  if (cfg.kvCacheFn) return cfg.kvCacheFn(seqLen_);
  const nLayersKV = cfg.nLKV || cfg.nL;
  return 2 * nLayersKV * cfg.batch * cfg.H * cfg.D * seqLen_ * kvBpe();
}

// KV cache sharding factor under TP. Standard practice (vLLM, Megatron, SGLang):
// K/V heads are sharded by min(nCards, nKV). When nCards > nKV, KV is REPLICATED on the
// surplus cards (each KV head feeds nCards/nKV query-head shards).
//   - Standard GQA/MQA: cfg.H == nKV → shard by min(nCards, nKV).
//   - MLA (DeepSeek-V3): cfg.H is set to 1 (latent KV) → shard factor = 1, i.e.
//     KV stays REPLICATED across TP — matches vLLM's MLA-DP behavior where MLA
//     latent isn't head-sharded under TP.
//   - DSV4 / Qwen3.5 hybrid: cfg.H is set to the full-attn nKV → same min(nCards,nKV).
function kvShardFactor() {
  return Math.max(1, Math.min(tpAttn, cfg.H || 1));
}

function updateCfgInfo() {
  const kvLen    = mode==='decode' ? cfg.seqLen + cfg.outTokens : cfg.seqLen;
  // NOTE on semantics: cfg.batch is treated as PER-DP-RANK batch throughout the
  // tool (MoE uses Meff = M*dp). kvCacheBytes(seqLen) therefore returns the KV
  // total for ONE DP replica (one rank's batch worth), NOT the cluster-wide KV.
  // Per-card KV (after intra-rank TP sharding) = perRankKv / ksf.
  // Cluster-wide KV (sum across all DP replicas, no sharing) = perRankKv * dp.
  const kvPerRank = kvCacheBytes(kvLen) / 1e9;          // KV for one DP rank's batch (NOT cluster-global)
  const ksf      = kvShardFactor();
  const kvGB     = kvPerRank / ksf;                     // per-card KV after TP sharding within rank
  const kvClusterGB = kvPerRank * dp;                   // total KV across all DP replicas combined
  const wGB      = modelWeightBytes() / 1e9;
  const wb = wBpe();
  let moeWeightBytes = 0;
  layers.forEach(l => {
    if (l.type === 'moe') {
      const r = getRepeat(l);
      const {hid:h, ffnDim, nExperts:ne, nShared=0} = l;
      // Includes router weights for consistency with the expanded sub-layer path
      // (router is a small col-parallel linear: hid → nExperts).
      moeWeightBytes += (ne + nShared) * (2*h*ffnDim + ffnDim*h) * wb * r;
      moeWeightBytes += h * ne * wb * r;  // router
    } else if (l.moeGroup === 'gate_up' || l.moeGroup === 'down') {
      moeWeightBytes += l.moeExperts * l.K * l.N * wb * getRepeat(l);
    } else if (l.moeGroup === 'router') {
      moeWeightBytes += l.K * l.N * wb * getRepeat(l);
    }
  });
  const nonMoeWeightBytes = modelWeightBytes() - moeWeightBytes;
  // Split the non-MoE bucket: TP-replicated dense linears (MLA a-projections)
  // sit in FULL on every card and must NOT be divided by tpAttn — only the
  // row/col-parallel remainder is sharded. This matches tpInfo's timing path.
  const replWeightBytes          = replicatedWeightBytes();
  const shardedNonMoeWeightBytes = nonMoeWeightBytes - replWeightBytes;
  // MoE weights per card depend on parallelism strategy:
  //   TP:     all experts sliced 1/nCards on ffnDim → total_moe / nCards
  //   EP:     nExperts/nCards experts at full weight → total_moe / nCards  (same total)
  //   Hybrid: routed /nCards, shared /(nCards/epDeg), router replicated under TP/hybrid
  // For TP and EP, moeDivisor = nCards is exact. For hybrid, shared experts and
  // router have higher per-card cost; we compute the correction here.
  let moePerCard;
  if (moePar === 'hybrid' && epDeg > 1 && nCards > 1) {
    // Split moeWeightBytes into routed, shared, router components
    let routedW = 0, sharedW = 0, routerW = 0;
    const expertTp = Math.max(1, nCards / epDeg);
    layers.forEach(l => {
      if (l.type === 'moe') {
        const r = getRepeat(l);
        const {hid:h, ffnDim, nExperts:ne, nShared:ns=0} = l;
        const ew = (2*h*ffnDim + ffnDim*h) * wBpe();
        routedW += ne * ew * r;
        sharedW += ns * ew * r;
        routerW += h * ne * wBpe() * r;
      } else if (l.moeGroup === 'gate_up' || l.moeGroup === 'down') {
        routedW += l.moeExperts * l.K * l.N * wBpe() * getRepeat(l);
      } else if (l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down') {
        sharedW += l.K * l.N * wBpe() * getRepeat(l);
      } else if (l.moeGroup === 'router') {
        routerW += l.K * l.N * wBpe() * getRepeat(l);
      }
    });
    // Routed: distributed by EP, sliced by expertTp → /nCards
    // Shared: replicated across EP, sliced by expertTp → /expertTp
    // Router: replicated under hybrid → full (not sharded)
    moePerCard = routedW / nCards + sharedW / expertTp + routerW;
  } else {
    moePerCard = moeWeightBytes / nCards;
  }
  const wPerCard = (shardedNonMoeWeightBytes / tpAttn + replWeightBytes + moePerCard) / 1e9;
  const totalUsedGB = kvPerRank + wGB;
  const kvPerCard = kvGB;
  const usedPerCard = wPerCard + kvPerCard;
  const freePerCard = selChip.mem - usedPerCard;
  // bytesPerTok is the PER-CARD KV growth per additional context token, used to
  // compute how many more tokens fit in the remaining VRAM budget on a card.
  const bytesPerTokGlobal = cfg.kvCacheFn
    ? Math.max(1, kvCacheBytes(10000) - kvCacheBytes(9999))
    : 2 * (cfg.nLKV||cfg.nL) * cfg.batch * cfg.H * cfg.D * kvBpe();
  const bytesPerTok = bytesPerTokGlobal / ksf;
  const maxSeq = Math.max(0, Math.floor((selChip.mem - wPerCard) * 1e9 / Math.max(1, bytesPerTok)));
  const pTag = cfg.precision !== 'fp16' ? ` · <span style="color:#185FA5">${precisionLabel()}</span>` : '';
  const shTag = ksf > 1 ? ` <span style="font-size:10px;color:#888">(sharded ${ksf}× across TP)</span>` : '';

  const el = document.getElementById('cfgInfo');
  let cls = 'ok', lines = [];

  if (mode === 'prefill') {
    const globalBatchTag = dp > 1
      ? ` · global batch=<span>${cfg.batch*dp}</span> (${cfg.batch}×${dp} DP)`
      : ` · global batch=<span>${cfg.batch}</span> (pure TP)`;
    lines.push(`M = <span>${(cfg.batch*cfg.seqLen).toLocaleString()}</span> (batch×seq, per rank)${globalBatchTag}${pTag}`);
    lines.push(`Weights: <span>~${wGB.toFixed(1)} GB</span> total · <span>${wPerCard.toFixed(1)} GB</span>/card (TP=${tpAttn})`);
    const clusterKvTag = dp > 1
      ? ` · cluster KV <span>${kvClusterGB.toFixed(2)} GB</span> (=${kvPerRank.toFixed(2)} GB × ${dp} DP)`
      : '';
    lines.push(`KV cache: <span>${kvGB.toFixed(2)} GB</span>/card (nKV=${cfg.H}, D=${cfg.D})${shTag}${clusterKvTag} · Used: <span>${usedPerCard.toFixed(1)} GB</span> / ${selChip.mem} GB`);
    lines.push(`Max seq @ batch=${cfg.batch}: <span>~${maxSeq.toLocaleString()} tokens</span>`);
    if (usedPerCard > selChip.mem) {
      cls = 'err';
      lines.push(`<span class="err-line">⚠ Total VRAM ${usedPerCard.toFixed(1)} GB exceeds ${selChip.mem} GB per card. Reduce batch, seq, or use more TP.</span>`);
    } else if (usedPerCard > selChip.mem * 0.85) {
      cls = 'warn';
      lines.push(`<span class="warn-line">△ VRAM ${(usedPerCard/selChip.mem*100).toFixed(0)}% used — near limit.</span>`);
    }
  } else {
    const genKvPerRank = kvCacheBytes(cfg.seqLen + cfg.outTokens) / 1e9;
    const genKvGB = genKvPerRank / ksf;
    const genKvCluster = genKvPerRank * dp;
    const usedDecodePerCard = wPerCard + genKvGB;
    const maxOut = Math.max(0, Math.floor(((selChip.mem - wPerCard) * 1e9 / Math.max(1,bytesPerTok)) - cfg.seqLen));
    const globalBatchTag = dp > 1
      ? ` · global batch=<span>${cfg.batch*dp}</span> (${cfg.batch}×${dp} DP)`
      : ` · global batch=<span>${cfg.batch}</span> (pure TP)`;
    lines.push(`Query M = <span>${cfg.batch}</span> (per rank)${globalBatchTag} · KV context = <span>${kvLen.toLocaleString()}</span> tokens`);
    const clusterKvTag = dp > 1
      ? ` · cluster KV <span>${genKvCluster.toFixed(2)} GB</span> (=${genKvPerRank.toFixed(2)} GB × ${dp} DP)`
      : '';
    lines.push(`Weights: <span>~${wPerCard.toFixed(1)} GB</span>/card · KV: <span>${genKvGB.toFixed(2)} GB</span>/card (nKV=${cfg.H}, D=${cfg.D})${shTag}${clusterKvTag}`);
    lines.push(`Total used: <span>${usedDecodePerCard.toFixed(1)} GB</span> / ${selChip.mem} GB · Max output: <span>~${maxOut.toLocaleString()} tokens</span>`);
    if (usedDecodePerCard > selChip.mem) {
      cls = 'err';
      lines.push(`<span class="err-line">⚠ VRAM ${usedDecodePerCard.toFixed(1)} GB exceeds ${selChip.mem} GB. Reduce batch, seq, output tokens, or add TP.</span>`);
    } else if (usedDecodePerCard > selChip.mem * 0.85) {
      cls = 'warn';
      lines.push(`<span class="warn-line">△ VRAM ${(usedDecodePerCard/selChip.mem*100).toFixed(0)}% used — near limit.</span>`);
    }
  }
  el.className = 'info-strip ' + cls;
  el.innerHTML = lines.join('<br>');
}


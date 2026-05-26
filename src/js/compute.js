function attnDims(l) {
  const Sq = l.S, SkFull = l.S + (l.kvCache||0);
  const topk = (sparseMLA && l.sparseTopK) ? l.sparseTopK : 0;
  const Sk   = topk ? Math.min(SkFull, topk) : SkFull;   // core-attention span
  const kvHeadDim = l.D + (l.ropeExtra||0);              // MLA: latent + rope key
  const kvStreams = (l.mla || l.scoreOnly) ? 1 : 2;       // MLA / indexer: one stream
  return {Sq, Sk, SkFull, kvHeadDim, kvStreams};
}
function calcStdAttn(l) {
  const {B,H,D}=l;
  const ab=aBpe(), kvb=kvBpe();
  const nKV = l.nKV || H;
  const {Sq,Sk,kvHeadDim,kvStreams} = attnDims(l);
  const qk=2*B*H*Sq*kvHeadDim*Sk, sm=5*B*H*Sq*Sk, av=2*B*H*Sq*Sk*D;
  const kvBytes = kvStreams*B*nKV*Sk*kvHeadDim*kvb;       // K(+V) latent read
  if (l.scoreOnly) {  // DSA lightning indexer: QK^T scoring only, no softmax/AV
    const by = B*H*Sq*D*ab + kvBytes + 2*B*H*Sq*Sk*ab;
    return {flops:qk, bytes:by, ai:qk/by,
      sub:{'Index QK^T':{flops:qk,bytes:B*H*Sq*D*ab + kvBytes}}};
  }
  const fl=qk+sm+av;
  // HBM: Q + K + V + O reads/writes + score matrix (read+write for non-FA2)
  const by = (B*H*Sq*D + B*H*Sq*D)*ab           // Q + O (nH heads)
           + kvBytes                            // K(+V) latent (MLA: 1 stream)
           + 2*B*H*Sq*Sk*ab;                    // score matrix (rd+wr)
  return {flops:fl, bytes:by, ai:fl/by,
    sub:{'QK^T':{flops:qk,bytes:B*H*Sq*D*ab + kvBytes},
         'Softmax':{flops:sm,bytes:2*B*H*Sq*Sk*ab},
         'AV':{flops:av,bytes:2*B*H*Sq*Sk*ab + kvBytes + B*H*Sq*D*ab}}};
}
function calcFA2Attn(l) {
  const {B,H,D}=l;
  const ab=aBpe(), kvb=kvBpe();
  const nKV = l.nKV || H;
  const {Sq,Sk,kvHeadDim,kvStreams} = attnDims(l);
  const qk=2*B*H*Sq*kvHeadDim*Sk, sm=5*B*H*Sq*Sk, av=2*B*H*Sq*Sk*D;
  const kvBytes = kvStreams*B*nKV*Sk*kvHeadDim*kvb;       // K(+V) latent read
  if (l.scoreOnly) {  // DSA lightning indexer: QK^T scoring only, no softmax/AV
    const by = B*H*Sq*D*ab + kvBytes;                     // Q read + index-key read
    return {flops:qk, bytes:by, ai:qk/by, fa2:true,
      sub:{'Index QK^T':{flops:qk,bytes:by}}};
  }
  const fl=qk+sm+av;
  // FA2: score matrix stays in SRAM; only Q/K/V/O hit HBM.
  const by = (B*H*Sq*D + B*H*Sq*D)*ab           // Q + O (nH heads)
           + kvBytes;                           // K(+V) latent (MLA: 1 stream)
  return {flops:fl, bytes:by, ai:fl/by, fa2:true,
    sub:{'QK^T':{flops:qk,bytes:B*H*Sq*D*ab + kvBytes},
         'Softmax (SRAM)':{flops:sm,bytes:0},
         'AV':{flops:av,bytes:kvBytes + B*H*Sq*D*ab}}};
}
function calcL(l) {
  if (l.type==='attention') return l.fa2 ? calcFA2Attn(l) : calcStdAttn(l);
  if (l.type==='linear_attn') return calcLinearAttn(l);
  if (l.type==='moe') return calcMoE(l);
  let fl,by;
  const wb=wBpe(), ab=aBpe();
  if (l.type==='linear') {
    if (l.moeGroup === 'gate_up' || l.moeGroup === 'down') {
      const {M, K, N, moeExperts:nE, moeTopK:topK, moeFfnDim:ffnDim, moeHid:hid} = l;
      fl = 2 * M * K * N * topK;
      const distinct = expectedActivatedExperts(M, topK, nE);
      const wBytes = distinct * K * N * wb;
      // gate_up: input [M,hid] read once, output [M,topK,2*ffn]
      // down:    input [M,topK,ffn] read,   output [M,hid] (combined post-AR)
      const aBytes = l.moeGroup === 'gate_up'
        ? (M*K + M*N*topK) * ab
        : (M*K*topK + M*N) * ab;
      by = wBytes + aBytes;
    } else if (l.moeGroup === 'router') {
      fl = 2*l.M*l.K*l.N;
      by = l.M*l.K*ab + l.K*l.N*wb + l.M*l.N*ab;
    } else {
      fl = 2*l.M*l.K*l.N;
      by = l.M*l.K*ab + l.K*l.N*wb + l.M*l.N*ab;
    }
  }
  else {
    // Embedding lookup: gather M rows of [N=hid] from the [K=vocab, N] table.
    // No math (FLOPs ≈ 0 — just gather), HBM = M·N·wb (rows read) + M·N·ab (output write) + M·4 (indices).
    fl = 0;
    const N = (l.N && l.N > 1) ? l.N : 4096;
    by = l.M * N * (wb + ab) + l.M * 4;
  }
  return {flops:fl, bytes:by, ai:fl/by};
}

// Expected number of distinct experts activated by M tokens, each picking topK
// uniformly at random from nExperts. Classical balls-in-bins approximation:
//   E[activated] ≈ nExperts × (1 - (1 - topK/nExperts)^M)
// For M=1: returns topK. For large M: saturates at nExperts.
function expectedActivatedExperts(M, topK, nExperts) {
  if (M <= 0 || topK <= 0 || nExperts <= 0) return 0;
  if (topK >= nExperts) return nExperts;
  const pMiss = 1 - topK / nExperts;
  return nExperts * (1 - Math.pow(pMiss, M));
}

// MoE FFN compute model
//
// FLOPs: per active expert per token = 6 × hid × ffnDim (SwiGLU: gate+up+down, 2 matmuls + 1 output)
//   Total = 6 × M × hid × ffnDim × (topK + nShared)  — each token activates topK+shared
//   Router = 2 × M × hid × nExperts  (gating projection over all experts)
//
// HBM bytes: the subtlety — weight reads depend on how many *distinct* experts
//   actually get hit across the batch. For decode (M=1): exactly topK experts.
//   For prefill (M=seq): approaches nExperts (saturation). Use expected value.
//
// Sub-operators (what the GPU actually schedules):
//   1. Router gate          : small matmul [M,hid]×[hid,nExperts]
//   2. GroupedGEMM gate+up  : per-token topK experts, [hid]→[2*ffnDim] (SwiGLU fused)
//   3. GroupedGEMM down     : per-token topK experts, [ffnDim]→[hid]
//   These have very different AI and may hit different roofline regions.
//
// NOTE: all nExperts weights must live in VRAM (see modelWeightBytes),
//       but only expectedActivated experts are *streamed* per forward pass.
function calcMoE(l) {
  const {M, hid, ffnDim, topK, nExperts, nShared=0} = l;
  const wb=wBpe(), ab=aBpe();
  const activePerToken = topK + nShared;
  const distinctRouted = expectedActivatedExperts(M, topK, nExperts);
  const distinctTotal  = distinctRouted + nShared;

  const routerFlops = 2 * M * hid * nExperts;
  const routerBytes = M*hid*ab + hid*nExperts*wb + M*nExperts*ab;

  const gateUpFlops = 2 * M * hid * (2*ffnDim) * activePerToken;
  const gateUpWeightBytes = distinctTotal * 2 * hid * ffnDim * wb;
  const gateUpActBytes    = (M*hid + M*(2*ffnDim)*activePerToken) * ab;
  const gateUpBytes       = gateUpWeightBytes + gateUpActBytes;

  const downFlops = 2 * M * ffnDim * hid * activePerToken;
  const downWeightBytes = distinctTotal * ffnDim * hid * wb;
  const downActBytes    = (M*ffnDim*activePerToken + M*hid) * ab;
  const downBytes       = downWeightBytes + downActBytes;

  const flops = routerFlops + gateUpFlops + downFlops;
  const totalWeightStream = (hid*nExperts +
                             distinctTotal * (2*hid*ffnDim + ffnDim*hid)) * wb;
  const totalActBytes     = (M*hid*2 +
                             M*(2*ffnDim)*activePerToken +
                             M*ffnDim*activePerToken) * ab;
  const bytes = totalWeightStream + totalActBytes;

  return {flops, bytes, ai: flops/bytes, isMoE: true,
    distinctRouted,
    totalWeightBytes: (nExperts + nShared) * (2*hid*ffnDim + ffnDim*hid) * wb,
    sub: {
      'Router':          {flops: routerFlops, bytes: routerBytes},
      'GroupedGEMM gate+up': {flops: gateUpFlops, bytes: gateUpBytes},
      'GroupedGEMM down':    {flops: downFlops,   bytes: downBytes},
    }};
}

// GatedDeltaNet (linear attention) compute model:
// State size = nV*D × D (the delta-rule state matrix, D = head_dim)
// Per token per head: O(D²) operations for state update + O(nQK*D + nV*D) for QKV read
// FLOPs ≈ 2 * B * S * nV * D * D  (state update matmul, dominant term)
//        + 2 * B * S * nQK * D * D (query against state)
// Memory: read input activation M*K + write output M*nV*D; state stays in SRAM/registers
function calcLinearAttn(l) {
  const {M, nQK, nV, D} = l;
  // State update: each of nV heads does D×D matmul per token
  const stateUpdate = 2 * M * nV * D * D;
  // Query readout: each of nQK heads reads state
  const queryRead   = 2 * M * nQK * D * D;
  const flops = stateUpdate + queryRead;
  // HBM: read input (M*K) + write output (M*nV*D), state matrix is in SRAM
  const K = l.K || (nQK + nV) * D;
  const bytes = (M * K + M * nV * D) * aBpe();
  return {flops, bytes, ai: flops/bytes, isLinearAttn: true};
}
// Total FLOPs/bytes including repeat — returns zeros for repeat=0 entries
function calcLTotal(l) {
  const base = calcL(l); const r = getRepeat(l);
  return {flops:base.flops*r, bytes:base.bytes*r, ai:base.ai};
}

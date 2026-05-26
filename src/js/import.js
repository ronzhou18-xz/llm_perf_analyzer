function inferFromConfig(cfg_) {
  // ── Multimodal wrapper unwrapping ───────────────────────────────────────────
  // Qwen3.5 / Qwen3-VL / some LLaVA-family models use a multimodal wrapper config
  // where the real LLM fields live under cfg_.text_config (or llm_config).
  // The top-level only has {architectures, model_type, text_config, vision_config}.
  // We detect this by looking for a text_config / llm_config object and merging
  // it up so the rest of the logic below finds the LLM hyperparameters normally.
  // Preserve the top-level architectures/model_type since inner text_config may
  // have its own (e.g. inner model_type='qwen3_5_moe_text') and we want the outer.
  if (cfg_ && typeof cfg_ === 'object') {
    const inner = cfg_.text_config || cfg_.llm_config || cfg_.language_config;
    if (inner && typeof inner === 'object' && (inner.hidden_size != null || inner.num_hidden_layers != null)) {
      // Shallow-merge: inner fields win for LLM params, outer wins for arch/model_type
      // (because the outer arch like 'Qwen3_5MoeForConditionalGeneration' is the one
      //  that tells us it's a hybrid MoE — the inner model_type may just say 'qwen3_5_moe_text').
      const merged = {...inner, ...cfg_};
      // But we specifically want the *inner* values for core LLM dims
      const override = ['hidden_size','num_hidden_layers','num_attention_heads',
                        'num_key_value_heads','head_dim','intermediate_size','vocab_size',
                        'max_position_embeddings','hidden_act','tie_word_embeddings',
                        'layer_types','linear_num_key_heads','linear_num_value_heads',
                        'linear_key_head_dim','linear_value_head_dim','linear_num_qk_heads',
                        'linear_head_dim','num_experts','num_experts_per_tok',
                        'moe_intermediate_size','shared_expert_intermediate_size',
                        'decoder_sparse_step','n_routed_experts','n_shared_experts',
                        'first_k_dense_replace','kv_lora_rank','q_lora_rank',
                        'v_head_dim','qk_nope_head_dim','qk_rope_head_dim',
                        'num_dense_layers_in_shallow_end',
                        'index_n_heads','index_head_dim','index_topk',
                        'num_local_experts','shared_intermediate_size',
                        'attn_type_list'];
      for (const k of override) {
        if (inner[k] !== undefined) merged[k] = inner[k];
      }
      cfg_ = merged;
    }
  }

  cfg.kvCacheFn = undefined;  // clear any model-specific KV cache function from prior loads

  // Auto-detect FP8 from quantization_config
  const qc = cfg_.quantization_config;
  if (qc && (qc.quant_method === 'fp8' || qc.fmt === 'e4m3')) {
    cfg.precision = 'fp8w';
    const sel = document.getElementById('precisionSel');
    if (sel) sel.value = 'fp8w';
  }

  const arch=(cfg_.architectures||[cfg_.model_type||''])[0]||'unknown';
  const hid=cfg_.hidden_size||cfg_.d_model||cfg_.n_embd||1024;
  const nL=cfg_.num_hidden_layers||cfg_.n_layer||12;
  const nH=cfg_.num_attention_heads||cfg_.n_head||16;
  const nKV=cfg_.num_key_value_heads||nH;
  const hd=cfg_.head_dim||Math.floor(hid/nH);
  const ffn=cfg_.intermediate_size||cfg_.ffn_dim||(hid*4);
  const vocab=cfg_.vocab_size||32000;
  const seq=cfg_.max_position_embeddings||cfg_.max_seq_len||2048;
  const M=cfg.batch*(mode==='prefill'?cfg.seqLen:1);
  const act=cfg_.hidden_act||'';
  const isGated=act==='silu'||act==='swiglu'||/llama|mistral|gemma|qwen/i.test(arch);
  const Sattn=mode==='decode'?1:cfg.seqLen;
  const kvC=mode==='decode'?cfg.seqLen+Math.floor(cfg.outTokens/2):0;

  // ── Detect Qwen3.5 / Qwen3-Next hybrid architecture ────────────────────────
  // Qwen3.5 MoE (35B-A3B, 122B-A10B, 397B-A17B) and Qwen3-Next use a hybrid of
  // Gated DeltaNet (linear attention) + full attention in alternating groups,
  // combined with sparse MoE FFN (not dense!).
  //
  // Real field names (from Qwen3NextConfig / Qwen3_5 source):
  //   linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim
  //   layer_types: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', ...]
  //   num_experts, num_experts_per_tok, moe_intermediate_size, shared_expert_intermediate_size
  //   decoder_sparse_step (1 = every layer is MoE), first_k_dense_replace may or may not exist
  //
  // Even small Qwen3.5 variants (e.g. 9B-Base) ARE hybrid — the config carries
  // a 32-element layer_types (24 linear + 8 full) under text_config; only the
  // FFN is dense. Plain Qwen3 (3, 8, 14, 32B without ".5" in name) lacks both
  // layer_types AND num_experts and falls through to the standard dense branch.
  const hasLayerTypes = Array.isArray(cfg_.layer_types) && cfg_.layer_types.length > 0;
  const hasMoEFields  = cfg_.num_experts != null || cfg_.n_routed_experts != null;
  const isQwen35Hybrid = /qwen3[_.]?5|qwen3.?next/i.test(arch) && (hasLayerTypes || hasMoEFields);

  cfg.nH=nH; cfg.H=nKV; cfg.D=hd; cfg.nL=nL;

  if (isQwen35Hybrid) {
    // DeltaNet dimensions — use correct field names, with sensible fallbacks
    const nKeyH  = cfg_.linear_num_key_heads   || cfg_.linear_num_qk_heads || 16;
    const nValH  = cfg_.linear_num_value_heads || nKeyH*2                  || 32;
    const hdLinK = cfg_.linear_key_head_dim    || cfg_.linear_head_dim     || 128;
    const hdLinV = cfg_.linear_value_head_dim  || hdLinK                    || 128;

    // Count full-attn vs linear-attn layers
    let nFullAttn, nLinAttn;
    if (hasLayerTypes) {
      nFullAttn = cfg_.layer_types.filter(t => t==='attention' || t==='full_attention').length;
      nLinAttn  = cfg_.layer_types.filter(t => t==='linear' || t==='linear_attention' || t==='delta_net' || t==='gated_delta_net').length;
      if (nFullAttn === 0 && nLinAttn === 0) {
        nFullAttn = Math.round(nL / 4);
        nLinAttn  = nL - nFullAttn;
      }
    } else {
      // Qwen3-Next default pattern: 3 DeltaNet + 1 full-attn repeated
      nFullAttn = Math.round(nL / 4);
      nLinAttn  = nL - nFullAttn;
    }

    // MoE config (if present) — fall back to dense FFN if absent
    const hasMoE      = hasMoEFields;
    const nExperts    = cfg_.num_experts          || cfg_.n_routed_experts  || 0;
    const topK        = cfg_.num_experts_per_tok  || 8;
    const moeFfn      = cfg_.moe_intermediate_size || ffn;
    const sharedFfn   = cfg_.shared_expert_intermediate_size || 0;
    const nShared     = sharedFfn > 0 ? 1 : 0;
    // Dense replacement layers (rare in Qwen3.5, but handle it)
    const denseLayersTotal = cfg_.first_k_dense_replace || cfg_.num_dense_layers_in_shallow_end || 0;
    // Sparse step: if decoder_sparse_step=1, every non-dense-replaced layer is MoE
    const sparseStep  = cfg_.decoder_sparse_step  || 1;

    // Qwen3-Next/3.5 specific extras:
    //   attn_output_gate=true → Gated Attention: a learned sigmoid gate
    //     g = silu(W_g · x), W_g shape hid → nH·hd, applied element-wise to the
    //     attention output before O proj. Adds nFullAttn · hid · nH·hd weight.
    //   GatedDeltaNet ALWAYS has a learned output gate (that's what makes it
    //     "gated" — not just plain DeltaNet). g_proj shape hid → nValH·hdV per
    //     linear-attn layer. This is the dominant extra weight beyond Q/K/V/O.
    //   mtp_num_hidden_layers — auxiliary Multi-Token-Prediction head used in
    //     speculative decoding. Not modelled in the layer list (used only in
    //     MTP-mode inference); we surface its presence in the meta panel.
    const attnOutputGate = cfg_.attn_output_gate === true;
    const mtpLayers      = cfg_.mtp_num_hidden_layers || 0;

    cfg.nH=nH; cfg.H=nKV; cfg.D=hd; cfg.nL=nL; cfg.nLKV=nFullAttn;
    const qkvNFull = nH*hd + 2*nKV*hd;

    // Build layer list: embed → DeltaNet stack → full-attn stack → (MoE FFN across ALL layers) → LM head
    // We model FFN as one big MoE block with repeat=nL-denseLayersTotal for simplicity,
    // because both DeltaNet and full-attn layers share the same MoE FFN in Qwen3.5.
    const moeLayers = Math.max(0, nL - denseLayersTotal);

    const res=[
      {id:Date.now()+1,  name:'Token embed',                       type:'embedding',  M, K:vocab, N:hid,       repeat:1},
      // DeltaNet (linear attention) block
      {id:Date.now()+2,  name:`DeltaNet QK proj (×${nLinAttn})`,    type:'linear',     M, K:hid,  N:nKeyH*hdLinK*2, repeat:nLinAttn},
      {id:Date.now()+3,  name:`DeltaNet V proj (×${nLinAttn})`,     type:'linear',     M, K:hid,  N:nValH*hdLinV,   repeat:nLinAttn},
      // GatedDeltaNet output gate (g): col-par linear (matches "gate" → no AR)
      {id:Date.now()+14, name:`DeltaNet g_proj (gate, ×${nLinAttn})`, type:'linear',   M, K:hid,  N:nValH*hdLinV,   repeat:nLinAttn},
      {id:Date.now()+4,  name:`GatedDeltaNet (×${nLinAttn})`,       type:'linear_attn', M, K:hid, nQK:nKeyH, nV:nValH, D:hdLinV, repeat:nLinAttn},
      {id:Date.now()+5,  name:`DeltaNet O proj (×${nLinAttn})`,     type:'linear',     M, K:nValH*hdLinV, N:hid,    repeat:nLinAttn},
      // Full attention block (Gated Attention if attn_output_gate=true)
      {id:Date.now()+6,  name:`Attn QKV (×${nFullAttn})`,           type:'linear',     M, K:hid, N:qkvNFull,       repeat:nFullAttn},
      {id:Date.now()+7,  name:`Attention (×${nFullAttn})`,          type:'attention',  B:cfg.batch, S:Sattn, H:nH, nKV, D:hd, kvCache:kvC, fa2:true, repeat:nFullAttn},
    ];
    // Gated Attention output gate (Qwen3-Next/3.5): col-par linear, no AR.
    if (attnOutputGate) {
      res.push({id:Date.now()+15, name:`Attn output gate (×${nFullAttn})`, type:'linear', M, K:hid, N:nH*hd, repeat:nFullAttn});
    }
    res.push({id:Date.now()+8, name:`Attn O proj (×${nFullAttn})`, type:'linear', M, K:nH*hd, N:hid, repeat:nFullAttn});

    // FFN: MoE or dense
    if (hasMoE) {
      if (denseLayersTotal > 0) {
        res.push({id:Date.now()+9,  name:`Dense FFN gate+up (×${denseLayersTotal})`, type:'linear', M, K:hid, N:ffn*2, repeat:denseLayersTotal});
        res.push({id:Date.now()+10, name:`Dense FFN down (×${denseLayersTotal})`,    type:'linear', M, K:ffn, N:hid,   repeat:denseLayersTotal});
      }
      res.push({id:Date.now()+11, name:`MoE FFN (×${moeLayers})`, type:'moe', M, hid, ffnDim:moeFfn, topK, nExperts, nShared, repeat:moeLayers});
    } else {
      // Hybrid DeltaNet but dense FFN (rare but possible)
      res.push({id:Date.now()+11, name:`FFN gate+up (×${nL})`, type:'linear', M, K:hid, N:ffn*2, repeat:nL});
      res.push({id:Date.now()+12, name:`FFN down (×${nL})`,    type:'linear', M, K:ffn, N:hid,   repeat:nL});
    }

    // LM head: always emit (the matmul still happens at runtime).
    // When tie_word_embeddings=true, weights are shared with the embedding table
    // so modelWeightBytes() must skip the tied entry to avoid double-counting params.
    res.push({id:Date.now()+13, name:'LM head', type:'linear', M, K:hid, N:vocab,
              repeat:1, tied: cfg_.tie_word_embeddings === true});

    return {layers:res, meta:{arch, hid, nL, nH, nKV, hd, ffn:hasMoE?moeFfn:ffn, vocab, seq, isGated:true,
                              isQwen35:true, nFullAttn, nLinAttn, nKeyH, nValH, hdLinK, hdLinV,
                              hasMoE, nExperts, topK, moeFfn, nShared, denseLayers:denseLayersTotal,
                              attnOutputGate, mtpLayers}};
  }

  // ── Qwen3-MoE (Qwen3-235B-A22B etc.) ────────────────────────────────────────
  // Identified by model_type=qwen3_moe or architectures containing Qwen3Moe
  // Config fields: num_experts, num_experts_per_tok, moe_intermediate_size, decoder_sparse_step
  // Dense-vs-MoE per-layer rule from transformers/Qwen3MoeDecoderLayer:
  //   if (layer_idx not in mlp_only_layers) and (num_experts > 0) and
  //      ((layer_idx + 1) % decoder_sparse_step == 0):
  //     use SparseMoeBlock
  //   else:
  //     use dense Qwen3MoeMLP(intermediate_size)
  // So `mlp_only_layers` lists DENSE layer indices. (decoder_sparse_step=1 +
  // empty mlp_only_layers ⇒ every layer is MoE — the canonical Qwen3-235B-A22B.)
  const isQwen3MoE = /qwen3.*moe|qwen3moe/i.test(arch);
  if (isQwen3MoE) {
    const nExperts    = cfg_.num_experts         || 128;
    const topK        = cfg_.num_experts_per_tok || 8;
    const moeFfn      = cfg_.moe_intermediate_size || 1536;
    // Dense-layer count: prefer explicit mlp_only_layers; fall back to legacy fields.
    const mlpOnly     = Array.isArray(cfg_.mlp_only_layers) ? cfg_.mlp_only_layers : null;
    const sparseStep  = cfg_.decoder_sparse_step || 1;
    let denseLayers;
    if (mlpOnly !== null) {
      // Layers in mlp_only_layers are dense; plus any layer where (idx+1) % step != 0.
      let cnt = 0;
      for (let i = 0; i < nL; i++) {
        const isMoE = !mlpOnly.includes(i) && ((i + 1) % sparseStep === 0);
        if (!isMoE) cnt++;
      }
      denseLayers = cnt;
    } else {
      denseLayers = sparseStep === 1 ? 0 : (cfg_.num_dense_layers_in_shallow_end || 1);
    }
    const moeLayers   = nL - denseLayers;
    cfg.nH=nH; cfg.H=nKV; cfg.D=hd; cfg.nL=nL; cfg.nLKV=undefined;
    const qkvN2 = nH*hd + 2*nKV*hd;
    const res=[
      {id:Date.now()+1, name:'Token embed',          type:'embedding', M, K:vocab,  N:hid,   repeat:1},
      {id:Date.now()+2, name:`Attn QKV (×${nL})`,    type:'linear',    M, K:hid,    N:qkvN2, repeat:nL},
      {id:Date.now()+3, name:`Attention (×${nL})`,    type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV, D:hd, kvCache:kvC, fa2:true, repeat:nL},
      {id:Date.now()+4, name:`Attn O proj (×${nL})`,  type:'linear',    M, K:nH*hd, N:hid,   repeat:nL},
    ];
    if(denseLayers>0){
      res.push({id:Date.now()+5,name:`Dense FFN gate+up (×${denseLayers})`,type:'linear',M,K:hid,N:ffn*2,repeat:denseLayers});
      res.push({id:Date.now()+6,name:`Dense FFN down (×${denseLayers})`,   type:'linear',M,K:ffn,N:hid,  repeat:denseLayers});
    }
    res.push({id:Date.now()+7, name:`MoE FFN (×${moeLayers})`,  type:'moe', M, hid, ffnDim:moeFfn, topK, nExperts, nShared:cfg_.shared_expert_intermediate_size?1:0, repeat:moeLayers});
    res.push({id:Date.now()+8,name:'LM head',type:'linear',M,K:hid,N:vocab,repeat:1,tied:cfg_.tie_word_embeddings===true});
    return {layers:res, meta:{arch,hid,nL,nH,nKV,hd,ffn:moeFfn,vocab,seq,isQwen3MoE:true,nExperts,topK,moeFfn,denseLayers}};
  }

  // ── DeepSeek-V3 / V3.1 / V3.2-Exp / R1 / Kimi-K2 / K2.5 (MLA + MoE) ──────
  // Identified by: model_type=deepseek_v3 / deepseek_v32, arch contains DeepSeek,
  //   or model_type=kimi_k2 (Kimi K2 uses a DeepseekV3-style architecture).
  // MLA (Multi-head Latent Attention): compressed KV cache via low-rank projection
  //   q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128, nH=128
  //   KV cache per layer = batch * seq * kv_lora_rank * 2 bytes (compressed) — far smaller than GQA
  // MoE: 256 routed experts + 1 shared, top-8 active, moe_intermediate_size=2048
  // Dense layers: first 3 layers use dense FFN (intermediate_size=18432)
  // Kimi K2: 384 routed + 1 shared, top-8, hid=7168, 61 layers, first_k_dense_replace=1.
  //   Kimi K2 / K2.5 use PLAIN MLA — no DSA (its indexer fields are absent).
  //
  // v83: DeepSeek-V3.2-Exp (model_type=deepseek_v32, arch=DeepseekV32ForCausalLM)
  //   keeps the V3 dims but adds DeepSeek Sparse Attention (DSA): a lightning
  //   indexer (index_n_heads=64 small heads, index_head_dim=128, FP8) scores the
  //   full token history, and the MLA core attends only the top-k=index_topk
  //   (2048) selected tokens. Modelled with the SAME generic DSA machinery the
  //   GLM-5 path uses — indexer Q/K proj + scoreOnly indexer-score layer, both
  //   indexerLayer:true (→ getRepeat 0 when the sparse-MLA toggle is off), and
  //   sparseTopK on the core MLA attention. Toggle off → dense-MLA upper bound.
  const isDSV3 = /deepseek.?v3|deepseekv3|kimi.?k2/i.test(arch)
              || cfg_.model_type === 'kimi_k2'
              || cfg_.model_type === 'deepseek_v3'
              || cfg_.model_type === 'deepseek_v32';
  if (isDSV3) {
    const nExperts    = cfg_.n_routed_experts     || 256;
    const topK        = cfg_.num_experts_per_tok  || 8;
    const nShared     = cfg_.n_shared_experts     || 1;
    const moeFfn      = cfg_.moe_intermediate_size || 2048;
    const denseLayers = cfg_.first_k_dense_replace || 3;
    const denseFfn    = cfg_.intermediate_size     || 18432;
    const moeLayers   = nL - denseLayers;
    // MLA dimensions
    const kvRank      = cfg_.kv_lora_rank         || 512;
    const qRank       = cfg_.q_lora_rank           || 1536;
    const ropeHd      = cfg_.qk_rope_head_dim      || 64;
    const nopeHd      = cfg_.qk_nope_head_dim      || 128;
    const vHd         = cfg_.v_head_dim            || 128;
    // DSA detection — DeepSeek-V3.2-Exp reports model_type=deepseek_v32 /
    // arch=DeepseekV32ForCausalLM and carries index_n_heads / index_head_dim /
    // index_topk. Presence of the indexer fields is the surest signal; plain V3
    // / R1 / Kimi-K2 lack them and stay dense MLA.
    const isDSA       = cfg_.model_type === 'deepseek_v32'
                     || /deepseek.?v3\.?2|deepseekv32/i.test(arch)
                     || (cfg_.index_topk != null && cfg_.index_n_heads != null);
    // DSA lightning-indexer params (only meaningful when isDSA).
    // DeepSeek-V3.2-Exp: 64 indexer heads, head_dim 128, top-k 2048.
    const idxNHeads   = cfg_.index_n_heads  || 64;
    const idxHeadDim  = cfg_.index_head_dim || 128;
    const idxTopK     = cfg_.index_topk     || 2048;
    // KV cache compresses to a single latent stream of (kv_lora_rank+qk_rope_head_dim);
    // we use kv_lora_rank as the per-head effective dim for the Attention sub-call.
    cfg.nH=nH; cfg.H=1; cfg.D=kvRank; cfg.nL=nL; cfg.nLKV=undefined;
    // v80 (Fix B): MLA caches ONE compressed latent stream per token per layer of
    // (kv_lora_rank + qk_rope_head_dim) elements — there is no separate K and V
    // cache. The generic kvCacheBytes() 2× (K+V) factor would overstate MLA KV by
    // 2·kv_lora_rank / (kv_lora_rank+qk_rope) ≈ 1.78×, so MLA needs its own fn.
    // v83: a DSA model additionally stores ONE small MQA indexer key of
    // index_head_dim per token per layer (the indexer's selection key store).
    // KV-cache VRAM is full-sequence regardless of the toggle — sparsity caps
    // what is READ, not what is STORED — but with sparse modelling off there is
    // no indexer at all, so its key store is dropped too.
    cfg.kvCacheFn = function(seqLen_) {
      let b = nL * cfg.batch * (kvRank + ropeHd) * seqLen_ * kvBpe();
      if (isDSA && sparseMLA) b += nL * cfg.batch * idxHeadDim * seqLen_ * kvBpe();
      return b;
    };
    // DSA lightning indexer (V3.2-Exp): a lightweight MQA-style scorer.
    //   Q/K proj: hid → idxNHeads·idxHeadDim (queries) + idxHeadDim (one shared key).
    //   Score:    QK^T over the FULL sequence (O(L²) — the price of sparsity),
    //             scoreOnly (no softmax/AV). Both are TP-replicated (the per-head
    //             score sum + top-k select doesn't head-split cleanly) and carry
    //             indexerLayer:true → getRepeat()→0 when sparse-MLA is off.
    const idxLayers = isDSA ? [
      {id:Date.now()+11, name:`DSA indexer Q/K proj (×${nL})`, type:'linear',
       M, K:hid, N:idxNHeads*idxHeadDim + idxHeadDim, repeat:nL,
       tpMode:'replicated', indexerLayer:true},
      {id:Date.now()+12, name:`DSA indexer score (×${nL})`, type:'attention',
       B:cfg.batch, S:Sattn, H:idxNHeads, nKV:1, D:idxHeadDim, kvCache:kvC,
       fa2:true, repeat:nL, scoreOnly:true, indexerLayer:true, tpReplicated:true},
    ] : [];
    const res=[
      {id:Date.now()+1, name:'Token embed',           type:'embedding', M, K:vocab, N:hid,     repeat:1},
      // MLA Q is factored: hid → q_lora_rank → nH·(nope+rope)
      //   q_a_proj (Q down) is REPLICATED — its output feeds q_a_layernorm over the
      //     full latent dim, so DeepSeek/vLLM keep it replicated rather than reduce.
      //   q_b_proj (Q up) is COLUMN-parallel — output split cleanly by nH, no AllReduce.
      {id:Date.now()+2, name:`MLA Q down (×${nL})`,   type:'linear',    M, K:hid,         N:qRank,                repeat:nL, tpMode:'replicated'},
      {id:Date.now()+10,name:`MLA Q up (×${nL})`,     type:'linear',    M, K:qRank,       N:nH*(nopeHd+ropeHd),   repeat:nL, tpMode:'col'},
      // MLA KV_a projects hid → (kv_lora_rank + qk_rope_head_dim) — the rope head is
      // shared across query heads and appended to the latent K cache. REPLICATED —
      // its output feeds kv_a_layernorm over the full latent stream, so no AllReduce.
      {id:Date.now()+3, name:`MLA KV down (×${nL})`,  type:'linear',    M, K:hid,         N:kvRank+ropeHd,        repeat:nL, tpMode:'replicated'},
      // DSA indexer layers (V3.2-Exp only) sit between KV_a and the core attention.
      ...idxLayers,
      // Attention with absorption: effective K/V latent dim = kv_lora_rank. The
      // kv_b_proj weight (kv_lora → nH·(nope+v_head)) is fused into Q/O — its FLOPs
      // are accounted for via the larger Q_up output dim and the O_proj output dim.
      // mla:true → single latent KV stream (kvStreams=1); ropeExtra → QK^T score
      // uses (kv_lora_rank + qk_rope_head_dim), AV uses kv_lora_rank only.
      // sparseTopK (DSA) → with the sparse-MLA toggle on, the core attends only the
      // top-k indexer-selected tokens (Sk capped to min(Sk, index_topk)).
      {id:Date.now()+4, name:`MLA Attention (×${nL})`,  type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV:1, D:kvRank, kvCache:kvC, fa2:true, repeat:nL, mla:true, ropeExtra:ropeHd, ...(isDSA ? {sparseTopK:idxTopK} : {})},
      // O proj is ROW-parallel (input split by nH) → exactly ONE AllReduce per MLA block.
      {id:Date.now()+5, name:`MLA O proj (×${nL})`,    type:'linear',    M, K:nH*vHd,N:hid,    repeat:nL, tpMode:'row'},
      {id:Date.now()+6, name:`Dense FFN gate+up (×${denseLayers})`, type:'linear', M, K:hid, N:denseFfn*2, repeat:denseLayers},
      {id:Date.now()+7, name:`Dense FFN down (×${denseLayers})`,    type:'linear', M, K:denseFfn, N:hid,    repeat:denseLayers},
      {id:Date.now()+8, name:`MoE FFN (×${moeLayers})`, type:'moe',  M, hid, ffnDim:moeFfn, topK, nExperts, nShared, repeat:moeLayers},
    ];
    res.push({id:Date.now()+9,name:'LM head',type:'linear',M,K:hid,N:vocab,repeat:1,tied:cfg_.tie_word_embeddings===true});
    return {layers:res, meta:{arch,hid,nL,nH,nKV:1,hd:kvRank,ffn:moeFfn,vocab,seq,isDSV3:true,
                              isKimiK2: /kimi/i.test(arch) || cfg_.model_type === 'kimi_k2',
                              isDSA, idxNHeads, idxHeadDim, idxTopK,
                              nExperts,topK,nShared,moeFfn,denseLayers,kvRank}};
  }

  // ── DeepSeek-V4 (Pro / Flash) — Hybrid HCA/CSA/SWA-only + MoE ────────────
  // Identified by: model_type=deepseek_v4 or architecture contains DeepseekV4.
  // Unlike V3 (MLA), V4 uses MQA (num_key_value_heads=1) with a large head_dim=512
  // plus a hybrid attention mechanism. Per-layer kind is encoded in compress_ratios[]:
  //   r >= 32 → HCA      (Heavily Compressed Attention, ratio≈128, full-context)
  //   1<r<32  → CSA      (Compressed Sparse Attention, ratio≈4, full-context + top-k)
  //   r == 0  → SWA-only (local sliding-window attention, win=128, no global path)
  // The trailing 0 in compress_ratios is the MTP head (auxiliary, not a layer here).
  // V4-Pro is all HCA/CSA (no SWA-only). V4-Flash uses SWA-only on layers 0,1
  //   per the HF blog: "the MTP block at the end runs sliding-window only" and
  //   the same window applies to those leading no-compression entries.
  // New projections vs V3 (BOTH V4-Pro and V4-Flash carry these):
  //   q_lora_rank  — Q is factored: hid → qRank → nH*head_dim
  //   o_lora_rank  — O is factored: nH*head_dim → oRank → hid
  //   index_head_dim, index_n_heads, index_topk — DSA-style sparse indexer (CSA)
  // KV cache: accurately modelled via custom kvCacheFn that sums per-layer
  //   compressed KV bytes = 2 * batch * numKV * head_dim * effSeq * kvBpe,
  //   where effSeq = ceil(seq/ratio) for HCA/CSA, min(seq, sliding_window) for SWA-only.
  // Attention COMPUTE:
  //   SWA-only layers carry a swaWindow flag honored by applyConfigToLayers, so
  //     S/kvCache stay window-bounded across batch/seq/mode changes (compute is
  //     ~window-bounded rather than full-seq).
  //   HCA/CSA still use full uncompressed Sk in compute (overestimates long-ctx
  //     attention FLOPs; same caveat GLM-5 DSA carries). KV cache is precise.
  // Reference configs (from HF):
  //   V4-Pro:   nL=61, hid=7168, nH=128, head_dim=512, 384e top-6 + 1 shared,
  //             moe_ffn=3072, q_lora_rank=1536, o_lora_rank=1024, o_groups=16.
  //             compress_ratios alternates 128/4 — all HCA/CSA, no SWA-only.
  //   V4-Flash: nL=43, hid=4096, nH=64,  head_dim=512, 256e top-6 + 1 shared,
  //             moe_ffn=2048, q_lora_rank=1024, o_lora_rank=1024, o_groups=8.
  //             compress_ratios = [0, 0, 128, 4, ..., 4, 0] — layers 0,1 SWA-only.
  const isDSV4 = /deepseek.?v4|deepseekv4/i.test(arch)
              || cfg_.model_type === 'deepseek_v4';
  if (isDSV4) {
    const nExperts    = cfg_.n_routed_experts     || 384;
    const topK        = cfg_.num_experts_per_tok  || 6;
    const nShared     = cfg_.n_shared_experts     || 1;
    const moeFfn      = cfg_.moe_intermediate_size || 3072;
    const denseLayers = cfg_.first_k_dense_replace || 0;   // V4 typically has no dense FFN
    const denseFfn    = cfg_.intermediate_size    || moeFfn;
    const moeLayers   = Math.max(0, nL - denseLayers);

    const headDim   = cfg_.head_dim             || 512;
    const numKV     = cfg_.num_key_value_heads  || 1;
    const qRank     = cfg_.q_lora_rank          || 1536;
    const oRank     = cfg_.o_lora_rank          || null;   // both Pro and Flash carry this (1024)
    const indexTopK = cfg_.index_topk           || 512;
    const slidingWindow  = cfg_.sliding_window_size || cfg_.sliding_window || 128;
    const compressRatios = Array.isArray(cfg_.compress_ratios) ? cfg_.compress_ratios : [];
    const mtpLayers = cfg_.num_nextn_predict_layers || 0;

    // Classify each of the nL transformer layers by its compression ratio.
    // Convention: ratio>=32 → HCA (heavy), 1<ratio<32 → CSA (light/sparse),
    // ratio==0 → SWA-only (local sliding window). The trailing 0 in compress_ratios
    // for V4 corresponds to the MTP layer (not counted here since we iterate 0..nL-1).
    const HCA_THRESHOLD = 32;
    let nHCA = 0, nCSA = 0, nSWAOnly = 0;
    for (let i = 0; i < nL; i++) {
      const r = compressRatios[i] != null ? compressRatios[i] : 0;
      if (r === 0)                  nSWAOnly++;
      else if (r >= HCA_THRESHOLD)  nHCA++;
      else                          nCSA++;
    }
    // Fallback when config omits compress_ratios: detect Pro (61L, 384e) vs
    // Flash (43L, 256e — layers 0,1 are SWA-only) and apply the matching pattern.
    if (compressRatios.length === 0) {
      const looksLikeFlash = /v4.?flash/i.test(arch) || nL <= 50 || nExperts <= 256;
      if (looksLikeFlash) {
        // V4-Flash: 2 leading SWA-only + remainder alternating HCA/CSA
        nSWAOnly = Math.min(2, nL);
        const remaining = Math.max(0, nL - nSWAOnly);
        nHCA = Math.floor(remaining / 2);
        nCSA = remaining - nHCA;
      } else {
        // V4-Pro: ~half HCA / ~half CSA, no SWA-only
        nHCA = Math.ceil(nL / 2);
        nCSA = nL - nHCA;
        nSWAOnly = 0;
      }
    }

    // Effective per-layer KV head_dim (averaged, for display only).
    // For SWA-only layers, the equivalent compression ratio is seq/window (i.e.
    // KV held = headDim * window per token, vs headDim * seq for uncompressed).
    // KV bytes in VRAM are computed precisely via cfg.kvCacheFn below.
    const swaEffRatio = Math.max(1, seq / slidingWindow);
    let totalEffPerLayer = 0;
    if (compressRatios.length > 0) {
      for (let i = 0; i < nL; i++) {
        const r = compressRatios[i] != null ? compressRatios[i] : 0;
        totalEffPerLayer += (r > 0) ? (headDim / r) : (headDim / swaEffRatio);
      }
    } else {
      // Fallback: weight by inferred nHCA/nCSA/nSWAOnly using representative ratios.
      totalEffPerLayer = nHCA * (headDim / 128)
                       + nCSA * (headDim / 4)
                       + nSWAOnly * (headDim / swaEffRatio);
    }
    const avgEffD = Math.max(1, Math.round(totalEffPerLayer / nL));

    // cfg.H / cfg.D drive the simple VRAM summary. Set them to the averaged
    // effective compressed dim so the on-screen "KV cache (nKV=.., D=..)" line
    // roughly matches reality. Precise numbers come from kvCacheFn.
    cfg.nH = nH; cfg.H = numKV; cfg.D = avgEffD; cfg.nL = nL; cfg.nLKV = undefined;

    // Precise KV cache: sum across layers honoring each layer's mode.
    // HCA/CSA → ceil(seq/ratio); SWA-only (r==0) → min(seq, sliding_window).
    cfg.kvCacheFn = function(seqLen_) {
      const kb = kvBpe();
      const swaSeq = Math.min(seqLen_, slidingWindow);
      if (compressRatios.length === 0) {
        // Fallback: use the inferred Pro/Flash pattern with representative ratios
        // (HCA=128, CSA=4) plus SWA-only window contribution.
        const hcaSeq = Math.ceil(seqLen_ / 128);
        const csaSeq = Math.ceil(seqLen_ / 4);
        const totalSeq = nHCA * hcaSeq + nCSA * csaSeq + nSWAOnly * swaSeq;
        return 2 * cfg.batch * numKV * headDim * totalSeq * kb;
      }
      let totalBytes = 0;
      for (let i = 0; i < nL; i++) {
        const r = compressRatios[i] != null ? compressRatios[i] : 0;
        const layerSeq = (r > 0) ? Math.ceil(seqLen_ / r) : swaSeq;
        totalBytes += 2 * cfg.batch * numKV * headDim * layerSeq * kb;
      }
      return totalBytes;
    };

    // Per-layer projection dims
    const qUpOut  = nH * headDim;           // qRank → nH*head_dim
    const kvOut   = 2 * numKV * headDim;    // combined K and V, direct proj

    const res = [
      {id:Date.now()+1,  name:'Token embed',                 type:'embedding', M, K:vocab,  N:hid,    repeat:1},
      // Q path (two-stage low-rank). Named 'a-proj' / 'b-proj' to match DeepSeek's
      // own q_a_proj / q_b_proj nomenclature AND to avoid the row-par regex trigger
      // on the word 'down' — Q's down-projection is col-parallel (splits qRank output
      // across TPs, no AllReduce), unlike O's down-projection which is row-parallel.
      {id:Date.now()+2,  name:`Q a-proj (×${nL})`,           type:'linear',    M, K:hid,    N:qRank,  repeat:nL},
      {id:Date.now()+3,  name:`Q b-proj (×${nL})`,           type:'linear',    M, K:qRank,  N:qUpOut, repeat:nL},
      // KV path (direct, MQA)
      {id:Date.now()+4,  name:`KV proj (×${nL})`,            type:'linear',    M, K:hid,    N:kvOut,  repeat:nL},
    ];

    // Attention compute, split by type for visibility.
    // HCA/CSA: full Sk in compute (KV cache itself is correctly compressed via
    //   kvCacheFn above; the compute side overestimates long-ctx FLOPs by a
    //   factor of ~ratio — known caveat shared with GLM-5 DSA).
    if (nHCA > 0) {
      res.push({id:Date.now()+5, name:`HCA Attention (×${nHCA})`, type:'attention',
                B:cfg.batch, S:Sattn, H:nH, nKV:numKV, D:headDim, kvCache:kvC, fa2:true, repeat:nHCA});
    }
    if (nCSA > 0) {
      res.push({id:Date.now()+6, name:`CSA Attention (×${nCSA})`, type:'attention',
                B:cfg.batch, S:Sattn, H:nH, nKV:numKV, D:headDim, kvCache:kvC, fa2:true, repeat:nCSA});
    }
    // SWA-only layers (V4-Flash layers 0,1; absent on V4-Pro): local window
    //   attention. swaWindow is honored by applyConfigToLayers so S/kvCache
    //   stay window-bounded across batch/seq/mode changes:
    //     prefill → S = min(seqLen, win), kvCache = 0
    //     decode  → S = 1,                kvCache = min(win, full_kv)
    if (nSWAOnly > 0) {
      const swaS  = mode==='prefill' ? Math.min(cfg.seqLen, slidingWindow) : 1;
      const swaKv = mode==='decode'  ? Math.min(slidingWindow, kvC) : 0;
      res.push({id:Date.now()+7, name:`SWA-only Attention (×${nSWAOnly})`, type:'attention',
                B:cfg.batch, S:swaS, H:nH, nKV:numKV, D:headDim, kvCache:swaKv, fa2:true,
                swaWindow:slidingWindow, repeat:nSWAOnly});
    }

    // O projection (may be low-rank via o_lora_rank)
    if (oRank) {
      res.push({id:Date.now()+8, name:`O down-proj (×${nL})`, type:'linear', M, K:nH*headDim, N:oRank, repeat:nL});
      res.push({id:Date.now()+9, name:`O up-proj (×${nL})`,   type:'linear', M, K:oRank,      N:hid,   repeat:nL});
    } else {
      res.push({id:Date.now()+8, name:`O proj (×${nL})`,      type:'linear', M, K:nH*headDim, N:hid,   repeat:nL});
    }

    // Dense FFN (almost always 0 in V4, but keep the branch for future configs)
    if (denseLayers > 0) {
      res.push({id:Date.now()+10, name:`Dense FFN gate+up (×${denseLayers})`, type:'linear', M, K:hid, N:denseFfn*2, repeat:denseLayers});
      res.push({id:Date.now()+11, name:`Dense FFN down (×${denseLayers})`,    type:'linear', M, K:denseFfn, N:hid,    repeat:denseLayers});
    }
    // MoE FFN (every non-dense layer)
    if (moeLayers > 0) {
      res.push({id:Date.now()+12, name:`MoE FFN (×${moeLayers})`, type:'moe', M, hid, ffnDim:moeFfn, topK, nExperts, nShared, repeat:moeLayers});
    }

    res.push({id:Date.now()+13, name:'LM head', type:'linear', M, K:hid, N:vocab,
              repeat:1, tied: cfg_.tie_word_embeddings === true});

    const isFlash = /v4.?flash/i.test(arch) || nExperts <= 256;
    return {layers:res, meta:{arch, hid, nL, nH, nKV:numKV, hd:headDim, ffn:moeFfn, vocab, seq,
                              isDSV4:true, isFlash,
                              nExperts, topK, nShared, moeFfn, denseLayers,
                              nHCA, nCSA, nSWAOnly, slidingWindow, indexTopK,
                              qRank, oRank, hasORank: oRank != null,
                              mtpLayers, avgEffD}};
  }

  // ── GLM-4 MoE family (GLM-4.5, GLM-4.5-Air, GLM-4.6, GLM-4.7, GLM-4.7-Flash, GLM-5) ──
  // Identified by: model_type=glm4_moe / glm4_moe_lite / glm_moe_dsa, or arch
  //   contains Glm4Moe / GlmMoeDsa.
  // Field names: n_routed_experts, n_shared_experts, first_k_dense_replace,
  //              num_experts_per_tok, moe_intermediate_size.
  // GLM-4.5/4.6/4.7 use plain GQA. GLM-4.7-Flash uses MLA (kv_lora_rank, q_lora_rank,
  //   v_head_dim, qk_nope_head_dim, qk_rope_head_dim) just like DeepSeek-V3.
  // GLM-5 / GLM-5.1 (744B/40B, model_type=glm_moe_dsa, arch=GlmMoeDsaForCausalLM)
  //   pair MLA with DeepSeek Sparse Attention (DSA): a lightning indexer scores the
  //   full token history and the MLA core attends only the top-k selected tokens.
  //   v80 models DSA explicitly (indexer layer + top-k-restricted core attention)
  //   when the sparse-MLA toggle is on; off → conservative dense-MLA upper bound.
  const isGLM4Moe = /glm.?4.?moe|glm4moe|glm.?4.?moe.?lite|glm.?moe.?dsa|glm-?dsa/i.test(arch)
                 || cfg_.model_type === 'glm4_moe'
                 || cfg_.model_type === 'glm4_moe_lite'
                 || cfg_.model_type === 'glm-dsa'
                 || cfg_.model_type === 'glm_moe_dsa'
                 || (/glm/i.test(arch) && (cfg_.n_routed_experts != null));
  if (isGLM4Moe) {
    const nExperts    = cfg_.n_routed_experts    || 128;
    const topK        = cfg_.num_experts_per_tok || 8;
    const nShared     = cfg_.n_shared_experts    || 1;
    const moeFfn      = cfg_.moe_intermediate_size || 1408;
    const denseLayers = cfg_.first_k_dense_replace || 1;
    const denseFfn    = cfg_.intermediate_size    || 10944;
    const moeLayers   = Math.max(0, nL - denseLayers);
    // DSA detection — real GLM-5 reports model_type=glm_moe_dsa / arch GlmMoeDsa;
    // the regex must match those, not just a literal "GLM-5" / "glm-dsa" string.
    const isDSA       = /glm.?5|glm.?moe.?dsa|glm-?dsa/i.test(arch)
                     || cfg_.model_type === 'glm-dsa'
                     || cfg_.model_type === 'glm_moe_dsa';
    // MLA detection (GLM-4.7-Flash, GLM-5, …)
    const hasMLA      = cfg_.kv_lora_rank != null && cfg_.q_lora_rank != null;
    // DSA lightning-indexer params (only meaningful when isDSA)
    const idxNHeads   = cfg_.index_n_heads  || 32;
    const idxHeadDim  = cfg_.index_head_dim || 128;
    const idxTopK     = cfg_.index_topk     || 2048;

    const res = [
      {id:Date.now()+1, name:'Token embed', type:'embedding', M, K:vocab, N:hid, repeat:1},
    ];

    if (hasMLA) {
      // MLA path (GLM-4.7-Flash, GLM-5, etc.): Q and KV are factored through
      // low-rank projections; KV cache is a single compressed latent stream.
      //   GLM-4.7-Flash: kv_lora_rank=512, q_lora_rank=768, v_head_dim=256,
      //                  qk_nope_head_dim=192, qk_rope_head_dim=64
      //   GLM-5:         kv_lora_rank=512, q_lora_rank=2048, v_head_dim=256,
      //                  qk_nope_head_dim=192, qk_rope_head_dim=64
      const kvRank = cfg_.kv_lora_rank;
      const qRank  = cfg_.q_lora_rank;
      const vHd    = cfg_.v_head_dim         || 128;
      const ropeHd = cfg_.qk_rope_head_dim   || 64;
      const nopeHd = cfg_.qk_nope_head_dim   || 128;
      // KV cache stores the compressed latent stream of kv_lora_rank dims per token
      // per layer (+ qk_rope_head_dim shared across heads); effective per-head dim
      // for the Attention sub-call is kv_lora_rank under absorption.
      cfg.nH=nH; cfg.H=1; cfg.D=kvRank; cfg.nL=nL; cfg.nLKV=undefined;
      // v80 (Fix B): MLA caches ONE latent stream of (kv_lora_rank+qk_rope_head_dim)
      // per token per layer — not two K/V streams. For DSA (GLM-5) the lightning
      // indexer also keeps one small MQA key of index_head_dim per token per layer.
      // KV-cache VRAM is full-sequence regardless of the sparse toggle (sparsity
      // restricts what is *read*, not what is *stored*) — but with sparse modelling
      // off there is no indexer at all, so its key store is dropped too.
      cfg.kvCacheFn = function(seqLen_) {
        let b = nL * cfg.batch * (kvRank + ropeHd) * seqLen_ * kvBpe();
        if (isDSA && sparseMLA) b += nL * cfg.batch * idxHeadDim * seqLen_ * kvBpe();
        return b;
      };
      // MLA Q is factored: hid → q_lora_rank → nH·(nope+rope)
      //   q_a_proj REPLICATED (feeds q_a_layernorm over full latent) — no AllReduce.
      //   q_b_proj COLUMN-parallel (output split by nH) — no AllReduce.
      res.push({id:Date.now()+2,  name:`MLA Q down (×${nL})`,   type:'linear', M, K:hid,         N:qRank,                repeat:nL, tpMode:'replicated'});
      res.push({id:Date.now()+10, name:`MLA Q up (×${nL})`,     type:'linear', M, K:qRank,       N:nH*(nopeHd+ropeHd),   repeat:nL, tpMode:'col'});
      // MLA KV_a: hid → (kv_lora_rank + qk_rope_head_dim). REPLICATED — feeds
      // kv_a_layernorm over the full latent stream, so no AllReduce.
      res.push({id:Date.now()+3,  name:`MLA KV down (×${nL})`,  type:'linear', M, K:hid,         N:kvRank+ropeHd,        repeat:nL, tpMode:'replicated'});
      // DSA lightning indexer (GLM-5): a lightweight MQA-style scorer. Q/K proj is
      // hid → idxNHeads·idxHeadDim (queries) + idxHeadDim (one shared MQA key).
      // Replicated under TP (the indexer head-score sum + top-k doesn't split
      // cleanly). indexerLayer:true → getRepeat()→0 when sparse-MLA is off.
      if (isDSA) {
        res.push({id:Date.now()+11, name:`DSA indexer Q/K proj (×${nL})`, type:'linear',
                  M, K:hid, N:idxNHeads*idxHeadDim + idxHeadDim, repeat:nL,
                  tpMode:'replicated', indexerLayer:true});
        // Indexer score: QK^T over the FULL sequence (O(L²) — this is the price of
        // sparsity), scoreOnly (no softmax/AV), tpReplicated.
        res.push({id:Date.now()+12, name:`DSA indexer score (×${nL})`, type:'attention',
                  B:cfg.batch, S:Sattn, H:idxNHeads, nKV:1, D:idxHeadDim, kvCache:kvC,
                  fa2:true, repeat:nL, scoreOnly:true, indexerLayer:true, tpReplicated:true});
      }
      // Attention with absorption: effective K/V latent dim = kv_lora_rank. The
      // kv_b_proj weight is fused into Q/O — its FLOPs are accounted for via the
      // larger Q_up output dim and the O_proj output dim.
      // mla:true → single latent KV stream; ropeExtra → QK^T uses kv_lora+rope.
      // sparseTopK (DSA) → with sparse-MLA on, the core attends only top-k tokens.
      res.push({id:Date.now()+4,  name:`MLA Attention (×${nL})`,type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV:1, D:kvRank, kvCache:kvC, fa2:true, repeat:nL,
                mla:true, ropeExtra:ropeHd, ...(isDSA ? {sparseTopK:idxTopK} : {})});
      // O proj ROW-parallel (input split by nH) → exactly ONE AllReduce per MLA block.
      res.push({id:Date.now()+5,  name:`MLA O proj (×${nL})`,   type:'linear', M, K:nH*vHd, N:hid, repeat:nL, tpMode:'row'});
    } else {
      // Standard GQA (GLM-4.5, 4.6, 4.7, GLM-5)
      cfg.nH=nH; cfg.H=nKV; cfg.D=hd; cfg.nL=nL; cfg.nLKV=undefined;
      const qkvNGlm = nH*hd + 2*nKV*hd;
      res.push({id:Date.now()+2, name:`Attn QKV (×${nL})`,    type:'linear', M, K:hid, N:qkvNGlm, repeat:nL});
      res.push({id:Date.now()+3, name:`Attention (×${nL})`,    type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV, D:hd, kvCache:kvC, fa2:true, repeat:nL});
      res.push({id:Date.now()+4, name:`Attn O proj (×${nL})`,  type:'linear', M, K:nH*hd, N:hid,    repeat:nL});
    }

    if (denseLayers > 0) {
      res.push({id:Date.now()+6, name:`Dense FFN gate+up (×${denseLayers})`, type:'linear', M, K:hid, N:denseFfn*2, repeat:denseLayers});
      res.push({id:Date.now()+7, name:`Dense FFN down (×${denseLayers})`,    type:'linear', M, K:denseFfn, N:hid,    repeat:denseLayers});
    }
    if (moeLayers > 0) {
      res.push({id:Date.now()+8, name:`MoE FFN (×${moeLayers})`, type:'moe', M, hid, ffnDim:moeFfn, topK, nExperts, nShared, repeat:moeLayers});
    }
    res.push({id:Date.now()+9, name:'LM head', type:'linear', M, K:hid, N:vocab,
              repeat:1, tied: cfg_.tie_word_embeddings === true});

    return {layers:res, meta:{arch, hid, nL, nH, nKV, hd, ffn:moeFfn, vocab, seq,
                              isGLM4Moe:true, nExperts, topK, nShared, moeFfn, denseLayers, isDSA, hasMLA,
                              idxNHeads, idxHeadDim, idxTopK}};
  }

  // ── Xiaomi MiMo-V2-Flash / StepFun Step-3.5-Flash (hybrid SWA/GA + MoE) ───
  // Both models use hybrid Sliding Window Attention / Global Attention + MoE FFN.
  // MiMo-V2: model_type=mimo_v2_flash, 309B, 48L, 1:5 GA:SWA, win=128, 256e top-8
  //   Asymmetric heads: GA KV=4, SWA KV=8, Q/K hd=192, V hd=128
  // Step-3.5-Flash: model_type=step35, 196B, 45L, 1:3 GA:SWA, win=512, 288e+1sh top-8
  //   Symmetric GQA: same head config for SWA and GA
  //
  // Detection:
  //   hybrid_layer_pattern (per-layer array) or sliding_window_pattern (integer ratio)
  //   moe_layer_freq (per-layer array or integer)
  const isMiMoV2 = /mimo.?v2|mimov2/i.test(arch) || cfg_.model_type === 'mimo_v2_flash';
  const isStep35 = /step.?3.?5|step35/i.test(arch) || cfg_.model_type === 'step35';
  if (isMiMoV2 || isStep35) {
    // ── Attention dimensions ──
    // MiMo-V2 has different KV head counts for GA vs SWA; Step 3.5 uses same for both
    const gaKV    = cfg_.num_key_value_heads      || nKV;
    const swaKV   = cfg_.swa_num_key_value_heads  || gaKV;
    const qHd     = cfg_.head_dim || hd;
    const vHd     = cfg_.v_head_dim               || qHd;  // MiMo-V2: 128 vs 192; Step 3.5: same
    const swaVHd  = cfg_.swa_v_head_dim           || vHd;
    const swaWin  = cfg_.sliding_window_size || cfg_.sliding_window || 128;

    // ── Hybrid layer pattern ──
    let nGA = 0, nSWA = 0;
    if (Array.isArray(cfg_.hybrid_layer_pattern) && cfg_.hybrid_layer_pattern.length === nL) {
      nGA  = cfg_.hybrid_layer_pattern.filter(v => v === 0).length;
      nSWA = cfg_.hybrid_layer_pattern.filter(v => v === 1).length;
    } else if (cfg_.sliding_window_pattern > 0) {
      // Step 3.5 style: sliding_window_pattern=N means N SWA layers per 1 GA layer
      // e.g. sliding_window_pattern=3 → 3:1 SWA:GA ratio
      const swp = cfg_.sliding_window_pattern;
      const groupSize = swp + 1;  // SWA + GA per group
      nGA  = Math.max(1, Math.floor(nL / groupSize));
      nSWA = nL - nGA;
    } else if (isMiMoV2) {
      // Fallback MiMo-V2: assume 1:5 GA:SWA ratio
      nGA  = Math.max(1, Math.round(nL / 6));
      nSWA = nL - nGA;
    } else if (isStep35) {
      // Fallback Step 3.5: assume 1:3 GA:SWA ratio (from architecture spec)
      nGA  = Math.max(1, Math.round(nL / 4));
      nSWA = nL - nGA;
    } else {
      nGA  = Math.max(1, Math.round(nL / 4));
      nSWA = nL - nGA;
    }

    // ── MoE config ──
    const nExperts = cfg_.n_routed_experts || 256;
    const topK     = cfg_.num_experts_per_tok || 8;
    const nShared  = cfg_.n_shared_experts || 0;
    const moeFfn   = cfg_.moe_intermediate_size || 2048;
    const denseFfn = cfg_.intermediate_size || 16384;

    // Dense vs MoE layers from moe_layer_freq
    let denseLayers = 0, moeLayers = 0;
    if (Array.isArray(cfg_.moe_layer_freq) && cfg_.moe_layer_freq.length === nL) {
      denseLayers = cfg_.moe_layer_freq.filter(v => v === 0).length;
      moeLayers   = cfg_.moe_layer_freq.filter(v => v === 1).length;
    } else {
      denseLayers = 1;  // default: layer 0 dense
      moeLayers   = nL - 1;
    }

    // KV cache: GA layers use the same vHd.
    // For cfg.H/D we set GA KV heads (smaller), used for VRAM summary.
    // The custom kvCacheFn handles both GA and SWA contributions precisely.
    cfg.nH=nH; cfg.H=gaKV; cfg.D=vHd; cfg.nL=nL; cfg.nLKV=undefined;
    cfg.kvCacheFn = function(seqLen_) {
      const kb = kvBpe();
      const gaBytes  = 2 * nGA  * cfg.batch * gaKV  * vHd    * seqLen_               * kb;
      const swaBytes = 2 * nSWA * cfg.batch * swaKV * swaVHd * Math.min(swaWin, seqLen_) * kb;
      return gaBytes + swaBytes;
    };

    // ── Build layers ──
    // GA QKV: Q = nH*qHd, K = gaKV*qHd, V = gaKV*vHd
    const gaQKV  = nH*qHd + gaKV*qHd + gaKV*vHd;
    // SWA QKV: Q = nH*qHd (swa_num_attention_heads defaults to nH), K = swaKV*qHd, V = swaKV*swaVHd
    const swaQHd = cfg_.swa_head_dim || qHd;
    const swaQKV = nH*swaQHd + swaKV*swaQHd + swaKV*swaVHd;
    // O proj: nH * v_head_dim → hidden (both GA and SWA use their respective v_head_dim)

    const res = [
      {id:Date.now()+1,  name:'Token embed',                    type:'embedding', M, K:vocab, N:hid, repeat:1},
      // SWA attention block
      {id:Date.now()+2,  name:`SWA Attn QKV (×${nSWA})`,       type:'linear',    M, K:hid,         N:swaQKV,       repeat:nSWA},
      {id:Date.now()+3,  name:`SWA Attention (×${nSWA})`,       type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV:swaKV, D:swaVHd,
                          kvCache: mode==='decode' ? Math.min(swaWin, kvC) : 0, fa2:true, repeat:nSWA},
      {id:Date.now()+4,  name:`SWA Attn O proj (×${nSWA})`,    type:'linear',    M, K:nH*swaVHd,   N:hid,          repeat:nSWA},
      // GA (Global Attention) block
      {id:Date.now()+5,  name:`GA Attn QKV (×${nGA})`,         type:'linear',    M, K:hid,         N:gaQKV,        repeat:nGA},
      {id:Date.now()+6,  name:`GA Attention (×${nGA})`,         type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV:gaKV, D:vHd,
                          kvCache:kvC, fa2:true, repeat:nGA},
      {id:Date.now()+7,  name:`GA Attn O proj (×${nGA})`,      type:'linear',    M, K:nH*vHd,      N:hid,          repeat:nGA},
    ];

    // Dense FFN layers
    if (denseLayers > 0) {
      res.push({id:Date.now()+8,  name:`Dense FFN gate+up (×${denseLayers})`, type:'linear', M, K:hid, N:denseFfn*2, repeat:denseLayers});
      res.push({id:Date.now()+9,  name:`Dense FFN down (×${denseLayers})`,    type:'linear', M, K:denseFfn, N:hid,   repeat:denseLayers});
    }
    // MoE FFN layers
    if (moeLayers > 0) {
      res.push({id:Date.now()+10, name:`MoE FFN (×${moeLayers})`, type:'moe', M, hid, ffnDim:moeFfn, topK, nExperts, nShared, repeat:moeLayers});
    }

    res.push({id:Date.now()+11, name:'LM head', type:'linear', M, K:hid, N:vocab,
              repeat:1, tied: cfg_.tie_word_embeddings === true});

    return {layers:res, meta:{arch, hid, nL, nH, nKV:gaKV, hd:qHd, ffn:moeFfn, vocab, seq,
                              isMiMoV2: isMiMoV2, isStep35: isStep35,
                              nGA, nSWA, gaKV, swaKV, qHd, vHd, swaVHd, swaWin,
                              nExperts, topK, nShared, moeFfn, denseLayers, denseFfn}};
  }

  // ── Xiaomi MiMo-7B (dense, Qwen2-like) ──────────────────────────────────────
  // Identified by: model_type=mimo or architectures containing MiMoForCausalLM.
  // Structurally identical to a dense Qwen2/LLaMA model with SiLU gated FFN.
  // Extra field num_nextn_predict_layers (MTP head) is auxiliary and not modelled.
  // Falls through to the standard dense branch below; this detection just ensures
  // isGated is correctly set (silu triggers it, but we guard against edge cases).
  const isMiMo7B = /^mimo$/i.test(cfg_.model_type||'') || /MiMoForCausalLM/i.test(arch);
  if (isMiMo7B && !isMiMoV2 && !isStep35) {
    // Force-set isGated and fall through to the standard dense path.
    // The only thing MiMo-7B adds is num_nextn_predict_layers for MTP, which is
    // a lightweight speculative decoding module — we note it but don't model it.
    cfg.nH=nH; cfg.H=nKV; cfg.D=hd; cfg.nL=nL; cfg.nLKV=undefined;
    cfg.kvCacheFn = undefined;  // standard KV formula
    const qkvN=nH*hd+2*nKV*hd;
    const res=[
      {id:Date.now()+1, name:'Token embed',         type:'embedding', M, K:vocab, N:hid,  repeat:1},
      {id:Date.now()+2, name:`Attn QKV (×${nL})`,   type:'linear',    M, K:hid,   N:qkvN, repeat:nL},
      {id:Date.now()+3, name:`Attention (×${nL})`,   type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV, D:hd, kvCache:kvC, fa2:true, repeat:nL},
      {id:Date.now()+4, name:`Attn O proj (×${nL})`, type:'linear',    M, K:nH*hd, N:hid, repeat:nL},
      {id:Date.now()+5, name:`FFN gate+up (×${nL})`, type:'linear',    M, K:hid,   N:ffn*2, repeat:nL},
      {id:Date.now()+6, name:`FFN down (×${nL})`,    type:'linear',    M, K:ffn,   N:hid,   repeat:nL},
    ];
    res.push({id:Date.now()+7, name:'LM head', type:'linear', M, K:hid, N:vocab,
              repeat:1, tied: cfg_.tie_word_embeddings === true});
    const mtpLayers = cfg_.num_nextn_predict_layers || 0;
    return {layers:res, meta:{arch,hid,nL,nH,nKV,hd,ffn,vocab,seq,isGated:true,isMiMo7B:true,mtpLayers}};
  }

  // ── MiniMax-M2 family (M2, M2.1, M2.5, M2.7 ...) ───────────────────────────
  // Identified by: model_type=minimax_m2 or architectures contain MiniMaxM2.
  // Distinctive fields (NOT shared with Qwen/DeepSeek/GLM):
  //   num_local_experts      (NOT num_experts / n_routed_experts)
  //   num_experts_per_tok    (topK, typically 8)
  //   intermediate_size      ← PER-EXPERT FFN dim (HF Transformers MoE convention);
  //                            for M2 this is 1536, giving 228B total / 11B active.
  //                            Do NOT be tempted by mlp_intermediate_size (8192) —
  //                            that's a separate auxiliary/MTP path, not the expert FFN.
  //   shared_intermediate_size (shared-expert FFN dim, 0 = no shared expert in M2)
  //   attn_type_list         (per-layer: 1 = full attention, 0 would be linear)
  // Architecture: GQA full-attention on every layer + sparse MoE FFN on every layer,
  //   no dense-replacement layers, no MLA. M2-base: 62L, hid=3072, 256 experts top-8,
  //   expert ffn=1536, nH=48, nKV=8, hd=128 → 230B total / ~10B active.
  const isMiniMaxM2 = /minimax.?m2|minimaxm2/i.test(arch)
                   || cfg_.model_type === 'minimax_m2'
                   || cfg_.num_local_experts != null;
  if (isMiniMaxM2) {
    const nExperts  = cfg_.num_local_experts      || 256;
    const topK      = cfg_.num_experts_per_tok    || 8;
    // HF MoE convention: intermediate_size IS the per-expert FFN dim.
    const moeFfn    = cfg_.intermediate_size      || 1536;
    const sharedFfn = cfg_.shared_intermediate_size || 0;
    const nShared   = sharedFfn > 0 ? 1 : 0;
    // attn_type_list: currently all M2 variants are 1 (full attn) on every layer.
    // If a future variant mixes linear attn, warn but still model as full attn.
    let nFullAttn = nL, nonFullCount = 0;
    if (Array.isArray(cfg_.attn_type_list) && cfg_.attn_type_list.length === nL) {
      nFullAttn    = cfg_.attn_type_list.filter(t => t === 1).length;
      nonFullCount = nL - nFullAttn;
    }
    cfg.nH=nH; cfg.H=nKV; cfg.D=hd; cfg.nL=nL; cfg.nLKV=undefined;
    const qkvN_mm = nH*hd + 2*nKV*hd;
    const res = [
      {id:Date.now()+1, name:'Token embed',         type:'embedding', M, K:vocab,  N:hid,     repeat:1},
      {id:Date.now()+2, name:`Attn QKV (×${nL})`,   type:'linear',    M, K:hid,    N:qkvN_mm, repeat:nL},
      {id:Date.now()+3, name:`Attention (×${nL})`,   type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV, D:hd, kvCache:kvC, fa2:true, repeat:nL},
      {id:Date.now()+4, name:`Attn O proj (×${nL})`, type:'linear',    M, K:nH*hd, N:hid,     repeat:nL},
      {id:Date.now()+5, name:`MoE FFN (×${nL})`,     type:'moe',  M, hid, ffnDim:moeFfn, topK, nExperts, nShared, repeat:nL},
    ];
    res.push({id:Date.now()+6, name:'LM head', type:'linear', M, K:hid, N:vocab,
              repeat:1, tied: cfg_.tie_word_embeddings === true});
    return {layers:res, meta:{arch, hid, nL, nH, nKV, hd, ffn:moeFfn, vocab, seq,
                              isMiniMaxM2:true, nExperts, topK, nShared, moeFfn,
                              sharedFfn, denseLayers:0, nonFullCount}};
  }

  // ── Standard dense architecture (Qwen3, LLaMA, Mistral etc.) ────────────────
  cfg.nH=nH; cfg.H=nKV; cfg.D=hd; cfg.nL=nL; cfg.nLKV=undefined;  // no hybrid split
  const qkvN=nH*hd+2*nKV*hd;
  const res=[
    {id:Date.now()+1, name:'Token embed',         type:'embedding', M, K:vocab, N:hid,  repeat:1},
    {id:Date.now()+2, name:`Attn QKV (×${nL})`,   type:'linear',    M, K:hid,   N:qkvN, repeat:nL},
    {id:Date.now()+3, name:`Attention (×${nL})`,   type:'attention', B:cfg.batch, S:Sattn, H:nH, nKV, D:hd, kvCache:kvC, fa2:true, repeat:nL},
    {id:Date.now()+4, name:`Attn O proj (×${nL})`, type:'linear',    M, K:nH*hd, N:hid, repeat:nL},
  ];
  if(isGated){
    res.push({id:Date.now()+5,name:`FFN gate+up (×${nL})`,type:'linear',M,K:hid,N:ffn*2,repeat:nL});
    res.push({id:Date.now()+6,name:`FFN down (×${nL})`,   type:'linear',M,K:ffn,N:hid,  repeat:nL});
  } else {
    res.push({id:Date.now()+5,name:`FFN up (×${nL})`,  type:'linear',M,K:hid,N:ffn,repeat:nL});
    res.push({id:Date.now()+6,name:`FFN down (×${nL})`,type:'linear',M,K:ffn,N:hid,repeat:nL});
  }
  res.push({id:Date.now()+7, name:'LM head', type:'linear', M, K:hid, N:vocab,
            repeat:1, tied: cfg_.tie_word_embeddings === true});
  return {layers:res, meta:{arch,hid,nL,nH,nKV,hd,ffn,vocab,seq,isGated}};
}

// ── Post-process: expand type='moe' into individual linear sub-layers ──────
// Called on inferFromConfig output to convert each monolithic 'moe' layer into
// 3–5 individual linear layers:
//   1. Router:     [M, hid] × [hid, nExperts]   (small gating matmul)
//   2. Gate+Up:    [M, hid] × [hid, 2*ffnDim]   ×topK experts (GroupedGEMM)
//   3. Down:       [M, ffnDim] × [ffnDim, hid]   ×topK experts (GroupedGEMM)
//   4. Shared Up:  [M, hid] × [hid, 2*sharedFfn] (if nShared > 0)
//   5. Shared Down:[M, sharedFfn] × [sharedFfn, hid]
//
// Each sub-layer carries a `moeGroup` tag so downstream code (tpInfo,
// modelWeightBytes, rendering) can identify MoE-related weights for correct
// EP/TP sharding, VRAM accounting, and color-coding.
//
// FLOPs: the gate+up layer computes topK expert matmuls per token (grouped GEMM),
// so its effective FLOPs = 2 * M * hid * (2*ffnDim) * topK (not a simple [M,K]×[K,N]).
// We handle this by setting K and N to represent per-expert dims and adding a
// `moeActive` field = topK, which calcL multiplies into the FLOPs.
// Similarly, weight bytes = nExperts * K * N * 2 (all experts loaded into VRAM),
// but HBM streaming per forward = distinctActivated * K * N * 2.
function expandMoEToSubLayers(result) {
  const {layers: lyrs, meta} = result;
  const out = [];
  let id = Date.now();
  for (const l of lyrs) {
    if (l.type !== 'moe') { out.push(l); continue; }
    const {M, hid, ffnDim, topK, nExperts, nShared=0, repeat} = l;
    const rp = repeat;
    // Shared expert FFN dim: stored in meta.sharedFfn if available, else same as ffnDim
    const shFfn = meta.sharedFfn || ffnDim;

    // 1. Router
    out.push({id:++id, name:`MoE Router (×${rp})`, type:'linear', M, K:hid, N:nExperts,
              repeat:rp, moeGroup:'router', _moeParent:l});

    // 2. Gate+Up (GroupedGEMM over topK routed experts)
    //    Effective FLOPs = 2 * M * hid * (2*ffnDim) * topK
    //    Weight bytes stored in VRAM = nExperts * hid * (2*ffnDim) * 2
    //    Weight bytes streamed per fwd = distinctActivated * hid * (2*ffnDim) * 2
    out.push({id:++id, name:`MoE gate+up (×${rp})`, type:'linear', M, K:hid, N:ffnDim*2,
              repeat:rp, moeGroup:'gate_up',
              moeExperts:nExperts, moeTopK:topK, moeFfnDim:ffnDim, moeHid:hid, _moeParent:l});

    // 3. Down (GroupedGEMM over topK routed experts)
    out.push({id:++id, name:`MoE down (×${rp})`, type:'linear', M:M, K:ffnDim, N:hid,
              repeat:rp, moeGroup:'down',
              moeExperts:nExperts, moeTopK:topK, moeFfnDim:ffnDim, moeHid:hid, _moeParent:l});

    // 4–5. Shared expert (if present)
    if (nShared > 0) {
      out.push({id:++id, name:`Shared expert up (×${rp})`, type:'linear', M, K:hid, N:shFfn*2,
                repeat:rp, moeGroup:'shared_up'});
      out.push({id:++id, name:`Shared expert down (×${rp})`, type:'linear', M:M, K:shFfn, N:hid,
                repeat:rp, moeGroup:'shared_down'});
    }
  }
  return {layers: out, meta};
}

function inferFromSF(meta_) {
  const tensors=Object.entries(meta_).filter(([k])=>k!=='__metadata__');
  const res=[];const seen=new Set();
  tensors.forEach(([name,info])=>{const shape=info.shape||[];if(shape.length<2)return;const key=name.split('.').slice(0,-1).join('.');if(seen.has(key))return;seen.add(key);const r=shape[shape.length-2]||shape[0],c=shape[shape.length-1];const t=name.toLowerCase().includes('embed')?'embedding':'linear';res.push({id:Date.now()+res.length,name,type:t,M:cfg.batch*(mode==='prefill'?cfg.seqLen:1),K:t==='embedding'?Math.max(r,c):r,N:t==='embedding'?1:c,repeat:1});});
  return res.slice(0,60);
}

function openImportPreview(cfg_,src){
  // inferFromConfig mutates global cfg (nH, H, D, nL, nLKV).
  // Store the inferred layers on window._ip and the inferred cfg separately so
  // we can apply them only when the user confirms Import, not on Cancel.
  window._pendingCfg = null;
  window._pendingModelName = null;
  const savedCfg = {...cfg, nLKV: cfg.nLKV, kvCacheFn: cfg.kvCacheFn};  // explicit capture (may be undefined)
  const{layers:prop,meta}=expandMoEToSubLayers(inferFromConfig(cfg_));
  window._ip=prop;
  window._pendingCfg = {...cfg, kvCacheFn: cfg.kvCacheFn};  // capture the cfg inferFromConfig just wrote
  // Derive a clean model name from the source string (HF repo id or filename)
  window._pendingModelName = src.includes('/') ? src.split('/').pop() : src.replace(/\.json$/i,'');
  Object.assign(cfg, savedCfg);
  cfg.nLKV = savedCfg.nLKV;  // explicitly restore — Object.assign won't delete keys
  cfg.kvCacheFn = savedCfg.kvCacheFn;  // restore custom KV cache function
  const mb=document.getElementById('mb');mb.className='modal wide';document.getElementById('ov').style.display='flex';
  const extraInfo = meta.isQwen35
    ? (() => {
        const fullAttnLabel = meta.attnOutputGate ? 'Gated full-attn' : 'full-attn';
        const archType = meta.hasMoE
          ? `Hybrid (GatedDeltaNet + ${fullAttnLabel}) · MoE FFN`
          : `Hybrid (GatedDeltaNet + ${fullAttnLabel}) · Dense FFN`;
        const bgColor  = meta.hasMoE ? '#FAEEDA' : '#E1F5EE';
        const brColor  = meta.hasMoE ? '#EF9F27' : '#9FE1CB';
        const txColor  = meta.hasMoE ? '#633806' : '#0F6E56';
        const moeDetail = meta.hasMoE
          ? `MoE: ${meta.nExperts} experts top-${meta.topK}${meta.nShared?`+${meta.nShared} shared`:''}, ffn=${meta.moeFfn}`
          : `Dense FFN: ffn=${meta.ffn}`;
        const mtpDetail = meta.mtpLayers ? ` · MTP: ${meta.mtpLayers} layer(s) (not modelled)` : '';
        return `<div style="font-size:12px;color:${txColor};background:${bgColor};border:1px solid ${brColor};border-radius:6px;padding:8px 10px;margin-bottom:10px">
          <div style="font-weight:700;margin-bottom:2px">${archType}</div>
          <div style="opacity:.85">Layers: ${meta.nLinAttn} GatedDeltaNet + ${meta.nFullAttn} ${fullAttnLabel} = ${meta.nL} total · DeltaNet nK=${meta.nKeyH} nV=${meta.nValH} D=${meta.hdLinV} · ${moeDetail}${mtpDetail}</div>
        </div>`;
      })()
    : meta.isQwen3MoE
    ? `<div style="font-size:12px;color:#633806;background:#FAEEDA;border:1px solid #EF9F27;border-radius:6px;padding:8px 10px;margin-bottom:10px">
        <div style="font-weight:700;margin-bottom:2px">Full attention · MoE FFN</div>
        <div style="opacity:.85">Qwen3-MoE — ${meta.nExperts} experts, top-${meta.topK} active, FFN dim=${meta.moeFfn}, ${meta.denseLayers} dense layer(s)</div>
      </div>`
    : meta.isDSV3
    ? `<div style="font-size:12px;color:#633806;background:#FAEEDA;border:1px solid #EF9F27;border-radius:6px;padding:8px 10px;margin-bottom:10px">
        <div style="font-weight:700;margin-bottom:2px">${meta.isDSA?'Sparse MLA (DSA) · compressed KV':'MLA (compressed KV)'} · MoE FFN</div>
        <div style="opacity:.85">${meta.isKimiK2?'Kimi K2':(meta.isDSA?'DeepSeek-V3.2-Exp':'DeepSeek-V3')} — ${meta.nExperts} routed+${meta.nShared} shared experts, top-${meta.topK}, KV rank=${meta.kvRank}, ${meta.denseLayers} dense layer(s)${meta.isDSA?` · DSA lightning indexer: ${meta.idxNHeads} heads · d=${meta.idxHeadDim} · top-k=${meta.idxTopK}`:''}</div>
      </div>`
    : meta.isDSV4
    ? `<div style="font-size:12px;color:#633806;background:#FAEEDA;border:1px solid #EF9F27;border-radius:6px;padding:8px 10px;margin-bottom:10px">
        <div style="font-weight:700;margin-bottom:2px">Hybrid Attention (HCA + CSA${meta.nSWAOnly?' + SWA-only':''}) · MoE FFN</div>
        <div style="opacity:.85">DeepSeek-V4${meta.isFlash?'-Flash':'-Pro'} — ${meta.nHCA} HCA + ${meta.nCSA} CSA${meta.nSWAOnly?` + ${meta.nSWAOnly} SWA-only (win=${meta.slidingWindow})`:''} = ${meta.nL} layers · MQA nKV=${meta.nKV}, head_dim=${meta.hd} (avg eff ≈${meta.avgEffD}) · ${meta.nExperts} routed+${meta.nShared} shared experts, top-${meta.topK}, ffn=${meta.moeFfn}${meta.hasORank?` · Q-LoRA=${meta.qRank}, O-LoRA=${meta.oRank}`:` · Q-LoRA=${meta.qRank}`}${meta.mtpLayers?` · MTP: ${meta.mtpLayers} layer(s) (not modelled)`:''} · ⚠ HCA/CSA attention compute uses full Sk (KV cache itself is correctly compressed) — long-ctx attention FLOPs overestimated</div>
      </div>`
    : meta.isGLM4Moe
    ? `<div style="font-size:12px;color:#633806;background:#FAEEDA;border:1px solid #EF9F27;border-radius:6px;padding:8px 10px;margin-bottom:10px">
        <div style="font-weight:700;margin-bottom:2px">${meta.hasMLA?(meta.isDSA?'Sparse MLA (DSA) · compressed KV':'MLA (compressed KV)'):'Full attention (GQA)'} · MoE FFN</div>
        <div style="opacity:.85">${meta.isDSA?'GLM-5 (glm_moe_dsa)':'GLM-4 MoE'} — ${meta.nExperts} routed+${meta.nShared} shared experts, top-${meta.topK}, moe_ffn=${meta.moeFfn}, ${meta.denseLayers} dense layer(s)${meta.isDSA?` · DeepSeek Sparse Attention: lightning indexer (${meta.idxNHeads} heads, d=${meta.idxHeadDim}) + top-${meta.idxTopK} core attention — toggle "Sparse MLA" to switch between DSA modelling and a dense-MLA upper bound`:''}</div>
      </div>`
    : meta.isMiniMaxM2
    ? `<div style="font-size:12px;color:#633806;background:#FAEEDA;border:1px solid #EF9F27;border-radius:6px;padding:8px 10px;margin-bottom:10px">
        <div style="font-weight:700;margin-bottom:2px">Full attention (GQA) · MoE FFN</div>
        <div style="opacity:.85">MiniMax-M2 — ${meta.nExperts} experts${meta.nShared?`+${meta.nShared} shared (ffn=${meta.sharedFfn})`:''}, top-${meta.topK}, expert ffn=${meta.moeFfn}, MoE on every layer${meta.nonFullCount?` · ⚠ ${meta.nonFullCount} non-full-attn layers modelled as full attn`:''}</div>
      </div>`
    : (meta.isMiMoV2 || meta.isStep35)
    ? `<div style="font-size:12px;color:#633806;background:#FAEEDA;border:1px solid #EF9F27;border-radius:6px;padding:8px 10px;margin-bottom:10px">
        <div style="font-weight:700;margin-bottom:2px">Hybrid SWA/GA attention · MoE FFN</div>
        <div style="opacity:.85">${meta.isMiMoV2?'MiMo-V2-Flash':'Step-3.5-Flash'} — ${meta.nSWA} SWA (win=${meta.swaWin}) + ${meta.nGA} GA layers · ${meta.gaKV!==meta.swaKV?`GA KV=${meta.gaKV}, SWA KV=${meta.swaKV}`:`KV=${meta.gaKV}`}${meta.vHd!==meta.qHd?` · Q/K head=${meta.qHd}, V head=${meta.vHd}`:''} · ${meta.nExperts} experts top-${meta.topK}${meta.nShared?`+${meta.nShared} shared`:''}, ffn=${meta.moeFfn} · ${meta.denseLayers} dense layer(s)</div>
      </div>`
    : meta.isMiMo7B
    ? `<div style="font-size:12px;color:#0F6E56;background:#E1F5EE;border:1px solid #9FE1CB;border-radius:6px;padding:8px 10px;margin-bottom:10px">
        <div style="font-weight:700;margin-bottom:2px">Full attention (GQA) · Dense FFN</div>
        <div style="opacity:.85">MiMo-7B — ${meta.nH} heads, ${meta.nKV} KV heads, head_dim=${meta.hd}, FFN=${meta.ffn}${meta.mtpLayers?` · MTP: ${meta.mtpLayers} auxiliary layer(s) (not modelled)`:''}</div>
      </div>`
    : '';
  const sh=`<div class="sum-grid">${[['Arch',meta.arch],['Total layers',meta.nL],['Hidden',meta.hid],['Heads',meta.nH+(meta.nKV!==meta.nH?` (KV:${meta.nKV})`:'')] ].map(([l,v])=>`<div class="sum-cell"><div class="sum-label">${l}</div><div class="sum-val">${v}</div></div>`).join('')}</div>`;
  // Warn if values look like fallback defaults — sign of an empty/gated/wrong config
  const looksLikeFallback = (meta.hid <= 1024 || meta.nL <= 12 || meta.vocab <= 32000);
  const warnInfo = looksLikeFallback
    ? `<div style="font-size:12px;color:#854F0B;background:#FEF3E2;border:1px solid #F0A32A;border-radius:6px;padding:7px 10px;margin-bottom:10px">
        ⚠ <strong>Config looks incomplete</strong> — hidden=${meta.hid}, layers=${meta.nL}, vocab=${meta.vocab.toLocaleString()}.
        These match default fallback values, which usually means the model is <strong>gated</strong> (login required),
        the repo doesn't exist, or uses non-standard field names.
        Try the full instruct variant (e.g. <code style="background:#fff3;padding:1px 4px;border-radius:3px">${src}-Instruct</code>) or upload config.json manually.
      </div>`
    : '';
  const rows=prop.map((l,i)=>{
    const shape=l.type==='attention'?`B=${l.B} S=${l.S} H=${l.H} D=${l.D} kv=${l.kvCache||0}`
               :l.type==='linear_attn'?`M=${l.M} nQK=${l.nQK} nV=${l.nV} D=${l.D}`
               :l.type==='moe'?`M=${l.M} hid=${l.hid} ffn=${l.ffnDim} top${l.topK}/${l.nExperts}+${l.nShared??0}sh`
               :`M=${l.M} K=${l.K} N=${l.N}`;
    const bdCls=l.type==='attention'?'ba':l.type==='linear_attn'?'bla':(l.type==='moe'||l.moeGroup)?'bmoe':l.type==='embedding'?'bm':'bc';
    const bdTxt=l.type==='attention'?'attn':l.type==='linear_attn'?'δ-net':(l.type==='moe'||l.moeGroup)?'MoE':l.type;
    return`<tr><td style="width:28px;text-align:center"><input type="checkbox" class="ic" data-i="${i}" checked onchange="window._selectedIndices=[...document.querySelectorAll('.ic')].filter(c=>c.checked).map(c=>+c.dataset.i)"/></td><td>${l.name} <span style="font-size:10px;color:#aaa">×${l.repeat}</span></td><td><span class="bd ${bdCls}">${bdTxt}</span></td><td style="color:#888;font-size:11px">${shape}</td></tr>`;
  }).join('');
  // Capture the replace flag immediately at modal-build time into a JS variable.
  // doImport() must NOT query the DOM for this — the modal may have been closed or
  // re-rendered by the time the callback fires in some browser environments.
  window._replaceOnImport = true;   // default: replace
  window._selectedIndices = prop.map((_,i)=>i);  // all selected by default

  mb.innerHTML=`<h3>Import — <span style="color:#888;font-weight:400">${src}</span></h3>${sh}${warnInfo}${extraInfo}<div class="pt-wrap"><table class="pt"><thead><tr><th style="width:28px"><input type="checkbox" id="ca" checked onchange="
    const allChecked=this.checked;
    document.querySelectorAll('.ic').forEach(c=>{c.checked=allChecked;});
    window._selectedIndices=[...document.querySelectorAll('.ic')].filter(c=>c.checked).map(c=>+c.dataset.i);
  "/></th><th>Layer (repeat)</th><th>Type</th><th>Shape (${mode})</th></tr></thead><tbody>${rows}</tbody></table></div><div style="margin-top:10px"><label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer"><input type="checkbox" id="rc" checked onchange="window._replaceOnImport=this.checked"/>Replace current layers</label></div><div class="mbtns"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn g" onclick="doImport()">Import</button></div>`;
}
function openSFPreview(meta_,src){
  const prop=inferFromSF(meta_);window._ip=prop;
  const mb=document.getElementById('mb');mb.className='modal wide';document.getElementById('ov').style.display='flex';
  const rows=prop.map((l,i)=>`<tr><td style="width:28px;text-align:center"><input type="checkbox" class="ic" data-i="${i}" checked/></td><td style="font-size:11px">${l.name}</td><td><span class="bd ${l.type==='embedding'?'bm':'bc'}">${l.type}</span></td><td style="color:#888;font-size:11px">K=${l.K} N=${l.N}</td></tr>`).join('');
  mb.innerHTML=`<h3>Tensors — <span style="color:#888;font-weight:400">${src}</span></h3><p style="font-size:12px;color:#888;margin-bottom:10px">${Object.keys(meta_).filter(k=>k!=='__metadata__').length} tensors. Note: set repeat counts manually after import.</p><div class="pt-wrap"><table class="pt"><thead><tr><th style="width:28px"><input type="checkbox" id="ca" checked onchange="document.querySelectorAll('.ic').forEach(c=>c.checked=this.checked)"/></th><th>Tensor</th><th>Type</th><th>Shape</th></tr></thead><tbody>${rows}</tbody></table></div><div style="margin-top:10px"><label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer"><input type="checkbox" id="rc" checked/>Replace current layers</label></div><div class="mbtns"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn g" onclick="doImport()">Import</button></div>`;
}
function doImport(){
  const prop = window._ip || [];
  const sel  = window._selectedIndices || prop.map((_,i)=>i);  // default: all
  const replace = (window._replaceOnImport !== false);           // default: true
  const toAdd = sel.map(i=>({...prop[i], id:Date.now()+i}));
  // Clear state immediately so stale values can't leak into a future import
  window._ip = null; window._selectedIndices = null; window._replaceOnImport = true;
  closeModal();
  if(replace) layers = toAdd; else layers = [...layers, ...toAdd];
  // Apply the cfg that inferFromConfig computed (deferred from openImportPreview)
  if(window._pendingCfg) {
    Object.assign(cfg, window._pendingCfg);
    cfg.nLKV = window._pendingCfg.nLKV;  // preserve nLKV explicitly
    cfg.kvCacheFn = window._pendingCfg.kvCacheFn;  // preserve custom KV cache function
    // Sync precision selector UI with auto-detected value
    const pSel = document.getElementById('precisionSel');
    if (pSel) pSel.value = cfg.precision || 'fp16';
    window._pendingCfg = null;
  }
  // Update model name display
  if(window._pendingModelName) {
    currentModelName = window._pendingModelName;
    window._pendingModelName = null;
  }
  const lbl = document.getElementById('modelNameLabel');
  if(lbl) lbl.textContent = currentModelName;
  updateCfgInfo();
  renderArchInfo();
  renderLayerList();
  renderTab();
  // Force a second render on the next frame to guarantee Structure-tab SVG
  // and any deferred Chart.js canvases pick up the new state. Cheap insurance
  // against any timing issue where the first renderTab() ran before DOM settled.
  requestAnimationFrame(()=>{ try{ renderTab(); }catch(e){ console.warn('rAF renderTab:',e); } });
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

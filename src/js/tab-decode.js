function renderMoESweetSpot(chip) {
  // Sweet-spot analysis is meaningful under EP and Hybrid (where per-card weight
  // reads grow with M via activated experts until saturation). Under pure TP,
  // all experts' shards live on every card and the analysis is less actionable.
  if (moePar === 'tp') return '';

  // Find MoE parameters from either legacy type:'moe' or expanded gate_up sub-layer
  let hid, ffnDim, topK, nExperts, nShared=0;
  const legacyMoE = layers.find(l => l.type === 'moe');
  const expandedMoE = layers.find(l => l.moeGroup === 'gate_up');
  if (legacyMoE) {
    ({hid, ffnDim, topK, nExperts, nShared=0} = legacyMoE);
  } else if (expandedMoE) {
    hid = expandedMoE.moeHid;
    ffnDim = expandedMoE.moeFfnDim;
    topK = expandedMoE.moeTopK;
    nExperts = expandedMoE.moeExperts;
    nShared = layers.some(l => l.moeGroup === 'shared_up') ? 1 : 0;
  } else {
    return '';
  }
  const ridge = effectiveTflops(chip,'linear')*1e12 / (chip.bw*1e9);
  const wb = wBpe(), ab = aBpe();

  // Sweep M in log scale
  const Ms = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096];
  const rows = Ms.map(M => {
    const distinct = expectedActivatedExperts(M, topK, nExperts);
    const pctActivated = distinct / nExperts * 100;
    const flops = 6 * M * hid * ffnDim * (topK + nShared) + 2*M*hid*nExperts;
    const bytes = (distinct + nShared) * (2*hid*ffnDim + ffnDim*hid) * wb + M*hid*ab*2;
    const ai = flops/bytes;
    const eTF = effectiveTflops(chip,'linear')*1e12;
    const cT = flops/eTF, mT = bytes/(chip.bw*1e9);
    const t = Math.max(cT, mT);
    return {M, distinct, pctActivated, flops, bytes, ai, t, bound: cT>mT?'comp':'mem'};
  });

  // Find thresholds
  const satRow = rows.find(r => r.pctActivated >= 95);
  const compRow = rows.find(r => r.ai >= ridge);
  const M_sat = satRow ? satRow.M : null;
  const M_comp = compRow ? compRow.M : null;

  // Build table
  const body = rows.map(r => {
    const isSat = M_sat && r.M >= M_sat;
    const isComp = r.bound === 'comp';
    const bg = isComp ? 'background:#E1F5EE' : isSat ? 'background:#FEF3E2' : '';
    return `<tr style="${bg}">
      <td><strong>${r.M}</strong></td>
      <td>${r.distinct.toFixed(1)} / ${nExperts} <span style="color:#888">(${r.pctActivated.toFixed(0)}%)</span></td>
      <td>${(r.flops/1e9).toFixed(1)}G</td>
      <td>${(r.bytes/1e6).toFixed(1)}M</td>
      <td>${r.ai.toFixed(1)}</td>
      <td>${ft(r.t)}</td>
      <td>${ft(r.t/r.M)}</td>
      <td><span class="bd ${isComp?'bc':'bm'}">${r.bound}</span></td>
    </tr>`;
  }).join('');

  const ridgeStr = ridge.toFixed(1);
  const satStr  = M_sat  ? `<strong style="color:#854F0B">M ≈ ${M_sat}</strong>` : '<span style="color:#888">&gt; 4096</span>';
  const compStr = M_comp ? `<strong style="color:#0F6E56">M ≈ ${M_comp}</strong>` : '<span style="color:#888">&gt; 4096</span>';

  return `<div class="cw">
    <div class="ct">MoE sweet-spot analysis (single layer: hid=${hid}, ffn=${ffnDim}, top${topK}/${nExperts})</div>
    <div style="font-size:12px;color:#555;line-height:1.8;margin-bottom:10px">
      Ridge point for ${chip.name}: <strong>${ridgeStr}</strong> FLOP/byte<br>
      <strong>Expert saturation</strong> (&ge;95% experts activated): ${satStr} — beyond this, extra M adds no HBM weight traffic<br>
      <strong>Compute-bound threshold</strong> (AI ≥ ridge): ${compStr} — beyond this, FLOPs dominate the latency entirely<br>
      <strong>Sweet spot</strong>: starts at <em>M_saturate</em> — beyond it, per-token latency drops ~linearly with M (bytes are fixed, M grows) until either compute-bound or VRAM for activations / KV cache runs out. Below M_saturate, each extra token costs real bandwidth.
    </div>
    <div style="overflow-x:auto"><table class="lt"><thead><tr>
      <th>M</th><th>Activated experts</th><th>FLOPs</th><th>Bytes</th><th>AI</th><th>Time/step</th><th>Time/token</th><th>Bound</th>
    </tr></thead><tbody>${body}</tbody></table></div>
    <div style="font-size:11px;color:#888;margin-top:8px">
      <span style="display:inline-block;width:10px;height:10px;background:#FEF3E2;border:1px solid #F0A32A;vertical-align:middle"></span> expert saturation reached
      &nbsp;&nbsp;
      <span style="display:inline-block;width:10px;height:10px;background:#E1F5EE;border:1px solid #5DCAA5;vertical-align:middle"></span> compute-bound
    </div>
  </div>`;
}

function rDecode(){
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}
  const ch=selChip;
  const prevMode=mode;
  const prevSnap=JSON.parse(JSON.stringify(layers));

  // Prefill
  mode='prefill'; applyConfigToLayers();
  const prefillSnap=JSON.parse(JSON.stringify(layers));
  const prefillTime=prefillSnap.map(l=>tpInfoTotal(l,ch).effectiveTime).reduce((a,b)=>a+b,0);

  // Decode step
  mode='decode'; applyConfigToLayers();
  const decodeSnap=JSON.parse(JSON.stringify(layers));
  const decodeStepT=decodeSnap.map(l=>tpInfoTotal(l,ch).effectiveTime).reduce((a,b)=>a+b,0);

  // Restore
  mode=prevMode; layers.forEach((l,i)=>Object.assign(l,prevSnap[i]));

  const totalDecodeTime=decodeStepT*cfg.outTokens;
  const totalTime=prefillTime+totalDecodeTime;
  // Throughput decomposition (decode stage):
  //   per-seq tok/s   = 1 / step_time          (one sequence's generation rate)
  //   per-replica t/s = batch × per-seq        (one DP replica runs `batch` concurrent seqs)
  //   cluster t/s     = dp × per-replica       (sum across DP replicas)
  // The headline "Throughput" reported to the user is the CLUSTER total, since
  // that's what the deployment actually delivers. Earlier versions reported
  // only 1/step_time (single-seq rate), which understated by a factor of
  // batch × dp — for the screenshot (batch=240, dp=1) that's 240× low.
  const perSeqTokPerSec     = 1 / (decodeStepT || 1);
  const perReplicaTokPerSec = cfg.batch * perSeqTokPerSec;
  const clusterTokPerSec    = dp * perReplicaTokPerSec;
  // Global batch = per-rank batch × dp. When dp=1 (pure TP) this equals the
  // per-rank batch — the semantics stay consistent either way. Throughput is
  // ALWAYS reported against the global batch (clusterTokPerSec = globalBatch / step).
  const globalBatch = cfg.batch * dp;
  // Back-compat aliases used downstream:
  const tokensPerSec   = perReplicaTokPerSec;   // per-replica (batch × 1/step)
  const totalTokPerSec = clusterTokPerSec;      // cluster total (batch × dp × 1/step)

  // VRAM accounting — compute per-card weight with DP-aware sharding
  const wGB = modelWeightBytes()/1e9;
  let moeWB = 0;
  layers.forEach(l => {
    if (l.type==='moe') {
      const r=getRepeat(l), {hid:h,ffnDim:f,nExperts:ne,nShared:ns=0}=l, wb2=wBpe();
      moeWB += ((ne+ns)*(2*h*f+f*h) + h*ne) * wb2 * r;
    } else if (l.moeGroup==='gate_up'||l.moeGroup==='down') { moeWB += l.moeExperts*l.K*l.N*wBpe()*getRepeat(l); }
    else if (l.moeGroup==='shared_up'||l.moeGroup==='shared_down') { moeWB += l.K*l.N*wBpe()*getRepeat(l); }
    else if (l.moeGroup==='router') { moeWB += l.K*l.N*wBpe()*getRepeat(l); }
  });
  const nonMoeWB = modelWeightBytes() - moeWB;
  // TP-replicated dense linears (MLA a-projections) are held in FULL on every
  // card — split them out so they're not wrongly divided by tpAttn below.
  const replWB    = replicatedWeightBytes();
  const shardedNonMoeWB = nonMoeWB - replWB;
  let moePC;
  if (moePar === 'hybrid' && epDeg > 1 && nCards > 1) {
    let rW=0,sW=0,rtW=0; const eTp2=Math.max(1,nCards/epDeg);
    layers.forEach(l=>{
      if(l.type==='moe'){const r=getRepeat(l),{hid:h,ffnDim:f,nExperts:ne,nShared:ns=0}=l,e=(2*h*f+f*h)*wBpe();rW+=ne*e*r;sW+=ns*e*r;rtW+=h*ne*wBpe()*r;}
      else if(l.moeGroup==='gate_up'||l.moeGroup==='down'){rW+=l.moeExperts*l.K*l.N*wBpe()*getRepeat(l);}
      else if(l.moeGroup==='shared_up'||l.moeGroup==='shared_down'){sW+=l.K*l.N*wBpe()*getRepeat(l);}
      else if(l.moeGroup==='router'){rtW+=l.K*l.N*wBpe()*getRepeat(l);}
    });
    moePC = rW/nCards + sW/eTp2 + rtW;
  } else { moePC = moeWB / nCards; }
  const wPerCard = (shardedNonMoeWB / tpAttn + replWB + moePC) / 1e9;
  const ksf = kvShardFactor();
  // kvEndPerRank = KV total for ONE DP rank's batch (cfg.batch is per-rank).
  // kvEndGB     = per-card KV after intra-rank TP sharding.
  // kvEndCluster = KV summed across all DP replicas (no sharing).
  const kvEndPerRank = kvCacheBytes(cfg.seqLen+cfg.outTokens)/1e9;
  const kvEndGB      = kvEndPerRank / ksf;
  const kvEndCluster = kvEndPerRank * dp;
  const usedPerCard = wPerCard + kvEndGB;
  const kvFits = usedPerCard <= selChip.mem;

  const bytesPerTokGlobal = cfg.kvCacheFn
    ? Math.max(1, kvCacheBytes(10000) - kvCacheBytes(9999))
    : 2*(cfg.nLKV||cfg.nL)*cfg.batch*cfg.H*cfg.D*kvBpe();
  const bytesPerTok = bytesPerTokGlobal / ksf;
  const kvBudgetBytes = Math.max(0, (selChip.mem - wPerCard)*1e9);
  const maxTotalTok = Math.floor(kvBudgetBytes / Math.max(1, bytesPerTok));
  const maxOut = Math.max(0, maxTotalTok - cfg.seqLen);

  const shardNote = ksf > 1 ? ` (sharded ${ksf}× across TP)` : ' (replicated across TP)';
  const dpKvNote = dp > 1 ? ` · cluster total <strong>${kvEndCluster.toFixed(2)} GB</strong> (=${kvEndPerRank.toFixed(2)} × ${dp} DP replicas)` : '';
  const kvFormulaStr = cfg.kvCacheFn
    ? `hybrid SWA/GA KV cache = <strong>${kvEndGB.toFixed(2)} GB</strong>/card${shardNote} (GA full-seq + SWA capped at window, per-rank total ${kvEndPerRank.toFixed(2)} GB)${dpKvNote}`
    : `2 × ${cfg.nLKV||cfg.nL} layers × batch ${cfg.batch} × ${cfg.H} KV-heads × D${cfg.D} × (seq ${cfg.seqLen}+out ${cfg.outTokens}) × ${kvBpe()}B = <strong>${kvEndPerRank.toFixed(2)} GB</strong> per-rank → <strong>${kvEndGB.toFixed(2)} GB</strong>/card${shardNote}${cfg.nLKV&&cfg.nLKV!==cfg.nL?` <span style="font-size:11px;color:#888">(only ${cfg.nLKV} of ${cfg.nL} layers have KV cache)</span>`:''}${dpKvNote}`;

  const pLabel = cfg.precision !== 'fp16' ? ` · ${precisionLabel()}` : '';

  const ri=effectiveTflops(ch,'linear')*1e12/(ch.bw*1e9);
  const decRows=decodeSnap.map(l=>{
    const ti=tpInfoTotal(l,ch); const c=calcL(l); const r=getRepeat(l);
    const ia=l.type==='attention';
    const lRi=effectiveTflops(ch,l.type)*1e12/(ch.bw*1e9);
    // Per-call timing decomposition (single layer instance, before ×repeat).
    //   effectiveTime = max(computeTime, memTime) + commTime − hiddenComm
    // We surface the formula so the user can see exactly how time/call is derived.
    const one  = tpInfo(l,ch);
    const busy = Math.max(one.computeTime, one.memTime);
    const exComm = Math.max(0, one.commTime - one.hiddenComm);
    const busyLabel = one.computeTime >= one.memTime
      ? `max(<span style="color:#185FA5">compute ${ft(one.computeTime)}</span>, mem ${ft(one.memTime)})`
      : `max(compute ${ft(one.computeTime)}, <span style="color:#0F6E56">mem ${ft(one.memTime)}</span>)`;
    const commPart = exComm > 1e-12
      ? ` + <span style="color:#A06800">comm ${ft(exComm)}</span>`
      : (one.commTime > 1e-12 ? ` <span style="color:#aaa">(+comm ${ft(one.commTime)} fully overlapped)</span>` : '');
    const callFormula = `${busyLabel}${commPart} = <strong>${ft(one.effectiveTime)}</strong>`;
    // Total-per-step formula: time/call × repeat count.
    const stepFormula = r === 1
      ? `${ft(one.effectiveTime)} × 1 = <strong>${ft(ti.effectiveTime)}</strong>`
      : `${ft(one.effectiveTime)} × ${r} = <strong>${ft(ti.effectiveTime)}</strong>`;
    // ×outTokens formula: total/step × number of decode steps.
    const outFormula = `${ft(ti.effectiveTime)} × ${cfg.outTokens} = <strong>${ft(ti.effectiveTime*cfg.outTokens)}</strong>`;
    // v76: per-call / per-step inter-card communication volume (bytes).
    //   one.commBytes = AllReduce (row-par TP) or All-to-All (MoE EP) traffic for
    //                   a SINGLE layer instance; ti.commBytes = ×repeat per step.
    const ocb = one.commBytes || 0;
    const tcb = ti.commBytes  || 0;
    const commCell = nCards <= 1
      ? '<span style="color:#bbb">—</span>'
      : (ocb > 1e-9
          ? `${fmt(ocb)}<div style="font-size:10px;color:#999;margin-top:2px;line-height:1.5">×${r} = <strong>${fmt(tcb)}</strong>/step</div>`
          : '<span style="color:#bbb">—</span><div style="font-size:10px;color:#aaa;margin-top:2px">no cross-card comm</div>');
    return`<tr>
      <td>${l.name} <span style="font-size:10px;color:#aaa">×${r}</span> <span class="bd ${ia?'ba':c.ai<lRi?'bm':'bc'}">${ia?'attn':c.ai<lRi?'mem':'comp'}</span></td>
      <td>${c.ai.toFixed(2)}</td>
      <td>${fmt(c.bytes)}</td>
      <td>${commCell}</td>
      <td>${ft(one.effectiveTime)}<div style="font-size:10px;color:#999;margin-top:2px;line-height:1.5">${callFormula}</div></td>
      <td>${ft(ti.effectiveTime)}<div style="font-size:10px;color:#999;margin-top:2px;line-height:1.5">${stepFormula}</div></td>
      <td>${ft(ti.effectiveTime*cfg.outTokens)}<div style="font-size:10px;color:#999;margin-top:2px;line-height:1.5">${outFormula}</div></td>
      <td style="font-size:11px;color:${ti.memTime>ti.computeTime?'#A32D2D':'#0F6E56'}">${ti.memTime>ti.computeTime?'memory-bound':'compute-bound'}</td>
    </tr>`;
  }).join('');

  const genCurve=[]; let t=prefillTime;
  for(let i=0;i<=Math.min(cfg.outTokens,500);i++){genCurve.push({x:i,y:t*1000});t+=decodeStepT;}

  // Build the throughput formula string for the headline card.
  // Decode throughput = global batch × per-seq token rate
  //                   = (per-rank batch × dp) × (1 / decode_step_time)
  // When dp=1, global batch == per-rank batch (pure TP). The headline always
  // reports the GLOBAL throughput so the number reflects what the deployment
  // actually delivers, and stays consistent whether or not DP is configured.
  const dpFactorStr = dp > 1 ? ` × ${dp} DP` : '';
  const gbExplain = dp > 1 ? ` (= ${cfg.batch} per-rank × ${dp} DP)` : ' (= per-rank batch, pure TP)';
  const throughputFormula = `global batch ${globalBatch}${gbExplain} × (1 / ${ft(decodeStepT)}) = <strong>${Math.round(clusterTokPerSec).toLocaleString()} tok/s</strong>`;

  document.getElementById('mainContent').innerHTML=
    `<div class="banner dec">Decode${pLabel} · prompt=${cfg.seqLen} · global batch=${globalBatch}${dp>1?` (${cfg.batch} per rank × ${dp} DP)`:` (pure TP, = per-rank batch)`} · output=${cfg.outTokens} tokens · Weights ~${wPerCard.toFixed(1)} GB/card + KV ${kvEndGB.toFixed(2)} GB = <strong>${usedPerCard.toFixed(1)} GB</strong> / ${selChip.mem} GB ${!kvFits?'<strong style="color:#A32D2D">⚠ OOM</strong>':''}</div>`+
    `<div class="dec-summary">
      <div class="dec-card"><div class="lbl">Time to first token (TTFT)</div><div class="big">${ft(prefillTime)}</div><div class="sub">Prefill ${cfg.seqLen} tokens</div></div>
      <div class="dec-card hl"><div class="lbl">Total throughput</div><div class="big">${Math.round(clusterTokPerSec).toLocaleString()} tok/s</div><div class="sub" style="line-height:1.5">global batch ${globalBatch}${dp>1?` (${cfg.batch}×${dp} DP)`:''} × (1/${ft(decodeStepT)})${dp>1?` · ${Math.round(perReplicaTokPerSec).toLocaleString()} tok/s/replica`:''}<br><span style="color:#888;font-size:10px">Per-token decode: ${ft(decodeStepT)} · Per-seq rate: ${perSeqTokPerSec.toFixed(1)} tok/s</span></div></div>
      <div class="dec-card"><div class="lbl">Total generation time</div><div class="big">${ft(totalTime)}</div><div class="sub">Prefill + ${cfg.outTokens} decode steps</div></div>
    </div>`+
    mcs(mc('TTFT',ft(prefillTime),'')+mc('Decode/tok',ft(decodeStepT),'')+mc('Throughput',Math.round(clusterTokPerSec).toLocaleString(),`tok/s (global batch ${globalBatch}${dp>1?` = ${cfg.batch}×${dp} DP`:''})`)+mc('VRAM used',usedPerCard.toFixed(1),'GB/card',kvFits?'mc ok':'mc wa'))+
    `<div class="cw">
      <div class="ct">VRAM breakdown — ${ch.name} × ${nCards} cards${dp>1?` (DP=${dp}, Attn TP=${tpAttn})`:nCards>1?` (TP=${tpAttn})`:''} (${selChip.mem} GB/card)</div>
      <div style="font-size:12px;color:#555;line-height:1.9">
        <strong>Weights (${cfg.precision==='fp16'?'BF16':'FP8'}):</strong> ~${wGB.toFixed(1)} GB total${dp>1?` · Attn÷${tpAttn} MoE÷${nCards}`:` ÷ ${nCards} cards`}${replWB>0?` · MLA a-proj ${(replWB/1e9).toFixed(2)} GB replicated (full/card, not divided)`:''} = <strong>${wPerCard.toFixed(1)} GB/card</strong><br>
        <strong>KV cache:</strong> ${kvFormulaStr}<br>
        <strong>Total per card:</strong> ${wPerCard.toFixed(1)} + ${kvEndGB.toFixed(2)} = <strong>${usedPerCard.toFixed(1)} GB</strong> / ${selChip.mem} GB (${(usedPerCard/selChip.mem*100).toFixed(0)}%)<br>
        <strong>KV per token:</strong> ${(bytesPerTok/1e6).toFixed(3)} MB · Max output @ seq=${cfg.seqLen}: <strong>~${maxOut.toLocaleString()} tokens</strong>
        ${!kvFits?`<br><span style="color:#A32D2D;font-weight:600">⚠ OOM — total ${usedPerCard.toFixed(1)} GB exceeds ${selChip.mem} GB. Reduce batch, seq, output, or increase TP.</span>`:''}
      </div>
    </div>`+
    `<div class="cw">
      <div class="ct">Throughput breakdown</div>
      <div style="font-size:12px;color:#555;line-height:1.9">
        <strong>Per-token decode time:</strong> <strong>${ft(decodeStepT)}</strong> (one step generates 1 token for each of the ${cfg.batch} concurrent sequences in a replica)<br>
        <strong>Per-sequence rate:</strong> 1 / ${ft(decodeStepT)} = <strong>${perSeqTokPerSec.toFixed(2)} tok/s</strong> <span style="color:#888;font-size:11px">(rate of any single sequence's generation — what a user sees)</span><br>
        <strong>Per-replica throughput:</strong> per-rank batch ${cfg.batch} × ${perSeqTokPerSec.toFixed(2)} tok/s = <strong>${Math.round(perReplicaTokPerSec).toLocaleString()} tok/s</strong> <span style="color:#888;font-size:11px">(sum of ${cfg.batch} concurrent sequences in one DP replica)</span><br>
        ${dp>1
          ? `<strong>Global throughput:</strong> ${Math.round(perReplicaTokPerSec).toLocaleString()} tok/s × ${dp} DP replicas = global batch ${globalBatch} ÷ ${ft(decodeStepT)} = <strong style="color:#185FA5">${Math.round(clusterTokPerSec).toLocaleString()} tok/s</strong> <span style="color:#888;font-size:11px">(global batch = ${cfg.batch} per-rank × ${dp} DP — what the full deployment delivers)</span>`
          : `<strong>Global throughput:</strong> global batch ${globalBatch} ÷ ${ft(decodeStepT)} = <strong style="color:#185FA5">${Math.round(clusterTokPerSec).toLocaleString()} tok/s</strong> <span style="color:#888;font-size:11px">(DP=1, pure TP — global batch equals per-rank batch, so global = per-replica total)</span>`}
      </div>
    </div>`+
    `<div class="cw"><div class="ct">Decode step breakdown (per-call and total across ${cfg.outTokens} steps)</div><div style="font-size:11px;color:#999;margin-bottom:8px;line-height:1.6">Each timing cell shows its derivation. <strong>Comm/call</strong> = inter-card communication volume in bytes (AllReduce for row-parallel TP, All-to-All for MoE EP) per single layer instance; ×repeat gives the per-step total. <strong>Time/call</strong> = max(compute, memory) + exposed comm − hidden comm (per single layer instance). <strong>Total/step</strong> = time/call × repeat count. <strong>× ${cfg.outTokens} steps</strong> = total/step × output tokens.</div><div style="overflow-x:auto"><table class="lt"><thead><tr><th>Layer</th><th>AI</th><th>Bytes/call</th><th>Comm/call</th><th>Time/call</th><th>Total/step (×repeat)</th><th>× ${cfg.outTokens} steps</th><th>Bottleneck</th></tr></thead><tbody>${decRows}</tbody></table></div></div>`+
    `<div class="cw"><div class="ct">Token generation timeline (cumulative${cfg.outTokens>500?' — capped at 500 pts':''})</div><div style="position:relative;width:100%;height:240px"><canvas id="gc"></canvas></div></div>`+
    renderMoESweetSpot(ch);

  setTimeout(()=>{
    const c=document.getElementById('gc');if(!c)return;
    charts.gc=new Chart(c,{type:'line',data:{datasets:[
      {label:'Cumulative',data:genCurve,borderColor:'#1D9E75',backgroundColor:'rgba(29,158,117,.08)',fill:true,pointRadius:0,borderWidth:2,tension:0.1},
    ]},options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{title:d=>`Token ${d[0].raw.x}`,label:d=>`${d.raw.y.toFixed(2)} ms`}},annotation:{annotations:{ttft:{type:'line',xMin:0,xMax:0,borderColor:'#7F77DD',borderWidth:1.5,borderDash:[4,4]}}}},
      scales:{x:{title:{display:true,text:'Tokens generated',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}},y:{title:{display:true,text:'Cumulative time (ms)',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}}}
    }});
    c.parentNode.parentNode.appendChild(lgEl([['#1D9E75','Generation time',false],['#7F77DD','Prefill boundary',false]]));
  },50);
}

// ── Init ──────────────────────────────────────────────────────────────────────

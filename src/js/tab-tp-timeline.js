function rTPTimeline(){
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}
  const ch=selChip;
  const tis=layers.map(l=>tpInfoTotal(l,ch));
  // base1 = one card running ONE replica (per-rank batch) with no parallelism.
  const base1=layers.map(l=>{ const c=calcLTotal(l);return Math.max(c.flops/(effectiveTflops(ch,l.type)*1e12),c.bytes/(ch.bw*1e9));}).reduce((a,b)=>a+b,0);
  const totN=tis.map(t=>t.effectiveTime).reduce((a,b)=>a+b,0);   // per-card step time
  const commT=tis.map(t=>t.commTime).reduce((a,b)=>a+b,0);
  const hidT=tis.map(t=>t.hiddenComm).reduce((a,b)=>a+b,0);
  const hasMoE=layers.some(l=>l.type==='moe'||l.moeGroup);
  // v84: DP-aware semantics. With dp>1 the attention/dense blocks are TP-split
  // across tpAttn cards (not nCards), MoE spans all nCards (EP/TP). The timeline
  // below is PER CARD for ONE replica; the cluster runs `dp` such replicas in
  // lockstep — so per-token latency is unchanged but cluster throughput ×dp.
  const cardLatSpeedup=base1/totN;                    // per-card step speedup vs 1 card
  const clusterSpeedup=cardLatSpeedup*dp;             // throughput speedup (dp replicas)
  const isDP=dp>1;
  const rows=layers.map((l,i)=>{
    const ti=tis[i]; const r=getRepeat(l);
    const exC=Math.max(0,ti.commTime-ti.hiddenComm);
    const baseL=Math.max(calcLTotal(l).flops/(effectiveTflops(ch,l.type)*1e12),calcLTotal(l).bytes/(ch.bw*1e9));
    const sumTT=(ti.computeTime+ti.memTime+ti.hiddenComm+exC)||1;
    // per-layer parallel degree: MoE spans nCards, attention/dense span tpAttn
    const lDeg=(l.type==='moe'||l.moeGroup)?nCards:tpAttn;
    return`<tr><td>${l.name} <span style="font-size:10px;color:#aaa">×${r}</span>${isDP?` <span class="bd ${(l.type==='moe'||l.moeGroup)?'bmoe':'bc'}" title="parallel cards for this layer">${(l.type==='moe'||l.moeGroup)?'EP/TP':'TP'} ${lDeg}</span>`:''}</td><td>${ft(ti.computeTime)}</td><td>${ft(ti.memTime)}</td><td>${ft(ti.commTime)}<span style="font-size:10px;color:#aaa"> (${ft(ti.hiddenComm)} hid)</span></td><td><strong>${ft(ti.effectiveTime)}</strong></td><td>${(baseL/ti.effectiveTime).toFixed(2)}×</td>
      <td style="min-width:130px"><div class="tbar" style="height:13px"><div class="tbar-seg" style="flex:${ti.computeTime/sumTT*100};background:#185FA5"></div><div class="tbar-seg" style="flex:${ti.memTime/sumTT*100};background:#1D9E75"></div><div class="tbar-seg" style="flex:${ti.hiddenComm/sumTT*100};background:#b5d4f4"></div><div class="tbar-seg" style="flex:${exC/sumTT*100};background:#F0A32A"></div></div></td></tr>`;
  }).join('');
  // ── headline cards ───────────────────────────────────────────────────────
  const summary=isDP
    ? `<div class="mg5">${
        mc('1 card','1×','no parallel')
       +mc(`${nCards} cards`,ft(totN),`TP${tpAttn}·DP${dp}${hasMoE?` · EP/TP${nCards} MoE`:''}`,'mc hi')
       +mc('Latency speedup',cardLatSpeedup.toFixed(2)+'×','per card / replica','mc hi')
       +mc('Throughput speedup',clusterSpeedup.toFixed(2)+'×',`= ${cardLatSpeedup.toFixed(2)}× × ${dp} DP`,'mc ok')
       +mc('Exposed comm',ft(commT-hidT),`${ft(hidT)} hidden`,'mc'+((commT-hidT)>0?' wa':''))
      }</div>`
    : `<div class="mg5">${
        mc('TP=1',ft(base1),'1 card')
       +mc('TP='+nCards,ft(totN),'per card','mc hi')
       +mc('Speedup',cardLatSpeedup.toFixed(2)+'×','vs 1 card','mc hi')
       +mc('Exposed',ft(commT-hidT),'comm','mc wa')
       +mc('Hidden',ft(hidT),'comm','mc')
      }</div>`;
  // DP explainer banner
  const dpBanner=isDP
    ? `<div class="banner tp" style="display:block"><strong>Data-parallel layout — this timeline is for ONE card of ONE replica.</strong><br>
       <span style="font-size:11px">${dp} DP replicas run in lockstep, each on a per-rank batch of ${cfg.batch} (global batch ${cfg.batch*dp}). Attention &amp; dense blocks are tensor-parallel across <strong>TP=${tpAttn}</strong> cards within a replica; MoE${hasMoE?` spans all <strong>${nCards}</strong> cards (EP/TP)`:' (none in this model)'}.
       Per-card step time and per-token latency (<strong>${ft(totN)}</strong>) are <em>unchanged</em> by DP — replicas don't speed each other up. DP multiplies <em>throughput</em>: the cluster clears ${dp}× the tokens in the same wall-clock, so throughput speedup = ${cardLatSpeedup.toFixed(2)}× (TP) × ${dp} (DP) = <strong>${clusterSpeedup.toFixed(2)}×</strong>.</span></div>`
    : '';
  const titleStr=isDP
    ? `Per-card timeline — 1 of ${nCards} cards · TP${tpAttn} (attn/dense)${hasMoE?` · EP/TP${nCards} (MoE)`:''} · ${dp} DP replicas · total incl. ×repeat (${mode})`
    : `TP${nCards} timeline per card — total incl. ×repeat (${mode})`;
  document.getElementById('mainContent').innerHTML=modeB()+dpBanner+
    summary+
    `<div class="cw"><div class="ct">${titleStr}</div><div style="overflow-x:auto"><table class="lt"><thead><tr><th>Layer</th><th>Compute</th><th>Memory</th><th>Comm (AR/A2A)</th><th>Effective</th><th>Speedup<span style="font-weight:400;color:#aaa"> vs 1 card</span></th><th>Timeline</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div style="display:flex;gap:12px;margin-top:8px;font-size:12px;color:#888;flex-wrap:wrap">
      <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:#185FA5;display:inline-block"></span>Compute</span>
      <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:#1D9E75;display:inline-block"></span>Memory</span>
      <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:#b5d4f4;display:inline-block"></span>Comm (hidden)</span>
      <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:#F0A32A;display:inline-block"></span>Comm (exposed)</span>
    </div></div>`;
  setTimeout(()=>{
    const speedups=layers.map((l,i)=>{const b=Math.max(calcLTotal(l).flops/(effectiveTflops(ch,l.type)*1e12),calcLTotal(l).bytes/(ch.bw*1e9));return+(b/tis[i].effectiveTime).toFixed(3);});
    const div=document.createElement('div');div.className='cw';div.innerHTML=`<div class="ct">Layer speedup vs single card${isDP?' (per-card / per-replica — DP adds throughput on top, not latency)':''}</div><div style="position:relative;width:100%;height:${Math.max(200,layers.length*36+80)}px"><canvas id="spC"></canvas></div>`;
    document.getElementById('mainContent').appendChild(div);
    setTimeout(()=>{const c=document.getElementById('spC');if(!c)return;
      charts.sp=new Chart(c,{type:'bar',data:{labels:layers.map(l=>layerLabel(l)),datasets:[{data:speedups,backgroundColor:'#185FA5',borderRadius:4}]},
        options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:d=>`${d.raw}× speedup`}}},scales:{x:{title:{display:true,text:'Speedup factor',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}},y:{ticks:{color:'#555'},grid:{display:false}}}}
      });},10);
  },50);
}

// ── Batch sweep ───────────────────────────────────────────────────────────────
// Shows how throughput (tok/s) and per-token latency scale across batch sizes.
// Supports prefill and decode modes, and optionally overlays multiple chips.

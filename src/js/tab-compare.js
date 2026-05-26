let cmpSnapshots = [null, null];  // each: {label, layers, chips, cfg, nCards}

function cmpSnap(idx) {
  cmpSnapshots[idx] = {
    label: idx===0 ? (document.getElementById('cmpLbl0')||{value:selChip.name}).value : (document.getElementById('cmpLbl1')||{value:''}).value,
    layers: JSON.parse(JSON.stringify(layers)),
    chip: {...selChip},
    cfg: {...cfg},
    nCards,
    mode,
  };
  renderTab();
}

function rCompare() {
  const chipOpts0 = chips.map(c=>`<option value="${c.id}"${c.id===selChip.id?' selected':''}>${c.name}</option>`).join('');
  const chipOpts1 = chips.map(c=>`<option value="${c.id}"${c.id===chips[Math.min(1,chips.length-1)].id?' selected':''}>${c.name}</option>`).join('');

  const snap0 = cmpSnapshots[0];
  const snap1 = cmpSnapshots[1];

  document.getElementById('mainContent').innerHTML = `
    <div class="cmp-bar">
      <div class="cmp-col">
        <div class="cmp-col-hdr" style="color:#185FA5">▐ Side A</div>
        <select id="cmpMode0" style="margin-bottom:4px">
          <option value="chip">Compare chips (same model)</option>
          <option value="snap">Compare snapshots</option>
        </select>
        <select id="cmpChip0">${chipOpts0}</select>
        <input id="cmpLbl0" placeholder="Label (optional)" value="${snap0?snap0.label:'A'}"/>
        <button class="sweep-btn" style="margin-top:4px" onclick="cmpSnap(0)">📷 Snapshot A (current state)</button>
        ${snap0?`<div style="font-size:11px;color:#0F6E56;margin-top:2px">✓ Snapshot: ${snap0.label||snap0.chip.name} · ${snap0.layers.length} layers · TP=${snap0.nCards}</div>`:'<div style="font-size:11px;color:#aaa;margin-top:2px">No snapshot yet — will use current state on Run</div>'}
      </div>
      <div class="cmp-vsep">vs</div>
      <div class="cmp-col">
        <div class="cmp-col-hdr" style="color:#1D9E75">▐ Side B</div>
        <select id="cmpMode1" style="margin-bottom:4px">
          <option value="chip">Compare chips (same model)</option>
          <option value="snap">Compare snapshots</option>
        </select>
        <select id="cmpChip1">${chipOpts1}</select>
        <input id="cmpLbl1" placeholder="Label (optional)" value="${snap1?snap1.label:'B'}"/>
        <button class="sweep-btn" style="margin-top:4px;background:#1D9E75" onclick="cmpSnap(1)">📷 Snapshot B (current state)</button>
        ${snap1?`<div style="font-size:11px;color:#0F6E56;margin-top:2px">✓ Snapshot: ${snap1.label||snap1.chip.name} · ${snap1.layers.length} layers · TP=${snap1.nCards}</div>`:'<div style="font-size:11px;color:#aaa;margin-top:2px">No snapshot yet — will use current state on Run</div>'}
      </div>
      <button class="cmp-run" onclick="runCompare()">▶ Run comparison</button>
    </div>
    <div id="cmpOut"><div class="es" style="color:#aaa">Select chips and click Run, or take snapshots to compare different models / configs.</div></div>`;
}

function runCompare() {
  function resolveSide(idx) {
    const chipId = parseInt(document.getElementById(`cmpChip${idx}`).value);
    const chip = chips.find(c=>c.id===chipId)||selChip;
    const label = document.getElementById(`cmpLbl${idx}`).value.trim() || chip.name;
    const snap = cmpSnapshots[idx];
    if(snap) return {...snap, label: label||snap.label, chip};
    return {label, layers: JSON.parse(JSON.stringify(layers)), chip, cfg:{...cfg}, nCards, mode};
  }

  const A = resolveSide(0);
  const B = resolveSide(1);

  function computeSideStats(side) {
    const savedTp = nCards; nCards = side.nCards;
    const savedLayers = layers; layers = JSON.parse(JSON.stringify(side.layers));
    const M = side.mode==='prefill' ? side.cfg.batch * side.cfg.seqLen : side.cfg.batch;
    layers.forEach(l => {
      if(l.type==='linear'||l.type==='embedding') l.M = M;
      if(l.type==='moe'||l.moeGroup) l.M = M;
      if(l.type==='attention') {
        l.B = side.cfg.batch;
        if(side.mode==='prefill'){l.S=side.cfg.seqLen;l.kvCache=0;}
        else{l.S=1;l.kvCache=side.cfg.seqLen+Math.floor(side.cfg.outTokens/2);}
      }
    });
    const ch = side.chip;
    const tis = layers.map(l=>tpInfoTotal(l,ch));
    const tot = tis.map(t=>t.effectiveTime).reduce((a,b)=>a+b,0);
    // Per-rank tokens per step; cluster total = × dp DP replicas.
    // Compare tab is most commonly used without DP, but factor it in for
    // consistency with the Decode analysis page.
    const sideDp = Math.max(1, (typeof dp === 'number' && dp >= 1) ? dp : 1);
    const tokPerRank = side.mode==='prefill' ? side.cfg.batch*side.cfg.seqLen : side.cfg.batch;
    const tok = tokPerRank * sideDp;
    const thr = tok/(tot||1e-12);
    const gemmTF = effectiveTflops(ch,'linear');
    const ri  = gemmTF*1e12/(ch.bw*1e9);
    const totalFlops = layers.map(l=>calcLTotal(l).flops).reduce((a,b)=>a+b,0);
    const util = Math.min(100, totalFlops/(gemmTF*1e12*(tot||1e-12)*side.nCards)*100);
    const wGB = modelWeightBytes()/1e9;
    // Replicated dense linears (MLA a-projections) sit in FULL on every card —
    // keep them out of the /nCards split so this matches the Decode page's
    // per-card VRAM accounting. (The rest stays the Compare tab's simple
    // uniform-shard estimate.)
    const replWB = replicatedWeightBytes() / 1e9;
    const wPerCard = (wGB - replWB) / side.nCards + replWB;
    // KV sharded by min(side.nCards, side.cfg.H) — same convention as updateCfgInfo.
    const ksfSide = Math.max(1, Math.min(side.nCards, side.cfg.H || 1));
    const kvGB = kvCacheBytes(side.cfg.seqLen)/1e9 / ksfSide;
    const memPct = Math.min(100,(wPerCard+kvGB)/ch.mem*100);
    const memBound = layers.filter(l=>{const lRi=effectiveTflops(ch,l.type)*1e12/(ch.bw*1e9);return calcL(l).ai<lRi;}).length;
    const layerData = layers.map((l,i)=>({name:l.name, time:tis[i].effectiveTime, pct:tis[i].effectiveTime/(tot||1)*100}));
    nCards = savedTp; layers = savedLayers;
    return {tot, thr, util, wGB, wPerCard, kvGB, memPct, memBound, layerData, ri, chip:ch, side};
  }

  const sA = computeSideStats(A);
  const sB = computeSideStats(B);

  function winner(vA, vB, higherIsBetter=true) {
    if(higherIsBetter) return vA>vB ? ['<span class="cmp-winner">▲</span>',''] : vA<vB ? ['','<span class="cmp-winner">▲</span>'] : ['',''];
    else return vA<vB ? ['<span class="cmp-winner">▼</span>',''] : vA>vB ? ['','<span class="cmp-winner">▼</span>'] : ['',''];
  }

  const [wThrA, wThrB] = winner(sA.thr, sB.thr, true);
  const [wLatA, wLatB] = winner(sA.tot, sB.tot, false);
  const [wUtlA, wUtlB] = winner(sA.util, sB.util, true);
  const [wMemA, wMemB] = winner(sA.memPct, sB.memPct, false);

  const speedup = sA.tot>0 && sB.tot>0 ? (sA.tot/sB.tot).toFixed(2) : '—';
  const thrRatio = sA.thr>0 && sB.thr>0 ? (sB.thr/sA.thr).toFixed(2) : '—';

  function sideCard(s, color, label) {
    return `<div class="cmp-card">
      <div class="cmp-card-hdr"><span class="cmp-dot" style="background:${color}"></span>${label} — ${s.chip.name}</div>
      <div class="cmp-stat"><span class="lbl">Mode</span><span class="val">${s.side.mode.toUpperCase()} · batch=${s.side.cfg.batch}</span></div>
      <div class="cmp-stat"><span class="lbl">TP</span><span class="val">${s.side.nCards} card${s.side.nCards>1?'s':''}</span></div>
      <div class="cmp-stat"><span class="lbl">Chip</span><span class="val">${effectiveTflops(s.chip,'linear')}T · ${s.chip.bw}GB/s · ${s.chip.mem}GB</span></div>
      <div class="cmp-stat"><span class="lbl">Total latency</span><span class="val">${ft(s.tot)}</span></div>
      <div class="cmp-stat"><span class="lbl">Throughput</span><span class="val">${Math.round(s.thr)} tok/s</span></div>
      <div class="cmp-stat"><span class="lbl">GPU util</span><span class="val">${s.util.toFixed(1)}%</span></div>
      <div class="cmp-stat"><span class="lbl">Weights</span><span class="val">~${s.wGB.toFixed(1)} GB (${s.wPerCard.toFixed(1)} GB/card)</span></div>
      <div class="cmp-stat"><span class="lbl">VRAM used</span><span class="val">${s.memPct.toFixed(0)}% of ${s.chip.mem} GB</span></div>
      <div class="cmp-stat"><span class="lbl">Mem-bound layers</span><span class="val">${s.memBound} / ${s.side.layers.length}</span></div>
    </div>`;
  }

  // Layer comparison table
  const allNames = [...new Set([...sA.layerData.map(l=>l.name), ...sB.layerData.map(l=>l.name)])];
  const layerRows = allNames.map(name => {
    const la = sA.layerData.find(l=>l.name===name);
    const lb = sB.layerData.find(l=>l.name===name);
    const ta = la?la.time:null, tb = lb?lb.time:null;
    let diff = '';
    if(ta&&tb) {
      const r = tb/ta;
      diff = r<0.95 ? `<span style="color:#0F6E56;font-size:11px">B ${(1/r).toFixed(2)}× faster</span>` :
             r>1.05 ? `<span style="color:#A32D2D;font-size:11px">A ${r.toFixed(2)}× faster</span>` :
             `<span style="color:#888;font-size:11px">≈ equal</span>`;
    }
    return `<tr>
      <td style="font-size:12px">${name}</td>
      <td style="font-size:12px">${ta?ft(ta):'—'}</td>
      <td style="font-size:12px">${la?la.pct.toFixed(1)+'%':'—'}</td>
      <td style="font-size:12px">${tb?ft(tb):'—'}</td>
      <td style="font-size:12px">${lb?lb.pct.toFixed(1)+'%':'—'}</td>
      <td>${diff}</td>
    </tr>`;
  }).join('');

  // Bar chart datasets for per-layer time comparison
  const barLabels = allNames;
  const barA = allNames.map(n=>{ const l=sA.layerData.find(x=>x.name===n); return l?+(l.time*1e6).toFixed(3):0; });
  const barB = allNames.map(n=>{ const l=sB.layerData.find(x=>x.name===n); return l?+(l.time*1e6).toFixed(3):0; });

  document.getElementById('cmpOut').innerHTML =
    `<div class="cmp-grid">${sideCard(sA,'#185FA5',A.label)}${sideCard(sB,'#1D9E75',B.label)}</div>`+
    mcs(
      mc('B vs A latency', speedup+'×','','mc hi')+
      mc('B vs A throughput', thrRatio+'×','','mc hi')+
      mc('A throughput', Math.round(sA.thr),'tok/s')+
      mc('B throughput', Math.round(sB.thr),'tok/s')
    )+
    `<div class="cw"><div class="ct">Per-layer time — ${A.label} vs ${B.label} (µs, incl. repeat)</div>
      <div style="position:relative;width:100%;height:${Math.max(260,allNames.length*32+80)}px"><canvas id="cmpC"></canvas></div></div>`+
    `<div class="cw"><div class="ct">Layer-by-layer breakdown</div><div style="overflow-x:auto">
      <table class="lt"><thead><tr>
        <th>Layer</th>
        <th>${A.label} time</th><th>${A.label} %</th>
        <th>${B.label} time</th><th>${B.label} %</th>
        <th>Delta</th>
      </tr></thead><tbody>${layerRows}</tbody></table></div></div>`;

  setTimeout(()=>{
    const c=document.getElementById('cmpC'); if(!c) return;
    charts.cmp = new Chart(c,{type:'bar',
      data:{labels:barLabels,datasets:[
        {label:A.label,data:barA,backgroundColor:'#185FA580',borderColor:'#185FA5',borderWidth:1,borderRadius:3},
        {label:B.label,data:barB,backgroundColor:'#1D9E7580',borderColor:'#1D9E75',borderWidth:1,borderRadius:3},
      ]},
      options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:true,position:'top',labels:{color:'#555',font:{size:12}}},
                 tooltip:{callbacks:{label:d=>`${d.dataset.label}: ${d.raw} µs`}}},
        scales:{x:{title:{display:true,text:'Time (µs)',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}},
                y:{ticks:{color:'#555',font:{size:11}},grid:{display:false}}}
      }
    });
  },50);
}

// ── Structure diagram ─────────────────────────────────────────────────────────
// ── Architecture classifier ─────────────────────────────────────────────────
// Given the current layer list, return a short human-readable tag describing
// what kind of architecture it is. Used by rStructure() for the header banner.

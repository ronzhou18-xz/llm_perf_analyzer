let sweepState = { minB:1, maxB:128, steps:16, metric:'throughput', chips:[] };

function rSweep() {
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}

  // Build chip options for overlay
  const chipOpts = chips.map(c=>`<option value="${c.id}"${c.id===selChip.id?' selected':''}>${c.name}</option>`).join('');
  // Compute a set of batch sizes (log-spaced)
  const genBatches = (mn,mx,n) => {
    if(mn>=mx) return [mn];
    const arr=[];
    for(let i=0;i<n;i++) arr.push(Math.max(1,Math.round(Math.exp(Math.log(mn)+(i/(n-1))*(Math.log(mx)-Math.log(mn))))));
    return [...new Set(arr)].sort((a,b)=>a-b);
  };

  const html = `
    <div class="sweep-ctrl">
      <label>Min batch <input type="number" id="swMin" value="${sweepState.minB}" min="1" style="width:60px"/></label>
      <label>Max batch <input type="number" id="swMax" value="${sweepState.maxB}" min="1" style="width:70px"/></label>
      <label>Steps <input type="number" id="swSteps" value="${sweepState.steps}" min="2" max="64" style="width:55px"/></label>
      <label>Metric
        <select id="swMetric">
          <option value="throughput"${sweepState.metric==='throughput'?' selected':''}>Throughput (tok/s)</option>
          <option value="latency"${sweepState.metric==='latency'?' selected':''}>Latency (ms)</option>
          <option value="tokpersec_per_card"${sweepState.metric==='tokpersec_per_card'?' selected':''}>Tok/s per card</option>
        </select>
      </label>
      <label>Overlay chips
        <select id="swChips" multiple style="height:48px;width:160px;font-size:11px">${chipOpts}</select>
      </label>
      <button class="sweep-btn" onclick="runSweep()">Run sweep</button>
    </div>
    <div id="sweepOut"></div>`;
  document.getElementById('mainContent').innerHTML = html;
  // pre-select active chip
  const sel = document.getElementById('swChips');
  [...sel.options].forEach(o=>{ if(parseInt(o.value)===selChip.id) o.selected=true; });
}

function runSweep() {
  const minB  = Math.max(1, parseInt(document.getElementById('swMin').value)||1);
  const maxB  = Math.max(minB, parseInt(document.getElementById('swMax').value)||128);
  const steps = Math.min(64, Math.max(2, parseInt(document.getElementById('swSteps').value)||16));
  const metric= document.getElementById('swMetric').value;
  const selIds= [...document.getElementById('swChips').selectedOptions].map(o=>parseInt(o.value));
  const sweepChips = chips.filter(c=>selIds.includes(c.id));
  if(!sweepChips.length) return;

  sweepState = {minB, maxB, steps, metric, chips: selIds};

  const batches = (() => {
    if(minB>=maxB) return [minB];
    const arr=[];
    for(let i=0;i<steps;i++) arr.push(Math.max(1,Math.round(Math.exp(Math.log(minB)+(i/(steps-1))*(Math.log(maxB)-Math.log(minB))))));
    return [...new Set(arr)].sort((a,b)=>a-b);
  })();

  // For each chip × batch, compute total effective time
  function computeForBatch(chip, batch) {
    // Temporarily override layer M values for this batch
    const savedLayers = JSON.parse(JSON.stringify(layers));
    const M = mode==='prefill' ? batch * cfg.seqLen : batch;
    layers.forEach(l => {
      if(l.type==='linear'||l.type==='embedding') l.M=M;
      if(l.type==='moe'||l.moeGroup) l.M=M;
      if(l.type==='attention') { l.B=batch; if(mode==='prefill'){l.S=cfg.seqLen;l.kvCache=0;} else {l.S=1;l.kvCache=cfg.seqLen+Math.floor(cfg.outTokens/2);} }
    });
    const tot = layers.map(l=>tpInfoTotal(l,chip).effectiveTime).reduce((a,b)=>a+b,0);
    // restore
    layers.forEach((l,i)=>Object.assign(l,savedLayers[i]));
    return tot;
  }

  const COLORS = ['#185FA5','#1D9E75','#EF9F27','#7F77DD','#BA7517','#A32D2D'];
  const datasets = sweepChips.map((chip,ci) => {
    const color = COLORS[ci % COLORS.length];
    const data = batches.map(b => {
      const tot = computeForBatch(chip, b);
      // tok = tokens delivered ACROSS THE WHOLE CLUSTER per step
      //     = per-rank tokens (b for decode, b*seq for prefill) × dp DP replicas
      const tokPerRank = mode==='prefill' ? b*cfg.seqLen : b;
      const tok = tokPerRank * dp;
      if(metric==='throughput') return +(tok/tot).toFixed(2);     // cluster tok/s
      if(metric==='latency')    return +(tot*1000).toFixed(4);
      return +(tok/(tot*nCards)).toFixed(2);                       // tok/s per card
    });
    return {label: chip.name, data, borderColor: color, backgroundColor: color+'22',
            fill: false, tension: 0.3, pointRadius: 4, pointHoverRadius: 6, borderWidth: 2};
  });

  const yLabel = metric==='throughput' ? 'Tokens / second' : metric==='latency' ? 'Latency (ms)' : 'Tok/s per card';
  const title  = metric==='throughput' ? 'Throughput vs Batch size' : metric==='latency' ? 'Latency vs Batch size' : 'Tok/s per card vs Batch size';

  // Summary stats table
  const bestBatch = {};
  sweepChips.forEach((chip,ci) => {
    const data = datasets[ci].data;
    if(metric==='latency') bestBatch[chip.name] = batches[data.indexOf(Math.min(...data))];
    else                   bestBatch[chip.name] = batches[data.indexOf(Math.max(...data))];
  });

  const statsRows = sweepChips.map((chip,ci)=>{
    const data = datasets[ci].data;
    const peak = metric==='latency' ? Math.min(...data) : Math.max(...data);
    const at   = metric==='latency' ? batches[data.indexOf(peak)] : batches[data.indexOf(peak)];
    const atB1 = data[0];
    return`<tr>
      <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COLORS[ci%COLORS.length]};margin-right:5px"></span>${chip.name}</td>
      <td>${effectiveTflops(chip,'linear')} TFLOPS${cfg.precision!=='fp16'?' <span style="font-size:10px;color:#888">(FP8)</span>':''}</td><td>${chip.bw} GB/s</td><td>${chip.mem} GB</td>
      <td style="font-weight:600">${metric==='latency'?peak.toFixed(3)+' ms':Math.round(peak)+' tok/s'}</td>
      <td>batch=${at}</td>
      <td>${metric==='latency'?atB1.toFixed(3)+' ms':Math.round(atB1)+' tok/s'} @ batch=1</td>
    </tr>`;
  }).join('');

  document.getElementById('sweepOut').innerHTML =
    `<div class="cw"><div class="ct">${title} — ${mode} mode · seq=${cfg.seqLen} · TP=${nCards}</div>
      <div style="position:relative;width:100%;height:320px"><canvas id="swC"></canvas></div></div>` +
    `<div class="cw"><div class="ct">Peak summary</div><div style="overflow-x:auto">
      <table class="lt"><thead><tr><th>Chip</th><th>TFLOPS</th><th>BW</th><th>Mem</th><th>Peak ${yLabel}</th><th>@ batch</th><th>Batch=1 baseline</th></tr></thead>
      <tbody>${statsRows}</tbody></table></div></div>`;

  setTimeout(()=>{
    const c = document.getElementById('swC'); if(!c) return;
    charts.sw = new Chart(c, {type:'line',
      data: {labels: batches, datasets},
      options:{responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:sweepChips.length>1,position:'top',labels:{color:'#555',font:{size:12}}},
                 tooltip:{callbacks:{label:d=>`${d.dataset.label}: ${d.raw} ${yLabel}`}}},
        scales:{
          x:{title:{display:true,text:'Batch size',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}},
          y:{title:{display:true,text:yLabel,color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'},beginAtZero:true}
        }
      }
    });
  }, 50);
}

// ── Comparison mode ───────────────────────────────────────────────────────────
// Side-by-side analysis: compare two chips or two "snapshots" of the current model
// with different configs (e.g. different TP, batch, or model variant loaded from HF).


function setTab(btn,t){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));btn.classList.add('active');activeTab=t;renderTab();}
function renderTab(){
  dCharts();
  const t=activeTab;
  try {
    if(t==='roofline')rRoof();
    else if(t==='bottleneck')rBot();
    else if(t==='arithmetic')rAI();
    else if(t==='throughput')rThr();
    else if(t==='tp')rTPTimeline();
    else if(t==='sweep')rSweep();
    else if(t==='compare')rCompare();
    else if(t==='structure')rStructure();
    else rDecode();
  } catch(e) {
    console.error('renderTab['+t+'] failed:',e);
    document.getElementById('mainContent').innerHTML =
      `<div class="es" style="color:#A32D2D">Render error in "${t}" tab: ${e.message}<br><br>Try switching tabs or reloading the model.</div>`;
  }
}
function dCharts(){
  try {
    Object.values(charts).forEach(c=>{
      try{ c.destroy(); }catch(e){ console.warn('chart destroy failed:',e); }
    });
  } catch(e) { console.warn('dCharts outer:',e); }
  charts={};
}
function mcs(h,g='mg'){return`<div class="${g}">${h}</div>`;}
function mc(l,v,u,cls='mc'){return`<div class="${cls}"><div class="ml">${l}</div><div class="mv">${v}<span class="mu"> ${u}</span></div></div>`;}

function tpB(){
  if(nCards===1)return'';
  const tN=layers.map(l=>tpInfoTotal(l,selChip).effectiveTime).reduce((a,b)=>a+b,0);
  const b1=layers.map(l=>{const c=calcLTotal(l);return Math.max(c.flops/(effectiveTflops(selChip,l.type)*1e12),c.bytes/(selChip.bw*1e9));}).reduce((a,b)=>a+b,0);
  const hasMoE=layers.some(l=>l.type==='moe');
  const label=hasMoE?`TP=${nCards} (EP for MoE) · ${selChip.name} ${ibw} GB/s uni A2A/AR · ${Math.round(overlap*100)}% overlap`:`TP=${nCards} · ${selChip.name} ${ibw} GB/s uni · ${Math.round(overlap*100)}% overlap`;
  return`<div class="banner nCards"><span>${label}</span><span><strong>${nCards} cards</strong> · speedup <strong>${(b1/tN).toFixed(2)}×</strong> · total <strong>${ft(tN)}</strong></span></div>`;
}
function fa2B(){
  const on=layers.filter(l=>l.type==='attention'&&l.fa2);if(!on.length)return'';
  const s=on.map(l=>{const a=calcStdAttn(l),b=calcFA2Attn(l);return(a.bytes-b.bytes)*getRepeat(l);}).reduce((a,b)=>a+b,0);
  return`<div class="banner fa2">FA2 on ${on.length} attention type(s) — ~${fmt(s)} bytes HBM saved (total incl. repeat)</div>`;
}
function modeB(){
  const dpTag = dp > 1
    ? ` per rank (global batch=<strong>${cfg.batch*dp}</strong> = ${cfg.batch}×${dp} DP)`
    : ` (global batch=<strong>${cfg.batch}</strong>, pure TP)`;
  return mode==='decode'
    ?`<div class="banner dec">Decode mode · query M=<strong>${cfg.batch}</strong>${dpTag} · KV context=<strong>${cfg.seqLen+Math.floor(cfg.outTokens/2)}</strong> tokens · all times include repeat count</div>`
    :`<div class="banner pre">Prefill mode · M=<strong>${cfg.batch*cfg.seqLen}</strong> (batch ${cfg.batch} × seq ${cfg.seqLen})${dpTag} · times include layer repeat counts</div>`;
}
function layerLabel(l) {
  const r=getRepeat(l); const suf=r>1?` ×${r}`:'';
  if(l.type==='attention') return l.fa2?l.name+' [FA2]'+suf:l.name+' [std]'+suf;
  if(l.type==='linear_attn') return l.name+' [δ-net]'+suf;
  if(l.type==='moe') return l.name+` [top${l.topK}/${l.nExperts}]`+suf;
  if(l.moeGroup) return l.name+suf;
  return l.name+suf;
}
function layerComputeColor(l) {
  if(l.type==='attention') return l.fa2?'#FAC775':'#AFA9EC';
  if(l.type==='linear_attn') return '#9FE1CB';
  if(l.type==='moe'||l.moeGroup) return '#FAC775';
  return '#185FA5';
}
function layerMemColor(l) {
  if(l.type==='attention') return l.fa2?'#EF9F27':'#7F77DD';
  if(l.type==='linear_attn') return '#0F6E56';
  if(l.type==='moe'||l.moeGroup) return '#BA7517';
  return '#1D9E75';
}
function layerAIColor(l, ri) {
  if(l.type==='attention') return l.fa2?'#EF9F27':'#7F77DD';
  if(l.type==='linear_attn') return '#0F6E56';
  if(l.type==='moe'||l.moeGroup) return '#BA7517';
  return l.ai<ri?'#1D9E75':'#185FA5';
}
function lgEl(items){const el=document.createElement('div');el.style.cssText='display:flex;gap:12px;margin-top:8px;font-size:12px;color:#888;flex-wrap:wrap';el.innerHTML=items.map(([col,label,r])=>`<span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:${r?'50%':'2px'};background:${col};display:inline-block"></span>${label}</span>`).join('');return el;}

// ── Roofline (per-layer, not multiplied — shows single-layer AI) ──────────────

function renderChipList() {
  const defaultId = chips[0].id;  // BW1100 is always chips[0]
  document.getElementById('chipList').innerHTML = chips.map(c => `
    <div class="chip-card${c.id===selChip.id?' active':''}" onclick="selC(${c.id})">
      <div style="flex:1;min-width:0">
        <div class="cn" style="display:flex;align-items:center;gap:5px">
          ${c.name}
          ${c.id===defaultId?'<span style="font-size:9px;padding:1px 5px;border-radius:99px;background:#185FA5;color:#fff;font-weight:600;letter-spacing:.04em">DEFAULT</span>':''}
        </div>
        <div class="cm">${effectiveTflops(c,'linear')} TFLOPS${cfg.precision!=='fp16'?' (FP8)':''} · ${c.bw} GB/s HBM · ${c.mem} GB · ${c.ibw||'—'} GB/s link</div>
      </div>
      <div style="display:flex;gap:2px;flex-shrink:0">
        <button class="ibtn" onclick="event.stopPropagation();openEditChip(${c.id})" title="Edit">✎</button>
        <button class="ibtn del" onclick="event.stopPropagation();deleteChip(${c.id})" title="Delete chip">✕</button>
      </div>
    </div>`
  ).join('');
}

function renderLayerList() {
  // Refresh MoE parallelism selector availability whenever layers change
  refreshMoEParAvailability();
  // v84: refresh Sparse-MLA (DSA) toggle visibility — DSA models only
  refreshSparseMLAAvailability();
  const el=document.getElementById('layerList');
  const entryCount = layers.length;
  // Show cfg.nL as the "transformer layers" count (accurate for all architectures).
  // Sum-of-repeats is misleading for grouped entries (e.g. Qwen3.5 shows 186 not 32).
  const blockCount = cfg.nL;
  document.getElementById('lcount').textContent = entryCount
    ? (entryCount !== blockCount
        ? `${entryCount} entries · ${blockCount} transformer layers`
        : `${entryCount} layers`)
    : '';
  if(!layers.length){el.innerHTML='<div style="padding:10px 14px;font-size:12px;color:#aaa">No layers yet</div>';return;}
  const ridge=effectiveTflops(selChip,'linear')*1e12/(selChip.bw*1e9);
  // Filter out zero-repeat entries (defensive — shouldn't exist with correct imports)
  const visibleLayers = layers.filter(l => getRepeat(l) > 0);
  el.innerHTML=visibleLayers.map((l,i)=>{
    const c=calcL(l); const ia=l.type==='attention',f2=ia&&l.fa2;
    const lRidge = effectiveTflops(selChip, l.type)*1e12/(selChip.bw*1e9);
    const bnd=c.ai<lRidge?'mem':'comp';
    const r=getRepeat(l);
    const ti=tpInfoTotal(l,selChip);
    const ila=l.type==='linear_attn';
    const imoe=!!(l.type==='moe'||l.moeGroup);
    const meta=ia?`S=${l.S} kv=${l.kvCache||0} H=${l.H} D=${l.D}`
              :ila?`M=${l.M} nQK=${l.nQK} nV=${l.nV} D=${l.D}`
              :(l.type==='moe')?`M=${l.M} hid=${l.hid} ffn=${l.ffnDim} top${l.topK}/${l.nExperts}+${l.nShared??0}sh`
              :(l.moeGroup==='gate_up'||l.moeGroup==='down')?`M=${l.M} K=${l.K} N=${l.N} · top${l.moeTopK}/${l.moeExperts}`
              :l.type==='embedding'?`vocab=${l.K}`:`M=${l.M} K=${l.K} N=${l.N}`;
    const badgeCls = f2?'bf':ia?'ba':ila?'bla':imoe?'bmoe':bnd==='mem'?'bm':'bc';
    const badgeTxt = f2?'FA2':ia?'attn':ila?'δ-net':imoe?'MoE':bnd==='mem'?'mem':'comp';
    return`<div class="layer-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div style="display:flex;align-items:center;gap:5px;flex:1;min-width:0">
          <span style="font-size:10px;color:#aaa;min-width:16px">${i+1}</span>
          <span class="cn" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${l.name}</span>
          <span class="bd ${badgeCls}">${badgeTxt}</span>
          ${r>1?`<span style="font-size:10px;color:#aaa">×${r}</span>`:''}
        </div>
        <div style="display:flex;gap:3px;align-items:center;flex-shrink:0">
          ${ia?`<button class="fa2t${f2?' on':''}" onclick="togFA2(${l.id});event.stopPropagation()">FA2</button>`:''}
          <button class="ibtn" onclick="openEdit(${l.id})">✎</button>
          <button class="ibtn del" onclick="delL(${l.id})">✕</button>
        </div>
      </div>
      <div class="cm" style="padding-left:20px">AI: ${c.ai.toFixed(1)} · ${meta}${nCards>1?` · ${ft(ti.effectiveTime)} total`:''}</div>
      ${(() => { const sh = tpShardShape(l); return sh ? `<div class="cm" style="padding-left:20px;color:#a07722">${sh}</div>` : ''; })()}
    </div>`;
  }).join('');
}

function selC(id){
  selChip=chips.find(c=>c.id===id);
  syncIbwFromChip();
  renderChipList();updateCfgInfo();renderLayerList();renderTab();
}
function delL(id){layers=layers.filter(l=>l.id!==id);renderLayerList();renderTab();}

// ── Chip edit modal ───────────────────────────────────────────────────────────
function saveChip(id) {
  const c=chips.find(x=>x.id===id); if(!c) return;
  c.name   = document.getElementById('cN').value.trim()||c.name;
  c.bw     = parseFloat(document.getElementById('cBW').value)||c.bw;
  c.tflops = parseFloat(document.getElementById('cTF').value)||c.tflops;
  c.tflops_fp8 = parseFloat(document.getElementById('cTF8').value)||c.tflops*2;
  c.mem    = parseFloat(document.getElementById('cMEM').value)||c.mem;
  c.ibw    = parseFloat(document.getElementById('cIBW').value)||c.ibw;
  if(selChip.id===id) { selChip=c; syncIbwFromChip(); }
  closeModal(); renderChipList(); updateCfgInfo(); renderTab();
}

// ── Layer edit modal ──────────────────────────────────────────────────────────
function deleteChip(id) {
  if(chips.length <= 1) {
    // Show a brief inline message instead of alert()
    const card = document.querySelector(`.chip-card`);
    if(card) {
      const msg = document.createElement('div');
      msg.style.cssText='position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1a1a18;color:#fff;font-size:12px;padding:8px 16px;border-radius:8px;z-index:200;pointer-events:none';
      msg.textContent = 'At least one chip must remain.';
      document.body.appendChild(msg);
      setTimeout(()=>msg.remove(), 2200);
    }
    return;
  }
  const wasSelected = selChip.id === id;
  chips = chips.filter(c => c.id !== id);
  if(wasSelected) {
    // Prefer BW1100, otherwise fall back to first chip
    selChip = chips.find(c=>c.name==='BW1100') || chips[0];
    syncIbwFromChip();
  }
  closeModal();
  renderChipList(); updateCfgInfo(); renderLayerList(); renderTab();
}

function openEditChip(id) {
  const c = chips.find(x=>x.id===id); if(!c) return;
  document.getElementById('ov').style.display='flex';
  const mb=document.getElementById('mb'); mb.className='modal';
  mb.innerHTML=`<h3>Edit chip — ${c.name}</h3>
    <div class="f"><label>Name</label><input id="cN" value="${c.name}"/></div>
    <div class="f"><label>Memory bandwidth (GB/s)</label><input id="cBW" type="number" value="${c.bw}"/></div>
    <div class="f"><label>Compute TFLOPS (FP16)</label><input id="cTF" type="number" value="${c.tflops}"/></div>
    <div class="f"><label>Compute TFLOPS (FP8)</label><input id="cTF8" type="number" value="${c.tflops_fp8||c.tflops*2}"/></div>
    <div class="f"><label>Memory capacity (GB)</label><input id="cMEM" type="number" value="${c.mem}"/></div>
    <div class="f"><label>Interconnect per card — unidirectional (GB/s)</label><input id="cIBW" type="number" value="${c.ibw||224}"/><div style="font-size:11px;color:#aaa;margin-top:2px">Half of spec-sheet bidirectional (e.g. NVLink4=900 bi → 450 uni)</div></div>
    <div class="mbtns">
      <button class="btn" onclick="closeModal()">Cancel</button>
      <button class="btn p" onclick="saveChip(${id})">Save</button>
    </div>`;
}
function aF(l){
  const v=l||{B:1,S:2048,H:32,D:128,kvCache:0,fa2:false};
  return`<div class="abox"><div class="abox-title">Attention parameters</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div class="f" style="margin:0"><label>Batch (B)</label><input id="aB" type="number" value="${v.B}"/></div>
      <div class="f" style="margin:0"><label>Query seq (S)</label><input id="aS" type="number" value="${v.S}"/></div>
      <div class="f" style="margin:0"><label>Heads (H)</label><input id="aH" type="number" value="${v.H}"/></div>
      <div class="f" style="margin:0"><label>Head dim (D)</label><input id="aD" type="number" value="${v.D}"/></div>
    </div>
    <div class="f" style="margin-top:8px;margin-bottom:8px"><label>KV cache length</label><input id="aKV" type="number" value="${v.kvCache||0}"/><div style="font-size:11px;color:#aaa;margin-top:2px">0=prefill · >0 for decode</div></div>
    <label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer"><input type="checkbox" id="aFA2" ${v.fa2?'checked':''}/>FlashAttention-2</label>
  </div>`;
}

function moeF(l){
  const v=l||{M:2048,hid:4096,ffnDim:1536,topK:8,nExperts:128,nShared:1};
  return`<div class="abox"><div class="abox-title">MoE FFN parameters</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div class="f" style="margin:0"><label>M (batch×seq)</label><input id="moM" type="number" value="${v.M||2048}"/></div>
      <div class="f" style="margin:0"><label>Hidden dim</label><input id="moH" type="number" value="${v.hid||4096}"/></div>
      <div class="f" style="margin:0"><label>Expert FFN dim</label><input id="moF" type="number" value="${v.ffnDim||1536}"/></div>
      <div class="f" style="margin:0"><label>Top-K active</label><input id="moK" type="number" value="${v.topK||8}"/></div>
      <div class="f" style="margin:0"><label>Total experts</label><input id="moE" type="number" value="${v.nExperts||128}"/></div>
      <div class="f" style="margin:0"><label>Shared experts</label><input id="moS" type="number" value="${v.nShared!=null?v.nShared:1}"/></div>
    </div>
    <div style="font-size:11px;color:#aaa;margin-top:6px">Active FLOPs use top-K + shared. All expert weights load into VRAM.</div>
  </div>`;
}

function laF(l){
  const v=l||{M:2048,K:5120,nQK:16,nV:48,D:128};
  return`<div class="abox"><div class="abox-title">GatedDeltaNet parameters</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div class="f" style="margin:0"><label>M (batch×seq)</label><input id="laM" type="number" value="${v.M||2048}"/></div>
      <div class="f" style="margin:0"><label>K (input dim)</label><input id="laK" type="number" value="${v.K||5120}"/></div>
      <div class="f" style="margin:0"><label>nQK heads</label><input id="laNQK" type="number" value="${v.nQK||16}"/></div>
      <div class="f" style="margin:0"><label>nV heads</label><input id="laNV" type="number" value="${v.nV||48}"/></div>
      <div class="f" style="margin:0;grid-column:span 2"><label>Head dim (D)</label><input id="laD" type="number" value="${v.D||128}"/></div>
    </div>
    <div style="font-size:11px;color:#aaa;margin-top:6px">Linear complexity recurrent attention. No KV cache. State matrix (nV×D×D) stays in SRAM.</div>
  </div>`;
}

function openEdit(id){
  const l=layers.find(x=>x.id===id);if(!l)return;editId=id;
  const ia=l.type==='attention', ila=l.type==='linear_attn', imoe=l.type==='moe';
  document.getElementById('ov').style.display='flex';
  const mb=document.getElementById('mb');mb.className='modal';
  mb.innerHTML=`<h3>Edit layer</h3>
    <div class="f"><label>Name</label><input id="lN" value="${l.name}"/></div>
    <div class="f"><label>Type</label><select id="lT" onchange="onTC(this)">
      <option value="linear"${l.type==='linear'?' selected':''}>Linear (GEMM)</option>
      <option value="embedding"${l.type==='embedding'?' selected':''}>Embedding</option>
      <option value="attention"${ia?' selected':''}>Attention (full)</option>
      <option value="linear_attn"${ila?' selected':''}>GatedDeltaNet (linear attn)</option>
      <option value="moe"${imoe?' selected':''}>MoE FFN</option>
    </select></div>
    <div id="lf" style="${ia||ila||imoe?'display:none':''}">
      <div class="f"><label>M</label><input id="lM" type="number" value="${l.M||2048}"/></div>
      <div class="f"><label>K</label><input id="lK" type="number" value="${l.K||4096}"/></div>
      <div class="f"><label>N</label><input id="lNv" type="number" value="${l.N||4096}"/></div>
    </div>
    <div id="af" style="${ia?'':'display:none'}">${aF(l)}</div>
    <div id="laf" style="${ila?'':'display:none'}">${laF(l)}</div>
    <div id="mof" style="${imoe?'':'display:none'}">${moeF(l)}</div>
    <div class="f" style="margin-top:4px"><label>Repeat count (×)</label><input id="lR" type="number" min="1" value="${getRepeat(l)}"/><div style="font-size:11px;color:#aaa;margin-top:2px">Number of times this layer appears</div></div>
    <div class="mbtns"><button class="btn d" onclick="delL(${id});closeModal()">Delete</button><button class="btn" onclick="closeModal()">Cancel</button><button class="btn p" onclick="saveEdit()">Save</button></div>`;
}
function onTC(s){
  const t=s.value;
  document.getElementById('lf').style.display =(t==='attention'||t==='linear_attn'||t==='moe')?'none':'';
  document.getElementById('af').style.display  =(t==='attention')?'':'none';
  document.getElementById('laf').style.display =(t==='linear_attn')?'':'none';
  document.getElementById('mof').style.display =(t==='moe')?'':'none';
}
function saveEdit(){
  const l=layers.find(x=>x.id===editId);if(!l)return;
  l.name  =document.getElementById('lN').value.trim()||l.name;
  l.type  =document.getElementById('lT').value;
  l.repeat=Math.max(1,parseInt(document.getElementById('lR').value)||1);
  if(l.type==='attention'){l.B=parseInt(document.getElementById('aB').value)||1;l.S=parseInt(document.getElementById('aS').value)||1;l.H=parseInt(document.getElementById('aH').value)||32;l.D=parseInt(document.getElementById('aD').value)||128;l.kvCache=parseInt(document.getElementById('aKV').value)||0;l.fa2=document.getElementById('aFA2').checked;}
  else if(l.type==='linear_attn'){l.M=parseInt(document.getElementById('laM').value)||2048;l.K=parseInt(document.getElementById('laK').value)||5120;l.nQK=parseInt(document.getElementById('laNQK').value)||16;l.nV=parseInt(document.getElementById('laNV').value)||48;l.D=parseInt(document.getElementById('laD').value)||128;}
  else if(l.type==='moe'){l.M=parseInt(document.getElementById('moM').value)||2048;l.hid=parseInt(document.getElementById('moH').value)||4096;l.ffnDim=parseInt(document.getElementById('moF').value)||1536;l.topK=parseInt(document.getElementById('moK').value)||8;l.nExperts=parseInt(document.getElementById('moE').value)||128;l.nShared=parseInt(document.getElementById('moS').value)||1;}
  else{l.M=parseInt(document.getElementById('lM').value)||l.M;l.K=parseInt(document.getElementById('lK').value)||l.K;l.N=parseInt(document.getElementById('lNv').value)||l.N;}
  closeModal();renderLayerList();renderTab();
}

function openModal(t){
  editId=null;document.getElementById('ov').style.display='flex';const mb=document.getElementById('mb');mb.className='modal';
  if(t==='chip'){
    mb.innerHTML=`<h3>Add chip</h3>
      <div class="f"><label>Name</label><input id="cN" placeholder="e.g. A100 PCIe"/></div>
      <div class="f"><label>Memory bandwidth (GB/s)</label><input id="cBW" type="number" placeholder="900"/></div>
      <div class="f"><label>Compute TFLOPS (FP16)</label><input id="cTF" type="number" placeholder="312"/></div>
      <div class="f"><label>Compute TFLOPS (FP8)</label><input id="cTF8" type="number" placeholder="624"/></div>
      <div class="f"><label>Memory capacity (GB)</label><input id="cMEM" type="number" placeholder="40"/></div>
      <div class="f"><label>Interconnect per card — unidirectional (GB/s)</label><input id="cIBW" type="number" placeholder="224"/><div style="font-size:11px;color:#aaa;margin-top:2px">Half of spec-sheet bidirectional (NVLink4=900 bi → 450 uni; PCIe 4.0 x16≈16)</div></div>
      <div class="mbtns"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn p" onclick="addChip()">Add</button></div>`;
  } else {
    mb.innerHTML=`<h3>Add layer</h3>
      <div class="f"><label>Name</label><input id="lN" placeholder="e.g. FFN layer"/></div>
      <div class="f"><label>Type</label><select id="lT" onchange="onTC(this)">
        <option value="linear">Linear (GEMM)</option>
        <option value="embedding">Embedding</option>
        <option value="attention">Attention (full)</option>
        <option value="linear_attn">GatedDeltaNet (linear attn)</option>
        <option value="moe">MoE FFN</option>
      </select></div>
      <div id="lf"><div class="f"><label>M</label><input id="lM" type="number" placeholder="2048"/></div><div class="f"><label>K</label><input id="lK" type="number" placeholder="4096"/></div><div class="f"><label>N</label><input id="lNv" type="number" placeholder="16384"/></div></div>
      <div id="af" style="display:none">${aF(null)}</div>
      <div id="laf" style="display:none">${laF(null)}</div>
      <div id="mof" style="display:none">${moeF(null)}</div>
      <div class="f" style="margin-top:4px"><label>Repeat count (×)</label><input id="lR" type="number" min="1" value="1"/><div style="font-size:11px;color:#aaa;margin-top:2px">Number of times this layer appears</div></div>
      <div class="mbtns"><button class="btn" onclick="closeModal()">Cancel</button><button class="btn p" onclick="addLayer()">Add</button></div>`;
  }
}
function closeModal(){
  document.getElementById('ov').style.display='none';
  editId=null;
  // Reset import state so a cancelled import can't affect the next one
  window._replaceOnImport = true;
  window._selectedIndices = null;
}
function addChip(){
  const n=document.getElementById('cN').value.trim(),b=parseFloat(document.getElementById('cBW').value),t=parseFloat(document.getElementById('cTF').value),m=parseFloat(document.getElementById('cMEM').value);
  if(!n||isNaN(b)||isNaN(t)||isNaN(m))return;
  const t8=parseFloat(document.getElementById('cTF8').value)||(t*2);
  const ibwV=parseFloat(document.getElementById('cIBW').value)||224;
  chips.push({id:Date.now(),name:n,bw:b,tflops:t,tflops_fp8:t8,mem:m,ibw:ibwV});closeModal();renderChipList();
}
function addLayer(){
  const n=document.getElementById('lN').value.trim();if(!n)return;
  const t=document.getElementById('lT').value;
  const r=Math.max(1,parseInt(document.getElementById('lR').value)||1);
  let l={id:Date.now(),name:n,type:t,repeat:r};
  if(t==='attention'){l.B=parseInt(document.getElementById('aB').value)||1;l.S=parseInt(document.getElementById('aS').value)||1;l.H=parseInt(document.getElementById('aH').value)||32;l.D=parseInt(document.getElementById('aD').value)||128;l.kvCache=parseInt(document.getElementById('aKV').value)||0;l.fa2=document.getElementById('aFA2').checked;}
  else if(t==='linear_attn'){l.M=parseInt(document.getElementById('laM').value)||2048;l.K=parseInt(document.getElementById('laK').value)||5120;l.nQK=parseInt(document.getElementById('laNQK').value)||16;l.nV=parseInt(document.getElementById('laNV').value)||48;l.D=parseInt(document.getElementById('laD').value)||128;}
  else if(t==='moe'){l.M=parseInt(document.getElementById('moM').value)||2048;l.hid=parseInt(document.getElementById('moH').value)||4096;l.ffnDim=parseInt(document.getElementById('moF').value)||1536;l.topK=parseInt(document.getElementById('moK').value)||8;l.nExperts=parseInt(document.getElementById('moE').value)||128;l.nShared=parseInt(document.getElementById('moS').value)||1;}
  else{l.M=parseInt(document.getElementById('lM').value)||2048;l.K=parseInt(document.getElementById('lK').value)||4096;l.N=parseInt(document.getElementById('lNv').value)||4096;}
  layers.push(l);closeModal();renderLayerList();renderTab();
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

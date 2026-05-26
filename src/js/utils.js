function rlim(ai,c,lt) { const tf=effectiveTflops(c,lt||'linear'); return Math.min(tf*1e12, ai*c.bw*1e9); }
function fmt(n){if(n>=1e12)return(n/1e12).toFixed(1)+'T';if(n>=1e9)return(n/1e9).toFixed(1)+'G';if(n>=1e6)return(n/1e6).toFixed(1)+'M';return Math.round(n);}
function ft(s){if(!s||s<0)return'0';if(s<1e-9)return(s*1e12).toFixed(1)+'ps';if(s<1e-6)return(s*1e9).toFixed(1)+'ns';if(s<1e-3)return(s*1e6).toFixed(1)+'µs';if(s<1)return(s*1e3).toFixed(2)+'ms';return s.toFixed(3)+'s';}
function togFA2(id){const l=layers.find(x=>x.id===id);if(l&&l.type==='attention'){l.fa2=!l.fa2;renderLayerList();renderTab();}}

// ── Loader ────────────────────────────────────────────────────────────────────
function setLTab(btn,t){document.querySelectorAll('.ltb').forEach(x=>x.classList.remove('active'));btn.classList.add('active');document.getElementById('hfP').style.display=t==='hf'?'':'none';document.getElementById('fileP').style.display=t==='file'?'':'none';}
// HF endpoint + cache helpers
function saveHfEp(){try{localStorage.setItem('hfEp',document.getElementById('hfEp').value);}catch(e){}}
function loadHfEp(){try{const v=localStorage.getItem('hfEp');if(v){const s=document.getElementById('hfEp');if(s)s.value=v;}}catch(e){}}
function hfCacheKey(ep,id){return `hfcfg:${ep}:${id}`;}
function hfCacheGet(ep,id){
  try{
    const raw=localStorage.getItem(hfCacheKey(ep,id));if(!raw)return null;
    const o=JSON.parse(raw);
    // expire after 7 days
    if(Date.now()-o.ts>7*24*3600*1000)return null;
    return o;
  }catch(e){return null;}
}
function hfCacheSet(ep,id,cfgJson){
  try{localStorage.setItem(hfCacheKey(ep,id),JSON.stringify({ts:Date.now(),cfg:cfgJson}));}
  catch(e){/* quota full — silently ignore */}
}
function clearHfCache(){
  let n=0;
  try{
    const keys=[];
    for(let i=0;i<localStorage.length;i++){const k=localStorage.key(i);if(k&&k.startsWith('hfcfg:'))keys.push(k);}
    keys.forEach(k=>{localStorage.removeItem(k);n++;});
  }catch(e){}
  const st=document.getElementById('hfSt');
  if(st){st.className='lst ok';st.textContent=`✓ Cleared ${n} cached config(s)`;}
}

async function loadHF(){
  const id=document.getElementById('hfI').value.trim();if(!id)return;
  const st=document.getElementById('hfSt');
  const ep=(document.getElementById('hfEp')||{}).value||'https://huggingface.co';
  // 1) Cache hit — instant return
  const cached=hfCacheGet(ep,id);
  if(cached){
    st.className='lst ok';
    st.innerHTML=`✓ Cached (${new Date(cached.ts).toLocaleDateString()}) · <a href="#" onclick="event.preventDefault();refetchHF()" style="color:#185FA5">refetch</a>`;
    openImportPreview(cached.cfg,id);
    return;
  }
  await fetchHF(ep,id,st,false);
}

async function refetchHF(){
  const id=document.getElementById('hfI').value.trim();if(!id)return;
  const st=document.getElementById('hfSt');
  const ep=(document.getElementById('hfEp')||{}).value||'https://huggingface.co';
  await fetchHF(ep,id,st,true);
}

async function fetchHF(ep,id,st,force){
  st.className='lst ld';st.textContent='Fetching…';
  const t0=performance.now();
  try{
    // Use raw/main/ (not resolve/main/) to avoid LFS redirect hop.
    // config.json is never LFS so raw/main serves it directly and fast.
    const url=`${ep}/${id}/raw/main/config.json`;
    const r=await fetch(url,{cache:force?'reload':'default'});
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const c=await r.json();
    const dt=((performance.now()-t0)/1000).toFixed(2);
    hfCacheSet(ep,id,c);
    st.className='lst ok';
    st.textContent=`✓ Loaded in ${dt}s · cached`;
    openImportPreview(c,id);
  }
  catch(e){
    const dt=((performance.now()-t0)/1000).toFixed(2);
    st.className='lst err';st.textContent=`✗ ${e.message} (${dt}s)`;
  }
}
function handleDrop(e){e.preventDefault();document.getElementById('dz').classList.remove('drag');if(e.dataTransfer.files[0])parseFile(e.dataTransfer.files[0]);}
function handleFI(inp){if(inp.files[0])parseFile(inp.files[0]);}
async function parseFile(file){
  const st=document.getElementById('fileSt');st.className='lst ld';st.textContent='Parsing…';
  try{
    const n=file.name.toLowerCase();
    if(n.endsWith('.json')){const c=JSON.parse(await file.text());st.className='lst ok';st.textContent='✓ Parsed';openImportPreview(c,file.name);}
    else if(n.endsWith('.safetensors')){
      const hb=await file.slice(0,8).arrayBuffer();const hl=Number(new DataView(hb).getBigUint64(0,true));
      const meta=JSON.parse(new TextDecoder().decode(await file.slice(8,8+hl).arrayBuffer()));
      st.className='lst ok';st.textContent=`✓ ${Object.keys(meta).filter(k=>k!=='__metadata__').length} tensors`;openSFPreview(meta,file.name);
    }else{st.className='lst err';st.textContent='Use .safetensors or config.json';}
  }catch(e){st.className='lst err';st.textContent='✗ '+e.message;}
}


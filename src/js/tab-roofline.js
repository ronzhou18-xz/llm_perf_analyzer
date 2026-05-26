function rRoof(){
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}
  const ch=selChip, gemmTF=effectiveTflops(ch,'linear'), ri=gemmTF*1e12/(ch.bw*1e9);
  const ld=layers.map(l=>{const ti=tpInfo(l,ch);return{...l,ai:ti.aiPerCard};});
  const mx=Math.max(...ld.map(l=>l.ai),ri*3);
  const aR=Array.from({length:200},(_,i)=>Math.exp(Math.log(0.1)+(i/199)*(Math.log(mx*1.5)-Math.log(0.1))));
  const rV=aR.map(a=>Math.min(gemmTF*1e12,a*ch.bw*1e9)/1e12);
  const isMoE = l => !!(l.moeGroup || l.type==='moe');
  const mB=ld.filter(l=>l.type!=='attention'&&l.type!=='linear_attn'&&!isMoE(l)&&l.ai<ri);
  const cB=ld.filter(l=>l.type!=='attention'&&l.type!=='linear_attn'&&!isMoE(l)&&l.ai>=ri);
  const aS=ld.filter(l=>l.type==='attention'&&!l.fa2);
  const aF2=ld.filter(l=>l.type==='attention'&&l.fa2);
  const linA=ld.filter(l=>l.type==='linear_attn');
  const moe=ld.filter(l=>isMoE(l));
  const pLabel = cfg.precision!=='fp16' ? ` · ${precisionLabel()}` : '';
  document.getElementById('mainContent').innerHTML=modeB()+tpB()+fa2B()+
    mcs(mc('Peak/card',gemmTF,'TFLOPS')+mc('BW/card',ch.bw,'GB/s')+mc('Ridge',ri.toFixed(1),'FLOP/byte')+mc('Mode',mode.toUpperCase()+pLabel,'',mode==='decode'?'mc ok':'mc hi'))+
    `<div class="cw"><div class="ct">Roofline — ${ch.name}${pLabel} (single-layer AI, not multiplied by repeat)</div><div style="position:relative;width:100%;height:300px"><canvas id="rc"></canvas></div></div>`;
  setTimeout(()=>{const c=document.getElementById('rc');if(!c)return;
    charts.r=new Chart(c,{type:'scatter',data:{datasets:[
      {data:aR.map((x,i)=>({x,y:rV[i]})),type:'line',borderColor:'#aaa',borderWidth:1.5,pointRadius:0,fill:false},
      {data:mB.map(l=>({x:l.ai,y:rlim(l.ai,ch,l.type)/1e12,name:layerLabel(l)})),backgroundColor:'#1D9E75',pointRadius:7,pointHoverRadius:9},
      {data:cB.map(l=>({x:l.ai,y:rlim(l.ai,ch,l.type)/1e12,name:layerLabel(l)})),backgroundColor:'#185FA5',pointRadius:7},
      {data:aS.map(l=>({x:l.ai,y:rlim(l.ai,ch,l.type)/1e12,name:layerLabel(l)})),backgroundColor:'#7F77DD',pointRadius:9,pointStyle:'triangle'},
      {data:aF2.map(l=>({x:l.ai,y:rlim(l.ai,ch,l.type)/1e12,name:layerLabel(l)})),backgroundColor:'#EF9F27',pointRadius:9,pointStyle:'triangle'},
      {data:linA.map(l=>({x:l.ai,y:rlim(l.ai,ch,l.type)/1e12,name:layerLabel(l)})),backgroundColor:'#0F6E56',pointRadius:8,pointStyle:'rectRot'},
      {data:moe.map(l=>({x:l.ai,y:rlim(l.ai,ch,l.type)/1e12,name:layerLabel(l)})),backgroundColor:'#BA7517',pointRadius:8,pointStyle:'rectRot'},
    ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:d=>{const p=d.raw;return p.name?`${p.name}: AI=${p.x.toFixed(1)}, ${p.y.toFixed(1)} TFLOPS`:null;}}}},scales:{x:{type:'logarithmic',title:{display:true,text:'Arithmetic Intensity (FLOP/byte)',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}},y:{type:'logarithmic',title:{display:true,text:'Performance (TFLOPS)',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}}}
    }});c.parentNode.parentNode.appendChild(lgEl([['#aaa','Roofline',false],['#1D9E75','Mem-bound',true],['#185FA5','Comp-bound',true],['#7F77DD','Attn std',true],['#EF9F27','Attn FA2',true],['#0F6E56','GatedDeltaNet',true],['#BA7517','MoE',true]]));
  },50);
}

// ── Bottleneck ────────────────────────────────────────────────────────────────
// v85: per-cell formula derivations for the Bottleneck table. Returns, for one
// layer, the symbolic formula + numeric substitution behind every column so the
// user can see (and audit) exactly how each number was produced. Each field is a
// small HTML snippet rendered UNDER the headline result in its cell.
function bnDerive(l, ch, tiOne, base){
  const wb=wBpe(), ab=aBpe(), kvb=kvBpe();
  const eTF=effectiveTflops(ch,l.type)*1e12, eTFt=(eTF/1e12);
  const D=(x)=>`<div style="font-size:9.5px;color:#999;margin-top:2px;line-height:1.5">${x}</div>`;
  // a compact "= number" tail
  const R=(v,unit)=>`<span style="color:#1a1a18">= ${fmt(v)}${unit||''}</span>`;
  let flExpr='', byExpr='', commExpr='';
  const M=l.M, K=l.K, N=l.N;
  // ── FLOPs + Bytes ────────────────────────────────────────────────────────
  if(l.type==='linear'){
    if(l.moeGroup==='gate_up'||l.moeGroup==='down'){
      const tk=l.moeTopK, nE=l.moeExperts, dist=expectedActivatedExperts(M,tk,nE);
      flExpr=D(`2·M·K·N·topK = 2·${M}·${fmt(K)}·${fmt(N)}·${tk} ${R(base.flops)}`);
      // activation term: gate_up reads M·K writes M·N·topK · down reads M·K·topK writes M·N
      const aSym=l.moeGroup==='gate_up'?`M·K + M·N·topK`:`M·K·topK + M·N`;
      const aNum=l.moeGroup==='gate_up'
        ? `${M}·${fmt(K)} + ${M}·${fmt(N)}·${tk}`
        : `${M}·${fmt(K)}·${tk} + ${M}·${fmt(N)}`;
      byExpr=D(`distinct·K·N·wb + (${aSym})·ab<br>= ${dist.toFixed(1)}·${fmt(K)}·${fmt(N)}·${wb} + (${aNum})·${ab} ${R(base.bytes)}`);
    } else if(l.moeGroup==='router'){
      flExpr=D(`2·M·K·N = 2·${M}·${fmt(K)}·${fmt(N)} ${R(base.flops)}`);
      byExpr=D(`M·K·ab + K·N·wb + M·N·ab<br>= ${M}·${fmt(K)}·${ab}+${fmt(K)}·${fmt(N)}·${wb}+${M}·${fmt(N)}·${ab} ${R(base.bytes)}`);
    } else {
      flExpr=D(`2·M·K·N = 2·${M}·${fmt(K)}·${fmt(N)} ${R(base.flops)}`);
      byExpr=D(`X<sub>in</sub>+W+Y<sub>out</sub> = M·K·ab + K·N·wb + M·N·ab<br>= ${M}·${fmt(K)}·${ab}+${fmt(K)}·${fmt(N)}·${wb}+${M}·${fmt(N)}·${ab} ${R(base.bytes)}`);
    }
  } else if(l.type==='embedding'){
    flExpr=D(`table gather — no matmul ${R(0)}`);
    byExpr=D(`M·N·(wb+ab) + M·4 = ${M}·${fmt(N)}·(${wb}+${ab})+${M}·4 ${R(base.bytes)}`);
  } else if(l.type==='attention'){
    const {Sq,Sk,kvHeadDim,kvStreams}=attnDims(l), nKV=l.nKV||l.H, B=l.B,H=l.H,Dh=l.D;
    const qoN=l.scoreOnly?1:2;   // Q only for indexer, Q+O for full attention
    if(l.scoreOnly){
      flExpr=D(`QK<sup>T</sup> only: 2·B·H·Sq·d<sub>kv</sub>·Sk<br>= 2·${B}·${H}·${Sq}·${kvHeadDim}·${Sk} ${R(base.flops)}`);
    } else {
      flExpr=D(`QK<sup>T</sup>+softmax+AV: 2·B·H·Sq·d<sub>kv</sub>·Sk + 5·B·H·Sq·Sk + 2·B·H·Sq·Sk·D<br>= 2·${B}·${H}·${Sq}·${kvHeadDim}·${Sk} + 5·${B}·${H}·${Sq}·${Sk} + 2·${B}·${H}·${Sq}·${Sk}·${Dh} ${R(base.flops)}`);
    }
    // bytes: (Q[+O]) + KV stream [+ score matrix if not FA2]
    const scoreSym=l.fa2?'':' + 2·B·H·Sq·Sk·ab';
    const scoreNum=l.fa2?'':` + 2·${B}·${H}·${Sq}·${Sk}·${ab}`;
    byExpr=D(`${l.scoreOnly?'Q':'(Q+O)'} + KV${l.fa2?'':' + score'}: ${qoN}·B·H·Sq·D·ab + kvStreams·B·nKV·Sk·d<sub>kv</sub>·kvb${scoreSym}<br>= ${qoN}·${B}·${H}·${Sq}·${Dh}·${ab} + ${kvStreams}·${B}·${nKV}·${Sk}·${kvHeadDim}·${kvb}${scoreNum}${l.fa2?' <span style="color:#bbb">[FA2 — score in SRAM]</span>':''} ${R(base.bytes)}`);
  } else if(l.type==='linear_attn'){
    const {nQK,nV,D:Dh}=l, Kk=l.K||(nQK+nV)*Dh;
    flExpr=D(`state + readout = 2·M·nV·D² + 2·M·nQK·D²<br>= 2·${M}·${nV}·${Dh}² + 2·${M}·${nQK}·${Dh}² ${R(base.flops)}`);
    byExpr=D(`(M·K + M·nV·D)·ab = (${M}·${fmt(Kk)} + ${M}·${nV}·${Dh})·${ab} ${R(base.bytes)}`);
  } else if(l.type==='moe'){
    const {hid,ffnDim,topK,nExperts,nShared=0}=l, act=topK+nShared;
    const distR=expectedActivatedExperts(M,topK,nExperts), distT=distR+nShared;
    flExpr=D(`router + gate·up + down: 2·M·hid·E + 6·M·hid·ffn·(topK+sh)<br>= 2·${M}·${fmt(hid)}·${nExperts} + 6·${M}·${fmt(hid)}·${fmt(ffnDim)}·${act} ${R(base.flops)}`);
    // bytes: distinct-expert weight stream + activation r/w  (see calcMoE)
    byExpr=D(`(hid·E + distinct·3·hid·ffn)·wb + (2·M·hid + 3·M·ffn·act)·ab<br>distinct≈${distT.toFixed(1)}: (${fmt(hid)}·${nExperts}+${distT.toFixed(1)}·3·${fmt(hid)}·${fmt(ffnDim)})·${wb} + (2·${M}·${fmt(hid)}+3·${M}·${fmt(ffnDim)}·${act})·${ab} ${R(base.bytes)}`);
  } else {
    flExpr=D(`${R(base.flops)}`); byExpr=D(`${R(base.bytes)}`);
  }
  // ── Comm volume ──────────────────────────────────────────────────────────
  const cb=tiOne.commBytes||0;
  if(cb<=1e-9){
    commExpr = nCards>1 ? D('col-parallel / replicated → output already placed, no AllReduce') : '';
  } else if(l.type==='linear' && (l.moeGroup==='gate_up'||l.moeGroup==='down')){
    // MoE sub-layer comm — TP down AllReduce or EP All-to-All
    if(moePar==='tp'){
      commExpr=D(`AllReduce down: 2·(N−1)/N·M·N·ab = 2·(${nCards}−1)/${nCards}·${M}·${fmt(l.N)}·${ab} ${R(cb)}`);
    } else {
      const g=moePar==='hybrid'?epDeg:nCards, hd=l.moeGroup==='gate_up'?l.K:l.N;
      commExpr=D(`All-to-All: (M·dp/cards)·topK·hid·bytes·(g−1)/g<br>(${M}·${dp}/${nCards})·${l.moeTopK}·${fmt(hd)}·…·(${g}−1)/${g} ${R(cb)}`);
    }
  } else if(l.type==='linear'){
    commExpr=D(`AllReduce: 2·(tp−1)/tp·M·N·ab = 2·(${tpAttn}−1)/${tpAttn}·${M}·${fmt(N)}·${ab} ${R(cb)}`);
  } else if(l.type==='embedding'){
    commExpr=D(`AllReduce (vocab-parallel): 2·(tp−1)/tp·M·N·ab = 2·(${tpAttn}−1)/${tpAttn}·${M}·${fmt(N)}·${ab} ${R(cb)}`);
  } else if(l.type==='moe'){
    if(moePar==='tp'){
      commExpr=D(`AllReduce: 2·(N−1)/N·M<sub>eff</sub>·hid·ab, M<sub>eff</sub>=M·dp=${M*dp} ${R(cb)}`);
    } else {
      const g=moePar==='hybrid'?epDeg:nCards;
      commExpr=D(`All-to-All dispatch+combine: (M<sub>eff</sub>/cards)·topK·hid·bytes·(g−1)/g, M<sub>eff</sub>=${M*dp} ${R(cb)}`);
    }
  } else {
    commExpr=D(`${R(cb)}`);
  }
  // ── time derivations ─────────────────────────────────────────────────────
  const fpc=tiOne.flopsPerCard, bpc=tiOne.bytesPerCard;
  const flShard = (base.flops>0 && fpc>0 && fpc<base.flops*0.999) ? ` <span style="color:#aaa">(FLOPs ÷ ${(base.flops/fpc).toFixed(0)})</span>` : '';
  const byShard = (base.bytes>0 && bpc>0 && bpc<base.bytes*0.999) ? ` <span style="color:#aaa">(Bytes ÷ ${(base.bytes/bpc).toFixed(1)})</span>` : '';
  const ctExpr = fpc>0
    ? D(`FLOPs<sub>/card</sub> ÷ TFLOPS = ${fmt(fpc)} ÷ ${eTFt.toFixed(0)}T${flShard}`)
    : D(`no matmul (table gather) → 0`);
  const mtExpr = D(`Bytes<sub>/card</sub> ÷ HBM BW = ${fmt(bpc)} ÷ ${ch.bw}G${byShard}`);
  const commT=tiOne.commTime, hid=tiOne.hiddenComm;
  const commtExpr = commT>1e-12
    ? D(`comm ÷ link BW = ${fmt(cb)} ÷ ${ibw}G = ${ft(commT)}${hid>1e-12?`<br>exposed = ${ft(commT)} − ${ft(hid)} hidden`:''}`)
    : '';
  return {flExpr, byExpr, commExpr, ctExpr, mtExpr, commtExpr};
}

function rBot(){
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}
  const ch=selChip, gemmTF=effectiveTflops(ch,'linear'), ri=gemmTF*1e12/(ch.bw*1e9);
  const tot=layers.map(l=>tpInfoTotal(l,ch).effectiveTime).reduce((a,b)=>a+b,0);
  const multi=nCards>1;
  // v84: roll up the three time components across the whole model (per card,
  // ×repeat) so the headline cards can show where time is actually spent.
  const sumCT=layers.map(l=>tpInfoTotal(l,ch).computeTime).reduce((a,b)=>a+b,0);
  const sumMT=layers.map(l=>tpInfoTotal(l,ch).memTime).reduce((a,b)=>a+b,0);
  const sumComm=layers.map(l=>{const t=tpInfoTotal(l,ch);return Math.max(0,t.commTime-t.hiddenComm);}).reduce((a,b)=>a+b,0);
  const sumCommBytes=layers.map(l=>tpInfoTotal(l,ch).commBytes||0).reduce((a,b)=>a+b,0);
  const rows=layers.map(l=>{
    const base=calcL(l);const r=getRepeat(l);
    const eTF=effectiveTflops(ch,l.type)*1e12;
    const lRi=eTF/(ch.bw*1e9);
    const ia=l.type==='attention', ila=l.type==='linear_attn', imoe=!!(l.type==='moe'||l.moeGroup), f2=ia&&l.fa2, bnd=base.ai<lRi?'mem':'comp';
    const badgeCls=f2?'bf':ia?'ba':ila?'bla':imoe?'bmoe':bnd==='mem'?'bm':'bc';
    const badgeTxt=f2?'FA2':ia?'attn':ila?'δ-net':imoe?'MoE':bnd;
    const ti=tpInfoTotal(l,ch);
    const tiOne=tpInfo(l,ch);
    const der=bnDerive(l,ch,tiOne,base);
    let fa2c='';
    if(ia){const s=calcStdAttn(l),f=calcFA2Attn(l),sp=(Math.max(s.flops/eTF,s.bytes/(ch.bw*1e9)))/(Math.max(f.flops/eTF,f.bytes/(ch.bw*1e9)));fa2c=`<span style="font-size:10px;color:${f2?'#854F0B':'#aaa'}"> ${f2?`FA2 ${sp.toFixed(1)}×`:`(FA2: ${sp.toFixed(1)}×)`}</span>`;}
    // per-call time components
    const cT=tiOne.computeTime, mT=tiOne.memTime, commT=tiOne.commTime;
    const exC=Math.max(0,commT-tiOne.hiddenComm);
    const busy=Math.max(cT,mT);
    // dominant segment — comm can win once exposed comm exceeds the busy time
    const domTxt = exC>busy ? 'comm' : (cT>=mT ? 'compute' : 'memory');
    const domCol = exC>busy ? '#A06800' : (cT>=mT ? '#185FA5' : '#0F6E56');
    const sumT=(cT+mT+exC)||1;
    const tbar=`<div class="tbar" title="Compute (blue) · Memory (green) · exposed Comm (amber)"><div class="tbar-seg" style="flex:${cT/sumT*100};background:#185FA5"></div><div class="tbar-seg" style="flex:${mT/sumT*100};background:#1D9E75"></div><div class="tbar-seg" style="flex:${exC/sumT*100};background:#F0A32A"></div></div>`;
    // comm volume cell (per call → ×repeat)
    const ocb=tiOne.commBytes||0, tcb=ti.commBytes||0;
    const commVolCell = !multi ? '<td style="color:#bbb">— <span style="font-size:9.5px">TP=1</span></td>'
      : (ocb>1e-9
          ? `<td>${fmt(ocb)}<span style="font-size:10px;color:#aaa"> ·×${r}=${fmt(tcb)}</span>${der.commExpr}</td>`
          : `<td style="color:#bbb">—${der.commExpr}</td>`);
    const commTimeCell = !multi ? '<td style="color:#bbb">—</td>'
      : (commT>1e-12
          ? `<td>${ft(exC)}${tiOne.hiddenComm>1e-12?`<span style="font-size:10px;color:#aaa"> (+${ft(tiOne.hiddenComm)} hid)</span>`:''}${der.commtExpr}</td>`
          : '<td style="color:#bbb">—</td>');
    let sub='';
    if(ia&&base.sub){sub=Object.entries(base.sub).map(([k,v])=>{const sc=v.flops/eTF,sm=(v.bytes||0)/(ch.bw*1e9);return`<tr class="sr"><td style="padding-left:24px">${k}</td><td>${fmt(v.flops)}</td><td>${v.bytes>0?fmt(v.bytes):'SRAM'}</td>${multi?'<td></td>':'<td style="color:#bbb">—</td>'}<td>${v.bytes>0?(v.flops/v.bytes).toFixed(1):'—'}</td><td></td><td>${ft(sc)}</td><td>${v.bytes>0?ft(sm):'—'}</td>${multi?'<td></td>':'<td style="color:#bbb">—</td>'}<td>${ft(Math.max(sc,sm))} ·×${r}</td></tr>`;}).join('');}
    return`<tr style="border-top:2px solid #e5e3da">
      <td style="vertical-align:top">${l.name} <span class="bd ${badgeCls}">${badgeTxt}</span>${fa2c} <span style="font-size:10px;color:#aaa">×${r}</span></td>
      <td style="vertical-align:top">${fmt(base.flops)}${der.flExpr}</td>
      <td style="vertical-align:top">${fmt(base.bytes)}${der.byExpr}</td>
      ${commVolCell}
      <td style="vertical-align:top">${base.ai.toFixed(1)}</td>
      <td style="vertical-align:top"><span class="bd ${badgeCls}">${badgeTxt}</span></td>
      <td style="vertical-align:top">${ft(cT)}${der.ctExpr}</td>
      <td style="vertical-align:top">${ft(mT)}${der.mtExpr}</td>${commTimeCell}
      <td style="min-width:160px;vertical-align:top"><div style="font-size:11px;margin-bottom:2px"><strong style="color:${domCol}">${ft(tiOne.effectiveTime)}</strong>/call · ${ft(ti.effectiveTime)} total <span style="font-size:9px;color:#aaa">(${domTxt})</span></div>${tbar}<div style="font-size:9.5px;color:#999;margin-top:3px;line-height:1.5">max(compute,memory)+exposed comm<br>= max(${ft(cT)},${ft(mT)})${exC>1e-12?` + ${ft(exC)}`:''} ·×${r}</div></td>
    </tr>${sub}`;
  }).join('');
  // ── formula box (general reference — per-cell derivations are inline above) ──
  const formulaBox=`<div class="cw" style="background:#f9f9f8">
    <div class="ct">Formula reference — each table cell shows its own substituted derivation; this is the symbolic key</div>
    <div style="font-size:11.5px;color:#555;line-height:1.95">
      <strong>FLOPs/call</strong> — GEMM 2·M·K·N · attention 2·B·H·Sq·d<sub>kv</sub>·Sk + 5·B·H·Sq·Sk + 2·B·H·Sq·Sk·D · MoE 2·M·hid·E + 6·M·hid·ffn·(topK+sh) · δ-net 2·M·D²·(nQK+nV).<br>
      <strong>Bytes/call</strong> — HBM traffic: linear M·K·ab + K·N·wb + M·N·ab · attention (Q+O) + KV stream (+ score matrix if not FA2) · MoE distinct-expert weight stream + activation r/w. wb=${wBpe()}B aBpe=${aBpe()}B kvBpe=${kvBpe()}B.<br>
      <strong>Comm/call</strong> — row-parallel TP linear / vocab embedding → ring <strong>AllReduce</strong> 2·(tp−1)/tp·M·N·ab · MoE under EP → ring <strong>All-to-All</strong> (M<sub>eff</sub>/cards)·topK·hid·(dispatch+combine)·(g−1)/g, M<sub>eff</sub>=M·dp, g=EP groups · column-parallel / replicated → 0.<br>
      <strong>Compute time</strong> = FLOPs<sub>/card</sub> ÷ effective TFLOPS — here ${gemmTF.toFixed(0)} TFLOPS (${ch.name}${cfg.precision!=='fp16'?', FP8':''}). FLOPs<sub>/card</sub> = FLOPs ÷ TP degree (attn/dense ÷ ${tpAttn}, MoE ÷ ${nCards}).<br>
      <strong>Memory time</strong> = Bytes<sub>/card</sub> ÷ HBM bandwidth — here ${ch.bw} GB/s (${ch.name}).<br>
      <strong>Comm time</strong> = Comm bytes ÷ interconnect bandwidth — here ${ibw} GB/s unidirectional (${ch.name}). Exposed = Comm − hidden, hidden = min(Comm, busy)·${overlap.toFixed(2)} overlap.<br>
      <strong>Effective/call</strong> = max(Compute, Memory) + exposed Comm. <strong>Total</strong> = Effective/call × repeat count. The dominant term sets the bottleneck.
    </div></div>`;
  const headHi=multi?'mc hi':'mc';
  document.getElementById('mainContent').innerHTML=modeB()+tpB()+fa2B()+
    mcs(mc('Total time',ft(tot),'')
       +mc('Compute',ft(sumCT),'total/card','mc')
       +mc('Memory',ft(sumMT),'total/card','mc')
       +mc('Comm (exposed)',multi?ft(sumComm):'—',multi?`${fmt(sumCommBytes)} B total`:'TP=1','mc'+(multi&&sumComm>0?' wa':''))
       +mc('Cards',nCards,multi?`TP${tpAttn}${dp>1?'×DP'+dp:''}`:'× TP',headHi),'mg5')+
    `<div class="cw"><div class="ct">Bottleneck (${mode}) — each cell shows the result with its formula + substituted numbers underneath · bar: <span style="display:inline-block;width:10px;height:10px;background:#185FA5;vertical-align:middle;border-radius:2px"></span> compute  <span style="display:inline-block;width:10px;height:10px;background:#1D9E75;vertical-align:middle;border-radius:2px"></span> memory${multi?'  <span style="display:inline-block;width:10px;height:10px;background:#F0A32A;vertical-align:middle;border-radius:2px"></span> comm (AR/A2A)':''} · widest segment = dominant</div><div style="overflow-x:auto"><table class="lt"><thead><tr><th>Layer (×repeat)</th><th>FLOPs/call</th><th>Bytes/call</th><th>Comm/call</th><th>AI</th><th>Bound</th><th>Compute</th><th>Memory</th><th>Comm</th><th>Effective (call · total)</th></tr></thead><tbody>${rows}</tbody></table></div></div>`+
    formulaBox;
}

// ── Arithmetic ────────────────────────────────────────────────────────────────
function rAI(){
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}
  const ch=selChip, gemmTF=effectiveTflops(ch,'linear'), ri=gemmTF*1e12/(ch.bw*1e9);
  const ld=layers.map(l=>({...l,...(nCards>1?{ai:tpInfo(l,ch).aiPerCard}:calcL(l))}));
  document.getElementById('mainContent').innerHTML=modeB()+tpB()+
    `<div class="cw"><div class="ct">Arithmetic intensity — ridge: ${ri.toFixed(1)} FLOP/byte (${mode}, per single call)</div><div style="position:relative;width:100%;height:${Math.max(240,ld.length*40+80)}px"><canvas id="ac"></canvas></div></div>`;
  setTimeout(()=>{const c=document.getElementById('ac');if(!c)return;
    charts.a=new Chart(c,{type:'bar',data:{labels:ld.map(l=>layerLabel(l)),datasets:[{data:ld.map(l=>+l.ai.toFixed(2)),backgroundColor:ld.map(l=>layerAIColor(l,ri)),borderRadius:4}]},
      options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:d=>`${d.raw} FLOP/byte`}}},scales:{x:{title:{display:true,text:'FLOP/byte',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'},afterDataLimits:a=>{if(a.max<ri*1.2)a.max=ri*1.2;}},y:{ticks:{color:'#555'},grid:{display:false}}}}
    });c.parentNode.parentNode.appendChild(lgEl([['#1D9E75','Mem-bound',false],['#185FA5','Comp-bound',false],['#7F77DD','Attn std',false],['#EF9F27','Attn FA2',false],['#0F6E56','GatedDeltaNet',false],['#BA7517','MoE FFN',false]]));
  },50);
}

// ── Throughput ────────────────────────────────────────────────────────────────
function rThr(){
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}
  const ch=selChip;
  // Use total time (including repeat) for each layer
  const tis=layers.map(l=>tpInfoTotal(l,ch));
  const tot=tis.map(t=>t.effectiveTime).reduce((a,b)=>a+b,0);
  // Per-rank tokens processed per step:
  //   - prefill: batch sequences × seqLen tokens each
  //   - decode : batch sequences × 1 generated token each
  const tokPerRank = cfg.batch * (mode==='prefill' ? cfg.seqLen : 1);
  const clusterTok = tokPerRank * dp;                              // global batch across DP replicas
  const clusterThr = clusterTok / (tot || 1);                      // cluster tok/s
  const globalBatch = cfg.batch * dp;                              // = per-rank batch when dp=1 (pure TP)
  // v84: GPU utilisation — fix for DP. sum(calcLTotal.flops) is ONE replica's
  // per-step FLOPs (layers carry the per-rank batch). With dp>1 the cluster runs
  // `dp` replicas, so the work actually executed across all nCards cards in time
  // `tot` is sum(flops)×dp. The denominator (nCards × effTFLOPS × tot) is the
  // full-cluster capacity. Pre-v84 the numerator omitted ×dp → util understated
  // by exactly the DP factor (e.g. DP=8 reported 1/8 of the true utilisation).
  const clusterFlops = layers.map(l=>calcLTotal(l).flops).reduce((a,b)=>a+b,0) * dp;
  const clusterCap   = effectiveTflops(ch,'linear')*1e12 * (tot||1) * nCards;
  const util = Math.min(1, clusterFlops / clusterCap) * 100;
  const utilFormula = dp>1
    ? `Σ FLOPs × ${dp} DP ÷ (${nCards} cards × ${effectiveTflops(ch,'linear').toFixed(0)} TFLOPS × ${ft(tot)})`
    : `Σ FLOPs ÷ (${nCards} card${nCards>1?'s':''} × ${effectiveTflops(ch,'linear').toFixed(0)} TFLOPS × ${ft(tot)})`;
  const thrFormula = dp > 1
    ? `global batch ${globalBatch} (${cfg.batch} per-rank × ${dp} DP)${mode==='prefill'?` × seq ${cfg.seqLen}`:''} ÷ ${ft(tot)}`
    : `global batch ${globalBatch}${mode==='prefill'?` × seq ${cfg.seqLen}`:''} ÷ ${ft(tot)}`;
  const thrCell = mode==='prefill'
    ? mc('Throughput', Math.round(clusterThr).toLocaleString(), `tok/s · ${thrFormula}`)
    : mc('Next token', ft(tot), '/ token');
  document.getElementById('mainContent').innerHTML=modeB()+tpB()+fa2B()+
    mcs(mc('Total latency',ft(tot),'')+thrCell+mc('Tokens',clusterTok.toLocaleString(),dp>1?`global batch (×${dp} DP)`:'global batch / step')+mc('GPU util.',util.toFixed(1),`% · ${utilFormula}`))+
    `<div class="cw"><div class="ct">Time per layer type — ${ch.name} × ${nCards} card${nCards>1?'s':''}${dp>1?` (TP${tpAttn}·DP${dp})`:nCards>1?` (TP${nCards})`:''} (${mode}, per card, total incl. ×repeat)</div><div style="position:relative;width:100%;height:${Math.max(240,layers.length*40+80)}px"><canvas id="tc"></canvas></div></div>`;
  setTimeout(()=>{const c=document.getElementById('tc');if(!c)return;
    charts.t=new Chart(c,{type:'bar',
      data:{labels:layers.map(l=>layerLabel(l)),datasets:[
        {label:'Compute',data:tis.map(t=>+(t.computeTime*1e6).toFixed(3)),backgroundColor:layers.map(l=>layerComputeColor(l)),borderRadius:4,stack:'s'},
        {label:'Memory', data:tis.map(t=>+(t.memTime*1e6).toFixed(3)),  backgroundColor:layers.map(l=>layerMemColor(l)),borderRadius:4,stack:'s'},
        {label:'Comm (AR/A2A)',data:tis.map(t=>+(Math.max(0,t.commTime-t.hiddenComm)*1e6).toFixed(3)),backgroundColor:'#F0A32A',borderRadius:4,stack:'s'},
      ]},
      options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:d=>`${d.dataset.label}: ${d.raw} µs`}}},scales:{x:{stacked:true,title:{display:true,text:'Total time/card (µs, incl. repeat)',color:'#888'},grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#888'}},y:{stacked:true,ticks:{color:'#555'},grid:{display:false}}}}
    });c.parentNode.parentNode.appendChild(lgEl([['#185FA5','Compute (linear)',false],['#1D9E75','Memory (linear)',false],['#9FE1CB','Compute (δ-net)',false],['#0F6E56','Memory (δ-net)',false],['#FAC775','Compute (MoE)',false],['#BA7517','Memory (MoE)',false],['#AFA9EC','Compute (attn)',false],['#7F77DD','Memory (attn)',false],['#F0A32A','Comm (AR/A2A)',false]]));
  },50);
}

// ── TP Timeline ───────────────────────────────────────────────────────────────

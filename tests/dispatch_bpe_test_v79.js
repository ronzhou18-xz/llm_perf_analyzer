// v79: verify asymmetric dispatch/combine a2a byte width (FP8 dispatch).
const fs=require('fs'), vm=require('vm');
const FILE=process.argv[2]||'/home/claude/llm_perf_analyzer_v79.html';
const html=fs.readFileSync(FILE,'utf-8');
let code=''; for(const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) code+=m[1]+'\n';
code=code.replace(/^(\s*)let\s+/gm,'$1var ').replace(/^(\s*)const\s+/gm,'$1var ');
const sb={console,Math,Date,JSON,Number,Array,Object,String,parseInt,parseFloat,isNaN,
  setTimeout:()=>{},clearTimeout:()=>{},setInterval:()=>{},clearInterval:()=>{},
  document:{getElementById:()=>({value:'0',innerHTML:'',textContent:'',className:'',appendChild:()=>{},addEventListener:()=>{},querySelector:()=>null,querySelectorAll:()=>[],style:{},classList:{add:()=>{},remove:()=>{}},checked:false,files:[]}),
    createElement:()=>({appendChild:()=>{},addEventListener:()=>{},style:{},classList:{add:()=>{},remove:()=>{}},setAttribute:()=>{},textContent:'',innerHTML:'',value:'0'}),
    addEventListener:()=>{},querySelector:()=>null,querySelectorAll:()=>[]},
  window:{},localStorage:{getItem:()=>null,setItem:()=>{},removeItem:()=>{}},fetch:()=>Promise.reject(),
  Chart:function(){return{destroy:()=>{},update:()=>{}}},alert:()=>{},confirm:()=>true,prompt:()=>'',
  URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},Blob:function(){},FileReader:function(){return{readAsText:()=>{},readAsArrayBuffer:()=>{}}},
  navigator:{clipboard:{writeText:()=>Promise.resolve()}}};
sb.window=sb;sb.globalThis=sb;vm.createContext(sb);vm.runInContext(code,sb,{timeout:10000});

const hid=6144,topK=8,nE=256,nShared=1,moeFfn=2048;
sb.nCards=8;sb.dp=8;sb.tpAttn=1;sb.epDeg=8;sb.moePar='ep';sb.overlap=0;sb.ibw=450;
const chip=sb.selChip||{bw:3350,ibw:450};
let pass=0,fail=0;
const ok=(l,c,n)=>{c?pass++:fail++;console.log(`  ${c?'✓':'✗'} ${l}${n?'  '+n:''}`);};
const near=(a,b)=>Math.abs(a-b)/Math.abs(b)<1e-9;

function probe(prec,M){
  sb.cfg.precision=prec;
  const mkMoE=()=>({id:1,name:'MoE',type:'moe',M,hid,ffnDim:moeFfn,topK,nExperts:nE,nShared,repeat:75});
  const mkGU=()=>({id:2,name:'gu',type:'linear',moeGroup:'gate_up',M,K:hid,N:2*moeFfn,moeExperts:nE,moeTopK:topK,moeFfnDim:moeFfn,moeHid:hid,repeat:75});
  const mkDN=()=>({id:3,name:'dn',type:'linear',moeGroup:'down',M,K:moeFfn,N:hid,moeExperts:nE,moeTopK:topK,moeFfnDim:moeFfn,moeHid:hid,repeat:75});
  return {mono:sb.tpInfo(mkMoE(),chip),gu:sb.tpInfo(mkGU(),chip),dn:sb.tpInfo(mkDN(),chip)};
}
const base=(M)=>(M*sb.dp/sb.nCards)*topK*hid*(sb.nCards-1)/sb.nCards; // ×bpe gives a2a bytes

console.log('GLM-5.1 dp8/ep8 — dispatch/combine a2a byte width\n');
for(const M of [1,64]){
  console.log(`M=${M} (per-rank):`);
  const bf=probe('fp16',M), f8=probe('fp8',M), f8w=probe('fp8w',M);
  // BF16: dispatch=combine=2·base, mono=4·base
  ok(`  BF16 gate_up(dispatch)=2·base`, near(bf.gu.commBytes, 2*base(M)));
  ok(`  BF16 down(combine)   =2·base`, near(bf.dn.commBytes, 2*base(M)));
  ok(`  BF16 mono = gu+dn`,            near(bf.mono.commBytes, bf.gu.commBytes+bf.dn.commBytes));
  // W8A8: dispatch=1·base, combine=2·base, mono=3·base
  ok(`  W8A8 gate_up(dispatch)=1·base`, near(f8.gu.commBytes, 1*base(M)),
     `${(f8.gu.commBytes/1024).toFixed(1)} KiB`);
  ok(`  W8A8 down(combine)   =2·base`, near(f8.dn.commBytes, 2*base(M)),
     `${(f8.dn.commBytes/1024).toFixed(1)} KiB`);
  ok(`  W8A8 mono = gu+dn (=3·base)`,  near(f8.mono.commBytes, f8.gu.commBytes+f8.dn.commBytes) && near(f8.mono.commBytes,3*base(M)));
  // W8A16 keeps symmetric (activations 16-bit)
  ok(`  W8A16 dispatch stays 2·base`,  near(f8w.gu.commBytes, 2*base(M)));
  // ratio check
  const drop=1-f8.mono.commBytes/bf.mono.commBytes;
  ok(`  W8A8 a2a total = 75% of BF16`, Math.abs(drop-0.25)<1e-9, `drop=${(drop*100).toFixed(1)}%`);
  console.log(`     → ×75 layers/card: BF16 ${(bf.mono.commBytes*75/1048576).toFixed(2)} MiB`+
              ` | W8A8 ${(f8.mono.commBytes*75/1048576).toFixed(2)} MiB`+
              ` | time @450GB/s ${(bf.mono.commTime*75*1e6).toFixed(1)}→${(f8.mono.commTime*75*1e6).toFixed(1)} µs\n`);
}
console.log(`Summary: ${pass} pass, ${fail} fail`);
process.exit(fail?1:0);

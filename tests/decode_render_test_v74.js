// End-to-end runtime test: exercise rDecode() rendering path with a mock DOM
// to confirm the v74 changes (reordered sections, timing formulas, global batch)
// don't throw at runtime and that throughput uses global batch consistently.
const fs = require('fs');
const vm = require('vm');

const FILE = process.argv[2] || '/home/claude/llm_perf_analyzer_v74.html';
const html = fs.readFileSync(FILE, 'utf-8');
const scriptMatches = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
let allCode = '';
for (const m of scriptMatches) allCode += m[1] + '\n';
allCode = allCode.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');

// Stateful DOM mock that captures innerHTML writes
const elementStore = {};
function makeEl(id) {
  return {
    id, value: '1', textContent: '', _innerHTML: '',
    get innerHTML(){ return this._innerHTML; },
    set innerHTML(v){ this._innerHTML = v; },
    style: {}, classList: { add:()=>{}, remove:()=>{}, toggle:()=>{} },
    appendChild:()=>{}, addEventListener:()=>{}, querySelector:()=>null,
    querySelectorAll:()=>[], setAttribute:()=>{}, checked:false, files:[],
    parentNode: { parentNode: { appendChild:()=>{} } },
    getContext:()=>({}),
  };
}
function getEl(id){ if(!elementStore[id]) elementStore[id]=makeEl(id); return elementStore[id]; }

const sandbox = {
  console, Math, Date, JSON, Number, Array, Object, String, parseInt, parseFloat, isNaN,
  setTimeout:(fn)=>{ try{ fn(); }catch(e){} }, clearTimeout:()=>{},
  setInterval:()=>{}, clearInterval:()=>{},
  document: {
    getElementById:(id)=>getEl(id),
    createElement:()=>makeEl('new'),
    addEventListener:()=>{}, querySelector:()=>null, querySelectorAll:()=>[],
  },
  window:{}, localStorage:{ getItem:()=>null, setItem:()=>{}, removeItem:()=>{} },
  fetch:()=>Promise.reject(new Error('mock')),
  Chart:function(){ return { destroy:()=>{}, update:()=>{} }; },
  alert:()=>{}, confirm:()=>true, prompt:()=>'',
  URL:{ createObjectURL:()=>'blob:', revokeObjectURL:()=>{} }, Blob:function(){},
  FileReader:function(){ return {readAsText:()=>{}, readAsArrayBuffer:()=>{}}; },
  navigator:{ clipboard:{ writeText:()=>Promise.resolve() } },
};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
try { vm.runInContext(allCode, sandbox, { timeout: 10000 }); }
catch (e) { console.log('Load error:', e.message); process.exit(1); }

let nPass=0, nFail=0;
function expect(label, cond, detail){
  if(cond){ nPass++; console.log('  \u2713 '+label); }
  else { nFail++; console.log('  \u2717 '+label+'  ('+(detail||'')+')'); }
}

console.log('='.repeat(80));
console.log('rDecode() runtime + global-batch throughput consistency tests (v74)');
console.log('='.repeat(80));

// Stub render side-effects we don't need
sandbox.renderTab = ()=>{};
sandbox.renderLayerList = ()=>{};
sandbox.renderArchInfo = ()=>{};
sandbox.updateCfgInfo = ()=>{};

// ---- Build a representative dense model (Qwen3-32B-ish) ----
sandbox.cfg = { batch:1, seqLen:2048, outTokens:256, nH:64, H:8, D:128, nL:64,
                precision:'fp16', kvCacheFn:null };
sandbox.mode = 'decode';
sandbox.nCards = 1;

// Use the tool's own builder if available, else hand-build minimal layers
if (typeof sandbox.buildDenseLayers === 'function') {
  // not relying on exact name; fall through to manual build
}
sandbox.layers = [
  { id:1, name:'QKV proj', type:'linear', M:1, K:5120, N:7168, repeat:64 },
  { id:2, name:'Attention', type:'attention', B:1, S:1, H:64, nKV:8, D:128, kvCache:2048, fa2:true, repeat:64 },
  { id:3, name:'O proj', type:'linear', M:1, K:8192, N:5120, repeat:64 },
  { id:4, name:'Gate+Up', type:'linear', M:1, K:5120, N:51200, repeat:64 },
  { id:5, name:'Down', type:'linear', M:1, K:25600, N:5120, repeat:64 },
];

// ---- TEST A: dp=1 (pure TP), single card — rDecode must not throw ----
console.log('\n[A] dp=1, nCards=1 (pure TP) — rDecode renders without error');
sandbox.dp = 1; sandbox.tpAttn = 1; sandbox.nCards = 1;
let threwA = false, htmlA = '';
try { sandbox.rDecode(); htmlA = getEl('mainContent').innerHTML; }
catch(e){ threwA = true; console.log('    ERROR: '+e.message+'\n'+(e.stack||'').split('\n').slice(0,4).join('\n')); }
expect('rDecode() does not throw at dp=1', !threwA);
expect('mainContent populated', htmlA.length > 500, 'len='+htmlA.length);

// ---- TEST B: section ordering — step breakdown BEFORE timeline BEFORE moe sweet-spot ----
console.log('\n[B] Section order: Decode step breakdown -> Token generation timeline -> (MoE sweet-spot)');
const idxStep = htmlA.indexOf('Decode step breakdown');
const idxTimeline = htmlA.indexOf('Token generation timeline');
expect('"Decode step breakdown" present', idxStep > -1, 'idx='+idxStep);
expect('"Token generation timeline" present', idxTimeline > -1, 'idx='+idxTimeline);
expect('step breakdown comes BEFORE timeline', idxStep > -1 && idxTimeline > -1 && idxStep < idxTimeline,
  'step='+idxStep+' timeline='+idxTimeline);

// ---- TEST C: timing formula annotations present in step breakdown rows ----
console.log('\n[C] Timing formulas appended to decode step breakdown');
expect('formula header note present', /Each timing cell shows its derivation/.test(htmlA));
expect('per-call max() formula rendered', /max\(/.test(htmlA), 'no max( found');
expect('per-step "\u00d7 repeat" formula rendered', /\u00d7 64 =/.test(htmlA) || /\u00d7 1 =/.test(htmlA),
  'no repeat-multiply formula');
expect('"\u00d7 outTokens" formula rendered', new RegExp('\u00d7 '+sandbox.cfg.outTokens+' =').test(htmlA),
  'no outTokens-multiply formula');

// ---- TEST D: global batch wording at dp=1 ----
console.log('\n[D] dp=1 headline uses "global batch" terminology (= per-rank, pure TP)');
expect('headline mentions "global batch"', /global batch/i.test(htmlA));
expect('dp=1 labelled as pure TP', /pure TP/.test(htmlA), 'no "pure TP" label');

// ---- TEST E: throughput == global_batch / step_time, both dp=1 and dp>1 ----
console.log('\n[E] Throughput numeric consistency: clusterTok/s == globalBatch / decodeStepT');
// Recompute decodeStepT the same way rDecode does, for verification
function decodeStepTime(){
  const ch = sandbox.selChip;
  const prev = JSON.parse(JSON.stringify(sandbox.layers));
  sandbox.mode='decode'; sandbox.applyConfigToLayers();
  const t = sandbox.layers.map(l=>sandbox.tpInfoTotal(l,ch).effectiveTime).reduce((a,b)=>a+b,0);
  sandbox.layers.forEach((l,i)=>Object.assign(l,prev[i]));
  return t;
}
// dp=1 case
sandbox.dp=1; sandbox.tpAttn=1; sandbox.nCards=1;
sandbox.cfg.batch=8;
let stepT1 = decodeStepTime();
let expectedThr1 = (sandbox.cfg.batch * sandbox.dp) / stepT1;   // globalBatch / step
// Extract the headline throughput number from rendered HTML
sandbox.rDecode();
let h1 = getEl('mainContent').innerHTML;
let m1 = h1.match(/Total throughput<\/div><div class="big">([\d,]+) tok\/s/);
let renderedThr1 = m1 ? parseInt(m1[1].replace(/,/g,'')) : NaN;
expect('dp=1: rendered throughput == round(globalBatch/step)',
  !isNaN(renderedThr1) && Math.abs(renderedThr1 - Math.round(expectedThr1)) <= 1,
  'rendered='+renderedThr1+' expected='+Math.round(expectedThr1));

// dp=4 case
sandbox.dp=4; sandbox.tpAttn=1; sandbox.nCards=4;
sandbox.cfg.batch=8;   // per-rank
let stepT4 = decodeStepTime();
let expectedThr4 = (sandbox.cfg.batch * sandbox.dp) / stepT4;  // globalBatch=32 / step
sandbox.rDecode();
let h4 = getEl('mainContent').innerHTML;
let m4 = h4.match(/Total throughput<\/div><div class="big">([\d,]+) tok\/s/);
let renderedThr4 = m4 ? parseInt(m4[1].replace(/,/g,'')) : NaN;
expect('dp=4: rendered throughput == round(globalBatch/step) [globalBatch=32]',
  !isNaN(renderedThr4) && Math.abs(renderedThr4 - Math.round(expectedThr4)) <= 1,
  'rendered='+renderedThr4+' expected='+Math.round(expectedThr4));
expect('dp=4: throughput scales ~4x vs equivalent per-rank-only (sanity)',
  renderedThr4 > renderedThr1,  // more total work delivered
  'thr4='+renderedThr4+' thr1='+renderedThr1);

// ---- TEST F: dp=4 still renders all reordered sections ----
console.log('\n[F] dp=4 path also renders reordered sections without error');
const idxStep4 = h4.indexOf('Decode step breakdown');
const idxTimeline4 = h4.indexOf('Token generation timeline');
expect('dp=4: step breakdown before timeline',
  idxStep4 > -1 && idxTimeline4 > -1 && idxStep4 < idxTimeline4,
  'step='+idxStep4+' timeline='+idxTimeline4);

console.log('\n'+'='.repeat(80));
console.log('Summary: '+nPass+' pass, '+nFail+' fail (of '+(nPass+nFail)+' total)');
console.log('='.repeat(80));
process.exit(nFail ? 1 : 0);

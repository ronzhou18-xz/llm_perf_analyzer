function classifyArch() {
  const hasAttn     = layers.some(l => l.type==='attention');
  const hasLinAttn  = layers.some(l => l.type==='linear_attn');
  const hasMoE      = layers.some(l => l.moeGroup||l.type==='moe');
  const hasDenseFFN = layers.some(l =>
    l.type==='linear' && /ffn|mlp|gate|up|down/i.test(l.name) && !/deltanet|qkv|proj/i.test(l.name)
  );
  // MLA heuristic: attention layer where head_dim (D) is large AND there's a
  // layer named "MLA KV proj" or Q proj absorbed into q_lora — easiest signal
  // is whether any linear layer has "MLA" in name.
  const hasMLA = layers.some(l => /mla/i.test(l.name||''));
  // MiMo-V2 SWA/GA hybrid: both are type='attention', split by name
  const hasSWA = layers.some(l => l.type === 'attention' && /\bswa\b/i.test(l.name||''));
  const hasGA  = layers.some(l => l.type === 'attention' && /\bga\b/i.test(l.name||''));
  // DeepSeek-V4 family attention variants: HCA (Hybrid Core Attention),
  // CSA (Compressed Sparse Attention), DSA (Dense Sparse Attention),
  // SWA-only (Sliding-Window Attention without global). When two or more of
  // these (or in combination with SWA/GA) coexist we have a per-layer attn-
  // type fork — surface this in the banner instead of falling through to the
  // generic "Full attention (GQA/MHA)" label.
  const hasHCA     = layers.some(l => l.type === 'attention' && /\bhca\b/i.test(l.name||''));
  const hasCSA     = layers.some(l => l.type === 'attention' && /\bcsa\b/i.test(l.name||''));
  const hasDSA     = layers.some(l => l.type === 'attention' && /\bdsa\b/i.test(l.name||''));
  const hasSWAOnly = layers.some(l => l.type === 'attention' && /swa[-\s]?only/i.test(l.name||''));
  const v4Kinds = [
    hasHCA && 'HCA',
    hasCSA && 'CSA',
    hasDSA && 'DSA',
    hasSWAOnly && 'SWA-only',
  ].filter(Boolean);
  const isV4Hybrid = v4Kinds.length >= 2;

  const attnPart = hasLinAttn && hasAttn ? 'Hybrid (DeltaNet + full-attn)'
                 : isV4Hybrid             ? `Hybrid (${v4Kinds.join('/')})`
                 : hasSWA && hasGA        ? 'Hybrid (SWA + GA)'
                 : hasLinAttn             ? 'Linear attention only'
                 : hasMLA                 ? 'MLA (compressed KV)'
                 : hasAttn                ? 'Full attention (GQA/MHA)'
                 : '';
  const ffnPart  = hasMoE && hasDenseFFN ? 'MoE FFN (+ dense replace)'
                 : hasMoE                ? 'MoE FFN'
                 : hasDenseFFN           ? 'Dense FFN'
                 : '';
  if (!attnPart && !ffnPart) return '';
  if (attnPart && ffnPart)   return attnPart + ' · ' + ffnPart;
  return attnPart || ffnPart;
}

// ── Layer role classifier for structure rendering ─────────────────────────
// Splits layers into semantic groups in execution order:
//   pre:    embedding (before transformer block)
//   attn:   everything that participates in the attention/token-mixing sub-layer
//           including QKV projections, attention op itself, DeltaNet linear attn,
//           DeltaNet QK/V/O projections, and the final O proj
//   ffn:    FFN/MoE sub-layer and its projections (gate+up, down, moe)
//   post:   lm_head (after block)
function classifyLayerRole(l) {
  const n = (l.name||'').toLowerCase();
  if (l.type === 'embedding') return 'pre';
  if (n.includes('lm head'))  return 'post';
  if (l.type === 'attention' || l.type === 'linear_attn') return 'attn';
  // Linear layers: route by name. Recognise both legacy patterns ("Q proj",
  // "QKV", "O proj", "MLA Q proj"…) AND V4-style low-rank decompositions
  // ("Q a-proj", "Q b-proj", "O down-proj", "O up-proj"). The robust rule:
  // a name beginning with Q/K/V/O/QKV/KV followed by a separator (space,
  // hyphen, or underscore) is an attention sub-projection regardless of suffix.
  // This must be checked BEFORE the FFN regex below — otherwise "O down-proj"
  // would match "down" in the FFN regex and get mis-routed to ffn.
  if (/^(q|k|v|o|qkv|kv)[\s_-]/i.test(l.name||'')) return 'attn';
  if (/deltanet|qkv|q proj|k proj|v proj|o proj|attn.*proj|mla/i.test(l.name)) return 'attn';
  if (/ffn|mlp|moe|gate|up|down|expert/i.test(l.name)) return 'ffn';
  if (l.type === 'moe' || l.moeGroup) return 'ffn';
  // Unknown linear layer — default to ffn (so it doesn't disappear)
  return 'ffn';
}

function rStructure() {
  if(!layers.length){document.getElementById('mainContent').innerHTML='<div class="es">Add layers to begin.</div>';return;}

  const COL = {
    embedding:  {fill:'#E1F5EE', stroke:'#1D9E75', text:'#085041'},
    linear:     {fill:'#E6F1FB', stroke:'#185FA5', text:'#0c3d7a'},
    attention:  {fill:'#EEEDFE', stroke:'#7F77DD', text:'#26215C'},
    linear_attn:{fill:'#dcf5ec', stroke:'#0F6E56', text:'#085041'},
    moe:        {fill:'#FAEEDA', stroke:'#EF9F27', text:'#633806'},
    lmhead:     {fill:'#f0f0ee', stroke:'#999',    text:'#444'},
  };
  const colFor = l => l.name.toLowerCase().includes('lm head') ? COL.lmhead
                   : l.moeGroup ? COL.moe
                   : (COL[l.type]||COL.linear);
  const hasLinAttn = layers.some(l=>l.type==='linear_attn');
  const hasMoE     = layers.some(l=>l.moeGroup||l.type==='moe');

  // Unique marker id per render — prevents browser SVG <defs> id caching between redraws
  const MID = 'arr' + Date.now();

  // Classify layers by role.
  // v81: filter to getRepeat(l) > 0 — same predicate renderLayerList() uses for
  // its visibleLayers. Without this, DSA indexer layers (getRepeat → 0 when the
  // sparse-MLA toggle is off) stayed in the diagram: they were drawn as stale
  // boxes AND, because their l.repeat (78) differs from their getRepeat (0), the
  // fork detector saw two distinct repeat values in the attention section and
  // drew a spurious "other / other" branch diamond. Filtering here keeps the
  // architecture diagram consistent with the layer list and the timing/roofline
  // tabs (all of which already respect getRepeat).
  const roles = layers.filter(l => getRepeat(l) > 0).map(l => ({l, role: classifyLayerRole(l)}));
  const pre   = roles.filter(r => r.role==='pre').map(r=>r.l);
  const attn  = roles.filter(r => r.role==='attn').map(r=>r.l);
  const ffn   = roles.filter(r => r.role==='ffn').map(r=>r.l);
  const post  = roles.filter(r => r.role==='post').map(r=>r.l);

  // ── Hybrid serial detection (DeltaNet + FullAttn or SWA + GA interleaved) ──
  // Qwen3.5-style: 3×DeltaNet block → 1×FullAttn block, repeating.
  // MiMo-V2-style: 5×SWA block → 1×GA block, repeating.
  // These are SERIAL (sequential), not parallel alternatives. Detect and render
  // as ordered sub-blocks, NOT as a diamond fork.
  function isHybridSerial(grp) {
    const hasLin  = grp.some(l => l.type === 'linear_attn');
    const hasFull = grp.some(l => l.type === 'attention');
    if (hasLin && hasFull) return true;
    // MiMo-V2 SWA/GA: both are type='attention', differentiated by name
    const hasSWA = grp.some(l => l.type === 'attention' && /\bswa\b/i.test(l.name||''));
    const hasGA  = grp.some(l => l.type === 'attention' && /\bga\b/i.test(l.name||''));
    return hasSWA && hasGA;
  }

  // Build serial sub-block descriptors for hybrid attention + FFN.
  // Handles two hybrid patterns:
  //   1) DeltaNet (linear_attn) + FullAttn — Qwen3.5
  //   2) SWA + GA — MiMo-V2-Flash (both are type='attention', split by name)
  // Returns [{repeat, label, layers}, ...] in execution order.
  // When ffnLayers is provided, FFN layers are appended to each sub-block.
  function buildSerialGroups(grp, ffnLayers) {
    // Detect SWA/GA pattern (MiMo-V2)
    const hasSWA = grp.some(l => l.type === 'attention' && /\bswa\b/i.test(l.name||''));
    const hasGA  = grp.some(l => l.type === 'attention' && /\bga\b/i.test(l.name||''));
    const isSwaGa = hasSWA && hasGA;

    let blockA, blockB;  // blockA rendered first (majority), blockB second (minority)

    if (isSwaGa) {
      // MiMo-V2 SWA/GA split: both type='attention', differentiated by name
      const isSWAName = l => /\bswa\b/i.test(l.name||'');
      const isGAName  = l => /\bga\b/i.test(l.name||'');
      const swaLayers = grp.filter(l => isSWAName(l));
      const gaLayers  = grp.filter(l => isGAName(l));
      const swaRp = swaLayers.length ? getRepeat(swaLayers[0]) : 0;
      const gaRp  = gaLayers.length  ? getRepeat(gaLayers[0])  : 0;
      // Linear projections: classify by SWA/GA name prefix
      const linearProjs = grp.filter(l => l.type === 'linear');
      const swaProj = linearProjs.filter(isSWAName);
      const gaProj  = linearProjs.filter(isGAName);
      const isOProj = l => /o\s*proj/i.test(l.name||'');
      blockA = {
        attnLayers: swaLayers, projPre: swaProj.filter(l => !isOProj(l)), projPost: swaProj.filter(isOProj),
        rp: swaRp, typeLabel: 'SWA Block',
      };
      blockB = {
        attnLayers: gaLayers, projPre: gaProj.filter(l => !isOProj(l)), projPost: gaProj.filter(isOProj),
        rp: gaRp, typeLabel: 'GA Block',
      };
    } else {
      // DeltaNet + FullAttn split (Qwen3.5)
      const linLayers  = grp.filter(l => l.type === 'linear_attn');
      const fullLayers = grp.filter(l => l.type === 'attention');
      const linRp  = linLayers.length  ? getRepeat(linLayers[0])  : 0;
      const fullRp = fullLayers.length ? getRepeat(fullLayers[0]) : 0;
      const isOProj     = l => /o\s*proj/i.test(l.name||'');
      const isDeltaName = l => /deltanet|delta[_\s-]?net|gated[_\s-]?delta/i.test(l.name||'');
      const linearProjs = grp.filter(l => l.type === 'linear');
      const deltaProj   = linearProjs.filter(isDeltaName);
      const fullProj    = linearProjs.filter(l => !isDeltaName(l));
      blockA = {
        attnLayers: linLayers, projPre: deltaProj.filter(l => !isOProj(l)), projPost: deltaProj.filter(isOProj),
        rp: linRp, typeLabel: 'DeltaNet Block',
      };
      blockB = {
        attnLayers: fullLayers, projPre: fullProj.filter(l => !isOProj(l)), projPost: fullProj.filter(isOProj),
        rp: fullRp, typeLabel: 'Full Attn Block',
      };
    }

    function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }
    const groupCount = (blockA.rp > 0 && blockB.rp > 0) ? gcd(blockA.rp, blockB.rp) : Math.max(blockA.rp, blockB.rp);
    const aPerGroup = groupCount > 0 ? blockA.rp / groupCount : blockA.rp;
    const bPerGroup = groupCount > 0 ? blockB.rp / groupCount : blockB.rp;
    const totalRepeat = groupCount;

    // Classify FFN layers by repeat match
    const ffn = ffnLayers || [];
    const totalNL = blockA.rp + blockB.rp;
    let ffnForA = ffn.filter(l => { const r = getRepeat(l); return r === blockA.rp || r === totalNL; });
    let ffnForB = ffn.filter(l => { const r = getRepeat(l); return r === blockB.rp || r === totalNL; });
    if (ffn.length > 0 && ffnForA.length === 0) ffnForA = ffn;
    if (ffn.length > 0 && ffnForB.length === 0) ffnForB = ffn;

    const groups = [];
    if (blockA.attnLayers.length) {
      groups.push({repeat: aPerGroup, label: `×${aPerGroup}`, typeLabel: blockA.typeLabel, layers: [
        ...blockA.projPre.filter(l => getRepeat(l) === blockA.rp),
        ...blockA.attnLayers,
        ...blockA.projPost.filter(l => getRepeat(l) === blockA.rp),
      ], ffnLayers: ffnForA});
    }
    if (blockB.attnLayers.length) {
      groups.push({repeat: bPerGroup, label: `×${bPerGroup}`, typeLabel: blockB.typeLabel, layers: [
        ...blockB.projPre.filter(l => getRepeat(l) === blockB.rp),
        ...blockB.attnLayers,
        ...blockB.projPost.filter(l => getRepeat(l) === blockB.rp),
      ], ffnLayers: ffnForB});
    }
    return {serialGroups: groups, totalRepeat};
  }

  // Within each group, discover branches by repeat count.
  // Semantics: if ALL layers in the group share the same repeat count, there
  // are no branches — the whole group is a single trunk.
  // If it's a hybrid serial model (DeltaNet + FullAttn), render as sequential
  // sub-blocks rather than a fork diamond.
  // Otherwise fork into parallel columns (one per repeat value).
  function splitBranches(grp) {
    if (grp.length === 0) return {trunk:[], branches:[], totalRepeat:0, serialGroups:null};

    // ── Special handling for FFN group with MoE ──────────────────────────────
    // Routed + shared experts are INTERNAL to the MoE block (parallel internally),
    // but the whole MoE block is ONE branch alongside any dense-replace branch.
    // Result: at most 2 top-level branches — Dense (×k) vs MoE (×N-k).
    const hasRouted = grp.some(l => l.moeGroup === 'gate_up' || l.moeGroup === 'down' || l.moeGroup === 'router');
    const hasShared = grp.some(l => l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down');
    if (hasRouted || hasShared) {
      const moeLayers = grp.filter(l => l.moeGroup);   // all MoE sub-layers (routed + shared)
      const dense     = grp.filter(l => !l.moeGroup);  // dense FFN (first_k_dense_replace)
      const rp = getRepeat(moeLayers[0]);

      if (dense.length > 0 && getRepeat(dense[0]) !== rp) {
        // Two branches: Dense (×k) on the left, MoE block (×N-k) on the right
        const denseRp = getRepeat(dense[0]);
        return {trunk:[], branches:[
          {repeat: denseRp, layers: dense, isMoE: false},
          {repeat: rp,      layers: moeLayers, isMoE: true},
        ], totalRepeat: denseRp + rp, serialGroups:null};
      }
      // No dense-replace: all layers are MoE.
      if (hasShared) {
        // Has shared experts → need dual-column MoE wrapper (routed left, shared right)
        return {trunk:[], branches:[
          {repeat: rp, layers: moeLayers, isMoE: true},
        ], totalRepeat: rp, serialGroups:null};
      }
      // Single-column MoE (no shared expert, no dense branch): treat as plain trunk.
      // This renders identically to dense Attention/FFN sections — a dashed box
      // with a ×1 badge and vertically stacked layers, no special MoE wrapper.
      return {trunk: moeLayers, branches:[], totalRepeat: rp, serialGroups:null};
    }

    // ── Hybrid serial: DeltaNet + FullAttn → render as sequential sub-blocks ──
    if (isHybridSerial(grp)) {
      const {serialGroups, totalRepeat} = buildSerialGroups(grp);
      return {trunk:[], branches:[], totalRepeat, serialGroups};
    }

    // ── Standard repeat-based branching ────────────────────────────────────
    const byRepeat = new Map();
    grp.forEach(l => {
      const r = getRepeat(l);
      if (!byRepeat.has(r)) byRepeat.set(r, []);
      byRepeat.get(r).push(l);
    });
    const distinct = [...byRepeat.keys()];
    // All same repeat → single trunk, no branching
    if (distinct.length === 1) {
      return {trunk: grp, branches: [], totalRepeat: distinct[0], serialGroups:null};
    }
    // ── Trunk + conditional branches ────────────────────────────────────────
    // Pattern: ONE repeat value R equals the sum of all other repeats. Layers
    // with repeat=R run in EVERY transformer block (Q/KV/O projections), while
    // the others split that R into mutually-exclusive branches selected per
    // layer (e.g. V4: 43 layers, of which 21 use HCA + 20 use CSA + 2 use Dense
    // attention; total attention executions across the model = 43, matching the
    // 43-repeat trunk). This is distinct from the "first k dense → MoE" pattern
    // (handled above for FFN) where dense and MoE counts are sequential, not
    // mutually exclusive per layer.
    //
    // When detected, render preTrunk (layers preceding the first branch in the
    // original list order) above the diamond, branches with diamond fork in the
    // middle, and postTrunk (layers after the last branch) below — so V4 reads:
    //   Q a-proj → Q b-proj → KV proj → ◇ → {HCA | CSA | Dense} → O down-proj → O up-proj
    const sortedRpDesc = [...distinct].sort((a,b) => b-a);
    const dominantRp   = sortedRpDesc[0];
    const otherSum     = sortedRpDesc.slice(1).reduce((a,b) => a+b, 0);
    if (sortedRpDesc.length >= 2 && otherSum > 0 && dominantRp === otherSum) {
      const trunkLayers  = byRepeat.get(dominantRp);
      const branchLayers = grp.filter(l => getRepeat(l) !== dominantRp);
      const branches = sortedRpDesc.slice(1).map(rp => ({repeat: rp, layers: byRepeat.get(rp)}));
      // Split trunk by original position relative to the branches: layers that
      // appear before any branch layer go to preTrunk; remaining go to postTrunk.
      const firstBranchIdx = grp.findIndex(l => branchLayers.includes(l));
      const lastBranchIdx  = (() => { for (let i=grp.length-1; i>=0; i--) if (branchLayers.includes(grp[i])) return i; return -1; })();
      const preTrunk  = trunkLayers.filter(l => grp.indexOf(l) < firstBranchIdx);
      const postTrunk = trunkLayers.filter(l => grp.indexOf(l) > lastBranchIdx);
      return {trunk: preTrunk, branches, postTrunk, totalRepeat: dominantRp, serialGroups:null};
    }
    // Multiple distinct repeats → all branches, no trunk.
    const branches = [...byRepeat.entries()]
      .sort((a,b) => b[0]-a[0])
      .map(([repeat, lyrs]) => ({repeat, layers: lyrs}));
    const totalRepeat = distinct.reduce((a,b)=>a+b, 0);
    return {trunk: [], branches, totalRepeat, serialGroups:null};
  }

  let attnSplit = splitBranches(attn);
  let ffnSplit  = splitBranches(ffn);
  let ffnMerged = false;  // true when FFN layers are inside attn serial groups

  // For hybrid serial models (Qwen3.5: DeltaNet + FullAttn), merge FFN into
  // each attn sub-block so the diagram shows the correct data flow:
  //   DeltaNet Block = [QK proj → V proj → GatedDeltaNet → O proj → FFN]
  //   Full Attn Block = [QKV → Attention → O proj → FFN]
  // FFN layers (including MoE) are stored in each serial group's `ffnLayers`
  // and rendered with their own splitBranches inside the sub-block.
  if (attnSplit.serialGroups && ffn.length > 0) {
    const {serialGroups, totalRepeat} = buildSerialGroups(attn, ffn);
    attnSplit = {trunk:[], branches:[], totalRepeat, serialGroups};
    ffnMerged = true;
  }
  // Transformer block count for the outer ×N badge.
  // For hybrid serial models (Qwen3.5: 3×DeltaNet + 1×FullAttn repeated N times),
  // totalRepeat from buildSerialGroups is already the group count (N), which is what
  // we want for the outer badge. For non-hybrid models, attn/ffn totalRepeat equals
  // the layer count, so max() gives the right answer.
  const displayNL = attnSplit.serialGroups
    ? attnSplit.totalRepeat   // group count (e.g. 8 for Qwen3.5-9B)
    : Math.max(attnSplit.totalRepeat, ffnSplit.totalRepeat, 1);

  const BOX_H=38,BOX_R=8,GAP_V=10,ARROW=13;
  const PAD_TOP=30,BLOCK_PAD_H=14,BLOCK_PAD_TOP=26,BLOCK_PAD_BOT=5;
  const GAP_BR=14;
  // Dynamic width: each branch needs at least MIN_BR_W px for labels to fit.
  // SHARED_W is the width of the main (single-block) column; when forking into
  // N branches, we divide (SHARED_W - (N-1)*GAP_BR) / N per branch. Ensure this
  // per-branch width is at least MIN_BR_W.
  // MoE branches contain two inner columns — need more width per branch.
  const hasMoEBr = attnSplit.branches.some(g=>g.isMoE) || ffnSplit.branches.some(g=>g.isMoE);
  // Also check serial groups' embedded ffnLayers for MoE with shared experts
  const serialHasShared = attnSplit.serialGroups
    ? attnSplit.serialGroups.some(g => (g.ffnLayers||[]).some(l => l.moeGroup==='shared_up'))
    : false;
  const hasMoEShared = serialHasShared
    || (attnSplit.branches.some(g=>g.isMoE && g.layers.some(l=>l.moeGroup==='shared_up'))
     || ffnSplit.branches.some(g=>g.isMoE && g.layers.some(l=>l.moeGroup==='shared_up')));
  const MIN_BR_W = hasMoEShared ? 300 : (hasMoEBr ? 200 : 160);
  const maxBranches = Math.max(
    attnSplit.branches.length || 1,
    ffnSplit.branches.length  || 1,
    1
  );
  const SHARED_W = Math.max(260, maxBranches * MIN_BR_W + (maxBranches-1) * GAP_BR + 20);
  const SVG_W=SHARED_W+48,cx=SVG_W/2,PAD_X=(SVG_W-SHARED_W)/2;

  const stackH = n => n===0 ? 0 : (n-1)*(BOX_H+GAP_V+ARROW)+BOX_H;
  const DIAMOND_H=24,DIAMOND_W=148,FORK_GAP=8,LABEL_H=18,MERGE_PAD=14;

  // Height of one section (attn or ffn) including possible branching / serial sub-blocks
  function sectionH(split) {
    // Serial sub-blocks (hybrid: DeltaNet + FullAttn sequential)
    if (split.serialGroups) {
      const SUB_PAD_TOP = 22, SUB_PAD_BOT = 10, SUB_GAP = 10, LABEL_SPACE = 14;
      return split.serialGroups.reduce((acc, g, i) => {
        const attnLayersH = stackH(g.layers.length);
        // FFN part height — inline, no extra dashed-box wrapper
        let ffnPartH = 0;
        if (g.ffnLayers && g.ffnLayers.length > 0) {
          const fl = g.ffnLayers;
          const hasSharedFFN = fl.some(l => l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down');
          const hasRoutedFFN = fl.some(l => l.moeGroup === 'router' || l.moeGroup === 'gate_up' || l.moeGroup === 'down');
          if (hasRoutedFFN && hasSharedFFN) {
            // MoE dual-column wrapper height
            const MOE_PAD_TOP=18, MOE_PAD_BOT=10;
            const routed = fl.filter(l => l.moeGroup==='router'||l.moeGroup==='gate_up'||l.moeGroup==='down');
            const shared = fl.filter(l => l.moeGroup==='shared_up'||l.moeGroup==='shared_down');
            const rH = routed.length ? (routed.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            const sH = shared.length ? (shared.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            ffnPartH = MOE_PAD_TOP + Math.max(rH, sH) + MOE_PAD_BOT;
          } else {
            ffnPartH = stackH(fl.length);
          }
          ffnPartH += GAP_V + ARROW; // connector between attn and ffn
        }
        const h = LABEL_SPACE + SUB_PAD_TOP + attnLayersH + ffnPartH + SUB_PAD_BOT;
        return acc + h + (i < split.serialGroups.length-1 ? SUB_GAP + ARROW : 0);
      }, 0);
    }
    const trunkH = stackH(split.trunk.length);
    if (split.branches.length === 0) {
      // Trunk-only: wrapped in a dashed box with top/bottom padding
      if (split.trunk.length > 0) {
        const SEC_PAD_TOP = 14, SEC_PAD_BOT = 10;
        return SEC_PAD_TOP + trunkH + SEC_PAD_BOT;
      }
      return trunkH;
    }

    // MoE branch height: routed+shared inner columns inside wrapper
    const moeInnerH = (g) => {
      if (!g.isMoE) return stackH(g.layers.length);
      const INNER_PAD_TOP=22, INNER_PAD_BOT=12;
      const routed = g.layers.filter(l => l.moeGroup==='router'||l.moeGroup==='gate_up'||l.moeGroup==='down');
      const shared = g.layers.filter(l => l.moeGroup==='shared_up'||l.moeGroup==='shared_down');
      const rH = routed.length ? (routed.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
      if (shared.length === 0) {
        return INNER_PAD_TOP + rH + INNER_PAD_BOT;
      }
      const sH = shared.length ? (shared.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
      return INNER_PAD_TOP + Math.max(rH, sH) + INNER_PAD_BOT;
    };

    const isSingleMoE = split.branches.length === 1 && split.branches[0].isMoE;
    const tallestBr = Math.max(0, ...split.branches.map(g => moeInnerH(g)));
    const conn = split.trunk.length > 0 ? GAP_V + ARROW : 0;
    // postTrunk: layers after the branches (V4 O down/up). Render inside section wrapper.
    const postTrunkLen = (split.postTrunk || []).length;
    const postTrunkH = postTrunkLen > 0 ? stackH(postTrunkLen) + GAP_V + ARROW : 0;
    // Single-branch merge only needs a tiny margin (no horizontal merge line),
    // while multi-branch needs full MERGE_PAD for the horizontal collector line.
    const mergePad = split.branches.length <= 1 ? 4 : MERGE_PAD;
    // Section dashed-box wrapper padding (same values used in renderSection)
    const SEC_BR_PAD_TOP = 18, SEC_BR_PAD_BOT = 8;
    if (isSingleMoE) {
      // MoE wrapper provides its own padding — no extra section wrapper needed
      return trunkH + conn + LABEL_H + tallestBr + mergePad + postTrunkH;
    }
    return SEC_BR_PAD_TOP + trunkH + conn + DIAMOND_H + FORK_GAP + GAP_V + ARROW + LABEL_H + tallestBr + mergePad + postTrunkH + SEC_BR_PAD_BOT;
  }

  const attnH = sectionH(attnSplit);
  const ffnH  = sectionH(ffnSplit);
  // When FFN is merged into attn serial groups, skip FFN section in layout
  const effFfnH = ffnMerged ? 0 : ffnH;
  const sectionGap = 8;
  // Block contains both sections with a small gap between them, plus outer padding
  const blockInnerH = (attnH>0?attnH:0) + (attnH>0 && effFfnH>0 ? (GAP_V+ARROW+sectionGap) : 0) + (effFfnH>0?effFfnH:0);
  const blockH = BLOCK_PAD_TOP + blockInnerH + BLOCK_PAD_BOT;

  let totalH=PAD_TOP+4;
  pre.forEach(()=>{totalH+=BOX_H+ARROW+GAP_V;});
  if (attnH>0 || effFfnH>0) totalH += blockH + ARROW + GAP_V;
  post.forEach((l,i)=>{totalH+=BOX_H;if(i<post.length-1)totalH+=ARROW+GAP_V;});
  totalH+=PAD_TOP;
  const SVG_H=Math.max(totalH,200);

  let svg=`<svg viewBox="0 0 ${SVG_W} ${SVG_H}" xmlns="http://www.w3.org/2000/svg"
    style="width:100%;max-width:${Math.max(540, SVG_W+40)}px;display:block;margin:0 auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
    <defs><marker id="${MID}" markerWidth="8" markerHeight="8" refX="4" refY="3" orient="auto">
      <path d="M0,0 L0,6 L7,3 z" fill="#bbb"/></marker></defs>`;

  const arrowV=(x,y,len)=>
    `<line x1="${x}" y1="${y}" x2="${x}" y2="${y+len-6}" stroke="#bbb" stroke-width="1.5" marker-end="url(#${MID})"/>`;
  const seg=(x1,y1,x2,y2)=>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#bbb" stroke-width="1.5"/>`;
  // Strip the " (×N)" suffix from layer names — in the architecture diagram the
  // repeat count is shown on the section's ×N badge, so repeating it inside every
  // box adds clutter and can be confusing.
  const stripRepeat = s => s.replace(/\s*\(×\d+\)\s*$/, '');
  const drawBox=(bx,by,bw,bh,col,label,sublabel)=>{
    const bcx=bx+bw/2,ty=sublabel?by+bh*0.38:by+bh/2+4;
    return `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="${BOX_R}"
        fill="${col.fill}" stroke="${col.stroke}" stroke-width="1.3"/>
      <text x="${bcx}" y="${ty}" text-anchor="middle"
        font-size="11" font-weight="600" fill="${col.text}">${label}</text>`
      +(sublabel?`<text x="${bcx}" y="${by+bh*0.74}" text-anchor="middle"
        font-size="9" fill="${col.text}" opacity=".72">${sublabel}</text>`:'');
  };
  const diamond=(dx,dy,w,h,label)=>{
    const hw=w/2,hh=h/2;
    return `<polygon points="${dx},${dy-hh} ${dx+hw},${dy} ${dx},${dy+hh} ${dx-hw},${dy}"
        fill="#fff7e6" stroke="#EF9F27" stroke-width="1.5"/>
      <text x="${dx}" y="${dy+4}" text-anchor="middle"
        font-size="9.5" font-weight="700" fill="#854F0B">${label}</text>`;
  };

  const subOf = l =>
    l.type==='attention'?`H=${l.H} D=${l.D}${l.fa2?' FA2':''}`
    :l.type==='linear_attn'?`nK=${l.nQK||'?'} nV=${l.nV||'?'} D=${l.D}`
    :l.type==='moe'?`top${l.topK}/${l.nExperts} · ffn=${l.ffnDim}`
    :(l.moeGroup==='router')?`${l.K.toLocaleString()}→${l.N} experts`
    :(l.moeGroup==='gate_up'||l.moeGroup==='down')?`top${l.moeTopK}/${l.moeExperts} · ${l.K.toLocaleString()}→${l.N.toLocaleString()}`
    :(l.K!=null&&l.N!=null)?`${l.K.toLocaleString()}→${l.N.toLocaleString()}`
    :'';

  // Render a section (attn or ffn) — may include a branching fork if multiple
  // different repeat counts exist. Returns the Y coordinate at the bottom.
  function renderSection(split, startY, label) {
    if (split.trunk.length===0 && split.branches.length===0 && !split.serialGroups) return startY;
    let y = startY;
    // Section label — for serial groups each sub-block gets its own typeLabel instead
    if (!split.serialGroups) {
      svg += `<text x="${PAD_X+6}" y="${y-4}" font-size="9.5" font-weight="700" fill="#888" letter-spacing="0.5">${label}</text>`;
    }

    // ── Serial sub-blocks: DeltaNet → FullAttn sequential ──────────────────
    if (split.serialGroups) {
      const SUB_PAD_TOP=22, SUB_PAD_BOT=10, SUB_GAP=10, LABEL_SPACE=14;
      split.serialGroups.forEach((g, gi) => {
        const col0 = g.layers[0] ? colFor(g.layers[0]) : COL.linear;
        // FFN part height — inline (no extra dashed-box wrapper)
        let ffnPartH = 0;
        if (g.ffnLayers && g.ffnLayers.length > 0) {
          const fl = g.ffnLayers;
          const hasSharedFFN = fl.some(l => l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down');
          const hasRoutedFFN = fl.some(l => l.moeGroup === 'router' || l.moeGroup === 'gate_up' || l.moeGroup === 'down');
          if (hasRoutedFFN && hasSharedFFN) {
            const MOE_PAD_TOP=18, MOE_PAD_BOT=10;
            const routed = fl.filter(l => l.moeGroup==='router'||l.moeGroup==='gate_up'||l.moeGroup==='down');
            const shared = fl.filter(l => l.moeGroup==='shared_up'||l.moeGroup==='shared_down');
            const rH = routed.length ? (routed.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            const sH = shared.length ? (shared.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            ffnPartH = MOE_PAD_TOP + Math.max(rH, sH) + MOE_PAD_BOT;
          } else {
            ffnPartH = stackH(fl.length);
          }
          ffnPartH += GAP_V + ARROW;
        }
        const attnLayersH = stackH(g.layers.length);
        const subH  = SUB_PAD_TOP + attnLayersH + ffnPartH + SUB_PAD_BOT;
        // typeLabel above the sub-block
        if (g.typeLabel) {
          svg += `<text x="${PAD_X+6}" y="${y+LABEL_SPACE-4}" font-size="9.5" font-weight="700" fill="#888" letter-spacing="0.5">${g.typeLabel}</text>`;
        }
        y += LABEL_SPACE;
        // Dashed sub-block rect
        svg += `<rect x="${PAD_X+4}" y="${y}" width="${SHARED_W-8}" height="${subH}" rx="7"
          fill="${col0.fill}" fill-opacity=".35" stroke="${col0.stroke}" stroke-width="1.2" stroke-dasharray="4,2.5"/>`;
        // Repeat badge on top edge (centered) — per-block count
        const perBlockSG = g.repeat === displayNL ? 1 : g.repeat;
        const badgeW=36, badgeH=17, badgeX=cx-badgeW/2, badgeY=y-badgeH/2;
        svg += `<rect x="${badgeX}" y="${badgeY}" width="${badgeW}" height="${badgeH}" rx="${badgeH/2}"
          fill="${col0.stroke}"/>
          <text x="${cx}" y="${badgeY+12}" text-anchor="middle"
            font-size="9.5" font-weight="700" fill="#fff">×${perBlockSG}</text>`;
        let ly = y + SUB_PAD_TOP;
        // Render attn layers
        g.layers.forEach((l, li) => {
          svg += drawBox(PAD_X+10, ly, SHARED_W-20, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
          ly += BOX_H;
          if (li < g.layers.length-1) { svg += arrowV(cx, ly, GAP_V+ARROW); ly += GAP_V+ARROW; }
        });
        // Render FFN part inline — no extra dashed box, just the layers directly.
        // For MoE with shared experts: draw the MoE wrapper with dual columns.
        // For plain layers (dense FFN or MoE without shared): simple drawBox stack.
        if (g.ffnLayers && g.ffnLayers.length > 0) {
          const fl = g.ffnLayers;
          const hasSharedFFN = fl.some(l => l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down');
          const hasRoutedFFN = fl.some(l => l.moeGroup === 'router' || l.moeGroup === 'gate_up' || l.moeGroup === 'down');
          svg += arrowV(cx, ly, GAP_V+ARROW);
          ly += GAP_V+ARROW;

          if (hasRoutedFFN && hasSharedFFN) {
            // ── MoE dual-column: routed left, shared right ──────────────────
            const routed = fl.filter(l => l.moeGroup==='router'||l.moeGroup==='gate_up'||l.moeGroup==='down');
            const shared = fl.filter(l => l.moeGroup==='shared_up'||l.moeGroup==='shared_down');
            const MOE_PAD_H=8, MOE_PAD_TOP=18, MOE_PAD_BOT=10, MOE_GAP=8;
            const moeBoxX = PAD_X+10, moeBoxW = SHARED_W-20;
            const innerColW = Math.floor((moeBoxW - MOE_PAD_H*2 - MOE_GAP) / 2);
            const routedH = routed.length ? (routed.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            const sharedH = shared.length ? (shared.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            const innerH = Math.max(routedH, sharedH);
            const wrapH = MOE_PAD_TOP + innerH + MOE_PAD_BOT;
            // MoE wrapper rect
            svg += `<rect x="${moeBoxX}" y="${ly}" width="${moeBoxW}" height="${wrapH}" rx="7"
              fill="${COL.moe.fill}" fill-opacity=".5" stroke="${COL.moe.stroke}" stroke-width="1.2" stroke-dasharray="4,2.5"/>`;
            const rColX = moeBoxX + MOE_PAD_H;
            const sColX = moeBoxX + MOE_PAD_H + innerColW + MOE_GAP;
            const rColCX = rColX + innerColW/2;
            const sColCX = sColX + innerColW/2;
            const entryY = ly + 4;
            const firstY = ly + MOE_PAD_TOP;
            svg += seg(rColCX, entryY, sColCX, entryY);
            svg += arrowV(rColCX, entryY, firstY - entryY);
            svg += arrowV(sColCX, entryY, firstY - entryY);
            let rly = firstY;
            routed.forEach((l,li) => {
              svg += drawBox(rColX, rly, innerColW, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
              rly += BOX_H;
              if (li < routed.length-1) { svg += arrowV(rColCX, rly, GAP_V+ARROW); rly += GAP_V+ARROW; }
            });
            let sly = firstY;
            shared.forEach((l,li) => {
              svg += drawBox(sColX, sly, innerColW, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
              sly += BOX_H;
              if (li < shared.length-1) { svg += arrowV(sColCX, sly, GAP_V+ARROW); sly += GAP_V+ARROW; }
            });
            const exitY = ly + wrapH - 4;
            svg += seg(rColCX, rly, rColCX, exitY);
            svg += seg(sColCX, sly, sColCX, exitY);
            svg += seg(rColCX, exitY, sColCX, exitY);
            const moeCX = moeBoxX + moeBoxW/2;
            svg += seg(moeCX, exitY, moeCX, ly + wrapH);
            ly += wrapH;
          } else {
            // ── Plain stack: dense FFN or MoE without shared ────────────────
            fl.forEach((l, li) => {
              svg += drawBox(PAD_X+10, ly, SHARED_W-20, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
              ly += BOX_H;
              if (li < fl.length-1) { svg += arrowV(cx, ly, GAP_V+ARROW); ly += GAP_V+ARROW; }
            });
          }
        }
        y += subH;
        if (gi < split.serialGroups.length-1) {
          // Arrow between serial sub-blocks — tip above next sub-block's badge
          const subConnLen = SUB_GAP+ARROW;
          svg += arrowV(cx, y, Math.max(subConnLen - 10 + 6, 8));
          y += subConnLen;
        }
      });
      return y;
    }

    // ── Trunk only (no branches): wrap in a dashed box with ×N badge ─────────
    // This is the simple case: all layers have the same repeat count. We wrap
    // them in a dashed sub-block just like serial groups, so all sections look
    // visually consistent (label text + dashed box + badge).
    if (split.trunk.length > 0 && split.branches.length === 0) {
      const SEC_PAD_TOP = 14, SEC_PAD_BOT = 10;
      const col0 = colFor(split.trunk[0]);
      const contentH = stackH(split.trunk.length);
      const boxH = SEC_PAD_TOP + contentH + SEC_PAD_BOT;
      // Dashed section rect
      svg += `<rect x="${PAD_X+4}" y="${y}" width="${SHARED_W-8}" height="${boxH}" rx="7"
        fill="${col0.fill}" fill-opacity=".18" stroke="${col0.stroke}" stroke-width="1.2" stroke-dasharray="4,2.5"/>`;
      // ×N badge on top edge
      const rp = split.totalRepeat || getRepeat(split.trunk[0]);
      // Per-group count: divide by displayNL (group count for hybrid serial,
      // or total layers for non-hybrid). If rp equals displayNL → ×1.
      const perBlock = rp === displayNL ? 1
                     : (displayNL > 0 && rp > displayNL && rp % displayNL === 0)
                       ? rp / displayNL : rp;
      const badgeW=36, badgeH=17, badgeX=cx-badgeW/2, badgeY=y-badgeH/2;
      svg += `<rect x="${badgeX}" y="${badgeY}" width="${badgeW}" height="${badgeH}" rx="${badgeH/2}"
        fill="${col0.stroke}"/>
        <text x="${cx}" y="${badgeY+12}" text-anchor="middle"
          font-size="9.5" font-weight="700" fill="#fff">×${perBlock}</text>`;
      // No entry line inside the box — the badge IS the visual entry marker.
      let ly = y + SEC_PAD_TOP;
      split.trunk.forEach((l, li) => {
        svg += drawBox(PAD_X+10, ly, SHARED_W-20, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
        ly += BOX_H;
        if (li < split.trunk.length-1) { svg += arrowV(cx, ly, GAP_V+ARROW); ly += GAP_V+ARROW; }
      });
      y += boxH;
      return y;
    }

    // Trunk layers (before branches — shared across all block repetitions)
    // When branches follow, trunk layers will be inside the section dashed wrapper,
    // so we narrow them (PAD_X+10, SHARED_W-20) to fit inside the wrapper rect.
    const hasBranches = split.branches.length > 0;
    const trunkBoxX = hasBranches ? PAD_X+10 : PAD_X;
    const trunkBoxW = hasBranches ? SHARED_W-20 : SHARED_W;
    // Section-wrapper padding constants (also used inside the branches block below).
    // Hoisted here so we can apply SEC_BR_PAD_TOP padding ABOVE the first trunk
    // layer — giving the trunk visual breathing room from the wrapper top edge
    // (otherwise Q a-proj sits flush against the inner Attention dashed border).
    const SEC_BR_PAD_TOP = 18, SEC_BR_PAD_BOT = 8;
    // Capture wrapper-top y BEFORE we apply padding — the dashed wrapper rect
    // (drawn after branches) starts here, and the padding pushes trunk down.
    const sectionWrapperTop = hasBranches ? y : null;
    if (hasBranches && split.trunk.length > 0) {
      // For trunk+branches sections (V4-Flash: Q a-proj/Q b-proj/KV proj before
      // the diamond fork), give the wrapper top padding so the first trunk box
      // doesn't crowd against the wrapper border. The matching SEC_BR_PAD_TOP
      // gap that used to live between trunk and diamond is removed below — net
      // section height is unchanged, so sectionH() still matches.
      y += SEC_BR_PAD_TOP;
    }
    split.trunk.forEach((l,li)=>{
      svg += drawBox(trunkBoxX, y, trunkBoxW, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
      y += BOX_H;
      if (li < split.trunk.length-1 || split.branches.length>0) {
        svg += arrowV(cx, y, GAP_V+ARROW);
        y += GAP_V+ARROW;
      }
    });
    // Branches
    if (split.branches.length > 0) {
      // ── Section dashed-box wrapper ─────────────────────────────────────────
      // Record start y BEFORE trunk content so the wrapper covers everything.
      // Trunk layers were already drawn above (narrowed to fit); we retroactively
      // wrap the whole section in a dashed rect after branches finish.
      const N_BR = split.branches.length;
      const hasMoEBranch   = split.branches.some(g => g.isMoE);
      const isSingleMoE    = N_BR === 1 && hasMoEBranch;
      // Wrapper top:
      //  • With trunk: sectionWrapperTop is the y captured BEFORE we applied
      //    SEC_BR_PAD_TOP padding above trunk. Wrapper top sits there so the
      //    padding is INSIDE the wrapper, above the first trunk layer.
      //  • Without trunk: SEC_BR_PAD_TOP is applied below (between section
      //    start and the diamond) — wrapper top is still sectionWrapperTop.
      const sectionBoxStartY = sectionWrapperTop;
      // No-trunk case: still need padding between section start and diamond so
      // the diamond doesn't sit flush against the wrapper top edge. With trunk
      // the padding was already applied above the trunk, so skip it here.
      if (!isSingleMoE && split.trunk.length === 0) y += SEC_BR_PAD_TOP;

      // FFN-specific sort: dense FFN should render on the LEFT (executed first in
      // first_k_dense_replace pattern), MoE on the RIGHT. For other sections, fall
      // back to descending repeat order (larger branches first).
      const isDenseLike = g => g.layers.some(l => /dense/i.test(l.name));
      const isMoELike   = g => g.layers.some(l => l.type==='moe'||l.moeGroup);
      if (split === ffnSplit && split.branches.some(isDenseLike) && split.branches.some(isMoELike)) {
        split.branches.sort((a,b) => {
          const aDense = isDenseLike(a), bDense = isDenseLike(b);
          if (aDense && !bDense) return -1;
          if (!aDense && bDense) return 1;
          return b.repeat - a.repeat;
        });
      }
      // Branch widths and centers — fit inside the section wrapper (PAD_X+10 .. SHARED_W-20)
      const brInnerW = SHARED_W - 20;
      const brInnerX = PAD_X + 10;
      const BR_W = Math.floor((brInnerW - (N_BR-1)*GAP_BR) / N_BR);
      const brCX = split.branches.map((_,i) => brInnerX + i*(BR_W+GAP_BR) + BR_W/2);
      const dcy = y + DIAMOND_H/2;
      // Pick a clearer diamond label based on branch contents
      const isSharedBranch = g => g.layers.some(l => l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down');
      const isRoutedBranch = g => g.layers.some(l => l.moeGroup === 'router' || l.moeGroup === 'gate_up' || l.moeGroup === 'down');
      const brKinds = split.branches.map(g => {
        const first = g.layers[0];
        if (!first) return '?';
        if (isSharedBranch(g)) return 'shared';
        if (first.type === 'linear_attn' || /deltanet/i.test(first.name)) return 'linear';
        if (first.type === 'moe' || first.moeGroup || /moe/i.test(first.name)) return 'routed';
        // Attention type names — derived from the layer name so V4-style
        // HCA/CSA/SWA-only (and any future "<TYPE> Attention" labels) are
        // surfaced verbatim instead of being lumped into "full-attn"/"other".
        // Always returned UPPERCASE for case-consistent membership checks against
        // ATTN_KINDS below — failing this consistency in v55 caused the V4-Flash
        // 3-branch case (CSA/HCA/SWA-only) to lose its per-branch badges and
        // diamond label because the lowercase form didn't match the uppercase set.
        if (first.type === 'attention' || /\battention\b|\battn\b/i.test(first.name)) {
          const m = (first.name || '').match(/\b(HCA|CSA|DSA|MLA|SWA|GA|MQA|GQA|MHA|Dense|Full)\b/i);
          if (m) return m[1].toLowerCase() === 'full' ? 'FULL-ATTN' : m[1].toUpperCase();
          if (/dense/i.test(first.name)) return 'DENSE';
          return 'FULL-ATTN';
        }
        if (/dense/i.test(first.name)) return 'DENSE';
        return 'other';
      });
      const hasDenseBranch  = split.branches.some(g => !g.isMoE && g.layers.some(l => /dense/i.test(l.name||'')));
      // Detect "attention type fork" — multiple branches whose kinds are all
      // attention variants (HCA/CSA/SWA/etc). Used to pick a clearer
      // diamond label and to enable per-branch ×N badges (so every branch
      // shows its own layer count rather than relying on the diamond label).
      const ATTN_KINDS = new Set(['HCA','CSA','DSA','MLA','SWA','GA','MQA','GQA','MHA','DENSE','FULL-ATTN']);
      const isAttnTypeFork = split === attnSplit && brKinds.every(k => ATTN_KINDS.has(k));
      // Dense+MoE FFN fork (DeepSeek-V3/V3.2 first_k_dense_replace pattern):
      // first k layers run dense FFN, the remaining N-k layers run MoE. Same
      // "1-of-K per layer" semantics as the attn-type fork — each layer
      // dispatches to exactly one branch by index — so it gets the same
      // per-branch `N/L` badge treatment for visual consistency.
      const isFfnDenseMoEFork = split === ffnSplit && hasDenseBranch && hasMoEBranch;
      // Unified flag: any "1-of-K per layer" choice fork (sums to total layers,
      // not parallel). Drives per-branch `N/L` badge rendering, the extended
      // branch entry arrow that gives the badge its own vertical slot, and
      // the badge-above-box positioning.
      const isLayerChoiceFork = isAttnTypeFork || isFfnDenseMoEFork;

      let brTopY;
      if (isSingleMoE) {
        // No diamond — single MoE block. Skip the extra connector gap because
        // the section-transition arrow already provides visual separation.
        // brTopY sits right where the section starts — the ×N badge and wrapper
        // entry bridge handle the remaining space down to the first box.
        brTopY = y;
      } else {
        let diamondLabel;
        if (hasDenseBranch && hasMoEBranch) {
          const denseBranch = split.branches.find(g => !g.isMoE);
          const k = denseBranch ? denseBranch.repeat : 0;
          diamondLabel = `first ${k} dense → MoE`;
        } else if (isAttnTypeFork) {
          // Attention-type fork (V4-Flash 3-way HCA/CSA/SWA-only, V4-Pro 2-way
          // HCA/CSA, MiMo-V2 SWA/GA, etc). Each transformer layer dispatches
          // to exactly one branch based on its layer index — NOT a parallel
          // split. Diamond label says "1-of-K per layer" to convey choice
          // (rather than parallelism); per-branch badges below show layer-share
          // as a fraction `N/L` (rather than `×N`) to disambiguate from the
          // loop-count `×N` semantics used elsewhere on non-fork blocks.
          diamondLabel = `1-of-${N_BR} per layer`;
        } else {
          diamondLabel = brKinds.length >= 2
            ? brKinds.slice(0,2).join(' / ') + (brKinds.length>2?'/…':'')
            : (split===attnSplit ? 'attn type?' : 'FFN type?');
        }
        svg += diamond(cx, dcy, DIAMOND_W, DIAMOND_H, diamondLabel);
        const diamBottom = dcy + DIAMOND_H/2;
        y += DIAMOND_H + FORK_GAP;
        const roofY = y;
        svg += seg(cx, diamBottom, cx, roofY);
        // Branch entry arrow length — for layer-choice forks (attn-type or
        // dense+MoE FFN) we extend it by LABEL_H so the per-branch "N/L" pill
        // has its own dedicated vertical space above the box top (otherwise
        // the pill sits centered on the first box's title text and overlaps
        // it). LABEL_H is already budgeted into sectionH() above. Other forks
        // don't draw pills, so they keep the original short arrow.
        const brEntryLen = GAP_V + ARROW + (isLayerChoiceFork ? LABEL_H : 0);
        if (N_BR>1) {
          svg += seg(brCX[0], roofY, brCX[N_BR-1], roofY);
          brCX.forEach(bcx => { svg += arrowV(bcx, roofY, brEntryLen); });
        } else {
          svg += arrowV(cx, roofY, brEntryLen);
        }
        brTopY = roofY + brEntryLen;
      }
      const brBottomYs = split.branches.map((g,gi) => {
        const bcx = brCX[gi]||cx, bxb = bcx - BR_W/2;
        const col0 = g.layers[0] ? colFor(g.layers[0]) : COL.linear;
        // For multi-branch diamond forks: by default we do NOT draw per-branch
        // badges — the diamond label tells the story and badges would overlap
        // fork lines. EXCEPTIONS: layer-choice forks (V4 HCA/CSA/SWA-only attn
        // type fork, V3.x dense+MoE FFN fork) draw a per-branch `N/L` badge so
        // each branch's layer share is visually explicit.
        // Badge format depends on context:
        //   • Single branch (no fork)    → `×N`  (loop count, same convention
        //                                  as Llama "Attention (×64)" etc.)
        //   • Layer-choice fork branch   → `N/L` (this branch covers N of L
        //                                  total layers — fractions sum to L,
        //                                  visually distinct from `×N` so the
        //                                  reader doesn't read the fork as
        //                                  "all branches run in parallel each
        //                                  N times". Same N is still the
        //                                  loop count for that branch alone.)
        let badgeSvg = '';
        if (N_BR === 1 || isLayerChoiceFork) {
          const rpPerBlock = g.repeat === displayNL ? 1 : g.repeat;
          const text = isLayerChoiceFork
            ? `${rpPerBlock}/${displayNL}`
            : `×${rpPerBlock}`;
          // Pill widens for fork "N/L" form (e.g. "20/43" = 5 chars vs "×20"
          // = 3); fonts at 9.5pt fit ~6.5 chars in 50px. Below this width the
          // text would clip on 3-digit layer counts (rare but possible).
          const rpW = isLayerChoiceFork ? 50 : 44;
          const rpH = LABEL_H;
          // For layer-choice forks, the entry arrow was extended by LABEL_H so
          // the pill can sit ABOVE the box top (rather than on the box's title
          // text). Place the pill flush with the LABEL_H slot, leaving 2px
          // breathing room between the pill bottom and the box top edge. For
          // single-branch (non-fork) we keep the legacy position straddling
          // the box top — single boxes don't have title overlap because
          // they're standalone (no fork).
          const rpX = bcx - rpW/2;
          const rpY = isLayerChoiceFork
            ? brTopY - rpH - 2   // pill bottom 2px above the box top
            : brTopY - rpH/2;    // legacy: centered on box top
          badgeSvg = `<rect x="${rpX}" y="${rpY}" width="${rpW}" height="${rpH}" rx="${rpH/2}"
            fill="${col0.stroke}"/>
            <text x="${bcx}" y="${rpY+12}" text-anchor="middle"
              font-size="9.5" font-weight="700" fill="#fff">${text}</text>`;
        }
        let lby = brTopY;

        if (g.isMoE) {
          // ── MoE branch rendering ──────────────────────────────────────────
          const routed = g.layers.filter(l => l.moeGroup === 'router' || l.moeGroup === 'gate_up' || l.moeGroup === 'down');
          const shared = g.layers.filter(l => l.moeGroup === 'shared_up' || l.moeGroup === 'shared_down');
          const INNER_PAD_H = 10, INNER_PAD_TOP = 22, INNER_PAD_BOT = 12;

          if (shared.length === 0) {
            // ── Single-column MoE (no shared expert): render routed layers as simple stack ──
            const routedH = routed.length ? (routed.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            const wrapH = INNER_PAD_TOP + routedH + INNER_PAD_BOT;
            // Outer MoE wrapper box (dashed orange rect)
            svg += `<rect x="${bxb}" y="${lby}" width="${BR_W}" height="${wrapH}" rx="7"
              fill="${COL.moe.fill}" fill-opacity=".5" stroke="${COL.moe.stroke}" stroke-width="1.2" stroke-dasharray="4,2.5"/>`;
            // Single arrow entering the wrapper (offset inside border)
            const entryInside = lby + 4;
            const firstBoxY = lby + INNER_PAD_TOP;
            // Bridge from just below the ×N badge across the wrapper top border
            // into the internal entry arrow — avoids a visual gap at the badge.
            svg += seg(bcx, brTopY - 2, bcx, entryInside);
            svg += arrowV(bcx, entryInside, firstBoxY - entryInside);
            let rly = firstBoxY;
            const boxPad = INNER_PAD_H;
            routed.forEach((l,li) => {
              svg += drawBox(bxb + boxPad, rly, BR_W - boxPad*2, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
              rly += BOX_H;
              if (li < routed.length-1) { svg += arrowV(bcx, rly, GAP_V+ARROW); rly += GAP_V+ARROW; }
            });
            // Exit line (stop inside border)
            const exitInside = lby + wrapH - 4;
            svg += seg(bcx, rly, bcx, exitInside);
            // Bridge across the wrapper bottom border so the line is continuous
            // with the external branch-merge line drawn below.
            svg += seg(bcx, exitInside, bcx, lby + wrapH);
            lby += wrapH;
          } else {
            // ── Dual-column MoE: routed left, shared right ──────────────────
            const INNER_GAP = 10;
            const innerColW = Math.floor((BR_W - INNER_PAD_H*2 - INNER_GAP) / 2);
            const routedH = routed.length ? (routed.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            const sharedH = shared.length ? (shared.length-1)*(BOX_H+GAP_V+ARROW)+BOX_H : 0;
            const innerH = Math.max(routedH, sharedH);
            const wrapH = INNER_PAD_TOP + innerH + INNER_PAD_BOT;
            // Outer MoE wrapper box (dashed orange rect)
            svg += `<rect x="${bxb}" y="${lby}" width="${BR_W}" height="${wrapH}" rx="7"
              fill="${COL.moe.fill}" fill-opacity=".5" stroke="${COL.moe.stroke}" stroke-width="1.2" stroke-dasharray="4,2.5"/>`;
            // Two inner columns: routed left, shared right
            const rColX = bxb + INNER_PAD_H;
            const sColX = bxb + INNER_PAD_H + innerColW + INNER_GAP;
            const rColCX = rColX + innerColW/2;
            const sColCX = sColX + innerColW/2;
            // Entry: fork line INSIDE the wrapper (offset from top border)
            const entryInside = lby + 4;
            const firstBoxY = lby + INNER_PAD_TOP;
            // Bridge from just below the ×N badge across the wrapper top border
            // into the internal fork — avoids a visual gap at the badge.
            svg += seg(bcx, brTopY - 2, bcx, entryInside);
            // Horizontal split line and arrows into each column — inside the border
            svg += seg(rColCX, entryInside, sColCX, entryInside);
            svg += arrowV(rColCX, entryInside, firstBoxY - entryInside);
            svg += arrowV(sColCX, entryInside, firstBoxY - entryInside);
            let rly = firstBoxY;
            routed.forEach((l,li) => {
              svg += drawBox(rColX, rly, innerColW, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
              rly += BOX_H;
              if (li < routed.length-1) { svg += arrowV(rColCX, rly, GAP_V+ARROW); rly += GAP_V+ARROW; }
            });
            let sly = firstBoxY;
            shared.forEach((l,li) => {
              svg += drawBox(sColX, sly, innerColW, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
              sly += BOX_H;
              if (li < shared.length-1) { svg += arrowV(sColCX, sly, GAP_V+ARROW); sly += GAP_V+ARROW; }
            });
            // Exit: merge lines from both column bottoms — stop inside the border
            const exitInside = lby + wrapH - 4;
            svg += seg(rColCX, rly, rColCX, exitInside);
            svg += seg(sColCX, sly, sColCX, exitInside);
            svg += seg(rColCX, exitInside, sColCX, exitInside);
            // Bridge the inner horizontal merge down through the wrapper bottom
            // border so it connects to the external branch-merge line.
            svg += seg(bcx, exitInside, bcx, lby + wrapH);
            lby += wrapH;
          }
        } else {
          // Normal branch: stack layers vertically
          g.layers.forEach((l,li)=>{
            svg += drawBox(bxb, lby, BR_W, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
            lby += BOX_H;
            if (li < g.layers.length-1) { svg += arrowV(bcx, lby, GAP_V+ARROW); lby += GAP_V+ARROW; }
          });
        }
        // Draw badge AFTER wrapper so it renders on top in SVG z-order
        svg += badgeSvg;
        return lby;
      });
      // Single-branch merge only needs tiny margin; multi-branch needs full pad
      // for the horizontal collector line that joins all branches.
      const mergePadBr = N_BR <= 1 ? 4 : MERGE_PAD;
      const floorY = Math.max(...brBottomYs) + mergePadBr;
      if (N_BR>1) {
        svg += seg(brCX[0], floorY, brCX[N_BR-1], floorY);
        brCX.forEach((bcx,gi) => svg += seg(bcx, brBottomYs[gi], bcx, floorY));
      } else {
        // Single branch — draw a connecting line from branch bottom to merge point
        const bcx0 = brCX[0]||cx;
        svg += seg(bcx0, brBottomYs[0], bcx0, floorY);
      }
      y = floorY;
      // PostTrunk: layers that run AFTER the branches (V4 O down-proj / O up-proj).
      // Rendered inside the section dashed wrapper, narrowed like preTrunk.
      const postTrunk = split.postTrunk || [];
      if (postTrunk.length > 0) {
        // Connector arrow from branch-merge into the first postTrunk layer
        y += GAP_V + ARROW;
        svg += arrowV(cx, floorY, GAP_V + ARROW);
        const ptX = PAD_X+10, ptW = SHARED_W-20;
        postTrunk.forEach((l, li) => {
          svg += drawBox(ptX, y, ptW, BOX_H, colFor(l), stripRepeat(l.name), subOf(l));
          y += BOX_H;
          if (li < postTrunk.length - 1) {
            svg += arrowV(cx, y, GAP_V + ARROW);
            y += GAP_V + ARROW;
          }
        });
      }
      if (!isSingleMoE) y += SEC_BR_PAD_BOT;

      // Draw the section dashed wrapper rect behind all content.
      // Skip for isSingleMoE — the MoE wrapper (dashed orange rect) already
      // serves as the visual container; adding a second one causes overlap.
      if (!isSingleMoE) {
        const sectionBoxH = y - sectionBoxStartY;
        const secCol = split.trunk.length > 0 ? colFor(split.trunk[0])
                     : (split.branches[0]?.layers[0] ? colFor(split.branches[0].layers[0]) : COL.linear);
        svg += `<rect x="${PAD_X+4}" y="${sectionBoxStartY}" width="${SHARED_W-8}" height="${sectionBoxH}" rx="7"
          fill="${secCol.fill}" fill-opacity=".10" stroke="${secCol.stroke}" stroke-width="1.2" stroke-dasharray="4,2.5"/>`;
        // ×N badge on the top edge of the section wrapper — per-block count.
        // Always drawn (even when secPerBlock === 1) so every branched section
        // visually carries its repeat count, matching trunk-only sections that
        // also get a ×N badge unconditionally. Earlier versions suppressed
        // ×1 badges as "redundant" with the outer block badge — but that
        // dropped the FFN ×1 in V3.2-Exp and the Attn ×1 in V4-Pro. The
        // SEC_BR_PAD_TOP padding now applied above the first trunk layer
        // gives the badge ~9.5px clearance from Q a-proj, no more overlap.
        const secRp = split.totalRepeat || 1;
        const secPerBlock = secRp === displayNL ? 1 : secRp;
        const sBadgeW=44, sBadgeH=17, sBadgeX=cx-sBadgeW/2, sBadgeY=sectionBoxStartY-sBadgeH/2;
        svg += `<rect x="${sBadgeX}" y="${sBadgeY}" width="${sBadgeW}" height="${sBadgeH}" rx="${sBadgeH/2}"
          fill="${secCol.stroke}"/>
          <text x="${cx}" y="${sBadgeY+12}" text-anchor="middle"
            font-size="9.5" font-weight="700" fill="#fff">×${secPerBlock}</text>`;
      }
    }
    return y;
  }

  svg+=`<text x="${cx}" y="${PAD_TOP-6}" text-anchor="middle"
    font-size="13" font-weight="700" fill="#1a1a18">${currentModelName}</text>`;
  let y=PAD_TOP+4;

  pre.forEach((l,li)=>{
    const sub=l.K!=null?`vocab=${l.K.toLocaleString()} → hid=${l.N||'?'}`:'';
    svg+=drawBox(PAD_X,y,SHARED_W,BOX_H,colFor(l),l.name,sub);
    y+=BOX_H;
    // If block follows, draw a plain line (no arrowhead) from embed bottom
    // to the top of the ×N label — the arrowhead would get occluded by the
    // label anyway. The section-entry arrow below carries the actual head.
    if (li < pre.length-1) {
      svg += arrowV(cx, y, ARROW+GAP_V);
      y += ARROW+GAP_V;
    } else if (attnH>0 || effFfnH>0) {
      // Arrow from embed to block — tip stops just above the ×N badge
      // (badge sits centered on blockTopY, so its top edge is blockTopY - 11)
      const embedToBlock = ARROW+GAP_V;
      const badgeTopOffset = 11; // bpH/2
      svg += arrowV(cx, y, embedToBlock - badgeTopOffset + 6);
      y += embedToBlock;
    }
  });

  if (attnH>0 || effFfnH>0) {
    const blockTopY = y;
    // Draw the dashed block rectangle
    svg += `<rect x="${PAD_X-BLOCK_PAD_H}" y="${blockTopY}" width="${SHARED_W+BLOCK_PAD_H*2}" height="${blockH}" rx="12"
      fill="#f9f9f8" stroke="#c8c6be" stroke-width="1.5" stroke-dasharray="5,3"/>`;
    // ×N layers tag centered on top edge — drawn AFTER the dashed rect so it
    // sits on top of it. The segment from embed above stops at blockTopY
    // and this label visually caps it.
    const bpW=82,bpH=22,bpX=cx-bpW/2,bpY=blockTopY-bpH/2;
    svg+=`<rect x="${bpX}" y="${bpY}" width="${bpW}" height="${bpH}" rx="${bpH/2}" fill="#1a1a18"/>
      <text x="${cx}" y="${bpY+15}" text-anchor="middle"
        font-size="11" font-weight="700" fill="#fff">×${displayNL}</text>`;

    // No connecting line from the ×N block badge down to the first section —
    // the section's own dashed box + badge serves as the visual entry.
    // (Rule: no lines inside a dashed-box between badge and first content.)
    const firstSectionTop = blockTopY + BLOCK_PAD_TOP;

    // Render Attn section, then FFN section. Each renderSection returns the
    // Y at the bottom of its last drawn element (no trailing arrow).
    // When ffnMerged, the attn serial groups already contain FFN layers inside
    // each sub-block, so we skip the separate FFN section entirely.
    let innerY = firstSectionTop;
    const effectiveFfnH = ffnMerged ? 0 : ffnH;
    if (attnH>0) {
      innerY = renderSection(attnSplit, innerY, ffnMerged ? 'Transformer' : 'Attention');
      if (effectiveFfnH>0) {
        // Arrow between sections — tip stops just above the FFN section's badge.
        const sectionConnLen = GAP_V+ARROW+sectionGap;
        const badgeMargin = 10;
        svg += arrowV(cx, innerY, sectionConnLen - badgeMargin + 6);
        innerY += sectionConnLen;
      }
    }
    if (effectiveFfnH>0) {
      innerY = renderSection(ffnSplit, innerY, 'FFN');
    }
    // Exit the block: draw a single continuous arrow from innerY all the way
    // past the block's bottom border down to where the next element starts.
    const blockBottom = blockTopY + blockH;
    const exitTop = innerY;
    const exitBottom = blockBottom + GAP_V + ARROW;
    if (post.length > 0) {
      svg += arrowV(cx, exitTop, exitBottom - exitTop);
      y = exitBottom;
    } else {
      // No post layers — still draw the exit arrow to keep the block visually closed
      svg += arrowV(cx, exitTop, blockBottom - exitTop);
      y = blockBottom;
    }
  }

  post.forEach((l,li)=>{
    const sub=(l.K!=null&&l.N!=null)?`hid=${l.K.toLocaleString()} → vocab=${l.N.toLocaleString()}`:'';
    svg+=drawBox(PAD_X,y,SHARED_W,BOX_H,colFor(l),l.name,sub);
    y+=BOX_H;
    if(li<post.length-1){svg+=arrowV(cx,y,ARROW+GAP_V);y+=ARROW+GAP_V;}
  });
  svg+=`</svg>`;

  const legendItems=[
    {col:COL.embedding,label:'Embedding'},{col:COL.linear,label:'Linear (GEMM)'},
    {col:COL.attention,label:'Attention'},
    ...(hasLinAttn?[{col:COL.linear_attn,label:'GatedDeltaNet'}]:[]),
    ...(hasMoE?[{col:COL.moe,label:'MoE FFN'}]:[]),
  ];
  const hasBranches   = attnSplit.branches.length>0 || ffnSplit.branches.length>0;
  const hasSerialGrps = !!(attnSplit.serialGroups || ffnSplit.serialGroups);
  const legendHtml=`<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:14px;justify-content:center;font-size:12px;color:#555">
    ${legendItems.map(it=>`<span style="display:flex;align-items:center;gap:5px">
      <span style="width:12px;height:12px;border-radius:3px;background:${it.col.fill};border:1.5px solid ${it.col.stroke};display:inline-block"></span>${it.label}
    </span>`).join('')}
    ${hasBranches?`<span style="display:flex;align-items:center;gap:5px">
      <span style="width:14px;height:14px;background:#fff7e6;border:1.5px solid #EF9F27;transform:rotate(45deg);display:inline-block"></span>Parallel fork
    </span>`:''}
    ${hasSerialGrps?`<span style="display:flex;align-items:center;gap:5px">
      <span style="width:12px;height:12px;border-radius:3px;background:#dcf5ec;border:1.5px dashed #0F6E56;display:inline-block"></span>Serial sub-block
    </span>`:''}
  </div>`;

  const wGB=modelWeightBytes()/1e9;
  const totalFlops=layers.map(l=>calcLTotal(l).flops).reduce((a,b)=>a+b,0);
  const archTag = classifyArch();
  const pTag = cfg.precision !== 'fp16' ? ` (${precisionLabel()})` : '';
  const archBanner = archTag
    ? `<div style="background:#eef4fb;border:1px solid #b5d4f4;border-radius:8px;padding:8px 12px;margin-bottom:12px;font-size:12px;color:#0c447c;text-align:center">
         <strong>Architecture type:</strong> ${archTag}${pTag}
       </div>`
    : '';
  const statsHtml=mcs(
    mc('Model',currentModelName,'')+mc('Weights'+pTag,wGB.toFixed(1),'GB')+
    mc('Layers',displayNL,'transformer')+mc('Total FLOPs',fmt(totalFlops),'')
  );
  // Clear then set — forces a full DOM rebuild so browsers can't skip re-rendering
  const mc_el=document.getElementById('mainContent');
  mc_el.innerHTML='';
  mc_el.innerHTML=statsHtml+archBanner+`<div class="cw" style="text-align:center">
    <div class="ct" style="margin-bottom:14px">Architecture — ${currentModelName}</div>
    ${svg}${legendHtml}
  </div>`;
}


// ── MoE sweet-spot analyzer ───────────────────────────────────────────────────
// Scans a range of M (tokens per forward pass) and identifies:
//   M_saturate: smallest M where expected activated experts ≥ 95% of nExperts
//               (above this, further M doesn't increase HBM weight reads)
//   M_compute:  smallest M where MoE AI exceeds ridge (compute-bound kicks in)
// Between these two points is the "sweet spot": per-token latency is near minimum
// and throughput scales linearly with M.

# `tests/` — regression test suite

14 Node.js test files that load the built HTML via `vm.runInContext` with a
mock DOM, then assert on numerical and API invariants. Use `run_all.py` to
run the whole suite.

## Running

```bash
# from repo root
python3 tests/run_all.py                                    # tests dist/llm_perf_analyzer_v89.html
python3 tests/run_all.py dist/llm_perf_analyzer_v90.html    # specific build
python3 tests/run_all.py --filter feature_suite             # only matching tests
python3 tests/run_all.py --verbose                          # show full test output
```

Exit code 0 means every test passes its known baseline; 1 means a regression
was introduced relative to the baseline.

## What each file covers

| File                                | What it verifies |
|-------------------------------------|-------------------|
| `regression_test_v74.js`            | MoE TP/EP/hybrid FLOPs + bytes vs ground-truth analytical reference; `cfg.batch` + KV cache semantics under DP |
| `decode_render_test_v74.js`         | Decode tab renders correctly across model families |
| `global_batch_dom_test_v74.js`      | Global-vs-per-rank batch input field DOM behaviour |
| `comm_regression_test_v75.js`       | Per-card a2a/AR communication volume for pure-EP and hybrid (post-v75 fix); DeepEP reference comparison |
| `mla_tpmode_test_v78.js`            | MLA layer TP-mode handling (col-par latent_kv, row-par o_proj) |
| `update_verify_test_v78.js`         | `onTPChange` correctly rebuilds layer state across configs |
| `verify_unified_vram_test_v78.js`   | Unified VRAM accounting (weights + KV cache + activations) |
| `dispatch_bpe_test_v79.js`          | MoE dispatch a2a uses `dispatchBpe()` (FP8 under W8A8) while combine stays at `aBpe()` |
| `sparse_mla_test_v81.js`            | DSA / Sparse-MLA toggle gating; lightning-indexer + top-k-restricted core attention |
| `col_par_abytes_test_v83.js`        | Column-parallel activation byte calculation; **has 18 baseline fails** (test written against v81 conventions superseded by v83+) |
| `v83_verification_test.js`          | Hybrid attention (HCA/CSA/SWA) layer labeling; `compress_ratios` fallback; **has 6 baseline fails** (specific to v81 hybrid model rendering, since refined) |
| `feature_suite_test_v86.js`         | H200 chip specs, DSA toggle gating, bottleneck rollup, TP-timeline DP semantics, GPU util scaling |
| `precision_formula_test_v86.js`     | Bottleneck-table formula text matches numerical result across precision toggles (BF16 ↔ W8A16 ↔ W8A8) |

## Baseline failures (not regressions)

`run_all.py` knows two tests have pre-existing failures and reports
"pass" so long as the failure count doesn't grow. These represent assertions
against historical behaviour that has since been intentionally changed:

  - `col_par_abytes_test_v83.js`: 18 baseline fails
  - `v83_verification_test.js`:    6 baseline fails

If you're working on either of those areas and want a clean baseline, port
the relevant assertions to a new test file with current expected values.

## Adding a new test

Use any existing test as a template. Pattern:

```js
const fs = require('fs'); const vm = require('vm');
function load(file) {
  const html = fs.readFileSync(file, 'utf-8');
  let code = '';
  for (const m of [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]) code += m[1] + '\n';
  // sandbox isolation requires `let`/`const` → `var`
  code = code.replace(/^(\s*)let\s+/gm, '$1var ').replace(/^(\s*)const\s+/gm, '$1var ');
  /* ... mock DOM, vm.createContext, vm.runInContext ... */
  return sb;  // expose runtime globals
}

const V = load('/path/to/dist.html');   // run_all.py rewrites this path
// ... assert on V.tpInfo(...), V.modelWeightBytes(), etc.
console.log(`RESULT: ${nPass} passed, ${nFail} failed`);
process.exit(nFail ? 1 : 0);
```

Name the file `<topic>_test_v<N>.js` (or `v<N>_verification_test.js`). Place
it in this directory; `run_all.py` will pick it up automatically.

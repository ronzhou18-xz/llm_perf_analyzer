# `src/` — modular source layout

The tool was originally a single 5725-line HTML file. It is now split into a
shell + 15 JS modules under `js/`. `build.py` at the repo root concatenates
everything back into a single deployable HTML.

## Modules

| File                        | Lines | Purpose |
|-----------------------------|------:|---------|
| `index.html`                |   380 | HTML structure, `<style>`, `<!-- BUILD: INJECT_JS -->` placeholder |
| `js/state.js`               |   340 | `chips[]` / `layers[]` / `cfg`, precision helpers, mode events, `applyConfigToLayers` |
| `js/memory.js`              |   204 | `modelWeightBytes` / `kvCacheBytes` / `kvShardFactor` + `updateCfgInfo` display |
| `js/parallel.js`            |   986 | TP / EP / DP state, `tpInfo`, `tpShardShape`, sharding rules, **MoE M-convention** |
| `js/compute.js`             |   177 | Attention / MoE / linear-attn FLOP & byte math (`calcStdAttn`, `calcFA2Attn`, `calcMoE`, `calcLinearAttn`) |
| `js/utils.js`               |    93 | `rlim` / `fmt` / `ft`, HF cache (localStorage), file drop handlers |
| `js/import.js`              |  1091 | HF `config.json` & safetensors parser → layer list (`inferFromConfig`, `expandMoEToSubLayers`) |
| `js/ui-controls.js`         |   277 | chip / layer list rendering, edit modals, add / save / delete |
| `js/tabs-core.js`           |    81 | Tab dispatcher + metric-tile helpers + color helpers |
| `js/tab-roofline.js`        |   273 | Roofline / Bottleneck / AI / Throughput tabs |
| `js/tab-tp-timeline.js`     |    75 | TP timeline tab |
| `js/tab-sweep.js`           |   142 | Batch sweep tab |
| `js/tab-compare.js`         |   206 | Snapshot comparison tab |
| `js/tab-structure.js`       |  1150 | `classifyArch` + `rStructure` architecture visualisation |
| `js/tab-decode.js`          |   283 | `rDecode` + MoE sweet-spot block |
| `js/init.js`                |     7 | Bootstrap calls (**must be last in load order**) |

## Editing map

| Symptom                                  | Edit this module                          |
|------------------------------------------|-------------------------------------------|
| Wrong VRAM / KV cache size               | `memory.js`                               |
| TP sharding bug, AllReduce volume wrong  | `parallel.js`                             |
| MoE FLOPs / activated experts wrong      | `parallel.js` (`tpInfo` MoE branches), `compute.js` (`calcMoE`) |
| New model family not parsing             | `import.js` (`inferFromConfig`)           |
| Sparse-MLA / DSA misbehaviour            | `state.js` (toggle) + `parallel.js` (shard rules) |
| Wrong tab visualisation                  | corresponding `tab-*.js`                  |
| New chip spec                            | `state.js` (`chips` array)                |

## Module dependency rules

1. **`init.js` is loaded last.** It calls into other modules at script-end.
2. **All globals are declared in `state.js`** (`chips`, `layers`, `cfg`,
   `nCards`, `mode`, `sparseMLA`, etc.) and in `parallel.js` (`moePar`,
   `epDeg`, `dp`, `tpAttn`, `moePeakMode`). Other modules **mutate** them
   but never re-declare with `let` / `const`.
3. **All functions are global.** The bundle is one concatenated `<script>`,
   so every top-level `function foo() {}` is reachable from every other
   module without `import` / `export`.
4. **`build.py`'s `MODULE_ORDER` is load-bearing.** Do not reorder unless
   you have specifically validated against the dependency graph.

## Build

```bash
python3 build.py                 # default → dist/llm_perf_analyzer_v89.html
python3 build.py --version v90   # → dist/llm_perf_analyzer_v90.html
python3 build.py --no-banner     # omit section banners (byte-equal to monolith)
```

## Round-trip guarantee

`build.py --no-banner` produces a file functionally and byte-equivalent to
the original monolithic v88. The banner-on version is verified across 14
regression test files (~512 assertions) to behave identically.

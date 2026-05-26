# LLM Inference Performance Analyzer

A pure-frontend, single-file HTML/JS tool that models inference performance of
large language models across hardware, parallelism strategies, and architectures.
It is a **theoretical roofline / ceiling calculator** — not a system simulator.

The deployable artifact is `dist/llm_perf_analyzer_v<N>.html`. Open it in any
browser; no server, no install. The source is modular and assembled by a tiny
Python build script.

## Quick start

```bash
# Use the prebuilt HTML directly
open dist/llm_perf_analyzer_v89.html

# Or rebuild from source after editing
python3 build.py                 # → dist/llm_perf_analyzer_v89.html
python3 build.py --version v90   # → dist/llm_perf_analyzer_v90.html
```

## Layout

```
.
├── build.py                  # concatenates src/ → dist/
├── src/
│   ├── index.html            # HTML shell + <style> + JS injection point
│   └── js/                   # 15 JS modules (see src/README.md)
├── tests/                    # 14 regression test files + runner
│   ├── run_all.py            # run every test against dist/...html
│   └── *_test*.js
└── dist/
    └── llm_perf_analyzer_v89.html
```

See `src/README.md` for module breakdown and `tests/README.md` for what each
test covers.

## What it models

  - Parameter counts (with TP/EP weight sharding)
  - VRAM occupancy (full resident weights, KV cache, activations)
  - HBM bandwidth consumption per layer
  - Per-token / per-step latency under arithmetic-intensity rooflines
  - Cross-card communication: AllReduce (TP), AllToAll (MoE EP)
  - Architectures: dense, MoE, MLA, GDN, hybrid (HCA/CSA/SWA), DSA

## Key abstractions

  - `chips[]`: hardware specs (HBM BW, TFLOPS, interconnect BW)
  - `layers[]`: per-layer M/K/N + type + metadata
  - `cfg`: batch, seqLen, outTokens, precision, model dims
  - `tpInfo(layer, chip)`: the central per-layer compute/memory/comm calculator
  - `moePar ∈ {tp, ep, hybrid}` × `dp ∈ [1, nCards]`: parallelism config

## Conventions

  - **Precision modes**: `fp16` (BF16), `fp8w` (W8A16), `fp8` (W8A8), `fp8kv` (W8A8+FP8 KV)
  - **`cfg.batch` is PER-DP-RANK**; global batch = cfg.batch × dp
  - **Interconnect BW** is unidirectional GB/s
  - **MoE M-convention**: see `src/js/parallel.js` (search "M-convention")

## Adding a new model family

The Hugging Face `config.json` parser is in `src/js/import.js`
(`inferFromConfig`). Add a branch for the new architecture there. Run the
suite afterwards:

```bash
python3 tests/run_all.py dist/llm_perf_analyzer_v90.html
```

## Regression testing

```bash
python3 tests/run_all.py                                    # tests latest dist/
python3 tests/run_all.py dist/llm_perf_analyzer_v89.html    # tests specific file
```

Tests load the HTML, extract its `<script>` blocks, run them in `vm.runInContext`
with a mock DOM, and assert on numerical / API invariants.

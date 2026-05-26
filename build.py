#!/usr/bin/env python3
"""
build.py — assemble src/ into a single deployable HTML.

Reads:   src/index.html  (with <!-- BUILD: INJECT_JS --> placeholder)
         src/js/*.js     (in MODULE_ORDER below)
Writes:  dist/llm_perf_analyzer_v<VERSION>.html

Usage (from repo root):
    python3 build.py                # default version
    python3 build.py --version v90  # custom version
    python3 build.py --no-banner    # omit section banners (byte-equal to v88 monolith)
"""
import argparse, sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
SRC_DIR  = ROOT / 'src'
DIST_DIR = ROOT / 'dist'

# Module concatenation order is load-bearing: later modules reference globals
# declared in earlier ones. Do not reorder without checking dependencies.
MODULE_ORDER = [
    'state.js',           # chips/layers/cfg, precision helpers, mode events
    'memory.js',          # weight + KV math, updateCfgInfo
    'parallel.js',        # TP/EP/DP state and sharding logic
    'compute.js',         # attention / MoE / linear-attn math
    'utils.js',           # rlim, fmt, ft, HF cache, file drop
    'import.js',          # HF config parser
    'ui-controls.js',     # chip/layer rendering & modals
    'tabs-core.js',       # tab dispatcher + tile helpers
    'tab-roofline.js',
    'tab-tp-timeline.js',
    'tab-sweep.js',
    'tab-compare.js',
    'tab-structure.js',
    'tab-decode.js',
    'init.js',            # bootstrap — MUST be last
]

PLACEHOLDER = '<!-- BUILD: INJECT_JS -->'

def banner(name):
    bar = '═' * 78
    return (f"// {bar}\n"
            f"// MODULE: {name}\n"
            f"// {bar}\n")

def build(src_dir: Path, out_path: Path, *, include_banners: bool) -> None:
    index = (src_dir / 'index.html').read_text()
    parts = []
    for fn in MODULE_ORDER:
        content = (src_dir / 'js' / fn).read_text()
        parts.append((banner(fn) + content) if include_banners else content)
    blob = ''.join(parts)

    if PLACEHOLDER not in index:
        sys.exit(f"ERROR: {src_dir / 'index.html'} missing {PLACEHOLDER!r}")
    out = index.replace(PLACEHOLDER + '\n', blob)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out)
    rel = out_path.relative_to(ROOT) if str(out_path).startswith(str(ROOT)) else out_path
    print(f"\u2713 Built {rel} \u2014 {len(out)} bytes, {out.count(chr(10))+1} lines")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--version',   default='v89')
    ap.add_argument('--src',       default=str(SRC_DIR))
    ap.add_argument('--out-dir',   default=str(DIST_DIR))
    ap.add_argument('--no-banner', action='store_true',
                    help='Skip section banners — bit-exact to v88 monolithic layout')
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out_dir) / f'llm_perf_analyzer_{args.version}.html'
    build(src, out, include_banners=not args.no_banner)

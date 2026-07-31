# MLX on DGX Spark: SM121 quantized kernels (`sm121-port`)

NVIDIA shipped the DGX Spark (GB10) with ~500 TFLOPS of FP4 tensor compute
that no mainstream framework actually uses for quantized LLM inference.
This branch is the missing layer: native block-scaled CUTLASS GEMMs for the
SM120/SM121 (Desktop Blackwell) architecture, worth **10–16× on real-model
prefill** over stock MLX CUDA on the same silicon.

## Why the gap existed

Upstream MLX's CUDA effort targets Hopper (`qmm_sm90` checks
`compute_capability == 9`), and most serving stacks chase datacenter
Blackwell (B200/SM100), which uses a different instruction path entirely
(tcgen05/TMEM). Desktop Blackwell — GB10 and the RTX 50 series — fell in the
crack. On a Spark, stock MLX runs generic Ampere-era kernels that
dequantize in software while the FP4 units idle.

These kernels drive the SM120 `mma.block_scale` path directly
(MXFP4 / NVFP4 / MXFP8), plus a CUTLASS grouped GEMM for MoE expert
dispatch. Everything gates on compute capability ≥ 12 and is inert on
other GPUs. Tuning (tile configs, dispatch thresholds) is measured on
GB10's 48 SMs and LPDDR5x.

## Headline numbers

gpt-oss MXFP4, 2048-token prompt, 64 generated, 3 trials, median:

| gpt-oss-20b            | Prefill tok/s | Decode tok/s | vs stock |
|------------------------|--------------:|-------------:|---------:|
| Spark · stock MLX CUDA |           135 |           65 |     1.0× |
| **Spark · this branch**|     **2,221** |           66 |**16.5×** |
| Mac Studio M2 Ultra    |         1,075 |          105 |     8.0× |

| gpt-oss-120b           | Prefill tok/s | Decode tok/s | vs stock |
|------------------------|--------------:|-------------:|---------:|
| Spark · stock MLX CUDA |            87 |           42 |     1.0× |
| **Spark · this branch**|       **901** |           45 |**10.3×** |
| Mac Studio M2 Ultra    |           656 |           69 |     7.5× |

This is what makes disaggregated inference (exo: Spark prefills, Mac
decodes) real: prefill is compute-bound and the FP4 tensor cores win it
(Spark 1.4–2.1× over the Mac); decode is bandwidth-bound and the Mac's
800 GB/s wins it (1.5–1.6× over the Spark). Without these kernels the
Spark loses both halves.

## The dispatch is honest

The kernels are **not** faster than stock everywhere, and the dispatch
doesn't pretend they are. Across ~900 paired benchmark cells
(24 real model shapes × M ∈ 1…2048 × three fp modes):

**Ours** — large-M dense prefill on K%128 shapes (Llama, Qwen-72B,
Qwen3.6-27B: 1.1–4.2×), odd-N MXFP4 shards (DSv3's 1407×7168: 2.3–5.4×
at every M), DSv4's large GEMMs (1.4–3.0×), MoE prefill (grouped
CUTLASS + host-sync dequant paths), MXFP8 at M ∈ [512, 2048] on ≥32M-element
weights.

**Delegated to upstream** — decode and small batches (upstream's JIT
`fp_qmv` beat our old QMV by ~80%), the sub-M=64 band, gpt-oss's K=2880
dense projections, DSv4's compression stubs (N=576 / K=1024 — five tile
columns can't fill 48 SMs), and small-weight MXFP8. Where upstream
measured faster, upstream runs.

## Correctness

The March-era version of these kernels computed numerically wrong MXFP4
results (E8M0 activation scales quantized against a different scale than
was stored — up to 2× per-group inflation, invisible to perf-only
benchmarks). This branch is the first that is both fast and verified:

- E8M0 scale fix in dense + grouped paths (GEMM error now matches
  quantization theory: 0.16 mxfp4 / 0.14 nvfp4 / 0.04 mxfp8 vs unquantized
  ground truth)
- CUDA-graph ordering fix for direct-launched kernels
- use-after-free fix for allocations made in direct-launch windows
  (`commit()` + `synchronize()` before every window)
- weak_ptr-invalidated scale-factor caches (pool address reuse can no
  longer serve stale weights)

Regression tests: dense + gather correctness harnesses, a 60-iteration
input-corruption detector, and flake loops — all green on every commit
that touches the kernels.

## Layout

- `mlx/backend/cuda/quantized/qmm_sm120.cu` — dense block-scaled GEMM
  (MXFP4/NVFP4/MXFP8), activation quantization, SFB reformat + cache
- `mlx/backend/cuda/quantized/gather_qmm_grouped.cu` — CUTLASS grouped
  GEMM for MoE prefill
- `mlx/backend/cuda/quantized/gather_qmm.cu` — sorted per-expert and
  host-sync MoE paths
- `mlx/backend/cuda/quantized/quantized.cpp` — dispatch (every threshold
  is sweep-derived; see commit messages for the cells behind each rule)

Build is standard MLX CUDA (`MLX_BUILD_CUDA=ON`); arch auto-detects to
`121a`. CUTLASS v4.4.2 (upstream's pin). For a reproducible environment,
build inside `nvidia/cuda:13.2.0-devel-ubuntu24.04`.

## Known follow-ups

- Zero-drain refactor: pre-allocate direct-launch buffers while still in
  graph mode to remove the per-call pipeline drain (cost is ~0 for
  gpt-oss-class shapes; matters more for K%128-dense models)
- DSv4 routed-expert shapes through the grouped path (2048×4096 experts
  are small for 128-wide grouped tiles — not yet swept)
- Persistent QMV for >L2 dense decode (70B-class) — re-bench against
  upstream `fp_qmv` before porting
- Residual per-build overhead on delegated paths under load (suspect:
  graph exec-update cost scaling with graph size) — needs nsys on an
  idle box

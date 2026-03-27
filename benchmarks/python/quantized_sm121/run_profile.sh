#!/bin/bash
# SM121 Kernel Profiling — run on DGX Spark
#
# Usage: bash run_profile.sh [step]
#   step 1: nsys timeline (fast, ~30s) — shows kernel breakdown and pipeline gaps
#   step 2: ncu QMV (slow, ~2min)     — bandwidth utilization at M=1 (decode)
#   step 3: ncu QMM (slow, ~5min)     — compute utilization at M=1024 (prefill)
#   step 4: ncu dense baseline (~2min)
#   all:    run all steps
#
# Requirements: nsys and ncu must be in PATH (installed with CUDA toolkit)
#   which nsys && which ncu
#
# Reports go to $OUTDIR (~/.mlx-profile-results by default).
# View .nsys-rep files in Nsight Systems GUI.
# View .ncu-rep files in Nsight Compute GUI.
# CSV files are auto-exported for scripted analysis.

set -e
STEP="${1:-all}"

# --- Config ---
# Adjust these paths if your MLX env is different
MLX_REPO="${MLX_REPO:-/src/mlx}"
OUTDIR="${OUTDIR:-/results/profile}"
SCRIPT="$MLX_REPO/benchmarks/python/quantized_sm121/profile_kernels.py"

# Set up Python environment (adjust for your container/venv setup)
export PYTHONPATH="$MLX_REPO/python"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

mkdir -p "$OUTDIR"

echo "=== SM121 Kernel Profiling ==="
echo "MLX repo: $MLX_REPO"
echo "Output:   $OUTDIR"
echo ""

# ===== Step 1: nsys timeline =====
if [ "$STEP" = "1" ] || [ "$STEP" = "all" ]; then
  echo "=== Step 1: nsys timeline (shows kernel breakdown and gaps) ==="
  nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --output="$OUTDIR/qmm_timeline" \
    --force-overwrite=true \
    python3 "$SCRIPT" --mode all --iters 5 --warmup 2
  echo "Timeline: $OUTDIR/qmm_timeline.nsys-rep"

  # Export kernel summary to CSV
  nsys stats "$OUTDIR/qmm_timeline.nsys-rep" \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUTDIR/kernel_summary" 2>/dev/null || true
  echo "Kernel CSV: $OUTDIR/kernel_summary_cuda_gpu_kern_sum.csv"
  echo ""
fi

# ===== Step 2: ncu QMV (M=1 bandwidth — decode path) =====
if [ "$STEP" = "2" ] || [ "$STEP" = "all" ]; then
  echo "=== Step 2: ncu QMV (M=1, bandwidth utilization) ==="
  # Key metrics: DRAM throughput %, L2 hit rate, stall reasons
  ncu \
    --kernel-name "fp_qmv" \
    --launch-skip 3 --launch-count 10 \
    --metrics \
      dram__throughput.avg.pct_of_peak_sustained_elapsed,\
      dram__bytes_read.sum,\
      dram__bytes_write.sum,\
      lts__throughput.avg.pct_of_peak_sustained_elapsed,\
      sm__throughput.avg.pct_of_peak_sustained_elapsed,\
      smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
      smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
      smsp__warps_issue_stalled_wait_per_issue_active.ratio,\
      smsp__warps_issue_stalled_not_selected_per_issue_active.ratio,\
      smsp__warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
      sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
      l1tex__t_sector_hit_rate.pct,\
      l1tex__t_requests_pipe_lsu_mem_global.sum \
    --set full \
    --output "$OUTDIR/qmv_ncu" \
    --force-overwrite \
    python3 "$SCRIPT" --mode qmv --iters 5 --warmup 3
  echo "QMV report: $OUTDIR/qmv_ncu.ncu-rep"
  ncu --import "$OUTDIR/qmv_ncu.ncu-rep" --csv > "$OUTDIR/qmv_ncu.csv" 2>/dev/null || true
  echo "QMV CSV:    $OUTDIR/qmv_ncu.csv"
  echo ""
fi

# ===== Step 3: ncu QMM (M=1024, prefill — compute path) =====
if [ "$STEP" = "3" ] || [ "$STEP" = "all" ]; then
  echo "=== Step 3: ncu QMM (M=1024, compute utilization + tensor core usage) ==="
  # Key metrics: tensor core utilization, pipeline stalls, math throttle
  # Captures the 3-kernel pipeline: act quant → SF reformat → CUTLASS GEMM
  ncu \
    --launch-skip 9 --launch-count 30 \
    --metrics \
      dram__throughput.avg.pct_of_peak_sustained_elapsed,\
      sm__throughput.avg.pct_of_peak_sustained_elapsed,\
      sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
      smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
      smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
      smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
      smsp__warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
      sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
      sm__inst_executed_pipe_tensor.sum,\
      sm__inst_executed_pipe_fma.sum,\
      sm__sass_thread_inst_executed.sum \
    --set full \
    --output "$OUTDIR/qmm_ncu" \
    --force-overwrite \
    python3 "$SCRIPT" --mode qmm --iters 3 --warmup 3
  echo "QMM report: $OUTDIR/qmm_ncu.ncu-rep"
  ncu --import "$OUTDIR/qmm_ncu.ncu-rep" --csv > "$OUTDIR/qmm_ncu.csv" 2>/dev/null || true
  echo "QMM CSV:    $OUTDIR/qmm_ncu.csv"
  echo ""
fi

# ===== Step 4: ncu dense baseline =====
if [ "$STEP" = "4" ] || [ "$STEP" = "all" ]; then
  echo "=== Step 4: ncu dense matmul baseline ==="
  ncu \
    --launch-skip 3 --launch-count 10 \
    --set full \
    --output "$OUTDIR/dense_ncu" \
    --force-overwrite \
    python3 "$SCRIPT" --mode dense --iters 3 --warmup 3
  echo "Dense report: $OUTDIR/dense_ncu.ncu-rep"
  ncu --import "$OUTDIR/dense_ncu.ncu-rep" --csv > "$OUTDIR/dense_ncu.csv" 2>/dev/null || true
  echo "Dense CSV:    $OUTDIR/dense_ncu.csv"
  echo ""
fi

echo "=== Done ==="
echo "Reports in: $OUTDIR/"
ls -la "$OUTDIR/"

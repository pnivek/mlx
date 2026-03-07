#!/bin/bash
# SM121 Kernel Profiling — run on DGX Spark
# Usage: bash run_profile.sh [step]
#   step 1: nsys timeline (fast, ~30s) — shows kernel breakdown and gaps
#   step 2: ncu QMV (slow, ~2min) — bandwidth utilization at M=1
#   step 3: ncu QMM (slow, ~5min) — compute utilization at M=1024
#   step 4: ncu dense baseline (slow, ~2min)
#   all: run all steps

set -e
STEP="${1:-all}"

source ~/miniforge3/bin/activate mlx
export LD_LIBRARY_PATH=~/miniforge3/envs/mlx/lib:~/github/mlx/build:$LD_LIBRARY_PATH
export PYTHONPATH=~/github/mlx/python

SCRIPT=~/github/mlx/profile_kernels.py
OUTDIR=~/github/mlx/profile_results
mkdir -p "$OUTDIR"

# ===== Step 1: nsys timeline =====
if [ "$STEP" = "1" ] || [ "$STEP" = "all" ]; then
  echo "=== Step 1: nsys timeline profile ==="
  nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --output="$OUTDIR/qmm_timeline" \
    --force-overwrite=true \
    python3 "$SCRIPT" --mode all --iters 5 --warmup 2
  echo "Timeline saved to $OUTDIR/qmm_timeline.nsys-rep"
  echo ""

  # Export summary stats as text
  nsys stats "$OUTDIR/qmm_timeline.nsys-rep" \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUTDIR/kernel_summary"
  echo "Kernel summary CSV: $OUTDIR/kernel_summary_cuda_gpu_kern_sum.csv"
  echo ""
fi

# ===== Step 2: ncu QMV (M=1 bandwidth) =====
if [ "$STEP" = "2" ] || [ "$STEP" = "all" ]; then
  echo "=== Step 2: ncu QMV kernel profiling ==="
  # Target only QMV kernels (fp_qmv_single)
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
      sm__sass_thread_inst_executed_op_global_ld.sum,\
      sm__sass_thread_inst_executed_op_global_st.sum,\
      l1tex__t_sector_hit_rate.pct,\
      l1tex__t_requests_pipe_lsu_mem_global.sum \
    --set full \
    --output "$OUTDIR/qmv_ncu" \
    --force-overwrite \
    python3 "$SCRIPT" --mode qmv --iters 5 --warmup 3
  echo "QMV ncu report: $OUTDIR/qmv_ncu.ncu-rep"
  echo ""

  # Export as CSV for analysis
  ncu --import "$OUTDIR/qmv_ncu.ncu-rep" --csv > "$OUTDIR/qmv_ncu.csv" 2>/dev/null
  echo "QMV CSV: $OUTDIR/qmv_ncu.csv"
  echo ""
fi

# ===== Step 3: ncu QMM (M=1024 compute) =====
if [ "$STEP" = "3" ] || [ "$STEP" = "all" ]; then
  echo "=== Step 3: ncu QMM kernel profiling ==="
  # Profile the 3-kernel pipeline: act quant + sf reformat + CUTLASS GEMM
  # Skip warmup launches, capture main iterations
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
  echo "QMM ncu report: $OUTDIR/qmm_ncu.ncu-rep"
  echo ""

  ncu --import "$OUTDIR/qmm_ncu.ncu-rep" --csv > "$OUTDIR/qmm_ncu.csv" 2>/dev/null
  echo "QMM CSV: $OUTDIR/qmm_ncu.csv"
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
  echo "Dense ncu report: $OUTDIR/dense_ncu.ncu-rep"
  echo ""

  ncu --import "$OUTDIR/dense_ncu.ncu-rep" --csv > "$OUTDIR/dense_ncu.csv" 2>/dev/null
  echo "Dense CSV: $OUTDIR/dense_ncu.csv"
  echo ""
fi

echo "=== All profiling complete ==="
echo "Reports in: $OUTDIR/"
ls -la "$OUTDIR/"

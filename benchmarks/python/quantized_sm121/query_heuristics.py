"""Query nvMatmulHeuristics for optimal tile configs on SM120."""
from nvMatmulHeuristics import *

nvmmh = NvMatmulHeuristicsInterface(
    NvMatmulHeuristicsTarget.CUTLASS3,
    precision='HSH',
    flags=NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING
)
hw = nvmmh.createHardwareDescriptor()
nvmmh.setHardwarePredefinedGpu(hw, NvMatmulHeuristicsNvidiaGpu.RTX_5090)

layout = NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR
nvmmh.loadInternalDiscoverySet(layout, hw)

shapes = [
    (1, 4096, 4096),
    (1, 11008, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (1024, 11008, 4096),
    (1024, 4096, 11008),
    (2048, 4096, 4096),
    (4096, 4096, 4096),
]

for M, N, K in shapes:
    configs = nvmmh.get_with_mnk(M, N, K, layout, 8, hw)
    if configs:
        print(f"\nM={M}, N={N}, K={K} --- top 3:")
        for i, c in enumerate(sorted(configs, key=lambda d: d['runtime'])[:3]):
            print(f"  {i+1}. {c['kernel']}  ({c['runtime']*1000:.4f}ms)")
    else:
        print(f"\nM={M}, N={N}, K={K} --- no configs")

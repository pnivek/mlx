#!/usr/bin/env python3
"""MLX sm121 build + bench script. Run inside nvidia/cuda:13.2.0-devel-ubuntu24.04."""
import os, subprocess, sys, shutil, glob, datetime
from pathlib import Path

RESULTS = Path("/results")
LOG = RESULTS / "bench.log"
SRC = Path("/src/mlx")
BUILD = Path("/build/mlx")

RESULTS.mkdir(exist_ok=True)

def run(cmd, **kw):
    print(f">>> {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if r.returncode != 0:
        sys.exit(r.returncode)

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

log(f"=== MLX sm121 bench: start {datetime.datetime.now()} ===")

# Deps (idempotent)
run("apt-get update -qq && apt-get install -y --no-install-recommends "
    "cmake ninja-build build-essential git python3-dev python3-pip "
    "libopenblas-dev liblapack-dev liblapacke-dev libgfortran5 "
    "libnccl2 libnccl-dev libcudnn9-cuda-13 libcudnn9-dev-cuda-13 2>&1 | tail -5")
run("pip install --ignore-installed setuptools wheel 2>&1 | tail -2")
run("pip install numpy nanobind scikit-build-core 2>&1 | tail -2")

# Clone or pull source
if not (SRC / ".git").exists():
    log("=== Cloning pnivek/mlx@sm121 ===")
    SRC.parent.mkdir(parents=True, exist_ok=True)
    run(f"git clone --branch sm121 https://github.com/pnivek/mlx.git {SRC}")
else:
    log("=== Pulling latest sm121 ===")
    run(f"git -C {SRC} fetch origin sm121 && git -C {SRC} reset --hard origin/sm121")

commit = subprocess.check_output(f"git -C {SRC} rev-parse --short HEAD", shell=True).decode().strip()
log(f"Commit: {commit}")

# Configure and build (skip if cache is current)
commit_file = BUILD / ".built_commit"
BUILD.mkdir(parents=True, exist_ok=True)

if commit_file.exists() and commit_file.read_text().strip() == commit:
    log(f"=== Build cache hit for {commit}, skipping compile ===")
else:
    log(f"=== cmake configure (native arch via __nvcc_device_query) ===")
    # No explicit arch — CMakeLists.txt auto-detects via __nvcc_device_query
    # and promotes 121 -> 121a via UPGRADABLE_ARCHITECTURES
    run(f"""cmake {SRC} -B {BUILD} -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLX_BUILD_CUDA=ON \
        -DMLX_BUILD_PYTHON_BINDINGS=ON \
        -DMLX_BUILD_TESTS=OFF \
        -DMLX_BUILD_BENCHMARKS=OFF \
        -DMLX_BUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLAPACK_INCLUDE_DIRS=/usr/include/aarch64-linux-gnu 2>&1 | tee -a {LOG}""")

    log(f"=== cmake build ===")
    nproc = os.cpu_count() or 16
    run(f"cmake --build {BUILD} --parallel {nproc} 2>&1 | tee -a {LOG}")
    commit_file.write_text(commit)
    log(f"=== Build complete, cached as {commit} ===")

# Find core.so and set up pythonpath
matches = glob.glob(str(BUILD / "**/core.cpython*.so"), recursive=True)
if not matches:
    log("ERROR: core.cpython*.so not found in build tree")
    sys.exit(1)
core_so = matches[0]
log(f"core.so: {core_so}")

mlx_pkg = SRC / "python" / "mlx"
dest = mlx_pkg / Path(core_so).name
shutil.copy2(core_so, dest)

pythonpath = str(SRC / "python")
os.environ["PYTHONPATH"] = pythonpath
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
log(f"PYTHONPATH={pythonpath}")

log("=== smoke test ===")
run(f'python3 -c "import mlx.core as mx; print(\'MLX:\', mx.__version__); print(\'Device:\', mx.default_device())"')

log(f"=== benchmarks {datetime.datetime.now()} ===")
run(f"python3 {SRC}/benchmarks/python/quantized_sm121/bench_qmm.py 2>&1 | tee -a {LOG}")

log(f"=== Done {datetime.datetime.now()} ===")

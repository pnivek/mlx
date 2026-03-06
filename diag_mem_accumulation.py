"""Diagnose memory accumulation across benchmark trials on DGX Spark.

Run on sparkly/sparky:
  python3 diag_mem_accumulation.py --model ~/.local/share/exo/models/mlx-community--gpt-oss-120b-MXFP4-Q8

Tests 4 strategies:
  A: No clearing between trials (reproduce regression)
  B: mx.clear_cache() only
  C: mx.cuda.clear_graph_caches() only (clears graphs + trims pools)
  D: Both B + C
"""

import argparse
import time
import mlx.core as mx
import mlx_lm


def get_pool_memory():
    """Get CUDA memory pool reserved/used."""
    try:
        reserved, used = mx.cuda.get_pool_memory()
        return {"pool_reserved": reserved, "pool_used": used}
    except AttributeError:
        return {}


def read_meminfo():
    """Read /proc/meminfo MemAvailable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # Convert kB to bytes
    except Exception:
        return None


def snapshot(label):
    """Take a memory snapshot."""
    mx.synchronize()
    active = mx.get_active_memory()
    cache = mx.get_cache_memory()
    peak = mx.get_peak_memory()
    mem_avail = read_meminfo()
    pool = get_pool_memory()

    print(f"  [{label}]")
    print(f"    active={active / 1e9:.2f}GB  cache={cache / 1e9:.2f}GB  peak={peak / 1e9:.2f}GB")
    if mem_avail:
        print(f"    OS MemAvailable={mem_avail / 1e9:.2f}GB")
    if pool:
        for k, v in pool.items():
            if isinstance(v, int):
                print(f"    {k}={v / 1e9:.2f}GB")
            else:
                print(f"    {k}={v}")
    return {"active": active, "cache": cache, "peak": peak, "mem_avail": mem_avail}


def run_benchmark(model, tokenizer, prompt_tokens=512, gen_tokens=32):
    """Run a single generation trial and return throughput."""
    prompt = tokenizer.encode("Hello " * prompt_tokens, return_tensors="np")
    prompt = mx.array(prompt[0][:prompt_tokens])

    mx.synchronize()
    start = time.perf_counter()

    # Use generate_step for controlled generation
    tokens = []
    for token, _ in zip(
        mlx_lm.utils.generate_step(prompt, model),
        range(gen_tokens),
    ):
        tokens.append(token)

    mx.synchronize()
    elapsed = time.perf_counter() - start
    tps = gen_tokens / elapsed
    return tps, elapsed


def run_strategy(name, desc, model, tokenizer, num_trials, clear_fn, args):
    """Run a strategy with the given clearing function between trials."""
    print(f"\n{'='*60}")
    print(f"Strategy {name}: {desc}")
    print(f"{'='*60}")

    mx.reset_peak_memory()
    snapshot("before-trials")

    throughputs = []
    for trial in range(num_trials):
        print(f"\n  --- Trial {trial + 1} ---")
        tps, elapsed = run_benchmark(
            model, tokenizer, args.prompt_tokens, args.gen_tokens
        )
        throughputs.append(tps)
        print(f"    {tps:.1f} tok/s  ({elapsed:.2f}s)")
        snapshot(f"after-trial-{trial + 1}")

        if trial < num_trials - 1 and clear_fn:
            clear_fn()
            snapshot(f"after-clear-{trial + 1}")

    print(f"\n  Summary: {[f'{t:.1f}' for t in throughputs]} tok/s")
    if len(throughputs) > 1:
        ratio = throughputs[-1] / throughputs[0]
        print(f"  Last/First ratio: {ratio:.2f}x")
    return throughputs


def main():
    parser = argparse.ArgumentParser(description="Diagnose memory accumulation")
    parser.add_argument("--model", required=True, help="Path to MLX model")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument(
        "--strategies",
        default="A,B,C,D",
        help="Comma-separated strategies to run (default: A,B,C,D)",
    )
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = mlx_lm.load(args.model)
    mx.synchronize()
    snapshot("after-load")

    strategies = args.strategies.split(",")

    def clear_cache_only():
        mx.clear_cache()

    def clear_graphs_only():
        mx.cuda.clear_graph_caches()

    def clear_both():
        mx.clear_cache()
        mx.cuda.clear_graph_caches()

    strategy_map = {
        "A": ("No clearing", None),
        "B": ("mx.clear_cache() only", clear_cache_only),
        "C": ("mx.cuda.clear_graph_caches() only", clear_graphs_only),
        "D": ("clear_cache + clear_graph_caches", clear_both),
    }

    results = {}
    for s in strategies:
        s = s.strip().upper()
        if s not in strategy_map:
            print(f"Unknown strategy: {s}")
            continue
        desc, clear_fn = strategy_map[s]

        # Reset state before each strategy by clearing everything
        mx.clear_cache()
        mx.cuda.clear_graph_caches()
        mx.reset_peak_memory()
        mx.synchronize()

        results[s] = run_strategy(s, desc, model, tokenizer, args.num_trials, clear_fn, args)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    for s, tps_list in results.items():
        desc = strategy_map[s][0]
        ratio = tps_list[-1] / tps_list[0] if len(tps_list) > 1 else 1.0
        print(f"  {s} ({desc}): last/first={ratio:.2f}x  values={[f'{t:.1f}' for t in tps_list]}")


if __name__ == "__main__":
    main()

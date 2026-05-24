"""
Large-scale AMX BF16 GEMM benchmark (8K x 8K).

Compares:
  1. Pure C reference (single-thread, no AMX)
  2. AMX single-core
  3. AMX multi-core (OpenMP)

Theoretical peak for Xeon Gold 6526Y (32 cores, ~3.5 GHz, AMX-BF16):
  ~115 TFLOPS total system peak (fp32 accumulation counted as 2*M*N*K)
  ~3.6 TFLOPS per core

8K GEMM: 2 * 8192^3 = 1.1e12 FLOPs
  Single-core ideal: ~306 ms (3.6 TFLOPS)
  Full system ideal:  ~9.6 ms  (115 TFLOPS)

Usage:
    python3 bench_8k.py [--single-core] [--iter N]
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

# ────────────────────────────────────────────────────────────────
# AMX init
# ────────────────────────────────────────────────────────────────
def init_amx():
    libc = ctypes.CDLL("libc.so.6")
    bitmask = ctypes.c_uint64(0)
    ret = libc.syscall(158, 0x1022, ctypes.byref(bitmask))
    if ret == 0 and not (bitmask.value & (1 << 18)):
        libc.syscall(158, 0x1023, 18)


# ────────────────────────────────────────────────────────────────
# Compile optimized AMX library
# ────────────────────────────────────────────────────────────────
def compile_amx_lib():
    src = os.path.join(os.path.dirname(__file__), "amx_gemm_opt.c")
    lib = os.path.join(os.path.dirname(__file__), "libamx_gemm_opt.so")
    cmd = ("gcc -shared -fPIC -O3 -march=sapphirerapids -fopenmp "
           f"-o {lib} {src}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:", result.stderr)
        sys.exit(1)
    return lib


# ────────────────────────────────────────────────────────────────
# BF16 helpers
# ────────────────────────────────────────────────────────────────
def fp32_to_bf16(arr):
    return (arr.view(np.uint32) >> 16).astype(np.uint16)

def bf16_to_fp32(arr):
    return (arr.astype(np.uint32) << 16).view(np.float32)


# ────────────────────────────────────────────────────────────────
# Reference GEMM (pure NumPy, fp32)
# ────────────────────────────────────────────────────────────────
def ref_gemm_nt_numpy(A_bf16, B_bf16):
    A = bf16_to_fp32(A_bf16)
    B = bf16_to_fp32(B_bf16)
    return A @ B.T  # NT form: C[m][n] = sum_k A[m][k] * B[n][k]


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-core", action="store_true", help="Single core only")
    parser.add_argument("--iter", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=0, help="Num threads (0=all)")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K

    print(f"{'='*70}")
    print(f"AMX BF16 GEMM Benchmark: {M}x{N}x{K} (C[M][N] += A[M][K] * B[N][K])")
    print(f"{'='*70}")

    # CPU info
    cpu_count = os.cpu_count()
    print(f"CPU cores: {cpu_count}")

    # Init
    init_amx()
    lib_path = compile_amx_lib()
    lib = ctypes.CDLL(lib_path)

    # Function signatures
    lib.amx_bf16_gemm_large.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.amx_bf16_gemm_parallel.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int,
    ]

    # Generate test data (only first few rows to save memory for verification)
    np.random.seed(42)

    print(f"\nAllocating {M}x{N} fp32 + {M}x{K} bf16 + {N}x{K} bf16...")
    A_bf16 = fp32_to_bf16(np.random.randn(M, K).astype(np.float32))
    B_bf16 = fp32_to_bf16(np.random.randn(N, K).astype(np.float32))
    C_amx = np.zeros((M, N), dtype=np.float32)

    flops = 2.0 * M * N * K
    print(f"FLOPs: {flops:.2e} ({flops/1e9:.2f} GFLOPS)")

    # ─── Verify correctness on small subset first ───
    print("\n--- Correctness check (128x128 subset) ---")
    sub_M, sub_N, sub_K = 128, 128, 32
    A_sub = A_bf16[:sub_M, :sub_K].copy()
    B_sub = B_bf16[:sub_N, :sub_K].copy()
    C_sub_amx = np.zeros((sub_M, sub_N), dtype=np.float32)
    C_sub_ref = ref_gemm_nt_numpy(A_sub, B_sub)

    A_ptr = A_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    B_ptr = B_sub.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    C_ptr = C_sub_amx.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.amx_bf16_gemm_large(sub_M, sub_N, sub_K, A_ptr, sub_K, B_ptr, sub_K, C_ptr, sub_N)

    err = np.max(np.abs(C_sub_amx - C_sub_ref))
    print(f"  Max error: {err:.6f} {'OK' if err < 0.01 else 'FAIL'}")

    if err > 0.01:
        print("  Correctness check FAILED, aborting.")
        return

    # ─── Full-scale benchmark ───
    A_ptr = A_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    B_ptr = B_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    C_ptr = C_amx.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    print(f"\n--- Benchmark ({args.iter} warmups + {args.iter} iterations) ---")

    # Warmup single-core
    print("Warming up single-core...")
    for _ in range(args.iter):
        C_amx.fill(0.0)
        lib.amx_bf16_gemm_large(M, N, K, A_ptr, K, B_ptr, K, C_ptr, N)

    # Benchmark single-core
    times_sc = []
    print(f"Benchmarking single-core ({args.iter} iters)...")
    for i in range(args.iter):
        C_amx.fill(0.0)
        t0 = time.perf_counter()
        lib.amx_bf16_gemm_large(M, N, K, A_ptr, K, B_ptr, K, C_ptr, N)
        dt = time.perf_counter() - t0
        times_sc.append(dt)
        gflops = flops / dt / 1e9
        pct = gflops / 3600 * 100  # % of ~3.6 TFLOPS single-core peak
        print(f"  iter {i}: {dt*1000:.1f} ms  ({gflops:.1f} GFLOPS, {pct:.1f}% of single-core peak)")

    times_sc = np.array(times_sc)
    sc_mean = times_sc.mean()
    sc_gflops = flops / sc_mean / 1e9
    print(f"\nSingle-core: {sc_mean*1000:.1f} ms mean, {sc_gflops:.1f} GFLOPS")

    # Multi-core benchmark (if not single-core only)
    if not args.single_core:
        n_threads = args.threads if args.threads > 0 else cpu_count

        print(f"\nWarming up multi-core ({n_threads} threads)...")
        for _ in range(args.iter):
            C_amx.fill(0.0)
            lib.amx_bf16_gemm_parallel(M, N, K, A_ptr, K, B_ptr, K, C_ptr, N, n_threads)

        print(f"Benchmarking multi-core ({args.iter} iters)...")
        times_mc = []
        for i in range(args.iter):
            C_amx.fill(0.0)
            t0 = time.perf_counter()
            lib.amx_bf16_gemm_parallel(M, N, K, A_ptr, K, B_ptr, K, C_ptr, N, n_threads)
            dt = time.perf_counter() - t0
            times_mc.append(dt)
            gflops = flops / dt / 1e9
            pct = gflops / 115000 * 100  # % of ~115 TFLOPS system peak
            print(f"  iter {i}: {dt*1000:.1f} ms  ({gflops:.1f} GFLOPS, {pct:.2f}% of system peak)")

        times_mc = np.array(times_mc)
        mc_mean = times_mc.mean()
        mc_gflops = flops / mc_mean / 1e9
        mc_min = times_mc.min()
        mc_gflops_peak = flops / mc_min / 1e9

        print(f"\nMulti-core ({n_threads} threads): {mc_mean*1000:.1f} ms mean, "
              f"{mc_gflops:.1f} GFLOPS")
        print(f"Multi-core best:  {mc_min*1000:.1f} ms, {mc_gflops_peak:.1f} GFLOPS")

    # ─── Results ───
    print(f"\n{'='*70}")
    print(f"RESULTS ({M}x{N}x{K})")
    print(f"{'='*70}")
    print(f"FLOPs:                       {flops/1e9:.1f} G")
    print(f"Single-core:                 {sc_mean*1000:.1f} ms  ({sc_gflops:.1f} GFLOPS)")
    if not args.single_core:
        print(f"Multi-core ({n_threads} threads):    {mc_mean*1000:.1f} ms  ({mc_gflops:.1f} GFLOPS)")
        speedup = sc_mean / mc_mean
        print(f"Speedup:                     {speedup:.1f}x")
        print(f"Peak GFLOPS (best iter):     {mc_gflops_peak:.1f}")

    # Theoretical
    # TDPBF16PS: 16×16×32 = 8192 BF16 MACs/instruction
    # Throughput: ~16 cycles/instruction on Sapphire Rapids
    # Single-core: 3.5 GHz / 16 × 16384 FLOPs = ~3.6 TFLOPS
    # System: 32 cores → ~115 TFLOPS
    print(f"\nTheoretical peak for Xeon Gold 6526Y (32 cores, AMX-BF16):")
    print(f"  Per-core:  ~3.6 TFLOPS (TDPBF16PS @ 3.5 GHz, 16-cycle throughput)")
    print(f"  System:    ~115 TFLOPS (32 cores)")
    sc_eff = sc_gflops / 3600 * 100
    print(f"  Single-core achieved: {sc_gflops:.0f} GFLOPS ({sc_eff:.1f}% of peak)")
    if not args.single_core:
        sys_eff = mc_gflops / 115000 * 100
        print(f"  Multi-core achieved:  {mc_gflops:.0f} GFLOPS ({sys_eff:.1f}% of peak, {sc_mean/mc_mean:.1f}x speedup)")


if __name__ == "__main__":
    main()

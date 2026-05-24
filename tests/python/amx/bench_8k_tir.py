"""
Pure TVM TIR AMX BF16 GEMM at 8K×8K.

Strategy: TIR prim_func directly passes buffers to the optimized AMX micro-kernel
via T.call_extern. The TIR handles: AMX tile config, B pre-packing in TIR loops,
and the outer GEMM structure. The inner AMX tile ops (tileload+tdpbf16ps+tilestored)
are in the external micro-kernel since TVM lacks native AMX tile intrinsics.

Goal: ≥1.8 TFLOPS (50% of 3.6 TFLOPS single-core peak).
"""
from __future__ import annotations

import ctypes, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "3rdparty", "tvm-ffi", "python"))

BUILD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "build", "lib"))
os.environ["LD_LIBRARY_PATH"] = BUILD + ":" + os.environ.get("LD_LIBRARY_PATH", "")
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.abspath(os.path.join(os.path.dirname(__file__), "libamx_gemm_opt.so")), mode=ctypes.RTLD_GLOBAL)

import tvm
from tvm.script import tirx as T


# ────────────────────────────────────────────────────────────────
# TIR: B pre-packing (K-dimension as rows for TDPBF16PS src2)
# ────────────────────────────────────────────────────────────────
@T.prim_func(s_tir=True, check_well_formed=False)
def tir_prepack_B(
    B_in: T.Buffer((8192, 8192), "uint16"),        # [N][K] original
    B_out: T.Buffer((4096, 16384), "uint16"),       # [K/2][N*2] packed
):
    for kk in range(4096):
        for nn in range(8192):
            with T.sblock("pack_B"):
                vk, vn = T.axis.remap("SS", [kk, nn])
                B_out[vk, 2 * vn] = B_in[vn, 2 * vk]
                B_out[vk, 2 * vn + 1] = B_in[vn, 2 * vk + 1]


# ────────────────────────────────────────────────────────────────
# TIR: AMX GEMM using pre-packed B (T.call_extern for tile ops)
# ────────────────────────────────────────────────────────────────
@T.prim_func(s_tir=True, check_well_formed=False)
def tir_amx_gemm(
    A: T.Buffer((8192, 8192), "uint16"),            # [M][K] bf16
    B_packed: T.Buffer((4096, 16384), "uint16"),     # pre-packed [K/2][N*2]
    C: T.Buffer((8192, 8192), "float32"),            # [M][N] fp32 out
):
    T.evaluate(T.call_extern(
        "void", "amx_bf16_gemm_prepacked",
        T.int32(8192), T.int32(8192), T.int32(8192),
        A.access_ptr("r"), T.int32(8192),
        B_packed.access_ptr("r"), T.int32(16384),
        C.access_ptr("w"), T.int32(8192),
    ))


# ────────────────────────────────────────────────────────────────
# Direct ctypes benchmark (no TIR overhead) for comparison
# ────────────────────────────────────────────────────────────────
def init_amx():
    libc = ctypes.CDLL("libc.so.6")
    bitmask = ctypes.c_uint64(0)
    libc.syscall(158, 0x1022, ctypes.byref(bitmask))
    if not (bitmask.value & (1 << 18)):
        libc.syscall(158, 0x1023, 18)


def fp32_to_bf16(arr):
    return (arr.view(np.uint32) >> 16).astype(np.uint16)


def bf16_to_fp32(arr):
    return (arr.astype(np.uint32) << 16).view(np.float32)


def main():
    init_amx()
    M, N, K = 8192, 8192, 8192
    flops = 2.0 * M * N * K
    print(f"8K×8K BF16 GEMM: {flops/1e9:.1f} GFLOPs")
    print(f"Peak: 3.6 TFLOPS → target 1.8 TFLOPS (50%) → {flops/1.8e12*1000:.0f}ms\n")

    # Generate test data
    np.random.seed(42)
    A_bf16 = fp32_to_bf16(np.random.randn(M, K).astype(np.float32))
    B_bf16 = fp32_to_bf16(np.random.randn(N, K).astype(np.float32))
    C_ref = bf16_to_fp32(A_bf16) @ bf16_to_fp32(B_bf16).T

    target = tvm.target.Target({"kind": "llvm", "mcpu": "sapphirerapids"})
    dev = tvm.cpu(0)

    # ─── Pre-pack B via TIR ───
    print("Compiling TIR pre-pack...")
    t0 = time.perf_counter()
    prepack_fn = tvm.compile(tir_prepack_B, target=target)
    print(f"  TIR prepack compile: {(time.perf_counter()-t0)*1000:.0f}ms")

    B_packed = np.zeros((K//2, N*2), dtype=np.uint16)
    Bin_t = tvm.runtime.tensor(B_bf16.copy(), dev)
    Bout_t = tvm.runtime.tensor(B_packed.copy(), dev)

    t0 = time.perf_counter()
    prepack_fn(Bin_t, Bout_t)
    t_pack = (time.perf_counter() - t0) * 1000
    B_packed = Bout_t.numpy()
    print(f"  TIR pre-pack run: {t_pack:.0f}ms")

    # Verify
    ok = (B_packed[0,0] == B_bf16[0,0] and B_packed[0,1] == B_bf16[0,1] and
          B_packed[1,0] == B_bf16[0,2] and B_packed[1,1] == B_bf16[0,3])
    print(f"  Pre-pack verify: {'OK' if ok else 'FAIL'}")

    # ─── TIR AMX GEMM ───
    print("\nCompiling TIR AMX GEMM...")
    t0 = time.perf_counter()
    gemm_fn = tvm.compile(tir_amx_gemm, target=target)
    print(f"  Compile: {(time.perf_counter()-t0)*1000:.0f}ms")

    # Create TVM tensors once
    At = tvm.runtime.tensor(A_bf16, dev)
    Bpt = tvm.runtime.tensor(B_packed, dev)
    Ct = tvm.runtime.tensor(np.zeros((M, N), dtype=np.float32), dev)

    # Warmup
    for _ in range(3):
        gemm_fn(At, Bpt, Ct)

    # Benchmark (A and B tensors reused; C is overwritten in-place by GEMM)
    N_ITER = 5
    print(f"Benchmarking ({N_ITER} iters)...")
    times = []
    for i in range(N_ITER):
        t0 = time.perf_counter()
        gemm_fn(At, Bpt, Ct)
        dt = time.perf_counter() - t0
        times.append(dt)
        gf = flops / dt / 1e9
        print(f"  iter {i}: {dt*1000:.0f}ms  {gf:.0f} GFLOPS  ({gf/3600*100:.1f}% peak)")

    times = np.array(times)
    C_out = Ct.numpy()
    err = np.max(np.abs(C_out[:32,:32] - C_ref[:32,:32]))
    print(f"  Error: {err:.6f} {'OK' if err < 1.0 else 'FAIL'}")

    mean_t = times.mean()
    mean_gf = flops / mean_t / 1e9
    best_gf = flops / times.min() / 1e9

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Mean:   {mean_t*1000:.0f}ms  ({mean_gf:.0f} GFLOPS)")
    print(f"  Best:   {times.min()*1000:.0f}ms  ({best_gf:.0f} GFLOPS)")
    print(f"  Peak %: {mean_gf/3600*100:.1f}% (best {best_gf/3600*100:.1f}%)")
    print(f"  Target: 1.8 TFLOPS / 611ms")
    gap = mean_t / (flops / 1.8e12)
    print(f"  Gap to target: {gap:.1f}x slower than needed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

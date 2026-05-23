"""
MHA Attention Kernel using TVM Tensor IR with AMX optimization.

MHA: O = softmax(Q @ K^T / sqrt(d)) @ V

Two TVM TIR kernels:
  1. mha_baseline  — pure TIR loops (no AMX)
  2. mha_amx       — T.call_extern to AMX-accelerated BF16 GEMM

The AMX version calls amx_bf16_gemm_nt() from libamx_helpers.so.
Both kernels are compiled by TVM and run as native functions.

Usage:
    LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH python3 mha_tir.py

Parameters: seq_len=128, d_head=32, scale=1/sqrt(32)
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "3rdparty", "tvm-ffi", "python"))

BUILD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "build", "lib"))
os.environ["LD_LIBRARY_PATH"] = BUILD + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Pre-load TVM shared libraries
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)

import tvm
from tvm.script import tirx as T

# Parameters
SEQ_LEN = 128
D_HEAD = 32
SCALE = 1.0 / np.sqrt(D_HEAD).item()


# ────────────────────────────────────────────────────────────────
# AMX initialization via syscall (needed before any AMX instruction)
# ────────────────────────────────────────────────────────────────
def init_amx():
    SYS_arch_prctl = 158
    ARCH_GET_XCOMP_PERM = 0x1022
    ARCH_REQ_XCOMP_PERM = 0x1023
    XFEATURE_XTILEDATA = 18
    libc = ctypes.CDLL("libc.so.6")
    bitmask = ctypes.c_uint64(0)
    ret = libc.syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, ctypes.byref(bitmask))
    if ret == 0 and not (bitmask.value & (1 << XFEATURE_XTILEDATA)):
        ret = libc.syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)


# ────────────────────────────────────────────────────────────────
# TVM Tensor IR kernels
# ────────────────────────────────────────────────────────────────

@T.prim_func(s_tir=True)
def mha_baseline_tir(
    Q_bf16: T.Buffer((SEQ_LEN, D_HEAD), "uint16"),
    K_bf16: T.Buffer((SEQ_LEN, D_HEAD), "uint16"),
    V_bf16: T.Buffer((SEQ_LEN, D_HEAD), "uint16"),
    O_fp32: T.Buffer((SEQ_LEN, D_HEAD), "float32"),
):
    """Baseline MHA using pure TIR loops (no AMX)."""
    # Temporary buffers: S and P are [seq_len, seq_len] float32
    S = T.alloc_buffer((SEQ_LEN, SEQ_LEN), "float32")
    P = T.alloc_buffer((SEQ_LEN, SEQ_LEN), "float32")

    # Convert Q and K from bf16 (uint16) to fp32 via external helper
    Q_f32 = T.alloc_buffer((SEQ_LEN, D_HEAD), "float32")
    K_f32 = T.alloc_buffer((SEQ_LEN, D_HEAD), "float32")
    with T.sblock("bf16_to_fp32_Q"):
        T.evaluate(T.call_extern("void", "bf16_to_fp32_convert",
            Q_bf16.access_ptr("r"), Q_f32.access_ptr("w"),
            T.int32(SEQ_LEN * D_HEAD)))
    with T.sblock("bf16_to_fp32_K"):
        T.evaluate(T.call_extern("void", "bf16_to_fp32_convert",
            K_bf16.access_ptr("r"), K_f32.access_ptr("w"),
            T.int32(SEQ_LEN * D_HEAD)))

    # Step 1: S = Q @ K^T * scale  [SSR reduction over D_HEAD]
    for i, j, k in T.grid(SEQ_LEN, SEQ_LEN, D_HEAD):
        with T.sblock("S_compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                S[vi, vj] = T.float32(0)
            S[vi, vj] = S[vi, vj] + Q_f32[vi, vk] * K_f32[vj, vk] * T.float32(SCALE)

    # Step 2: P = softmax(S), row-wise
    mx_buf = T.alloc_buffer((1,), "float32")
    sm_buf = T.alloc_buffer((1,), "float32")
    for i in range(SEQ_LEN):
        # Find row max
        with T.sblock("softmax_max"):
            vi = T.axis.remap("S", [i])
            mx_buf[0] = S[vi, 0]
            for j in range(1, SEQ_LEN):
                mx_buf[0] = T.max(mx_buf[0], S[vi, j])
        # Compute exp and sum
        with T.sblock("softmax_exp"):
            vi = T.axis.remap("S", [i])
            sm_buf[0] = T.float32(0)
            for j in range(SEQ_LEN):
                P[vi, j] = T.exp(S[vi, j] - mx_buf[0])
                sm_buf[0] = sm_buf[0] + P[vi, j]
        # Normalize
        with T.sblock("softmax_norm"):
            vi = T.axis.remap("S", [i])
            for j in range(SEQ_LEN):
                P[vi, j] = P[vi, j] / sm_buf[0]

    # Convert V from bf16 to fp32
    V_f32 = T.alloc_buffer((SEQ_LEN, D_HEAD), "float32")
    with T.sblock("bf16_to_fp32_V"):
        T.evaluate(T.call_extern("void", "bf16_to_fp32_convert",
            V_bf16.access_ptr("r"), V_f32.access_ptr("w"),
            T.int32(SEQ_LEN * D_HEAD)))

    # Step 3: O = P @ V  [SSR reduction over SEQ_LEN]
    for i, j, k in T.grid(SEQ_LEN, D_HEAD, SEQ_LEN):
        with T.sblock("O_compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                O_fp32[vi, vj] = T.float32(0)
            O_fp32[vi, vj] = O_fp32[vi, vj] + P[vi, vk] * V_f32[vk, vj]


@T.prim_func(s_tir=True)
def mha_amx_tir(
    Q_bf16: T.Buffer((SEQ_LEN, D_HEAD), "uint16"),
    K_bf16: T.Buffer((SEQ_LEN, D_HEAD), "uint16"),
    V_bf16: T.Buffer((SEQ_LEN, D_HEAD), "uint16"),
    O_fp32: T.Buffer((SEQ_LEN, D_HEAD), "float32"),
):
    """AMX-accelerated MHA: T.call_extern for matmuls, TIR for softmax."""
    S = T.alloc_buffer((SEQ_LEN, SEQ_LEN), "float32")
    P_bf16 = T.alloc_buffer((SEQ_LEN, SEQ_LEN), "uint16")
    V_T = T.alloc_buffer((D_HEAD, SEQ_LEN), "uint16")

    # Step 1: S = Q @ K^T using AMX
    with T.sblock("S_amx"):
        T.evaluate(
            T.call_extern(
                "void",
                "amx_bf16_gemm_nt",
                T.int32(SEQ_LEN), T.int32(SEQ_LEN), T.int32(D_HEAD),
                Q_bf16.access_ptr("r"), T.int32(D_HEAD),
                K_bf16.access_ptr("r"), T.int32(D_HEAD),
                S.access_ptr("w"), T.int32(SEQ_LEN),
            )
        )

    # S *= scale
    for i, j in T.grid(SEQ_LEN, SEQ_LEN):
        with T.sblock("scale"):
            vi, vj = T.axis.remap("SS", [i, j])
            S[vi, vj] = S[vi, vj] * T.float32(SCALE)

    # Step 2: softmax(S) in-place
    mx_buf = T.alloc_buffer((1,), "float32")
    sm_buf = T.alloc_buffer((1,), "float32")
    for i in range(SEQ_LEN):
        with T.sblock("softmax_max"):
            vi = T.axis.remap("S", [i])
            mx_buf[0] = S[vi, 0]
            for j in range(1, SEQ_LEN):
                mx_buf[0] = T.max(mx_buf[0], S[vi, j])
        with T.sblock("softmax_exp"):
            vi = T.axis.remap("S", [i])
            sm_buf[0] = T.float32(0)
            for j in range(SEQ_LEN):
                S[vi, j] = T.exp(S[vi, j] - mx_buf[0])
                sm_buf[0] = sm_buf[0] + S[vi, j]
        with T.sblock("softmax_norm"):
            vi = T.axis.remap("S", [i])
            for j in range(SEQ_LEN):
                S[vi, j] = S[vi, j] / sm_buf[0]

    # Convert P (fp32 in S) to bf16 via external C helper
    with T.sblock("fp32_to_bf16"):
        T.evaluate(
            T.call_extern(
                "void",
                "fp32_to_bf16",
                S.access_ptr("r"),
                P_bf16.access_ptr("w"),
                T.int32(SEQ_LEN * SEQ_LEN),
            )
        )

    # Transpose V: V_T[d,s] = V[s,d]
    for d, s in T.grid(D_HEAD, SEQ_LEN):
        with T.sblock("transpose_V"):
            vd, vs = T.axis.remap("SS", [d, s])
            V_T[vd, vs] = V_bf16[vs, vd]

    # Step 3: O = P @ V using AMX
    with T.sblock("O_amx"):
        T.evaluate(
            T.call_extern(
                "void",
                "amx_bf16_gemm_nt",
                T.int32(SEQ_LEN), T.int32(D_HEAD), T.int32(SEQ_LEN),
                P_bf16.access_ptr("r"), T.int32(SEQ_LEN),
                V_T.access_ptr("r"), T.int32(SEQ_LEN),
                O_fp32.access_ptr("w"), T.int32(D_HEAD),
            )
        )


# ────────────────────────────────────────────────────────────────
# BF16 conversion helpers (NumPy)
# ────────────────────────────────────────────────────────────────
def bf16_to_fp32(arr_uint16):
    return (arr_uint16.astype(np.uint32) << 16).view(np.float32)


def fp32_to_bf16(arr):
    return (arr.view(np.uint32) >> 16).astype(np.uint16)


def mha_numpy(Q_bf16, K_bf16, V_bf16, scale):
    """Reference MHA in NumPy (fp32 math with bf16 inputs)."""
    Q = bf16_to_fp32(Q_bf16)
    K = bf16_to_fp32(K_bf16)
    V = bf16_to_fp32(V_bf16)
    S = Q @ K.T * scale
    S_max = S.max(axis=1, keepdims=True)
    P = np.exp(S - S_max) / np.exp(S - S_max).sum(axis=1, keepdims=True)
    return (P @ V).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Benchmark driver
# ────────────────────────────────────────────────────────────────
def run_benchmark():
    print(f"{'='*70}")
    print(f"MHA Attention Benchmark: TIR Baseline vs TIR+AMX")
    print(f"  seq_len={SEQ_LEN}, d_head={D_HEAD}, scale={SCALE:.6f}")
    print(f"{'='*70}")

    # Generate test data
    np.random.seed(42)
    Q_np = np.random.randn(SEQ_LEN, D_HEAD).astype(np.float32)
    K_np = np.random.randn(SEQ_LEN, D_HEAD).astype(np.float32)
    V_np = np.random.randn(SEQ_LEN, D_HEAD).astype(np.float32)

    Q_bf16 = fp32_to_bf16(Q_np)
    K_bf16 = fp32_to_bf16(K_np)
    V_bf16 = fp32_to_bf16(V_np)

    # Reference
    O_ref = mha_numpy(Q_bf16, K_bf16, V_bf16, SCALE)
    print(f"\nReference O: min={O_ref.min():.4f}, max={O_ref.max():.4f}, mean={O_ref.mean():.4f}")

    # Load AMX helper library (for linking at compile time)
    amx_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libamx_helpers.so"))
    amx_lib = ctypes.CDLL(amx_lib_path, mode=ctypes.RTLD_GLOBAL)
    init_amx()
    print("AMX hardware initialized")

    # ─── Compile TIR kernels ───
    target = tvm.target.Target({"kind": "llvm", "mcpu": "sapphirerapids"})

    print("\n--- Compiling TIR baseline kernel ---")
    t0 = time.perf_counter()
    baseline_mod = tvm.compile(mha_baseline_tir, target=target)
    t_comp_bl = (time.perf_counter() - t0) * 1000
    print(f"  Compiled in {t_comp_bl:.1f} ms")

    print("--- Compiling TIR AMX kernel ---")
    t0 = time.perf_counter()
    amx_mod = tvm.compile(mha_amx_tir, target=target)
    t_comp_amx = (time.perf_counter() - t0) * 1000
    print(f"  Compiled in {t_comp_amx:.1f} ms")

    # ─── Create TVM tensors ───
    dev = tvm.cpu(0)

    def make_tvm_tensor(np_arr):
        return tvm.runtime.tensor(np_arr.copy(), dev)

    # ─── Warmup ───
    N_WARMUP, N_ITER = 5, 50
    print(f"\n--- Warming up ({N_WARMUP} iters) ---")
    for _ in range(N_WARMUP):
        O_bl = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
        Q_t = make_tvm_tensor(Q_bf16); K_t = make_tvm_tensor(K_bf16)
        V_t = make_tvm_tensor(V_bf16); O_t = make_tvm_tensor(O_bl)
        baseline_mod(Q_t, K_t, V_t, O_t)

        O_ax = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
        Q_t = make_tvm_tensor(Q_bf16); K_t = make_tvm_tensor(K_bf16)
        V_t = make_tvm_tensor(V_bf16); O_t = make_tvm_tensor(O_ax)
        amx_mod(Q_t, K_t, V_t, O_t)

    # ─── Benchmark baseline TIR ───
    print(f"--- Benchmarking baseline TIR ({N_ITER} iters) ---")
    times_bl = []
    for _ in range(N_ITER):
        O_bl = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
        Q_t = make_tvm_tensor(Q_bf16); K_t = make_tvm_tensor(K_bf16)
        V_t = make_tvm_tensor(V_bf16); O_t = make_tvm_tensor(O_bl)
        t0 = time.perf_counter()
        baseline_mod(Q_t, K_t, V_t, O_t)
        times_bl.append((time.perf_counter() - t0) * 1000)
    times_bl = np.array(times_bl)

    # ─── Benchmark AMX TIR ───
    print(f"--- Benchmarking AMX TIR ({N_ITER} iters) ---")
    times_amx = []
    for _ in range(N_ITER):
        O_ax = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
        Q_t = make_tvm_tensor(Q_bf16); K_t = make_tvm_tensor(K_bf16)
        V_t = make_tvm_tensor(V_bf16); O_t = make_tvm_tensor(O_ax)
        t0 = time.perf_counter()
        amx_mod(Q_t, K_t, V_t, O_t)
        times_amx.append((time.perf_counter() - t0) * 1000)
    times_amx = np.array(times_amx)

    # ─── Verify correctness ───
    O_bl = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
    Q_t = make_tvm_tensor(Q_bf16); K_t = make_tvm_tensor(K_bf16)
    V_t = make_tvm_tensor(V_bf16); O_t = make_tvm_tensor(O_bl)
    baseline_mod(Q_t, K_t, V_t, O_t)
    O_baseline = O_t.numpy()

    O_ax = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
    Q_t = make_tvm_tensor(Q_bf16); K_t = make_tvm_tensor(K_bf16)
    V_t = make_tvm_tensor(V_bf16); O_t = make_tvm_tensor(O_ax)
    amx_mod(Q_t, K_t, V_t, O_t)
    O_amx = O_t.numpy()

    bl_err = np.max(np.abs(O_baseline - O_ref))
    amx_err = np.max(np.abs(O_amx - O_ref))

    # ─── Results ───
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"{'':<25} {'Baseline TIR':<25} {'AMX TIR':<25} {'Speedup'}")
    print(f"{'-'*70}")
    print(f"{'Mean time (ms)':<25} {times_bl.mean():<25.4f} {times_amx.mean():<25.4f} "
          f"{times_bl.mean()/times_amx.mean():<25.2f}x")
    print(f"{'Min time (ms)':<25} {times_bl.min():<25.4f} {times_amx.min():<25.4f} "
          f"{times_bl.min()/times_amx.min():<25.2f}x")
    print(f"{'Max abs error':<25} {bl_err:<25.6f} {amx_err:<25.6f}")
    print(f"{'Compile time (ms)':<25} {t_comp_bl:<25.1f} {t_comp_amx:<25.1f}")
    print(f"{'-'*70}")

    flops = 4 * SEQ_LEN * SEQ_LEN * D_HEAD
    bl_gflops = flops / (times_bl.mean() / 1000) / 1e9
    amx_gflops = flops / (times_amx.mean() / 1000) / 1e9
    print(f"{'GFLOPS':<25} {bl_gflops:<25.2f} {amx_gflops:<25.2f} "
          f"{amx_gflops/bl_gflops:<25.2f}x")

    speedup = times_bl.mean() / times_amx.mean()
    print(f"\n{'='*70}")
    print(f"AMX TIR is {speedup:.2f}x faster than baseline TIR!")
    print(f"\nTIR kernels show the correct TVM Tensor IR approach:")
    print(f"  - mha_baseline_tir: SSR loops for Q@K^T, softmax, and P@V")
    print(f"  - mha_amx_tir:      T.call_extern('amx_bf16_gemm_nt', ...) for matmuls")
    print(f"\nBoth kernels compile via tvm.compile() and link against libamx_helpers.so")

    return {
        "baseline_mean_ms": float(times_bl.mean()),
        "amx_mean_ms": float(times_amx.mean()),
        "speedup": float(speedup),
        "baseline_gflops": float(bl_gflops),
        "amx_gflops": float(amx_gflops),
        "baseline_error": float(bl_err),
        "amx_error": float(amx_err),
        "compile_time_baseline_ms": float(t_comp_bl),
        "compile_time_amx_ms": float(t_comp_amx),
    }


if __name__ == "__main__":
    results = run_benchmark()
    print(f"\nResults JSON: {json.dumps(results, indent=2)}")

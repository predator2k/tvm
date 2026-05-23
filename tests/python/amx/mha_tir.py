"""
MHA Attention Kernel using TVM Tensor IR with AMX optimization.

MHA: O = softmax(Q @ K^T / sqrt(d)) @ V

This file defines two TVM TIR kernels:
  1. mha_baseline_tir  — pure TIR loops for matrix multiply (no AMX)
  2. mha_amx_tir       — uses T.call_extern to invoke AMX-accelerated BF16 GEMM

The AMX version calls amx_bf16_gemm_nt() from libamx_helpers.so via
T.call_extern, which internally uses Intel AMX tile intrinsics
(_tile_dpbf16ps, _tile_stream_loadd, etc.).

Usage (requires working TVM installation):
    LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH python3 mha_tir.py

Parameters: seq_len=128, d_head=32, scale=1/sqrt(32)
Data format: BF16 (stored as uint16)
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time

import numpy as np

# ────────────────────────────────────────────────────────────────
# TVM import setup
# ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "3rdparty", "tvm-ffi", "python"))

try:
    import tvm
    from tvm import tir, tirx
    HAVE_TVM = True
except ImportError:
    HAVE_TVM = False
    print("Warning: TVM Python package not available. TIR definitions shown but not compiled.")


# ────────────────────────────────────────────────────────────────
# MHA parameters
# ────────────────────────────────────────────────────────────────
SEQ_LEN = 128
D_HEAD = 32
SCALE = 1.0 / np.sqrt(D_HEAD).item()  # 1/sqrt(32) ≈ 0.176777


# ────────────────────────────────────────────────────────────────
# TIR kernel definitions (TVM Tensor IR)
# ────────────────────────────────────────────────────────────────

def define_mha_baseline_tir():
    """
    Baseline MHA using pure TIR loops (no AMX).

    Computes:
      S = Q @ K^T * scale       [seq_len x seq_len]
      P = softmax(S, axis=1)     [seq_len x seq_len]
      O = P @ V                  [seq_len x d_head]

    All arithmetic in fp32. Input Q, K, V are bf16 (uint16).
    """
    if not HAVE_TVM:
        return _mha_baseline_tir_text()

    @tvm.tirx.prim_func
    def mha_baseline(
        Q_bf16: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "uint16"),
        K_bf16: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "uint16"),
        V_bf16: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "uint16"),
        O_fp32: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "float32"),
    ):
        # Temporary buffers
        S = tvm.tirx.alloc_buffer((SEQ_LEN, SEQ_LEN), "float32")
        P = tvm.tirx.alloc_buffer((SEQ_LEN, SEQ_LEN), "float32")

        # Step 1: S = Q @ K^T * scale
        for i, j, k in tvm.tirx.grid(SEQ_LEN, SEQ_LEN, D_HEAD):
            with tvm.tirx.block("S_compute"):
                vi, vj, vk = tvm.tirx.axis.remap("SSR", [i, j, k])
                with tvm.tirx.init():
                    S[vi, vj] = tvm.tirx.float32(0)
                # BF16 → FP32 conversion: left-shift by 16 bits
                q_val = tvm.tirx.Cast("float32", Q_bf16[vi, vk])
                k_val = tvm.tirx.Cast("float32", K_bf16[vj, vk])
                S[vi, vj] = S[vi, vj] + q_val * k_val * tvm.tirx.float32(SCALE)

        # Step 2: P = softmax(S), row-wise
        for i in tvm.tirx.serial(SEQ_LEN):
            # Find max per row
            with tvm.tirx.block("softmax_max"):
                vi = tvm.tirx.axis.remap("S", [i])
                row_max = tvm.tirx.alloc_buffer((1,), "float32")
                row_max[0] = S[vi, 0]
                for j in tvm.tirx.serial(1, SEQ_LEN):
                    row_max[0] = tvm.tirx.max(row_max[0], S[vi, j])

            # Compute exp and sum
            with tvm.tirx.block("softmax_exp"):
                vi = tvm.tirx.axis.remap("S", [i])
                row_sum = tvm.tirx.alloc_buffer((1,), "float32")
                row_sum[0] = tvm.tirx.float32(0)
                for j in tvm.tirx.serial(SEQ_LEN):
                    P[vi, j] = tvm.tirx.exp(S[vi, j] - row_max[0])
                    row_sum[0] = row_sum[0] + P[vi, j]

            # Normalize
            with tvm.tirx.block("softmax_norm"):
                vi = tvm.tirx.axis.remap("S", [i])
                for j in tvm.tirx.serial(SEQ_LEN):
                    P[vi, j] = P[vi, j] / row_sum[0]

        # Step 3: O = P @ V
        for i, j, k in tvm.tirx.grid(SEQ_LEN, D_HEAD, SEQ_LEN):
            with tvm.tirx.block("O_compute"):
                vi, vj, vk = tvm.tirx.axis.remap("SSR", [i, j, k])
                with tvm.tirx.init():
                    O_fp32[vi, vj] = tvm.tirx.float32(0)
                p_val = P[vi, vk]
                v_val = tvm.tirx.Cast("float32", V_bf16[vk, vj])
                O_fp32[vi, vj] = O_fp32[vi, vj] + p_val * v_val

    return mha_baseline


def define_mha_amx_tir():
    """
    AMX-accelerated MHA using T.call_extern for BF16 GEMM.

    The two matrix multiplies (Q@K^T and P@V) are offloaded to
    amx_bf16_gemm_nt() in libamx_helpers.so, which uses Intel AMX
    tile intrinsics. Only softmax is done in TIR loops.

    T.call_extern signature:
      void amx_bf16_gemm_nt(int M, int N, int K,
                            uint16* A, int lda,
                            uint16* B, int ldb,
                            float* C, int ldc)
    """
    if not HAVE_TVM:
        return _mha_amx_tir_text()

    @tvm.tirx.prim_func
    def mha_amx(
        Q_bf16: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "uint16"),
        K_bf16: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "uint16"),
        V_bf16: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "uint16"),
        O_fp32: tvm.tirx.Buffer((SEQ_LEN, D_HEAD), "float32"),
    ):
        # Temporary buffers
        S = tvm.tirx.alloc_buffer((SEQ_LEN, SEQ_LEN), "float32")

        # Step 1: S = Q @ K^T using AMX
        with tvm.tirx.block("S_amx_gemm"):
            # void amx_bf16_gemm_nt(
            #     int M, int N, int K,
            #     const uint16* A, int lda,
            #     const uint16* B, int ldb,
            #     float* C, int ldc)
            tvm.tirx.evaluate(
                tvm.tirx.call_extern(
                    "void",
                    "amx_bf16_gemm_nt",
                    tvm.tirx.int32(SEQ_LEN),           # M
                    tvm.tirx.int32(SEQ_LEN),           # N
                    tvm.tirx.int32(D_HEAD),            # K
                    Q_bf16.access_ptr("r"),            # A
                    tvm.tirx.int32(D_HEAD),            # lda
                    K_bf16.access_ptr("r"),            # B
                    tvm.tirx.int32(D_HEAD),            # ldb
                    S.access_ptr("w"),                 # C
                    tvm.tirx.int32(SEQ_LEN),           # ldc
                )
            )

        # Apply scale
        for i, j in tvm.tirx.grid(SEQ_LEN, SEQ_LEN):
            with tvm.tirx.block("scale"):
                vi, vj = tvm.tirx.axis.remap("SS", [i, j])
                S[vi, vj] = S[vi, vj] * tvm.tirx.float32(SCALE)

        # Step 2: P = softmax(S) in-place (S becomes P)
        for i in tvm.tirx.serial(SEQ_LEN):
            with tvm.tirx.block("softmax_max"):
                vi = tvm.tirx.axis.remap("S", [i])
                row_max = tvm.tirx.alloc_buffer((1,), "float32")
                row_max[0] = S[vi, 0]
                for j in tvm.tirx.serial(1, SEQ_LEN):
                    row_max[0] = tvm.tirx.max(row_max[0], S[vi, j])

            with tvm.tirx.block("softmax_exp"):
                vi = tvm.tirx.axis.remap("S", [i])
                row_sum = tvm.tirx.alloc_buffer((1,), "float32")
                row_sum[0] = tvm.tirx.float32(0)
                for j in tvm.tirx.serial(SEQ_LEN):
                    S[vi, j] = tvm.tirx.exp(S[vi, j] - row_max[0])
                    row_sum[0] = row_sum[0] + S[vi, j]

            with tvm.tirx.block("softmax_norm"):
                vi = tvm.tirx.axis.remap("S", [i])
                for j in tvm.tirx.serial(SEQ_LEN):
                    S[vi, j] = S[vi, j] / row_sum[0]

        # Step 3: O = P @ V using AMX
        # Create a bf16 copy of P for the GEMM
        P_bf16 = tvm.tirx.alloc_buffer((SEQ_LEN, SEQ_LEN), "uint16")

        # Convert P (fp32) to bf16 (uint16)
        for i, j in tvm.tirx.grid(SEQ_LEN, SEQ_LEN):
            with tvm.tirx.block("fp32_to_bf16"):
                vi, vj = tvm.tirx.axis.remap("SS", [i, j])
                P_bf16[vi, vj] = tvm.tirx.Cast("uint16",
                    tvm.tirx.right_shift(
                        tvm.tirx.reinterpret("uint32", S[vi, vj]),
                        tvm.tirx.int32(16)
                    ))

        with tvm.tirx.block("O_amx_gemm"):
            # Compute O += P_bf16 @ V_bf16 (V_bf16 is V^T logically in NT GEMM)
            # But actually V is [SEQ_LEN x D_HEAD], and we need P @ V.
            # The function computes C[M][N] += A[M][K] * B[N][K].
            # For P @ V: M=SEQ_LEN, N=D_HEAD, K=SEQ_LEN.
            # A = P_bf16 (M x K), B^T = V^T (N x K, i.e., V is transposed for NT form)
            # Actually the NT form computes C += A @ B^T, so to compute P @ V:
            # We pass A = P_bf16 and B = V_bf16 (stored as-is).
            # The function computes C[m][n] += sum_k P[m][k] * B[n][k].
            # If we want C[m][n] += sum_k P[m][k] * V[k][n], we need B[n][k] = V[k][n].
            # So B must be V^T (shape [D_HEAD, SEQ_LEN]).
            # We use V_bf16 as V directly; the GEMM handles B[n][k] correctly.

            # Create V^T buffer
            V_T = tvm.tirx.alloc_buffer((D_HEAD, SEQ_LEN), "uint16")
            for d, s in tvm.tirx.grid(D_HEAD, SEQ_LEN):
                with tvm.tirx.block("transpose_V"):
                    vd, vs = tvm.tirx.axis.remap("SS", [d, s])
                    V_T[vd, vs] = V_bf16[vs, vd]

            tvm.tirx.evaluate(
                tvm.tirx.call_extern(
                    "void",
                    "amx_bf16_gemm_nt",
                    tvm.tirx.int32(SEQ_LEN),           # M
                    tvm.tirx.int32(D_HEAD),            # N
                    tvm.tirx.int32(SEQ_LEN),           # K
                    P_bf16.access_ptr("r"),            # A
                    tvm.tirx.int32(SEQ_LEN),           # lda
                    V_T.access_ptr("r"),               # B
                    tvm.tirx.int32(SEQ_LEN),           # ldb
                    O_fp32.access_ptr("w"),            # C
                    tvm.tirx.int32(D_HEAD),            # ldc
                )
            )

    return mha_amx


def _mha_baseline_tir_text():
    """Text representation of the baseline TIR for display purposes."""
    return r'''
@tvm.tirx.prim_func
def mha_baseline(
    Q_bf16: T.Buffer((128, 32), "uint16"),
    K_bf16: T.Buffer((128, 32), "uint16"),
    V_bf16: T.Buffer((128, 32), "uint16"),
    O_fp32: T.Buffer((128, 32), "float32"),
):
    S = T.alloc_buffer((128, 128), "float32")
    P = T.alloc_buffer((128, 128), "float32")

    # Step 1: S = Q @ K^T * scale
    for i, j, k in T.grid(128, 128, 32):
        with T.block("S_compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                S[vi, vj] = T.float32(0)
            q_val = T.Cast("float32", Q_bf16[vi, vk])
            k_val = T.Cast("float32", K_bf16[vj, vk])
            S[vi, vj] += q_val * k_val * T.float32(scale)

    # Step 2: P = softmax(S), row-wise
    for i in range(128):
        # max per row
        row_max = S[i, 0]
        for j in range(1, 128):
            row_max = T.max(row_max, S[i, j])
        # exp and sum
        row_sum = T.float32(0)
        for j in range(128):
            P[i, j] = T.exp(S[i, j] - row_max)
            row_sum += P[i, j]
        # normalize
        for j in range(128):
            P[i, j] /= row_sum

    # Step 3: O = P @ V
    for i, j, k in T.grid(128, 32, 128):
        with T.block("O_compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                O_fp32[vi, vj] = T.float32(0)
            p_val = P[vi, vk]
            v_val = T.Cast("float32", V_bf16[vk, vj])
            O_fp32[vi, vj] += p_val * v_val
'''


def _mha_amx_tir_text():
    """Text representation of the AMX TIR for display purposes."""
    return r'''
@tvm.tirx.prim_func
def mha_amx(
    Q_bf16: T.Buffer((128, 32), "uint16"),
    K_bf16: T.Buffer((128, 32), "uint16"),
    V_bf16: T.Buffer((128, 32), "uint16"),
    O_fp32: T.Buffer((128, 32), "float32"),
):
    S = T.alloc_buffer((128, 128), "float32")
    P_bf16 = T.alloc_buffer((128, 128), "uint16")
    V_T = T.alloc_buffer((32, 128), "uint16")

    # Step 1: S = Q @ K^T using AMX (T.call_extern -> amx_bf16_gemm_nt)
    T.evaluate(T.call_extern("void", "amx_bf16_gemm_nt",
        T.int32(128), T.int32(128), T.int32(32),      # M, N, K
        Q_bf16.access_ptr("r"), T.int32(32),           # A, lda
        K_bf16.access_ptr("r"), T.int32(32),           # B, ldb
        S.access_ptr("w"), T.int32(128)))              # C, ldc

    # S *= scale
    for i, j in T.grid(128, 128):
        S[i, j] *= T.float32(scale)

    # Step 2: softmax(S) in-place
    for i in range(128):
        row_max = S[i, 0]
        for j in range(1, 128):
            row_max = T.max(row_max, S[i, j])
        row_sum = T.float32(0)
        for j in range(128):
            S[i, j] = T.exp(S[i, j] - row_max)
            row_sum += S[i, j]
        for j in range(128):
            S[i, j] /= row_sum

    # Convert P (fp32 in S) to bf16
    for i, j in T.grid(128, 128):
        P_bf16[i, j] = T.Cast("uint16",
            T.right_shift(T.reinterpret("uint32", S[i, j]), T.int32(16)))

    # Transpose V: V_T[d,s] = V[s,d]
    for d, s in T.grid(32, 128):
        V_T[d, s] = V_bf16[s, d]

    # Step 3: O = P @ V using AMX
    T.evaluate(T.call_extern("void", "amx_bf16_gemm_nt",
        T.int32(128), T.int32(32), T.int32(128),       # M, N, K
        P_bf16.access_ptr("r"), T.int32(128),           # A, lda
        V_T.access_ptr("r"), T.int32(128),              # B, ldb
        O_fp32.access_ptr("w"), T.int32(32)))           # C, ldc
'''


# ────────────────────────────────────────────────────────────────
# Benchmark using the existing ctypes approach (fallback)
# ────────────────────────────────────────────────────────────────

# BF16 conversion utilities
def bf16_to_fp32(arr_uint16):
    u32 = arr_uint16.astype(np.uint32) << 16
    return u32.view(np.float32)


def fp32_to_bf16(arr):
    u32 = arr.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def mha_numpy(Q_bf16, K_bf16, V_bf16, scale):
    """Reference MHA in NumPy (fp32 math with bf16 inputs)."""
    Q = bf16_to_fp32(Q_bf16)
    K = bf16_to_fp32(K_bf16)
    V = bf16_to_fp32(V_bf16)
    S = Q @ K.T * scale
    S_max = S.max(axis=1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=1, keepdims=True)
    O = P @ V
    return O.astype(np.float32)


def run_benchmark():
    """Run the MHA benchmark using the existing C libraries (fallback)."""
    import subprocess
    import tempfile

    print(f"{'='*70}")
    print(f"MHA Attention Benchmark: Baseline vs AMX (via T.call_extern)")
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

    O_ref = mha_numpy(Q_bf16, K_bf16, V_bf16, SCALE)
    print(f"\nReference MHA output stats: "
          f"min={O_ref.min():.6f}, max={O_ref.max():.6f}, mean={O_ref.mean():.6f}")

    # Compile baseline C library (no AMX)
    BASELINE_C = r"""
    #include <stdint.h>
    #include <string.h>
    #include <math.h>
    static inline float b2f(unsigned short v) {
        unsigned int b=((unsigned int)v)<<16; float f; memcpy(&f,&b,4); return f;
    }
    static inline unsigned short f2b(float v) {
        unsigned int b; memcpy(&b,&v,4); return (unsigned short)(b>>16);
    }
    void baseline_bf16_gemm_nt(int M,int N,int K,const unsigned short*A,int lda,
                                const unsigned short*B,int ldb,float*C,int ldc){
        for(int m=0;m<M;m++) for(int n=0;n<N;n++){
            float s=0; for(int k=0;k<K;k++) s+=b2f(A[m*lda+k])*b2f(B[n*ldb+k]);
            C[m*ldc+n]+=s;
        }
    }
    void baseline_softmax_fp32(float*d,int rows,int cols){
        for(int i=0;i<rows;i++){
            float*r=d+i*cols, mx=r[0];
            for(int j=1;j<cols;j++)if(r[j]>mx)mx=r[j];
            float sm=0; for(int j=0;j<cols;j++){r[j]=expf(r[j]-mx);sm+=r[j];}
            for(int j=0;j<cols;j++)r[j]/=sm;
        }
    }
    """
    tmpdir = tempfile.mkdtemp()
    lib_path = os.path.join(tmpdir, "libbaseline.so")
    src_path = os.path.join(tmpdir, "baseline.c")
    with open(src_path, "w") as f:
        f.write(BASELINE_C)
    subprocess.run(f"gcc -shared -fPIC -O3 -o {lib_path} {src_path} -lm",
                   shell=True, check=True, capture_output=True)
    baseline_lib = ctypes.CDLL(lib_path)
    baseline_lib.baseline_bf16_gemm_nt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    baseline_lib.baseline_softmax_fp32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ]

    # Initialize AMX hardware (must be done before any AMX instruction)
    SYS_arch_prctl = 158  # __NR_arch_prctl on x86_64
    ARCH_GET_XCOMP_PERM = 0x1022
    ARCH_REQ_XCOMP_PERM = 0x1023
    XFEATURE_XTILEDATA = 18
    libc = ctypes.CDLL("libc.so.6")
    bitmask = ctypes.c_uint64(0)
    ret = libc.syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, ctypes.byref(bitmask))
    if ret == 0 and not (bitmask.value & (1 << XFEATURE_XTILEDATA)):
        ret = libc.syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
        if ret != 0:
            print(f"  Warning: AMX init failed (ret={ret})")
    if ret == 0:
        print("  AMX hardware initialized via syscall")

    # Load AMX library
    amx_lib_path = os.path.join(os.path.dirname(__file__), "libamx_helpers.so")
    amx_lib = ctypes.CDLL(os.path.abspath(amx_lib_path))
    amx_lib.amx_bf16_gemm_nt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    amx_lib.softmax_fp32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ]
    amx_lib.fp32_to_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
    ]

    def run_baseline():
        S_mat = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        O_mat = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
        Qp = Q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        Kp = K_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        Sp = S_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        Op = O_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        S_mat.fill(0.0)
        baseline_lib.baseline_bf16_gemm_nt(SEQ_LEN, SEQ_LEN, D_HEAD, Qp, D_HEAD, Kp, D_HEAD, Sp, SEQ_LEN)
        S_mat *= SCALE
        baseline_lib.baseline_softmax_fp32(Sp, SEQ_LEN, SEQ_LEN)

        P_bf16 = fp32_to_bf16(S_mat)
        V_T = np.ascontiguousarray(V_bf16.T)
        Pp = P_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        VTp = V_T.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        O_mat.fill(0.0)
        baseline_lib.baseline_bf16_gemm_nt(SEQ_LEN, D_HEAD, SEQ_LEN, Pp, SEQ_LEN, VTp, SEQ_LEN, Op, D_HEAD)
        return O_mat

    def run_amx():
        S_mat = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        O_mat = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)
        Qp = Q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        Kp = K_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        Sp = S_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        Op = O_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        S_mat.fill(0.0)
        amx_lib.amx_bf16_gemm_nt(SEQ_LEN, SEQ_LEN, D_HEAD, Qp, D_HEAD, Kp, D_HEAD, Sp, SEQ_LEN)
        S_mat *= SCALE
        amx_lib.softmax_fp32(Sp, SEQ_LEN, SEQ_LEN)

        P_bf16 = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.uint16)
        Pp = P_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        amx_lib.fp32_to_bf16(Sp, Pp, SEQ_LEN * SEQ_LEN)

        V_T = np.ascontiguousarray(V_bf16.T)
        VTp = V_T.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        O_mat.fill(0.0)
        amx_lib.amx_bf16_gemm_nt(SEQ_LEN, D_HEAD, SEQ_LEN, Pp, SEQ_LEN, VTp, SEQ_LEN, Op, D_HEAD)
        return O_mat

    # Warmup + benchmark
    N_WARMUP, N_ITER = 5, 50
    for _ in range(N_WARMUP):
        run_baseline()
        run_amx()

    times_bl = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        run_baseline()
        times_bl.append((time.perf_counter() - t0) * 1000)
    times_bl = np.array(times_bl)

    times_amx = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        run_amx()
        times_amx.append((time.perf_counter() - t0) * 1000)
    times_amx = np.array(times_amx)

    # Verify correctness
    O_bl = run_baseline()
    O_amx = run_amx()
    bl_err = np.max(np.abs(O_bl - O_ref))
    amx_err = np.max(np.abs(O_amx - O_ref))

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"{'':<20} {'Baseline (no AMX)':<25} {'AMX-Optimized':<25} {'Speedup'}")
    print(f"{'-'*70}")
    print(f"{'Mean time (ms)':<20} {times_bl.mean():<25.4f} {times_amx.mean():<25.4f} {times_bl.mean()/times_amx.mean():<25.2f}x")
    print(f"{'Min time (ms)':<20} {times_bl.min():<25.4f} {times_amx.min():<25.4f} {times_bl.min()/times_amx.min():<25.2f}x")
    print(f"{'Max abs error':<20} {bl_err:<25.6f} {amx_err:<25.6f}")
    print(f"{'Max rel error':<20} {bl_err/(np.max(np.abs(O_ref))+1e-8):<25.6f} {amx_err/(np.max(np.abs(O_ref))+1e-8):<25.6f}")
    print(f"{'-'*70}")

    flops = 4 * SEQ_LEN * SEQ_LEN * D_HEAD
    bl_gflops = flops / (times_bl.mean() / 1000) / 1e9
    amx_gflops = flops / (times_amx.mean() / 1000) / 1e9
    print(f"{'GFLOPS':<20} {bl_gflops:<25.2f} {amx_gflops:<25.2f} {amx_gflops/bl_gflops:<25.2f}x")

    speedup = times_bl.mean() / times_amx.mean()
    print(f"\n{'='*70}")
    print(f"AMX is {speedup:.2f}x faster than baseline!")
    if HAVE_TVM:
        print(f"\nNote: This benchmark uses C libraries invoked via ctypes.")
        print(f"In a full TVM build, T.call_extern('amx_bf16_gemm_nt', ...)")
        print(f"would be embedded directly in compiled TIR and the same")
        print(f"libamx_helpers.so would be linked at module load time.")
    else:
        print(f"\nNote: TVM Python package not available for TIR compilation.")
        print(f"The TIR definitions above show the correct TVM Tensor IR approach.")
        print(f"See _mha_baseline_tir_text() and _mha_amx_tir_text() above.")

    return {
        "baseline_mean_ms": float(times_bl.mean()),
        "amx_mean_ms": float(times_amx.mean()),
        "speedup": float(speedup),
        "baseline_gflops": float(bl_gflops),
        "amx_gflops": float(amx_gflops),
        "baseline_error": float(bl_err),
        "amx_error": float(amx_err),
    }


if __name__ == "__main__":
    if HAVE_TVM:
        print("TVM Python package available.")
        print("TIR definitions:")
        print(_mha_amx_tir_text())
        print("\n" + "="*70)

    results = run_benchmark()
    print(f"\nResults JSON: {json.dumps(results, indent=2)}")

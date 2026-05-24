"""
Pure TIR AMX BF16 GEMM — N-blocked for cache efficiency.

Single-core GEMM: C[M][N] += A[M][K] * B[N][K] via AMX tile intrinsics.
All operations in pure Tensor IR (T.call_llvm_intrin).

Usage: LD_LIBRARY_PATH=build/lib python3 tir_amx_gemm.py [--M M] [--N N] [--K K]
"""
from __future__ import annotations

import ctypes, os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "3rdparty", "tvm-ffi", "python"))

BUILD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "build", "lib"))
os.environ["LD_LIBRARY_PATH"] = BUILD + ":" + os.environ.get("LD_LIBRARY_PATH", "")
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)

import tvm
from tvm.script import tirx as T


def make_tir_gemm(M, N, K, N_BLOCK=256):
    assert M % 16 == 0 and N % 16 == 0 and K % 32 == 0

    @T.prim_func(s_tir=True, check_well_formed=False)
    def amx_gemm(
        A: T.Buffer((M, K), "uint16"),
        B: T.Buffer((N, K), "uint16"),
        C: T.Buffer((M, N), "float32"),
        cfg: T.Buffer((64,), "uint8"),
        B_packed: T.Buffer((K // 2, N_BLOCK * 2), "uint16"),
    ):
        T.evaluate(T.call_llvm_intrin("void", "llvm.x86.ldtilecfg", cfg.access_ptr("r")))

        for nb in T.serial(N // N_BLOCK):
            # Pack B for this N-block
            for kk in T.serial(K // 2):
                for nn in T.serial(N_BLOCK):
                    n_idx = nb * N_BLOCK + nn
                    B_packed[kk, 2 * nn] = B[n_idx, 2 * kk]
                    B_packed[kk, 2 * nn + 1] = B[n_idx, 2 * kk + 1]

            # GEMM on this N-block
            for m in T.serial(M // 16):
                for n in T.serial(N_BLOCK // 16):
                    T.evaluate(T.call_llvm_intrin("void", "llvm.x86.tilezero", T.uint8(4)))
                    for kk in T.serial(K // 32):
                        T.evaluate(T.call_llvm_intrin("void", "llvm.x86.tileloadd64",
                            T.uint8(0),
                            A.access_ptr("r", offset=m*T.int32(16*K) + kk*T.int32(32)),
                            T.int64(K * 2)))
                        T.evaluate(T.call_llvm_intrin("void", "llvm.x86.tileloadd64",
                            T.uint8(1),
                            B_packed.access_ptr("r",
                                offset=kk*T.int32(16*N_BLOCK*2) + n*T.int32(32)),
                            T.int64(N_BLOCK * 4)))
                        T.evaluate(T.call_llvm_intrin("void", "llvm.x86.tdpbf16ps",
                            T.uint8(4), T.uint8(0), T.uint8(1)))
                    T.evaluate(T.call_llvm_intrin("void", "llvm.x86.tilestored64",
                        T.uint8(4),
                        C.access_ptr("w",
                            offset=m*T.int32(16*N) + nb*T.int32(N_BLOCK) + n*T.int32(16)),
                        T.int64(N * 4)))
    return amx_gemm


def init_amx():
    libc = ctypes.CDLL("libc.so.6")
    bm = ctypes.c_uint64(0)
    libc.syscall(158, 0x1022, ctypes.byref(bm))
    if not (bm.value & (1 << 18)): libc.syscall(158, 0x1023, 18)


def fp32_to_bf16(arr): return (arr.view(np.uint32)>>16).astype(np.uint16)
def bf16_to_fp32(arr): return (arr.astype(np.uint32)<<16).view(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=256); p.add_argument("--N", type=int, default=256)
    p.add_argument("--K", type=int, default=256); p.add_argument("--NB", type=int, default=256)
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()
    M,N,K,NB = args.M, args.N, args.K, args.NB
    init_amx()
    print(f"TIR AMX GEMM: M={M} N={N} K={K} NB={NB}")

    np.random.seed(42)
    A= fp32_to_bf16(np.random.randn(M,K).astype(np.float32))
    B= fp32_to_bf16(np.random.randn(N,K).astype(np.float32))
    C_ref = bf16_to_fp32(A) @ bf16_to_fp32(B).T

    cfg=np.zeros((64,),dtype=np.uint8); cfg[0]=1
    for i in range(8): cfg[16+2*i]=64; cfg[48+i]=16

    tgt=tvm.target.Target({"kind":"llvm","mcpu":"sapphirerapids"})
    t0=time.perf_counter()
    f=tvm.compile(make_tir_gemm(M,N,K,NB),target=tgt)
    print(f"  Compile: {(time.perf_counter()-t0)*1000:.0f}ms")

    dev=tvm.cpu(0)
    At=tvm.runtime.tensor(A.copy(),dev); Bt=tvm.runtime.tensor(B.copy(),dev)
    Ct=tvm.runtime.tensor(np.zeros((M,N),dtype=np.float32),dev)
    Cft=tvm.runtime.tensor(cfg.copy(),dev)
    Bpt=tvm.runtime.tensor(np.zeros((K//2,NB*2),dtype=np.uint16),dev)
    for _ in range(3): f(At,Bt,Ct,Cft,Bpt)

    ts=[]; flops=2.0*M*N*K
    for _ in range(args.iters):
        t0=time.perf_counter(); f(At,Bt,Ct,Cft,Bpt); ts.append(time.perf_counter()-t0)

    err=np.max(np.abs(Ct.numpy()-C_ref))
    gf=flops/np.mean(ts)/1e9
    print(f"  Err: {err:.6f} | Time: {np.mean(ts)*1000:.1f}ms | GFLOPS: {gf:.0f} ({gf/36:.1f}% peak)")


if __name__=="__main__": main()

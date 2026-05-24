"""
Final benchmark for pure-TIR MHA on AMX (single core).

Selects the best variant per (SEQ, D) and reports steady-state GFLOPS.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "3rdparty", "tvm-ffi", "python"))
_BUILD = os.path.abspath(os.path.join(_DIR, "..", "..", "..", "build", "lib"))
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)

import tvm
from mha_fa_tir import make_pack_K, make_pack_V, init_amx, fp32_to_bf16, mha_numpy
from mha_nf_tir import (
    make_mha_nonfused_one,
    make_mha_nf_fused_max,
    make_mha_mblock_fused,
)

init_amx()
target = tvm.target.Target({"kind": "llvm", "mcpu": "sapphirerapids"})
dev = tvm.cpu(0)


def run(SEQ, D, variant_factory, N_WARM=15, N_ITER=120):
    scale = 1.0 / math.sqrt(D)
    pack_K_fn = tvm.compile(make_pack_K(SEQ, D), target=target)
    pack_V_fn = tvm.compile(make_pack_V(SEQ, D), target=target)
    mha_fn = tvm.compile(variant_factory(SEQ, D), target=target)

    np.random.seed(42)
    Q = fp32_to_bf16(np.random.randn(SEQ, D).astype(np.float32) * 0.05)
    K = fp32_to_bf16(np.random.randn(SEQ, D).astype(np.float32) * 0.05)
    V = fp32_to_bf16(np.random.randn(SEQ, D).astype(np.float32) * 0.05)
    O_ref = mha_numpy(Q, K, V, scale)
    Qt = tvm.runtime.tensor(Q, dev)
    Kt = tvm.runtime.tensor(K, dev)
    Vt = tvm.runtime.tensor(V, dev)
    Kpt = tvm.runtime.tensor(np.zeros((D // 2, SEQ * 2), dtype=np.uint16), dev)
    Vpt = tvm.runtime.tensor(np.zeros((SEQ // 2, D * 2), dtype=np.uint16), dev)
    Ot = tvm.runtime.tensor(np.zeros((SEQ, D), dtype=np.float32), dev)
    pack_K_fn(Kt, Kpt)
    pack_V_fn(Vt, Vpt)

    mha_fn(Qt, Kpt, Vpt, Ot)
    rel = float(np.max(np.abs(Ot.numpy() - O_ref)) / (np.max(np.abs(O_ref)) + 1e-8))

    for _ in range(N_WARM):
        Ot.copyfrom(np.zeros((SEQ, D), dtype=np.float32))
        mha_fn(Qt, Kpt, Vpt, Ot)
    times = []
    for _ in range(N_ITER):
        Ot.copyfrom(np.zeros((SEQ, D), dtype=np.float32))
        t0 = time.perf_counter()
        mha_fn(Qt, Kpt, Vpt, Ot)
        times.append(time.perf_counter() - t0)
    return min(times), np.median(times), rel


def main():
    # (SEQ, D, variant_name, variant_factory)
    configs = [
        ("SEQ=512  D=1024", 512, 1024, "nf_fmax",
            make_mha_nf_fused_max),
        ("SEQ=512  D= 768", 512, 768, "nf_fmax",
            make_mha_nf_fused_max),
        ("SEQ=1024 D= 512", 1024, 512, "nf_fmax",
            make_mha_nf_fused_max),
        ("SEQ=1024 D= 256", 1024, 256, "mblock_Mq64",
            lambda S, D: make_mha_mblock_fused(S, D, 64)),
        ("SEQ=1024 D= 128", 1024, 128, "mblock_Mq64",
            lambda S, D: make_mha_mblock_fused(S, D, 64)),
        ("SEQ=2048 D= 128", 2048, 128, "mblock_Mq32",
            lambda S, D: make_mha_mblock_fused(S, D, 32)),
        ("SEQ=2048 D= 256", 2048, 256, "mblock_Mq32",
            lambda S, D: make_mha_mblock_fused(S, D, 32)),
    ]
    print("=" * 80)
    print("Pure-TVM-TIR MHA on AMX BF16 (Xeon Gold 6526Y, single core)")
    print("=" * 80)
    print(f"{'Config':<22} {'Variant':<14} {'Best':>9} {'Median':>9} {'GFLOPS':>8} {'rel err':>8}")
    print("-" * 80)
    for label, SEQ, D, name, factory in configs:
        t_best, t_med, rel = run(SEQ, D, factory)
        flops = 4.0 * SEQ * SEQ * D
        gf = flops / t_best / 1e9
        print(f"{label:<22} {name:<14} {t_best*1e6:>7.0f}us {t_med*1e6:>7.0f}us "
              f"{gf:>8.0f} {rel:>7.2%}")
    print("-" * 80)
    print("Notes: nf_fmax = non-fused GEMMs with pass-1 max fused into GEMM-A;")
    print("       mblock_MqN = per-M-block fused softmax with row-block size N.")
    print("       All variants are pure TIR using LLVM AMX/AVX-512 intrinsics.")


if __name__ == "__main__":
    main()

"""
Attempt to use TVM MetaSchedule (auto-scheduler) for AMX MHA.

Status (recorded so we don't repeat the attempt):
  - tvm.s_tir.meta_schedule.tir_integration.tune_tir works for vanilla TIR
    (e.g. fp32 matmul → ~5 GFLOPS on Xeon Gold 6526Y after 32 trials).
  - There is NO AMX BF16 tensor intrinsic registered in TVM.
    tvm/s_tir/tensor_intrin/x86.py only has VNNI int8 (vpdpbusd).
  - MS schedule rule MultiLevelTilingWithIntrin needs a registered intrinsic
    and a "decomposable" pattern (init / update / finalize). AMX TDPBF16PS
    needs persistent tile-state across the K loop, which a per-block tensor
    intrinsic cannot express — each invocation would have to tile_loadd C,
    dpbf16ps, tile_stored back, paying ~16x extra memory traffic.
  - tirx tile primitives (Gemm, Exp, Sum, Max, …) only have CUDA / Trainium
    backends; no x86 lowerings.

Conclusion: MS / autoscheduler cannot beat the hand-written AMX kernels in
mha_nf_tir.py / mha_fa_tir.py for our 1+ TFLOPS goal. Would need to land an
AMX bf16 tensor intrinsic upstream (significant work, not appropriate here).

This file records the sanity test for future reference.
"""
from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "3rdparty", "tvm-ffi", "python"))
_BUILD = os.path.abspath(os.path.join(_DIR, "..", "..", "..", "build", "lib"))
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)

import tvm
from tvm.s_tir.schedule import Schedule
from tvm.script import tirx as T
from tvm.target import Target


@T.prim_func(s_tir=True)
def matmul_fp32(a: T.handle, b: T.handle, c: T.handle):
    A = T.match_buffer(a, [512, 512], "float32")
    B = T.match_buffer(b, [512, 512], "float32")
    C = T.match_buffer(c, [512, 512], "float32")
    for i, j, k in T.grid(512, 512, 512):
        with T.sblock("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def main():
    target = Target({"kind": "llvm", "mcpu": "sapphirerapids", "num-cores": 1})

    np.random.seed(0)
    A = np.random.randn(512, 512).astype(np.float32)
    B = np.random.randn(512, 512).astype(np.float32)
    At = tvm.runtime.tensor(A, tvm.cpu(0))
    Bt = tvm.runtime.tensor(B, tvm.cpu(0))
    Ct = tvm.runtime.tensor(np.zeros((512, 512), dtype=np.float32), tvm.cpu(0))

    # Baseline: no schedule
    mod_base = tvm.compile(matmul_fp32, target=target)
    for _ in range(3): mod_base(At, Bt, Ct)
    times = [time.perf_counter() for _ in range(15)]
    for i in range(15):
        t0 = time.perf_counter(); mod_base(At, Bt, Ct)
        times[i] = time.perf_counter() - t0
    dt_base = min(times)
    print(f"baseline fp32 matmul (no schedule):  {dt_base*1e3:.1f}ms  "
          f"{2*512**3/dt_base/1e9:.1f} GFLOPS")

    # Manual schedule: tile + vectorize + unroll (poor man's auto-schedule)
    sch = Schedule(matmul_fp32)
    block = sch.get_sblock("update", "main")
    i, j, k = sch.get_loops(block)
    i_o, i_i = sch.split(i, factors=[None, 16])
    j_o, j_i = sch.split(j, factors=[None, 16])
    k_o, k_i = sch.split(k, factors=[None, 16])
    sch.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
    sch.vectorize(j_i)
    sch.unroll(i_i)
    sch.unroll(k_i)
    mod_tuned = tvm.compile(sch.mod, target=target)
    for _ in range(3): mod_tuned(At, Bt, Ct)
    times = [0.0] * 15
    for i in range(15):
        t0 = time.perf_counter(); mod_tuned(At, Bt, Ct)
        times[i] = time.perf_counter() - t0
    dt = min(times)
    print(f"manual tile+vec schedule:            {dt*1e3:.1f}ms  "
          f"{2*512**3/dt/1e9:.1f} GFLOPS")
    print(f"  speedup over baseline: {dt_base/dt:.2f}x")
    print()
    print("For context, our hand-written AMX BF16 kernel does ~1750 GFLOPS at")
    print("the same shape — auto-scheduler without AMX intrinsic registration")
    print("cannot match that.")


if __name__ == "__main__":
    main()

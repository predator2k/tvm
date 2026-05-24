"""
Pure TVM TIR primitives for AMX BF16 GEMM and MHA.

Everything below is expressed via TIR + LLVM intrinsics — no external C.
Uses T.call_llvm_intrin to emit AMX tile ops directly (tileloadd64, tdpbf16ps,
tilestored64, ldtilecfg, tilezero) and AVX-512 BF16 conversion intrinsics.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys

# TVM path setup (caller may have already set this)
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "3rdparty", "tvm-ffi", "python"))
_BUILD = os.path.abspath(os.path.join(_DIR, "..", "..", "..", "build", "lib"))

# Pre-load TVM shared libs
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)

import tvm
from tvm.script import tirx as T


# ─────────────────────────────────────────────────────────────────────
# AMX helpers — generate calls to LLVM AMX intrinsics from inside TIR
# ─────────────────────────────────────────────────────────────────────
def _ldtilecfg(cfg_ptr):
    return T.call_llvm_intrin(
        T.llvm_lookup_intrinsic_id("llvm.x86.ldtilecfg"),
        cfg_ptr, dtype="")


def _tilezero(t):
    return T.call_llvm_intrin(
        T.llvm_lookup_intrinsic_id("llvm.x86.tilezero"),
        T.uint8(t), dtype="")


def _tileloadd(t, ptr, stride_bytes):
    return T.call_llvm_intrin(
        T.llvm_lookup_intrinsic_id("llvm.x86.tileloadd64"),
        T.uint8(t), ptr, T.int64(stride_bytes), dtype="")


def _tilestored(t, ptr, stride_bytes):
    return T.call_llvm_intrin(
        T.llvm_lookup_intrinsic_id("llvm.x86.tilestored64"),
        T.uint8(t), ptr, T.int64(stride_bytes), dtype="")


def _tdpbf16ps(dst, a, b):
    return T.call_llvm_intrin(
        T.llvm_lookup_intrinsic_id("llvm.x86.tdpbf16ps"),
        T.uint8(dst), T.uint8(a), T.uint8(b), dtype="")


def _i64(x):
    return T.cast(x, "int64")


def make_pack_B_func(N, K):
    """Build a TIR prim_func that packs B[N, K] bf16 (NT form) into
       B_packed[K/2, N*2] using a sequential nested loop."""
    K_pairs = K // 2

    @T.prim_func(s_tir=True, check_well_formed=False)
    def pack_B(
        B: T.Buffer((N, K), "uint16"),
        B_packed: T.Buffer((K_pairs, N * 2), "uint16"),
    ):
        for kp in T.serial(K_pairs):
            for n in T.serial(N):
                with T.sblock("pack"):
                    vkp, vn = T.axis.remap("SS", [kp, n])
                    B_packed[vkp, 2 * vn]     = B[vn, 2 * vkp]
                    B_packed[vkp, 2 * vn + 1] = B[vn, 2 * vkp + 1]
    return pack_B


def make_gemm_2x2_func(M, N, K):
    """Build a pure-TIR AMX GEMM 2x2 micro-kernel for fixed M, N, K.
       Requires M % 32 == 0, N % 32 == 0, K % 32 == 0.

       C[M, N] (fp32) = A[M, K] (bf16) @ B_packed (B[N, K] pre-packed bf16).
       B_packed shape: [K/2, N*2] uint16.

       Tile assignment:
         tiles 0,1 = A_top, A_bot  (16x32 bf16)
         tiles 2,3 = B_left, B_right (16x16 bf16 paired)
         tiles 4,5,6,7 = C00, C01, C10, C11 (16x16 fp32 acc)
    """
    assert M % 32 == 0 and N % 32 == 0 and K % 32 == 0

    K_pairs = K // 2
    N_packed = N * 2   # uint16 cols per row of B_packed
    a_row_bytes = K * 2
    b_row_bytes = N_packed * 2
    c_row_bytes = N * 4

    Mo = M // 32
    No = N // 32
    Ko = K // 32

    @T.prim_func(s_tir=True, check_well_formed=False)
    def amx_gemm(
        A: T.Buffer((M, K), "uint16"),
        B_packed: T.Buffer((K_pairs, N_packed), "uint16"),
        C: T.Buffer((M, N), "float32"),
    ):
        cfg = T.alloc_buffer((64,), "uint8")
        # palette_id=1, all 8 tiles 16 rows × 64 bytes
        cfg[0] = T.uint8(1)
        for i in range(1, 16):
            cfg[i] = T.uint8(0)
        for i in range(8):
            cfg[16 + 2 * i]     = T.uint8(64)
            cfg[16 + 2 * i + 1] = T.uint8(0)
        for i in range(16):
            cfg[32 + i] = T.uint8(0)
        for i in range(8):
            cfg[48 + i] = T.uint8(16)
        for i in range(8):
            cfg[56 + i] = T.uint8(0)
        T.evaluate(_ldtilecfg(cfg.access_ptr("r")))

        for mo in T.serial(Mo):
            for no in T.serial(No):
                # zero accumulators
                T.evaluate(_tilezero(4))
                T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6))
                T.evaluate(_tilezero(7))

                for ko in T.serial(Ko):
                    # offsets in elements (uint16 for A,B_packed; fp32 for C)
                    a_top_off = _i64((mo * 32) * K + ko * 32)
                    a_bot_off = _i64((mo * 32 + 16) * K + ko * 32)
                    b_left_off  = _i64((ko * 16) * N_packed + (no * 32) * 2)
                    b_right_off = _i64((ko * 16) * N_packed + (no * 32 + 16) * 2)

                    T.evaluate(_tileloadd(0, A.access_ptr("r", offset=a_top_off), a_row_bytes))
                    T.evaluate(_tileloadd(1, A.access_ptr("r", offset=a_bot_off), a_row_bytes))
                    T.evaluate(_tileloadd(2, B_packed.access_ptr("r", offset=b_left_off),  b_row_bytes))
                    T.evaluate(_tileloadd(3, B_packed.access_ptr("r", offset=b_right_off), b_row_bytes))

                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))

                c_tl = _i64((mo * 32) * N + no * 32)
                c_tr = _i64((mo * 32) * N + no * 32 + 16)
                c_bl = _i64((mo * 32 + 16) * N + no * 32)
                c_br = _i64((mo * 32 + 16) * N + no * 32 + 16)
                T.evaluate(_tilestored(4, C.access_ptr("w", offset=c_tl), c_row_bytes))
                T.evaluate(_tilestored(5, C.access_ptr("w", offset=c_tr), c_row_bytes))
                T.evaluate(_tilestored(6, C.access_ptr("w", offset=c_bl), c_row_bytes))
                T.evaluate(_tilestored(7, C.access_ptr("w", offset=c_br), c_row_bytes))
    return amx_gemm

"""
Pure-TVM-TIR non-fused MHA (3 stages: Q@K^T, softmax+scale, P@V).

All AMX tile ops and AVX-512 BF16 conversions are emitted via T.call_llvm_intrin.
No C/C++ helpers anywhere.

Layout:
  Q          [SEQ, D]   bf16 (uint16)
  K_packed   [D/2,  SEQ*2]   bf16  — pre-packed for Q*K^T (B in GEMM-NT view)
  V_packed   [SEQ/2, D*2]    bf16  — pre-packed for P*V  (B in GEMM-NT view)
  O          [SEQ, D]   fp32
"""
from __future__ import annotations

import ctypes
import math
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", "3rdparty", "tvm-ffi", "python"))
_BUILD = os.path.abspath(os.path.join(_DIR, "..", "..", "..", "build", "lib"))
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(_BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)

import tvm
from tvm.script import tirx as T

from mha_fa_tir import (_ldtilecfg, _tilezero, _tileloadd, _tilestored, _tdpbf16ps,
                         _vec_round, _vec_reduce_fmax, _vec_reduce_fadd,
                         _vec_cvt32_to_bf16x32, _vec_exp,
                         make_pack_K, make_pack_V,
                         init_amx, fp32_to_bf16, bf16_to_fp32, mha_numpy)


def make_gemm_2x2(M, N, K, n_packed_u16):
    """Pure-TIR AMX 2x2 GEMM. C = A @ B (NT form). B is pre-packed.
       n_packed_u16 = row stride of B_packed in uint16 elements."""
    assert M % 32 == 0 and N % 32 == 0 and K % 32 == 0
    K_pairs = K // 2
    a_row_bytes = K * 2
    b_row_bytes = n_packed_u16 * 2
    c_row_bytes = N * 4
    Mo = M // 32; No = N // 32; Ko = K // 32

    @T.prim_func(s_tir=True, check_well_formed=False)
    def gemm(
        A: T.Buffer((M, K), "uint16"),
        B_packed: T.Buffer((K_pairs, n_packed_u16), "uint16"),
        C: T.Buffer((M, N), "float32"),
    ):
        cfg = T.alloc_buffer((64,), "uint8")
        cfg[0] = T.uint8(1)
        for i in range(1, 16): cfg[i] = T.uint8(0)
        for i in range(8):
            cfg[16 + 2 * i] = T.uint8(64); cfg[16 + 2 * i + 1] = T.uint8(0)
        for i in range(16): cfg[32 + i] = T.uint8(0)
        for i in range(8): cfg[48 + i] = T.uint8(16)
        for i in range(8): cfg[56 + i] = T.uint8(0)
        T.evaluate(_ldtilecfg(cfg.access_ptr("r")))

        for mo in T.serial(Mo):
            for no in T.serial(No):
                T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                for ko in T.serial(Ko):
                    a_top = (mo * 32) * K + ko * 32
                    a_bot = (mo * 32 + 16) * K + ko * 32
                    b_left = (ko * 16) * n_packed_u16 + (no * 32) * 2
                    b_right = (ko * 16) * n_packed_u16 + (no * 32 + 16) * 2
                    T.evaluate(_tileloadd(0, A.access_ptr("r", offset=a_top), a_row_bytes))
                    T.evaluate(_tileloadd(1, A.access_ptr("r", offset=a_bot), a_row_bytes))
                    T.evaluate(_tileloadd(2, B_packed.access_ptr("r", offset=b_left),  b_row_bytes))
                    T.evaluate(_tileloadd(3, B_packed.access_ptr("r", offset=b_right), b_row_bytes))
                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))
                c_tl = (mo * 32) * N + no * 32
                c_tr = (mo * 32) * N + no * 32 + 16
                c_bl = (mo * 32 + 16) * N + no * 32
                c_br = (mo * 32 + 16) * N + no * 32 + 16
                T.evaluate(_tilestored(4, C.access_ptr("w", offset=c_tl), c_row_bytes))
                T.evaluate(_tilestored(5, C.access_ptr("w", offset=c_tr), c_row_bytes))
                T.evaluate(_tilestored(6, C.access_ptr("w", offset=c_bl), c_row_bytes))
                T.evaluate(_tilestored(7, C.access_ptr("w", offset=c_br), c_row_bytes))
    return gemm


def make_softmax_to_bf16(rows, cols, scale):
    """Row-wise: P_bf16 = softmax(S * scale).  Vectorized along cols."""
    assert cols % 16 == 0

    @T.prim_func(s_tir=True, check_well_formed=False)
    def softmax(
        S: T.Buffer((rows, cols), "float32"),
        P: T.Buffer((rows, cols), "uint16"),
    ):
        vscale = T.broadcast(T.float32(scale), 16)
        NEG_INF = -1.0e30

        for i in T.serial(rows):
            # pass 1: max(s * scale)
            vmx = T.broadcast(T.float32(NEG_INF), 16)
            for jc in range(cols // 16):
                v = S.vload([i, jc * 16], "float32x16") * vscale
                vmx = T.max(vmx, v)
            mx = _vec_reduce_fmax(vmx)
            vmxb = T.broadcast(mx, 16)

            # pass 2: exp(s*scale - mx), sum; store fp32 back
            vsum = T.broadcast(T.float32(0), 16)
            for jc in range(cols // 16):
                v = S.vload([i, jc * 16], "float32x16") * vscale - vmxb
                v = _vec_exp(v)
                S[i, T.ramp(jc * 16, 1, 16)] = v
                vsum = vsum + v
            sum_p = _vec_reduce_fadd(vsum)
            vinv = T.broadcast(T.float32(1.0) / sum_p, 16)

            # pass 3: divide + convert to bf16 (32 lanes per zmm)
            if cols % 32 == 0:
                for jc in range(cols // 32):
                    v_lo = S.vload([i, jc * 32], "float32x16") * vinv
                    v_hi = S.vload([i, jc * 32 + 16], "float32x16") * vinv
                    bf32 = _vec_cvt32_to_bf16x32(v_lo, v_hi)
                    P[i, T.ramp(jc * 32, 1, 32)] = T.reinterpret(bf32, dtype="uint16x32")
            else:
                for jc in range(cols // 16):
                    v = S.vload([i, jc * 16], "float32x16") * vinv
                    bf16x16 = T.call_llvm_pure_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.x86.avx512bf16.cvtneps2bf16.512"),
                        v, dtype="int16x16")
                    P[i, T.ramp(jc * 16, 1, 16)] = T.reinterpret(bf16x16, dtype="uint16x16")
    return softmax


def make_mha_nonfused(SEQ, D):
    """Non-fused: Q@K^T → softmax → P@V."""
    scale = 1.0 / math.sqrt(D)

    gemm_QK = make_gemm_2x2(SEQ, SEQ, D, SEQ * 2)           # B_packed = K_packed, stride = SEQ*2
    gemm_PV = make_gemm_2x2(SEQ, D, SEQ, D * 2)             # B_packed = V_packed, stride = D*2
    sm      = make_softmax_to_bf16(SEQ, SEQ, scale)

    # Combine into a single prim_func that allocates S, P in scratch and calls the kernels
    # Actually, we'll keep them separate and orchestrate in Python (treating S, P as
    # caller-allocated). But that adds TVM call overhead. Let's compose into one func.
    return gemm_QK, sm, gemm_PV


# ─────────────────────────────────────────────────────────────────────
# Single fused prim_func (still "non-fused" algorithmically, but no
# call overhead between stages)
# ─────────────────────────────────────────────────────────────────────
def make_mha_nonfused_one(SEQ, D):
    """Single prim_func that runs Q@K^T → softmax → P@V end-to-end."""
    scale = 1.0 / math.sqrt(D)
    K_pairs_qk = D // 2          # K_packed row count
    K_pairs_pv = SEQ // 2        # V_packed row count
    Dp = D // 2
    Sp = SEQ // 2

    Mo = SEQ // 32  # M-blocks
    No_qk = SEQ // 32   # N-blocks of Q@K^T (N=SEQ)
    No_pv = D // 32     # N-blocks of P@V   (N=D)
    Ko_qk = D // 32     # K-iters of Q@K^T  (K=D)
    Ko_pv = SEQ // 32   # K-iters of P@V    (K=SEQ)

    a_row_bytes_Q = D * 2
    a_row_bytes_P = SEQ * 2
    b_row_bytes_K = SEQ * 2 * 2
    b_row_bytes_V = D * 2 * 2
    s_row_bytes = SEQ * 4
    o_row_bytes = D * 4

    NEG_INF = -1.0e30

    @T.prim_func(s_tir=True, check_well_formed=False)
    def mha(
        Q: T.Buffer((SEQ, D), "uint16"),
        K_packed: T.Buffer((Dp, SEQ * 2), "uint16"),
        V_packed: T.Buffer((Sp, D * 2), "uint16"),
        O: T.Buffer((SEQ, D), "float32"),
    ):
        cfg = T.alloc_buffer((64,), "uint8")
        cfg[0] = T.uint8(1)
        for i in range(1, 16): cfg[i] = T.uint8(0)
        for i in range(8):
            cfg[16 + 2 * i] = T.uint8(64); cfg[16 + 2 * i + 1] = T.uint8(0)
        for i in range(16): cfg[32 + i] = T.uint8(0)
        for i in range(8): cfg[48 + i] = T.uint8(16)
        for i in range(8): cfg[56 + i] = T.uint8(0)
        T.evaluate(_ldtilecfg(cfg.access_ptr("r")))

        S = T.alloc_buffer((SEQ, SEQ), "float32")
        P = T.alloc_buffer((SEQ, SEQ), "uint16")

        # ── Stage 1: S = Q @ K^T  ──────────────────────────────────────
        for mo in T.serial(Mo):
            for no in T.serial(No_qk):
                T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                for ko in T.serial(Ko_qk):
                    a_top = (mo * 32) * D + ko * 32
                    a_bot = (mo * 32 + 16) * D + ko * 32
                    b_left = (ko * 16) * (SEQ * 2) + (no * 32) * 2
                    b_right = (ko * 16) * (SEQ * 2) + (no * 32 + 16) * 2
                    T.evaluate(_tileloadd(0, Q.access_ptr("r", offset=a_top), a_row_bytes_Q))
                    T.evaluate(_tileloadd(1, Q.access_ptr("r", offset=a_bot), a_row_bytes_Q))
                    T.evaluate(_tileloadd(2, K_packed.access_ptr("r", offset=b_left),  b_row_bytes_K))
                    T.evaluate(_tileloadd(3, K_packed.access_ptr("r", offset=b_right), b_row_bytes_K))
                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))
                c_tl = (mo * 32) * SEQ + no * 32
                c_tr = (mo * 32) * SEQ + no * 32 + 16
                c_bl = (mo * 32 + 16) * SEQ + no * 32
                c_br = (mo * 32 + 16) * SEQ + no * 32 + 16
                T.evaluate(_tilestored(4, S.access_ptr("w", offset=c_tl), s_row_bytes))
                T.evaluate(_tilestored(5, S.access_ptr("w", offset=c_tr), s_row_bytes))
                T.evaluate(_tilestored(6, S.access_ptr("w", offset=c_bl), s_row_bytes))
                T.evaluate(_tilestored(7, S.access_ptr("w", offset=c_br), s_row_bytes))

        # ── Stage 2: P = softmax(S * scale)  ──────────────────────────
        vscale = T.broadcast(T.float32(scale), 16)
        for i in T.serial(SEQ):
            vmx = T.broadcast(T.float32(NEG_INF), 16)
            for jc in range(SEQ // 16):
                v = S.vload([i, jc * 16], "float32x16") * vscale
                vmx = T.max(vmx, v)
            mx = _vec_reduce_fmax(vmx)
            vmxb = T.broadcast(mx, 16)
            vsum = T.broadcast(T.float32(0), 16)
            for jc in range(SEQ // 16):
                v = S.vload([i, jc * 16], "float32x16") * vscale - vmxb
                v = _vec_exp(v)
                S[i, T.ramp(jc * 16, 1, 16)] = v
                vsum = vsum + v
            sum_p = _vec_reduce_fadd(vsum)
            vinv = T.broadcast(T.float32(1.0) / sum_p, 16)
            for jc in range(SEQ // 32):
                v_lo = S.vload([i, jc * 32], "float32x16") * vinv
                v_hi = S.vload([i, jc * 32 + 16], "float32x16") * vinv
                bf32 = _vec_cvt32_to_bf16x32(v_lo, v_hi)
                P[i, T.ramp(jc * 32, 1, 32)] = T.reinterpret(bf32, dtype="uint16x32")

        # ── Stage 3: O = P @ V  ───────────────────────────────────────
        for mo in T.serial(Mo):
            for no in T.serial(No_pv):
                T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                for ko in T.serial(Ko_pv):
                    a_top = (mo * 32) * SEQ + ko * 32
                    a_bot = (mo * 32 + 16) * SEQ + ko * 32
                    b_left = (ko * 16) * (D * 2) + (no * 32) * 2
                    b_right = (ko * 16) * (D * 2) + (no * 32 + 16) * 2
                    T.evaluate(_tileloadd(0, P.access_ptr("r", offset=a_top), a_row_bytes_P))
                    T.evaluate(_tileloadd(1, P.access_ptr("r", offset=a_bot), a_row_bytes_P))
                    T.evaluate(_tileloadd(2, V_packed.access_ptr("r", offset=b_left),  b_row_bytes_V))
                    T.evaluate(_tileloadd(3, V_packed.access_ptr("r", offset=b_right), b_row_bytes_V))
                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))
                o_tl = (mo * 32) * D + no * 32
                o_tr = (mo * 32) * D + no * 32 + 16
                o_bl = (mo * 32 + 16) * D + no * 32
                o_br = (mo * 32 + 16) * D + no * 32 + 16
                T.evaluate(_tilestored(4, O.access_ptr("w", offset=o_tl), o_row_bytes))
                T.evaluate(_tilestored(5, O.access_ptr("w", offset=o_tr), o_row_bytes))
                T.evaluate(_tilestored(6, O.access_ptr("w", offset=o_bl), o_row_bytes))
                T.evaluate(_tilestored(7, O.access_ptr("w", offset=o_br), o_row_bytes))
    return mha


def make_mha_mblock_fused(SEQ, D, Mq=32):
    """M-block-fused: for each Mq rows of Q, do GEMM-A + softmax + GEMM-B,
       keeping the per-block S (Mq × SEQ fp32) hot in L2 across the 3 stages.
    """
    assert SEQ % Mq == 0 and SEQ % 32 == 0 and D % 32 == 0
    assert Mq % 32 == 0
    M_pair = Mq // 32
    scale = 1.0 / math.sqrt(D)
    Dp = D // 2
    Sp = SEQ // 2

    Nq = SEQ // Mq
    No_qk = SEQ // 32       # GEMM-A N-blocks (N = SEQ)
    Ko_qk = D // 32         # GEMM-A K-iters (K = D)
    No_pv = D // 32         # GEMM-B N-blocks (N = D)
    Ko_pv = SEQ // 32       # GEMM-B K-iters (K = SEQ)

    a_row_bytes_Q = D * 2
    a_row_bytes_P = SEQ * 2
    b_row_bytes_K = SEQ * 2 * 2
    b_row_bytes_V = D * 2 * 2
    s_row_bytes = SEQ * 4
    o_row_bytes = D * 4

    NEG_INF = -1.0e30

    @T.prim_func(s_tir=True, check_well_formed=False)
    def mha(
        Q: T.Buffer((SEQ, D), "uint16"),
        K_packed: T.Buffer((Dp, SEQ * 2), "uint16"),
        V_packed: T.Buffer((Sp, D * 2), "uint16"),
        O: T.Buffer((SEQ, D), "float32"),
    ):
        cfg = T.alloc_buffer((64,), "uint8")
        cfg[0] = T.uint8(1)
        for i in range(1, 16): cfg[i] = T.uint8(0)
        for i in range(8):
            cfg[16 + 2 * i] = T.uint8(64); cfg[16 + 2 * i + 1] = T.uint8(0)
        for i in range(16): cfg[32 + i] = T.uint8(0)
        for i in range(8): cfg[48 + i] = T.uint8(16)
        for i in range(8): cfg[56 + i] = T.uint8(0)
        T.evaluate(_ldtilecfg(cfg.access_ptr("r")))

        # Per-M-block scratch (kept in L2 across the 3 stages)
        S_mb = T.alloc_buffer((Mq, SEQ), "float32")
        P_mb = T.alloc_buffer((Mq, SEQ), "uint16")

        for mo in T.serial(Nq):
            # ─── Stage A: S_mb = Q[mo*Mq:..., :] @ K^T ─────────────────
            for mbo in T.serial(M_pair):
                for no in T.serial(No_qk):
                    T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                    T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                    for ko in T.serial(Ko_qk):
                        a_top = (mo * Mq + mbo * 32) * D + ko * 32
                        a_bot = (mo * Mq + mbo * 32 + 16) * D + ko * 32
                        b_left = (ko * 16) * (SEQ * 2) + (no * 32) * 2
                        b_right = (ko * 16) * (SEQ * 2) + (no * 32 + 16) * 2
                        T.evaluate(_tileloadd(0, Q.access_ptr("r", offset=a_top), a_row_bytes_Q))
                        T.evaluate(_tileloadd(1, Q.access_ptr("r", offset=a_bot), a_row_bytes_Q))
                        T.evaluate(_tileloadd(2, K_packed.access_ptr("r", offset=b_left),  b_row_bytes_K))
                        T.evaluate(_tileloadd(3, K_packed.access_ptr("r", offset=b_right), b_row_bytes_K))
                        T.evaluate(_tdpbf16ps(4, 0, 2))
                        T.evaluate(_tdpbf16ps(5, 0, 3))
                        T.evaluate(_tdpbf16ps(6, 1, 2))
                        T.evaluate(_tdpbf16ps(7, 1, 3))
                    c_tl = (mbo * 32     ) * SEQ + no * 32
                    c_tr = (mbo * 32     ) * SEQ + no * 32 + 16
                    c_bl = (mbo * 32 + 16) * SEQ + no * 32
                    c_br = (mbo * 32 + 16) * SEQ + no * 32 + 16
                    T.evaluate(_tilestored(4, S_mb.access_ptr("w", offset=c_tl), s_row_bytes))
                    T.evaluate(_tilestored(5, S_mb.access_ptr("w", offset=c_tr), s_row_bytes))
                    T.evaluate(_tilestored(6, S_mb.access_ptr("w", offset=c_bl), s_row_bytes))
                    T.evaluate(_tilestored(7, S_mb.access_ptr("w", offset=c_br), s_row_bytes))

            # ─── Stage B: P_mb = softmax(S_mb * scale)  (Mq rows) ─────
            vscale = T.broadcast(T.float32(scale), 16)
            for i in T.serial(Mq):
                # Pass 1: max(s * scale)
                vmx = T.broadcast(T.float32(NEG_INF), 16)
                for jc in range(SEQ // 16):
                    v = S_mb.vload([i, jc * 16], "float32x16") * vscale
                    vmx = T.max(vmx, v)
                mx = _vec_reduce_fmax(vmx)
                vmxb = T.broadcast(mx, 16)
                # Pass 2: exp(s*scale - mx), store fp32 back to S_mb, accumulate sum
                vsum = T.broadcast(T.float32(0), 16)
                for jc in range(SEQ // 16):
                    v = S_mb.vload([i, jc * 16], "float32x16") * vscale - vmxb
                    v = _vec_exp(v)
                    S_mb[i, T.ramp(jc * 16, 1, 16)] = v
                    vsum = vsum + v
                sum_p = _vec_reduce_fadd(vsum)
                vinv = T.broadcast(T.float32(1.0) / sum_p, 16)
                # Pass 3: normalize + convert to bf16 (32 lanes per zmm)
                for jc in range(SEQ // 32):
                    v_lo = S_mb.vload([i, jc * 32], "float32x16") * vinv
                    v_hi = S_mb.vload([i, jc * 32 + 16], "float32x16") * vinv
                    bf32 = _vec_cvt32_to_bf16x32(v_lo, v_hi)
                    P_mb[i, T.ramp(jc * 32, 1, 32)] = T.reinterpret(bf32, dtype="uint16x32")

            # ─── Stage C: O[mo*Mq:..., :] = P_mb @ V  ─────────────────
            for mbo in T.serial(M_pair):
                for no in T.serial(No_pv):
                    T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                    T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                    for ko in T.serial(Ko_pv):
                        a_top = (mbo * 32     ) * SEQ + ko * 32
                        a_bot = (mbo * 32 + 16) * SEQ + ko * 32
                        b_left = (ko * 16) * (D * 2) + (no * 32) * 2
                        b_right = (ko * 16) * (D * 2) + (no * 32 + 16) * 2
                        T.evaluate(_tileloadd(0, P_mb.access_ptr("r", offset=a_top), a_row_bytes_P))
                        T.evaluate(_tileloadd(1, P_mb.access_ptr("r", offset=a_bot), a_row_bytes_P))
                        T.evaluate(_tileloadd(2, V_packed.access_ptr("r", offset=b_left),  b_row_bytes_V))
                        T.evaluate(_tileloadd(3, V_packed.access_ptr("r", offset=b_right), b_row_bytes_V))
                        T.evaluate(_tdpbf16ps(4, 0, 2))
                        T.evaluate(_tdpbf16ps(5, 0, 3))
                        T.evaluate(_tdpbf16ps(6, 1, 2))
                        T.evaluate(_tdpbf16ps(7, 1, 3))
                    o_tl = (mo * Mq + mbo * 32     ) * D + no * 32
                    o_tr = (mo * Mq + mbo * 32     ) * D + no * 32 + 16
                    o_bl = (mo * Mq + mbo * 32 + 16) * D + no * 32
                    o_br = (mo * Mq + mbo * 32 + 16) * D + no * 32 + 16
                    T.evaluate(_tilestored(4, O.access_ptr("w", offset=o_tl), o_row_bytes))
                    T.evaluate(_tilestored(5, O.access_ptr("w", offset=o_tr), o_row_bytes))
                    T.evaluate(_tilestored(6, O.access_ptr("w", offset=o_bl), o_row_bytes))
                    T.evaluate(_tilestored(7, O.access_ptr("w", offset=o_br), o_row_bytes))
    return mha


def make_mha_nf_fused_max(SEQ, D):
    """Non-fused but with pass-1 max FUSED into GEMM-A at M-block granularity:
       After all 32 rows of an M-block of S are fully written, do pass-1 max
       on them while data is still hot in L1/L2."""
    assert SEQ % 32 == 0 and D % 32 == 0
    scale = 1.0 / math.sqrt(D)
    Dp = D // 2; Sp = SEQ // 2
    Mo = SEQ // 32
    NEG_INF = -1.0e30

    @T.prim_func(s_tir=True, check_well_formed=False)
    def mha(
        Q: T.Buffer((SEQ, D), "uint16"),
        K_packed: T.Buffer((Dp, SEQ * 2), "uint16"),
        V_packed: T.Buffer((Sp, D * 2), "uint16"),
        O: T.Buffer((SEQ, D), "float32"),
    ):
        cfg = T.alloc_buffer((64,), "uint8")
        cfg[0] = T.uint8(1)
        for i in range(1, 16): cfg[i] = T.uint8(0)
        for i in range(8):
            cfg[16 + 2 * i] = T.uint8(64); cfg[16 + 2 * i + 1] = T.uint8(0)
        for i in range(16): cfg[32 + i] = T.uint8(0)
        for i in range(8): cfg[48 + i] = T.uint8(16)
        for i in range(8): cfg[56 + i] = T.uint8(0)
        T.evaluate(_ldtilecfg(cfg.access_ptr("r")))

        S = T.alloc_buffer((SEQ, SEQ), "float32")
        P = T.alloc_buffer((SEQ, SEQ), "uint16")
        m_buf = T.alloc_buffer((SEQ,), "float32")

        vscale = T.broadcast(T.float32(scale), 16)

        # init m_buf = -inf
        for i in T.serial(SEQ // 16):
            m_buf[T.ramp(i * 16, 1, 16)] = T.broadcast(T.float32(NEG_INF), 16)

        # ── Stage A: S = Q @ K^T, with pass-1 max fused at M-block granularity ───
        for mo in T.serial(Mo):
            for no in T.serial(SEQ // 32):
                T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                for ko in T.serial(D // 32):
                    a_top = (mo * 32) * D + ko * 32
                    a_bot = (mo * 32 + 16) * D + ko * 32
                    b_left = (ko * 16) * (SEQ * 2) + (no * 32) * 2
                    b_right = (ko * 16) * (SEQ * 2) + (no * 32 + 16) * 2
                    T.evaluate(_tileloadd(0, Q.access_ptr("r", offset=a_top), D * 2))
                    T.evaluate(_tileloadd(1, Q.access_ptr("r", offset=a_bot), D * 2))
                    T.evaluate(_tileloadd(2, K_packed.access_ptr("r", offset=b_left), SEQ * 4))
                    T.evaluate(_tileloadd(3, K_packed.access_ptr("r", offset=b_right), SEQ * 4))
                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))
                c_tl = (mo * 32) * SEQ + no * 32
                c_tr = (mo * 32) * SEQ + no * 32 + 16
                c_bl = (mo * 32 + 16) * SEQ + no * 32
                c_br = (mo * 32 + 16) * SEQ + no * 32 + 16
                T.evaluate(_tilestored(4, S.access_ptr("w", offset=c_tl), SEQ * 4))
                T.evaluate(_tilestored(5, S.access_ptr("w", offset=c_tr), SEQ * 4))
                T.evaluate(_tilestored(6, S.access_ptr("w", offset=c_bl), SEQ * 4))
                T.evaluate(_tilestored(7, S.access_ptr("w", offset=c_br), SEQ * 4))

            # After all N-blocks: 32 rows of S complete. Run pass-1 max while hot.
            for ii in T.serial(32):
                row = mo * 32 + ii
                vmx = T.broadcast(T.float32(NEG_INF), 16)
                for jc in range(SEQ // 16):
                    v = S.vload([row, jc * 16], "float32x16") * vscale
                    vmx = T.max(vmx, v)
                m_buf[row] = _vec_reduce_fmax(vmx)

        # ── Stage B: pass 2 (exp + sum) + pass 3 (normalize + bf16) ──────
        for i in T.serial(SEQ):
            vmxb = T.broadcast(m_buf[i], 16)
            vsum = T.broadcast(T.float32(0), 16)
            for jc in range(SEQ // 16):
                v = S.vload([i, jc * 16], "float32x16") * vscale - vmxb
                v = _vec_exp(v)
                S[i, T.ramp(jc * 16, 1, 16)] = v
                vsum = vsum + v
            sum_p = _vec_reduce_fadd(vsum)
            vinv = T.broadcast(T.float32(1.0) / sum_p, 16)
            for jc in range(SEQ // 32):
                v_lo = S.vload([i, jc * 32], "float32x16") * vinv
                v_hi = S.vload([i, jc * 32 + 16], "float32x16") * vinv
                bf32 = _vec_cvt32_to_bf16x32(v_lo, v_hi)
                P[i, T.ramp(jc * 32, 1, 32)] = T.reinterpret(bf32, dtype="uint16x32")

        # ── Stage C: O = P @ V (same as before) ──────────────────────────
        for mo in T.serial(Mo):
            for no in T.serial(D // 32):
                T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                for ko in T.serial(SEQ // 32):
                    a_top = (mo * 32) * SEQ + ko * 32
                    a_bot = (mo * 32 + 16) * SEQ + ko * 32
                    b_left = (ko * 16) * (D * 2) + (no * 32) * 2
                    b_right = (ko * 16) * (D * 2) + (no * 32 + 16) * 2
                    T.evaluate(_tileloadd(0, P.access_ptr("r", offset=a_top), SEQ * 2))
                    T.evaluate(_tileloadd(1, P.access_ptr("r", offset=a_bot), SEQ * 2))
                    T.evaluate(_tileloadd(2, V_packed.access_ptr("r", offset=b_left), D * 4))
                    T.evaluate(_tileloadd(3, V_packed.access_ptr("r", offset=b_right), D * 4))
                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))
                o_tl = (mo * 32) * D + no * 32
                o_tr = (mo * 32) * D + no * 32 + 16
                o_bl = (mo * 32 + 16) * D + no * 32
                o_br = (mo * 32 + 16) * D + no * 32 + 16
                T.evaluate(_tilestored(4, O.access_ptr("w", offset=o_tl), D * 4))
                T.evaluate(_tilestored(5, O.access_ptr("w", offset=o_tr), D * 4))
                T.evaluate(_tilestored(6, O.access_ptr("w", offset=o_bl), D * 4))
                T.evaluate(_tilestored(7, O.access_ptr("w", offset=o_br), D * 4))
    return mha


def make_mha_nf_full_softmax_fused(SEQ, D):
    """Non-fused GEMM-A + full softmax per M-block + non-fused GEMM-B.

    Within each Mq=32 row M-block:
      1. GEMM-A produces 32 rows of S
      2. pass-1 max (S hot in cache)
      3. pass-2/3 exp+sum+norm+bf16 (S hot, P stored to full buffer)
    GEMM-B runs once at end with full P (cache-friendly K reuse).
    """
    assert SEQ % 32 == 0 and D % 32 == 0
    scale = 1.0 / math.sqrt(D)
    Dp = D // 2; Sp = SEQ // 2
    Mo = SEQ // 32
    NEG_INF = -1.0e30

    @T.prim_func(s_tir=True, check_well_formed=False)
    def mha(
        Q: T.Buffer((SEQ, D), "uint16"),
        K_packed: T.Buffer((Dp, SEQ * 2), "uint16"),
        V_packed: T.Buffer((Sp, D * 2), "uint16"),
        O: T.Buffer((SEQ, D), "float32"),
    ):
        cfg = T.alloc_buffer((64,), "uint8")
        cfg[0] = T.uint8(1)
        for i in range(1, 16): cfg[i] = T.uint8(0)
        for i in range(8):
            cfg[16 + 2 * i] = T.uint8(64); cfg[16 + 2 * i + 1] = T.uint8(0)
        for i in range(16): cfg[32 + i] = T.uint8(0)
        for i in range(8): cfg[48 + i] = T.uint8(16)
        for i in range(8): cfg[56 + i] = T.uint8(0)
        T.evaluate(_ldtilecfg(cfg.access_ptr("r")))

        S = T.alloc_buffer((SEQ, SEQ), "float32")
        P = T.alloc_buffer((SEQ, SEQ), "uint16")
        vscale = T.broadcast(T.float32(scale), 16)

        # ── Stage A+softmax: per-M-block ──────────────────────────────────
        for mo in T.serial(Mo):
            # GEMM A for this M-block: produce 32 rows of S
            for no in T.serial(SEQ // 32):
                T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                for ko in T.serial(D // 32):
                    a_top = (mo * 32) * D + ko * 32
                    a_bot = (mo * 32 + 16) * D + ko * 32
                    b_left = (ko * 16) * (SEQ * 2) + (no * 32) * 2
                    b_right = (ko * 16) * (SEQ * 2) + (no * 32 + 16) * 2
                    T.evaluate(_tileloadd(0, Q.access_ptr("r", offset=a_top), D * 2))
                    T.evaluate(_tileloadd(1, Q.access_ptr("r", offset=a_bot), D * 2))
                    T.evaluate(_tileloadd(2, K_packed.access_ptr("r", offset=b_left), SEQ * 4))
                    T.evaluate(_tileloadd(3, K_packed.access_ptr("r", offset=b_right), SEQ * 4))
                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))
                c_tl = (mo * 32) * SEQ + no * 32
                c_tr = (mo * 32) * SEQ + no * 32 + 16
                c_bl = (mo * 32 + 16) * SEQ + no * 32
                c_br = (mo * 32 + 16) * SEQ + no * 32 + 16
                T.evaluate(_tilestored(4, S.access_ptr("w", offset=c_tl), SEQ * 4))
                T.evaluate(_tilestored(5, S.access_ptr("w", offset=c_tr), SEQ * 4))
                T.evaluate(_tilestored(6, S.access_ptr("w", offset=c_bl), SEQ * 4))
                T.evaluate(_tilestored(7, S.access_ptr("w", offset=c_br), SEQ * 4))

            # Full softmax for these 32 rows (S hot in L1/L2)
            for ii in T.serial(32):
                row = mo * 32 + ii
                # Pass 1: max
                vmx = T.broadcast(T.float32(NEG_INF), 16)
                for jc in range(SEQ // 16):
                    v = S.vload([row, jc * 16], "float32x16") * vscale
                    vmx = T.max(vmx, v)
                mx = _vec_reduce_fmax(vmx)
                vmxb = T.broadcast(mx, 16)
                # Pass 2: exp + sum + writeback fp32
                vsum = T.broadcast(T.float32(0), 16)
                for jc in range(SEQ // 16):
                    v = S.vload([row, jc * 16], "float32x16") * vscale - vmxb
                    v = _vec_exp(v)
                    S[row, T.ramp(jc * 16, 1, 16)] = v
                    vsum = vsum + v
                sum_p = _vec_reduce_fadd(vsum)
                vinv = T.broadcast(T.float32(1.0) / sum_p, 16)
                # Pass 3: normalize + bf16
                for jc in range(SEQ // 32):
                    v_lo = S.vload([row, jc * 32], "float32x16") * vinv
                    v_hi = S.vload([row, jc * 32 + 16], "float32x16") * vinv
                    bf32 = _vec_cvt32_to_bf16x32(v_lo, v_hi)
                    P[row, T.ramp(jc * 32, 1, 32)] = T.reinterpret(bf32, dtype="uint16x32")

        # ── Stage B: O = P @ V (full GEMM at end) ────────────────────────
        for mo in T.serial(Mo):
            for no in T.serial(D // 32):
                T.evaluate(_tilezero(4)); T.evaluate(_tilezero(5))
                T.evaluate(_tilezero(6)); T.evaluate(_tilezero(7))
                for ko in T.serial(SEQ // 32):
                    a_top = (mo * 32) * SEQ + ko * 32
                    a_bot = (mo * 32 + 16) * SEQ + ko * 32
                    b_left = (ko * 16) * (D * 2) + (no * 32) * 2
                    b_right = (ko * 16) * (D * 2) + (no * 32 + 16) * 2
                    T.evaluate(_tileloadd(0, P.access_ptr("r", offset=a_top), SEQ * 2))
                    T.evaluate(_tileloadd(1, P.access_ptr("r", offset=a_bot), SEQ * 2))
                    T.evaluate(_tileloadd(2, V_packed.access_ptr("r", offset=b_left), D * 4))
                    T.evaluate(_tileloadd(3, V_packed.access_ptr("r", offset=b_right), D * 4))
                    T.evaluate(_tdpbf16ps(4, 0, 2))
                    T.evaluate(_tdpbf16ps(5, 0, 3))
                    T.evaluate(_tdpbf16ps(6, 1, 2))
                    T.evaluate(_tdpbf16ps(7, 1, 3))
                o_tl = (mo * 32) * D + no * 32
                o_tr = (mo * 32) * D + no * 32 + 16
                o_bl = (mo * 32 + 16) * D + no * 32
                o_br = (mo * 32 + 16) * D + no * 32 + 16
                T.evaluate(_tilestored(4, O.access_ptr("w", offset=o_tl), D * 4))
                T.evaluate(_tilestored(5, O.access_ptr("w", offset=o_tr), D * 4))
                T.evaluate(_tilestored(6, O.access_ptr("w", offset=o_bl), D * 4))
                T.evaluate(_tilestored(7, O.access_ptr("w", offset=o_br), D * 4))
    return mha


if __name__ == "__main__":
    import time
    import numpy as np
    init_amx()
    target = tvm.target.Target({"kind": "llvm", "mcpu": "sapphirerapids"})
    dev = tvm.cpu(0)

    configs = [
        ("nonfused",  make_mha_nonfused_one,            1024, 128, None),
        ("nf_fmax",   make_mha_nf_fused_max,            1024, 128, None),
        ("nf_fsoft",  make_mha_nf_full_softmax_fused,   1024, 128, None),
        ("mblock",    make_mha_mblock_fused,            1024, 128, 64),
        ("nonfused",  make_mha_nonfused_one,            1024, 256, None),
        ("nf_fmax",   make_mha_nf_fused_max,            1024, 256, None),
        ("nf_fsoft",  make_mha_nf_full_softmax_fused,   1024, 256, None),
        ("nonfused",  make_mha_nonfused_one,            1024, 512, None),
        ("nf_fmax",   make_mha_nf_fused_max,            1024, 512, None),
        ("nf_fsoft",  make_mha_nf_full_softmax_fused,   1024, 512, None),
        ("nonfused",  make_mha_nonfused_one,            2048, 128, None),
        ("nf_fmax",   make_mha_nf_fused_max,            2048, 128, None),
        ("nf_fsoft",  make_mha_nf_full_softmax_fused,   2048, 128, None),
        ("nonfused",  make_mha_nonfused_one,            2048, 256, None),
        ("nf_fsoft",  make_mha_nf_full_softmax_fused,   2048, 256, None),
    ]
    for tag, factory, SEQ, D, Mq in configs:
        scale = 1.0 / math.sqrt(D)
        if Mq is None:
            kfn = factory(SEQ, D)
            label = f"SEQ={SEQ}  D={D}  {tag}"
        else:
            kfn = factory(SEQ, D, Mq)
            label = f"SEQ={SEQ}  D={D}  Mq={Mq}  {tag}"
        print(f"\n=== {label} ===")
        t0 = time.perf_counter()
        pack_K_fn = tvm.compile(make_pack_K(SEQ, D), target=target)
        pack_V_fn = tvm.compile(make_pack_V(SEQ, D), target=target)
        mha_fn = tvm.compile(kfn, target=target)
        t_comp = (time.perf_counter() - t0) * 1000

        np.random.seed(42)
        Q_np = np.random.randn(SEQ, D).astype(np.float32) * 0.05
        K_np = np.random.randn(SEQ, D).astype(np.float32) * 0.05
        V_np = np.random.randn(SEQ, D).astype(np.float32) * 0.05
        Q_bf = fp32_to_bf16(Q_np); K_bf = fp32_to_bf16(K_np); V_bf = fp32_to_bf16(V_np)
        O_ref = mha_numpy(Q_bf, K_bf, V_bf, scale)

        Qt = tvm.runtime.tensor(Q_bf, dev)
        Kt = tvm.runtime.tensor(K_bf, dev)
        Vt = tvm.runtime.tensor(V_bf, dev)
        Kpt = tvm.runtime.tensor(np.zeros((D // 2, SEQ * 2), dtype=np.uint16), dev)
        Vpt = tvm.runtime.tensor(np.zeros((SEQ // 2, D * 2), dtype=np.uint16), dev)
        Ot = tvm.runtime.tensor(np.zeros((SEQ, D), dtype=np.float32), dev)

        pack_K_fn(Kt, Kpt); pack_V_fn(Vt, Vpt)
        mha_fn(Qt, Kpt, Vpt, Ot)
        err = np.max(np.abs(Ot.numpy() - O_ref))
        rel = err / (np.max(np.abs(O_ref)) + 1e-8)

        for _ in range(5):
            Ot.copyfrom(np.zeros((SEQ, D), dtype=np.float32))
            mha_fn(Qt, Kpt, Vpt, Ot)
        times = []
        for _ in range(20):
            Ot.copyfrom(np.zeros((SEQ, D), dtype=np.float32))
            t0 = time.perf_counter()
            mha_fn(Qt, Kpt, Vpt, Ot)
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        flops = 4.0 * SEQ * SEQ * D
        gf_mean = flops / times.mean() / 1e9
        gf_best = flops / times.min() / 1e9
        print(f"  compile {t_comp:.0f}ms  err {err:.4f} rel {rel:.4%}")
        print(f"  mean {times.mean()*1e6:.1f}us  best {times.min()*1e6:.1f}us")
        print(f"  GFLOPS: mean {gf_mean:.1f}  best {gf_best:.1f}")

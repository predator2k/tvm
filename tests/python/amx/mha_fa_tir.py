"""
Pure-TVM-TIR FlashAttention-style MHA on AMX BF16.

No C/C++ helpers — all AMX tile ops and AVX-512 BF16 conversions are emitted via
T.call_llvm_intrin from inside TIR.

Forward pass:
  S = Q @ K^T * scale          [Mq × Nk]   per K-block
  online softmax updates m, l, O
  O += softmax(S) @ V_block    [Mq × D]

Layout (all bf16 stored as uint16):
  Q          [SEQ, D]
  K_packed   [D/2,   SEQ*2]   pre-packed: row k_pair contains  K[2*kp][n],K[2*kp+1][n] for each n
  V_packed   [SEQ/2, D*2]     pre-packed: row k_pair contains  V[2*kp][n],V[2*kp+1][n] for each n
  O          [SEQ, D]   fp32 output

Block sizes:
  Mq = 32     (= 1 AMX M-pair: 16 + 16 rows)
  Nk = 128    (= 4 AMX N-pairs of 32 each)
  D must be a multiple of 32
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


# ─────────────────────────────────────────────────────────────────────
# LLVM intrinsic call helpers (return TIR expressions)
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
    """Cast an offset to i64. Avoids LLVM i32→i64 sign-ext spills around AMX intrinsics."""
    return T.cast(x, "int64")


# ─────────────────────────────────────────────────────────────────────
# Vectorized helpers (return TIR expressions) — used inside prim_funcs
# ─────────────────────────────────────────────────────────────────────
def _vec_round(x):
    """Round float32x16 to nearest integer (still fp32)."""
    return T.call_llvm_pure_intrin(
        T.llvm_lookup_intrinsic_id("llvm.nearbyint.v16f32"),
        x, dtype="float32x16")

def _vec_reduce_fmax(v):
    return T.call_llvm_pure_intrin(
        T.llvm_lookup_intrinsic_id("llvm.vector.reduce.fmax.v16f32"),
        v, dtype="float32")

def _vec_reduce_fadd(v):
    return T.call_llvm_pure_intrin(
        T.llvm_lookup_intrinsic_id("llvm.vector.reduce.fadd.v16f32"),
        T.float32(0), v, dtype="float32")

def _vec_cvt32_to_bf16x32(vlo, vhi):
    """Convert 32 fp32 (vlo + vhi) -> 32 bf16 packed in 32 x i16 vector."""
    return T.call_llvm_pure_intrin(
        T.llvm_lookup_intrinsic_id("llvm.x86.avx512bf16.cvtne2ps2bf16.512"),
        vhi, vlo, dtype="int16x32")

def _vec_exp(x):
    """Vectorized exp on float32x16 — fast degree-3 2^frac polynomial.

    Accuracy: ~0.2% rel err, sufficient for softmax bf16 output (~0.4%).
    Throughput: ~5 FMAs per 16-lane vector.
    """
    # Clamp to avoid over/underflow in ldexp
    x = T.max(x, T.broadcast(T.float32(-87.0), 16))
    x = T.min(x, T.broadcast(T.float32(88.0),  16))
    log2e = T.broadcast(T.float32(1.4426950408889634), 16)
    fx = x * log2e
    fxr = _vec_round(fx)
    frac = fx - fxr  # in [-0.5, 0.5]
    # 2^frac degree-3 polynomial (~0.2% rel error)
    c0 = T.broadcast(T.float32(1.0),         16)
    c1 = T.broadcast(T.float32(0.69314718),  16)
    c2 = T.broadcast(T.float32(0.24022651),  16)
    c3 = T.broadcast(T.float32(0.0555041),   16)
    p = c3 * frac + c2
    p = p  * frac + c1
    p = p  * frac + c0
    # 2^ipart by adding to IEEE-754 exponent
    ipart = T.cast(fxr, "int32x16")
    bias  = T.broadcast(T.int32(127), 16)
    expon = (ipart + bias) << T.broadcast(T.int32(23), 16)
    ldexp = T.reinterpret(expon, dtype="float32x16")
    return p * ldexp


# ─────────────────────────────────────────────────────────────────────
# Packing kernels (TIR)
# ─────────────────────────────────────────────────────────────────────
def make_pack_K(SEQ, D):
    """Pack K[SEQ, D] -> K_packed[D/2, SEQ*2]."""
    Dp = D // 2

    @T.prim_func(s_tir=True, check_well_formed=False)
    def pack_K(
        K_in: T.Buffer((SEQ, D), "uint16"),
        K_out: T.Buffer((Dp, SEQ * 2), "uint16"),
    ):
        for kp in T.serial(Dp):
            for n in T.serial(SEQ):
                with T.sblock("pk"):
                    vkp, vn = T.axis.remap("SS", [kp, n])
                    K_out[vkp, 2 * vn]     = K_in[vn, 2 * vkp]
                    K_out[vkp, 2 * vn + 1] = K_in[vn, 2 * vkp + 1]
    return pack_K


def make_pack_V(SEQ, D):
    """Pack V[SEQ, D] -> V_packed[SEQ/2, D*2]."""
    Sp = SEQ // 2

    @T.prim_func(s_tir=True, check_well_formed=False)
    def pack_V(
        V_in: T.Buffer((SEQ, D), "uint16"),
        V_out: T.Buffer((Sp, D * 2), "uint16"),
    ):
        for kp in T.serial(Sp):
            for n in T.serial(D):
                with T.sblock("pv"):
                    vkp, vn = T.axis.remap("SS", [kp, n])
                    V_out[vkp, 2 * vn]     = V_in[2 * vkp,     vn]
                    V_out[vkp, 2 * vn + 1] = V_in[2 * vkp + 1, vn]
    return pack_V


# ─────────────────────────────────────────────────────────────────────
# FlashAttention MHA kernel
# ─────────────────────────────────────────────────────────────────────
def make_mha_fa(SEQ, D, Mq=32, Nk=128):
    """Pure-TIR FA-style MHA kernel.
    SEQ % Mq == 0, SEQ % Nk == 0, D % 32 == 0.
    """
    assert SEQ % Mq == 0 and SEQ % Nk == 0
    assert D % 32 == 0
    assert Mq % 32 == 0, "Mq must be multiple of 32"
    assert Nk % 32 == 0
    M_pair = Mq // 32  # number of M-pairs per Q-block

    SCALE = 1.0 / math.sqrt(D)
    Dp = D // 2
    Sp = SEQ // 2

    Nq = SEQ // Mq      # number of Q-blocks
    Nkb = SEQ // Nk     # number of K-blocks per Q-block
    Nk_nb = Nk // 32    # N-blocks per inner GEMM A (S)
    D_nb  = D  // 32    # N-blocks per inner GEMM B (O)
    D_kb  = D  // 32    # K-iters per inner GEMM A (K=D)
    Nk_kb = Nk // 32    # K-iters per inner GEMM B (K=Nk)

    N_packed_K = SEQ * 2  # row stride (uint16) of K_packed
    N_packed_V = D * 2    # row stride (uint16) of V_packed
    a_row_bytes_Q  = D * 2
    b_row_bytes_K  = N_packed_K * 2
    s_row_bytes    = Nk * 4
    a_row_bytes_P  = Nk * 2
    b_row_bytes_V  = N_packed_V * 2
    o_row_bytes    = D * 4

    NEG_INF = -1.0e30

    @T.prim_func(s_tir=True, check_well_formed=False)
    def mha_fa(
        Q: T.Buffer((SEQ, D), "uint16"),
        K_packed: T.Buffer((Dp, SEQ * 2), "uint16"),
        V_packed: T.Buffer((Sp, D * 2), "uint16"),
        O: T.Buffer((SEQ, D), "float32"),
    ):
        # ── Tile config (palette 1, 8 tiles × 16 rows × 64 bytes) ─────────
        cfg = T.alloc_buffer((64,), "uint8")
        cfg[0] = T.uint8(1)
        for i in range(1, 16): cfg[i] = T.uint8(0)
        for i in range(8):
            cfg[16 + 2 * i]     = T.uint8(64)
            cfg[16 + 2 * i + 1] = T.uint8(0)
        for i in range(16): cfg[32 + i] = T.uint8(0)
        for i in range(8):  cfg[48 + i] = T.uint8(16)
        for i in range(8):  cfg[56 + i] = T.uint8(0)
        T.evaluate(_ldtilecfg(cfg.access_ptr("r")))

        # ── Per-Q-block scratch (small, lives in L1) ──────────────────────
        m_buf = T.alloc_buffer((Mq,),       "float32")
        l_buf = T.alloc_buffer((Mq,),       "float32")
        O_acc = T.alloc_buffer((Mq, D),     "float32")
        S_buf = T.alloc_buffer((Mq, Nk),    "float32")
        P_buf = T.alloc_buffer((Mq, Nk),    "uint16")  # bf16

        for mo in T.serial(Nq):
            # init m=-inf, l=0, O_acc=0 (vectorized)
            for i in T.serial(Mq):
                m_buf[i] = T.float32(NEG_INF)
                l_buf[i] = T.float32(0)
            vzero = T.broadcast(T.float32(0), 16)
            for i in T.serial(Mq):
                for jc in range(D // 16):
                    O_acc[i, T.ramp(jc * 16, 1, 16)] = vzero

            for no in T.serial(Nkb):
                # ───── Step A: S = Q_block @ K_subblock^T  ────────────────
                for mbo in T.serial(M_pair):
                    for nbo in T.serial(Nk_nb):
                        T.evaluate(_tilezero(4))
                        T.evaluate(_tilezero(5))
                        T.evaluate(_tilezero(6))
                        T.evaluate(_tilezero(7))
                        for ko in T.serial(D_kb):
                            q_top_off = _i64((mo * Mq + mbo * 32) * D + ko * 32)
                            q_bot_off = _i64((mo * Mq + mbo * 32 + 16) * D + ko * 32)
                            k_left_off  = _i64((ko * 16) * N_packed_K + (no * Nk + nbo * 32) * 2)
                            k_right_off = _i64((ko * 16) * N_packed_K + (no * Nk + nbo * 32 + 16) * 2)
                            T.evaluate(_tileloadd(0, Q.access_ptr("r", offset=q_top_off), a_row_bytes_Q))
                            T.evaluate(_tileloadd(1, Q.access_ptr("r", offset=q_bot_off), a_row_bytes_Q))
                            T.evaluate(_tileloadd(2, K_packed.access_ptr("r", offset=k_left_off),  b_row_bytes_K))
                            T.evaluate(_tileloadd(3, K_packed.access_ptr("r", offset=k_right_off), b_row_bytes_K))
                            T.evaluate(_tdpbf16ps(4, 0, 2))
                            T.evaluate(_tdpbf16ps(5, 0, 3))
                            T.evaluate(_tdpbf16ps(6, 1, 2))
                            T.evaluate(_tdpbf16ps(7, 1, 3))
                        s_tl_off = _i64((mbo * 32     ) * Nk + nbo * 32)
                        s_tr_off = _i64((mbo * 32     ) * Nk + nbo * 32 + 16)
                        s_bl_off = _i64((mbo * 32 + 16) * Nk + nbo * 32)
                        s_br_off = _i64((mbo * 32 + 16) * Nk + nbo * 32 + 16)
                        T.evaluate(_tilestored(4, S_buf.access_ptr("w", offset=s_tl_off), s_row_bytes))
                        T.evaluate(_tilestored(5, S_buf.access_ptr("w", offset=s_tr_off), s_row_bytes))
                        T.evaluate(_tilestored(6, S_buf.access_ptr("w", offset=s_bl_off), s_row_bytes))
                        T.evaluate(_tilestored(7, S_buf.access_ptr("w", offset=s_br_off), s_row_bytes))

                # ───── Step B: online softmax update + P bf16 + O_acc scale ──
                vscale = T.broadcast(T.float32(SCALE), 16)
                for i in T.serial(Mq):
                    # pass 1: vectorized max over Nk
                    vmx = T.broadcast(T.float32(NEG_INF), 16)
                    for jc in range(Nk // 16):
                        v = S_buf.vload([i, jc * 16], "float32x16") * vscale
                        vmx = T.max(vmx, v)
                    mx_local = _vec_reduce_fmax(vmx)
                    mx_new = T.max(mx_local, m_buf[i])

                    # alpha (scalar)
                    alpha = T.exp(m_buf[i] - mx_new)

                    # pass 2: vectorized exp + sum + bf16 cvt (no fp32 writeback)
                    vmxb = T.broadcast(mx_new, 16)
                    vsum = T.broadcast(T.float32(0), 16)
                    for jc in range(Nk // 32):
                        v_lo = S_buf.vload([i, jc * 32], "float32x16") * vscale - vmxb
                        v_hi = S_buf.vload([i, jc * 32 + 16], "float32x16") * vscale - vmxb
                        v_lo = _vec_exp(v_lo)
                        v_hi = _vec_exp(v_hi)
                        vsum = vsum + v_lo
                        vsum = vsum + v_hi
                        bf32 = _vec_cvt32_to_bf16x32(v_lo, v_hi)
                        P_buf[i, T.ramp(jc * 32, 1, 32)] = T.reinterpret(bf32, dtype="uint16x32")

                    sum_p = _vec_reduce_fadd(vsum)
                    l_buf[i] = l_buf[i] * alpha + sum_p
                    m_buf[i] = mx_new

                    # vectorized O_acc row scale
                    valpha = T.broadcast(alpha, 16)
                    for jc in range(D // 16):
                        v = O_acc.vload([i, jc * 16], "float32x16") * valpha
                        O_acc[i, T.ramp(jc * 16, 1, 16)] = v

                # ───── Step C: O_acc += P_buf @ V_subblock  ───────────────
                for mbo in T.serial(M_pair):
                    for nbo in T.serial(D_nb):
                        o_tl_off = _i64((mbo * 32     ) * D + nbo * 32)
                        o_tr_off = _i64((mbo * 32     ) * D + nbo * 32 + 16)
                        o_bl_off = _i64((mbo * 32 + 16) * D + nbo * 32)
                        o_br_off = _i64((mbo * 32 + 16) * D + nbo * 32 + 16)
                        T.evaluate(_tileloadd(4, O_acc.access_ptr("rw", offset=o_tl_off), o_row_bytes))
                        T.evaluate(_tileloadd(5, O_acc.access_ptr("rw", offset=o_tr_off), o_row_bytes))
                        T.evaluate(_tileloadd(6, O_acc.access_ptr("rw", offset=o_bl_off), o_row_bytes))
                        T.evaluate(_tileloadd(7, O_acc.access_ptr("rw", offset=o_br_off), o_row_bytes))
                        for ko in T.serial(Nk_kb):
                            p_top_off = _i64((mbo * 32     ) * Nk + ko * 32)
                            p_bot_off = _i64((mbo * 32 + 16) * Nk + ko * 32)
                            v_left_off  = _i64((no * (Nk // 2) + ko * 16) * N_packed_V + (nbo * 32) * 2)
                            v_right_off = _i64((no * (Nk // 2) + ko * 16) * N_packed_V + (nbo * 32 + 16) * 2)
                            T.evaluate(_tileloadd(0, P_buf.access_ptr("r", offset=p_top_off), a_row_bytes_P))
                            T.evaluate(_tileloadd(1, P_buf.access_ptr("r", offset=p_bot_off), a_row_bytes_P))
                            T.evaluate(_tileloadd(2, V_packed.access_ptr("r", offset=v_left_off),  b_row_bytes_V))
                            T.evaluate(_tileloadd(3, V_packed.access_ptr("r", offset=v_right_off), b_row_bytes_V))
                            T.evaluate(_tdpbf16ps(4, 0, 2))
                            T.evaluate(_tdpbf16ps(5, 0, 3))
                            T.evaluate(_tdpbf16ps(6, 1, 2))
                            T.evaluate(_tdpbf16ps(7, 1, 3))
                        T.evaluate(_tilestored(4, O_acc.access_ptr("w", offset=o_tl_off), o_row_bytes))
                        T.evaluate(_tilestored(5, O_acc.access_ptr("w", offset=o_tr_off), o_row_bytes))
                        T.evaluate(_tilestored(6, O_acc.access_ptr("w", offset=o_bl_off), o_row_bytes))
                        T.evaluate(_tilestored(7, O_acc.access_ptr("w", offset=o_br_off), o_row_bytes))

            # ── normalize: O = O_acc / l (vectorized) ──
            for i in T.serial(Mq):
                inv_l = T.float32(1.0) / l_buf[i]
                vinv = T.broadcast(inv_l, 16)
                for jc in range(D // 16):
                    v = O_acc.vload([i, jc * 16], "float32x16") * vinv
                    O[mo * Mq + i, T.ramp(jc * 16, 1, 16)] = v

    return mha_fa


# ─────────────────────────────────────────────────────────────────────
# Benchmark driver
# ─────────────────────────────────────────────────────────────────────
def init_amx():
    libc = ctypes.CDLL("libc.so.6")
    bm = ctypes.c_uint64(0); libc.syscall(158, 0x1022, ctypes.byref(bm))
    if not (bm.value & (1 << 18)): libc.syscall(158, 0x1023, 18)


def fp32_to_bf16(arr):
    import numpy as np
    return (arr.view(np.uint32) >> 16).astype(np.uint16)

def bf16_to_fp32(arr):
    import numpy as np
    return (arr.astype(np.uint32) << 16).view(np.float32)


def mha_numpy(Q_bf16, K_bf16, V_bf16, scale):
    import numpy as np
    Q = bf16_to_fp32(Q_bf16)
    K = bf16_to_fp32(K_bf16)
    V = bf16_to_fp32(V_bf16)
    S = Q @ K.T * scale
    S_mx = S.max(axis=1, keepdims=True)
    P = np.exp(S - S_mx)
    P = P / P.sum(axis=1, keepdims=True)
    return (P @ V).astype(np.float32)


if __name__ == "__main__":
    import time
    import json
    import numpy as np

    init_amx()
    target = tvm.target.Target({"kind": "llvm", "mcpu": "sapphirerapids"})
    dev = tvm.cpu(0)

    results = []
    for SEQ, D, Mq, Nk in [(1024, 128, 32, 256), (1024, 128, 64, 256), (1024, 128, 64, 512),
                            (2048, 128, 64, 256), (2048, 128, 64, 512), (2048, 128, 128, 512),
                            (1024, 256, 64, 256), (1024, 256, 128, 256), (1024, 512, 64, 256)]:
        scale = 1.0 / math.sqrt(D)
        print(f"\n=== SEQ={SEQ}  D={D}  Mq={Mq}  Nk={Nk} ===")

        # Compile
        t0 = time.perf_counter()
        pack_K_fn = tvm.compile(make_pack_K(SEQ, D), target=target)
        pack_V_fn = tvm.compile(make_pack_V(SEQ, D), target=target)
        mha_fn    = tvm.compile(make_mha_fa(SEQ, D, Mq, Nk), target=target)
        t_comp = (time.perf_counter() - t0) * 1000

        # Data
        np.random.seed(42)
        Q_np = np.random.randn(SEQ, D).astype(np.float32) * 0.05
        K_np = np.random.randn(SEQ, D).astype(np.float32) * 0.05
        V_np = np.random.randn(SEQ, D).astype(np.float32) * 0.05
        Q_bf = fp32_to_bf16(Q_np)
        K_bf = fp32_to_bf16(K_np)
        V_bf = fp32_to_bf16(V_np)
        O_ref = mha_numpy(Q_bf, K_bf, V_bf, scale)
        print(f"  ref O stats: min={O_ref.min():.4f} max={O_ref.max():.4f}")

        Qt  = tvm.runtime.tensor(Q_bf, dev)
        Kt  = tvm.runtime.tensor(K_bf, dev)
        Vt  = tvm.runtime.tensor(V_bf, dev)
        Kpt = tvm.runtime.tensor(np.zeros((D // 2, SEQ * 2), dtype=np.uint16), dev)
        Vpt = tvm.runtime.tensor(np.zeros((SEQ // 2, D * 2), dtype=np.uint16), dev)
        Ot  = tvm.runtime.tensor(np.zeros((SEQ, D), dtype=np.float32), dev)

        # Pre-pack (outside timed kernel)
        t0 = time.perf_counter()
        pack_K_fn(Kt, Kpt)
        pack_V_fn(Vt, Vpt)
        t_pack = (time.perf_counter() - t0) * 1000

        # Run + check correctness
        mha_fn(Qt, Kpt, Vpt, Ot)
        O = Ot.numpy()
        err = np.max(np.abs(O - O_ref))
        rel = err / (np.max(np.abs(O_ref)) + 1e-8)
        print(f"  compile {t_comp:.1f}ms  pack {t_pack:.2f}ms  err {err:.4f}  rel {rel:.4%}")

        # Warmup
        for _ in range(5):
            Ot.copyfrom(np.zeros((SEQ, D), dtype=np.float32))
            mha_fn(Qt, Kpt, Vpt, Ot)

        # Benchmark
        N_ITER = 30
        times = []
        for _ in range(N_ITER):
            Ot.copyfrom(np.zeros((SEQ, D), dtype=np.float32))
            t0 = time.perf_counter()
            mha_fn(Qt, Kpt, Vpt, Ot)
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        flops = 4.0 * SEQ * SEQ * D
        gf_mean = flops / times.mean() / 1e9
        gf_best = flops / times.min() / 1e9
        print(f"  mean {times.mean()*1e6:.1f}us  best {times.min()*1e6:.1f}us")
        print(f"  GFLOPS: mean {gf_mean:.1f}  best {gf_best:.1f}")
        results.append({"SEQ": SEQ, "D": D, "Mq": Mq, "Nk": Nk,
                        "mean_us": float(times.mean() * 1e6),
                        "best_us": float(times.min() * 1e6),
                        "gflops_mean": float(gf_mean),
                        "gflops_best": float(gf_best),
                        "max_abs_err": float(err),
                        "rel_err": float(rel)})

    print(f"\nJSON: {json.dumps(results, indent=2)}")

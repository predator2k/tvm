"""
MHA Attention Kernel Benchmark: Baseline (regular loops) vs AMX (tile intrinsics).

MHA:  O = softmax(Q @ K^T / sqrt(d)) @ V
Parameters: batch=1, heads=1, seq_len=128, d_head=32 (BF16)

The AMX version calls hand-optimized C functions (libamx_helpers.so) that use
Intel AMX tile intrinsics (_tile_dpbf16ps, _tile_stream_loadd, etc.) for the
two matrix multiplies. The baseline uses standard TIR loops.
"""

import ctypes
import json
import os
import sys
import time

import numpy as np

# Add TVM python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "3rdparty", "tvm-ffi", "python"))


def load_tvm_libraries():
    """Load TVM shared libraries via ctypes so we can call registered functions."""
    build_lib = os.path.join(os.path.dirname(__file__), "..", "..", "..", "build", "lib")
    build_lib = os.path.abspath(build_lib)

    # Set LD_LIBRARY_PATH so sub-libraries can be found
    os.environ["LD_LIBRARY_PATH"] = build_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    ffi = ctypes.CDLL(os.path.join(build_lib, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
    runtime_lib = ctypes.CDLL(os.path.join(build_lib, "libtvm_runtime.so"), mode=ctypes.RTLD_GLOBAL)
    compiler = ctypes.CDLL(os.path.join(build_lib, "libtvm_compiler.so"), mode=ctypes.RTLD_GLOBAL)
    return ffi, runtime_lib, compiler


# ─────────────────────────────────────────────────────────────────────
# TVM FFI types (ctypes)
# ─────────────────────────────────────────────────────────────────────

class TVMFFIAny(ctypes.Structure):
    _fields_ = [
        ("type_index", ctypes.c_int32),
        ("_pad", ctypes.c_uint32),
        ("data", ctypes.c_uint8 * 8),
    ]

    @property
    def v_int64(self):
        return ctypes.c_int64.from_buffer(self.data).value

    @v_int64.setter
    def v_int64(self, val):
        ctypes.c_int64.from_buffer(self.data).value = val

    @property
    def v_ptr(self):
        return ctypes.c_void_p.from_buffer(self.data).value

    @v_ptr.setter
    def v_ptr(self, val):
        ctypes.c_void_p.from_buffer(self.data).value = val


TVMFFIObjectHandle = ctypes.c_void_p


class TVMFFIByteArray(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char_p), ("size", ctypes.c_size_t)]


def setup_ffi_api(ffi):
    """Set up function signatures for the TVM FFI C API."""
    ffi.TVMFFIFunctionGetGlobal.argtypes = [
        ctypes.POINTER(TVMFFIByteArray), ctypes.POINTER(TVMFFIObjectHandle)
    ]
    ffi.TVMFFIFunctionGetGlobal.restype = ctypes.c_int
    ffi.TVMFFIFunctionCall.argtypes = [
        TVMFFIObjectHandle, ctypes.POINTER(TVMFFIAny), ctypes.c_int32, ctypes.POINTER(TVMFFIAny)
    ]
    ffi.TVMFFIFunctionCall.restype = ctypes.c_int


def get_global_func(ffi, name):
    """Look up a global packed function by name."""
    ba = TVMFFIByteArray(name.encode(), len(name))
    handle = TVMFFIObjectHandle()
    ret = ffi.TVMFFIFunctionGetGlobal(ctypes.byref(ba), ctypes.byref(handle))
    if ret != 0 or not handle.value:
        raise RuntimeError(f"Function '{name}' not found (ret={ret})")
    return handle


def call_packed_func(ffi, handle, *args):
    """Call a packed function with the given arguments (ints/pointers only)."""
    nargs = len(args)
    any_args = (TVMFFIAny * nargs)()
    for i, arg in enumerate(args):
        if isinstance(arg, int):
            any_args[i].type_index = 1  # kTVMFFIInt
            any_args[i]._pad = 0
            any_args[i].v_int64 = arg
        elif isinstance(arg, float):
            any_args[i].type_index = 3  # kTVMFFIFloat
            any_args[i]._pad = 0
            ctypes.c_double.from_buffer(any_args[i].data).value = arg
        elif isinstance(arg, ctypes.c_void_p):
            any_args[i].type_index = 4  # kTVMFFIOpaquePtr
            any_args[i]._pad = 0
            any_args[i].v_ptr = arg.value
        elif hasattr(arg, 'value'):  # ctypes pointer
            any_args[i].type_index = 4
            any_args[i]._pad = 0
            any_args[i].v_ptr = arg.value or 0
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")

    result = TVMFFIAny()
    ret = ffi.TVMFFIFunctionCall(handle, any_args, nargs, ctypes.byref(result))
    return ret, result


# ─────────────────────────────────────────────────────────────────────
# Pure NumPy MHA (reference implementation, for correctness checking)
# ─────────────────────────────────────────────────────────────────────

def bf16_to_fp32(arr_uint16):
    """Convert uint16 (bf16 bits) to float32 by shifting left 16 bits."""
    u32 = arr_uint16.astype(np.uint32) << 16
    return u32.view(np.float32)

def mha_numpy(Q_bf16, K_bf16, V_bf16, scale):
    """Reference MHA implementation in NumPy (fp32 math with bf16 inputs)."""
    Q = bf16_to_fp32(Q_bf16)
    K = bf16_to_fp32(K_bf16)
    V = bf16_to_fp32(V_bf16)

    S = Q @ K.T * scale           # [S, S]
    S_max = S.max(axis=1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=1, keepdims=True)
    O = P @ V                     # [S, D]
    return O.astype(np.float32), S.astype(np.float32), P.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Pure C baseline (no AMX) using standard loops
# ─────────────────────────────────────────────────────────────────────

BASELINE_C_SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Convert bf16 (stored as uint16) to float
static inline float bf16_to_f32(unsigned short v) {
    unsigned int bits = ((unsigned int)v) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Convert float to bf16
static inline unsigned short f32_to_bf16(float v) {
    unsigned int bits;
    memcpy(&bits, &v, sizeof(bits));
    return (unsigned short)(bits >> 16);
}

// MHA baseline: Q,K,V in bf16 [S,D], output in fp32 [S,D]
// S=seq_len, D=d_head
void mha_baseline(int S, int D, const unsigned short *Q, const unsigned short *K,
                  const unsigned short *V, float scale, float *O) {
    // Allocate temporary buffers
    float *S_mat = (float *)aligned_alloc(64, S * S * sizeof(float));
    float *P_mat = (float *)aligned_alloc(64, S * S * sizeof(float));

    // S_mat = Q @ K^T * scale
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += bf16_to_f32(Q[i * D + k]) * bf16_to_f32(K[j * D + k]);
            }
            S_mat[i * S + j] = sum * scale;
        }
    }

    // P = softmax(S_mat), row-wise
    for (int i = 0; i < S; i++) {
        float mx = S_mat[i * S];
        for (int j = 1; j < S; j++)
            if (S_mat[i * S + j] > mx) mx = S_mat[i * S + j];
        float sum = 0.0f;
        for (int j = 0; j < S; j++) {
            P_mat[i * S + j] = expf(S_mat[i * S + j] - mx);
            sum += P_mat[i * S + j];
        }
        for (int j = 0; j < S; j++)
            P_mat[i * S + j] /= sum;
    }

    // O = P @ V
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            for (int k = 0; k < S; k++) {
                sum += P_mat[i * S + k] * bf16_to_f32(V[k * D + j]);
            }
            O[i * D + j] = sum;
        }
    }

    free(S_mat);
    free(P_mat);
}
"""

# ─────────────────────────────────────────────────────────────────────
# Pure C baseline without AMX also (fair comparison: both C, one with AMX intrinsics, one without)
# ─────────────────────────────────────────────────────────────────────

BASELINE_GEMM_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>
#include <math.h>

static inline float bf16_to_f32(unsigned short v) {
    unsigned int bits = ((unsigned int)v) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline unsigned short f32_to_bf16(float v) {
    unsigned int bits;
    memcpy(&bits, &v, sizeof(bits));
    return (unsigned short)(bits >> 16);
}

// Standard BF16 GEMM: C[M][N] += A[M][K] * B[N][K]  (NT form, same API as AMX version)
void baseline_bf16_gemm_nt(int M, int N, int K, const unsigned short *A, int lda,
                           const unsigned short *B, int ldb, float *C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += bf16_to_f32(A[m * lda + k]) * bf16_to_f32(B[n * ldb + k]);
            }
            C[m * ldc + n] += sum;
        }
    }
}

void baseline_softmax_fp32(float *data, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float *row = data + i * cols;
        float mx = row[0];
        for (int j = 1; j < cols; j++)
            if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        for (int j = 0; j < cols; j++)
            row[j] /= sum;
    }
}
"""


def compile_c_lib(source, name, extra_flags=""):
    """Compile a C source string into a shared library."""
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp()
    src_path = os.path.join(tmpdir, f"{name}.c")
    lib_path = os.path.join(tmpdir, f"lib{name}.so")

    with open(src_path, "w") as f:
        f.write(source)

    cmd = f"gcc -shared -fPIC -O3 {extra_flags} -o {lib_path} {src_path} -lm"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed for {name}:\n{result.stderr}")

    return lib_path


# ─────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────

def benchmark_mha():
    # Parameters
    SEQ_LEN = 128
    D_HEAD = 32
    SCALE = 1.0 / np.sqrt(D_HEAD)
    N_WARMUP = 5
    N_ITER = 50

    print(f"{'='*70}")
    print(f"MHA Attention Benchmark: Baseline vs AMX")
    print(f"  seq_len={SEQ_LEN}, d_head={D_HEAD}, scale={SCALE:.6f}")
    print(f"  warmup={N_WARMUP}, iterations={N_ITER}")
    print(f"{'='*70}")

    # Generate test data
    np.random.seed(42)
    Q_np = np.random.randn(SEQ_LEN, D_HEAD).astype(np.float32)
    K_np = np.random.randn(SEQ_LEN, D_HEAD).astype(np.float32)
    V_np = np.random.randn(SEQ_LEN, D_HEAD).astype(np.float32)

    # Convert to BF16 (stored as uint16)
    def fp32_to_bf16_bytes(arr):
        u32 = arr.view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        return u16

    Q_bf16 = fp32_to_bf16_bytes(Q_np)
    K_bf16 = fp32_to_bf16_bytes(K_np)
    V_bf16 = fp32_to_bf16_bytes(V_np)

    # Compute reference result
    O_ref, S_ref, P_ref = mha_numpy(Q_bf16, K_bf16, V_bf16, SCALE)
    print(f"\nReference MHA output stats: "
          f"min={O_ref.min():.6f}, max={O_ref.max():.6f}, mean={O_ref.mean():.6f}")

    # ─── Compile baseline C library (no AMX) ───
    print("\n--- Compiling baseline (standard loops, no AMX) ---")
    baseline_lib_path = compile_c_lib(BASELINE_GEMM_C_SOURCE, "baseline_gemm", "")
    baseline_lib = ctypes.CDLL(baseline_lib_path)

    baseline_lib.baseline_bf16_gemm_nt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    baseline_lib.baseline_softmax_fp32.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
    ]

    # ─── Load AMX C library ───
    print("--- Loading AMX library ---")
    amx_lib_path = os.path.join(os.path.dirname(__file__), "libamx_helpers.so")
    amx_lib_path = os.path.abspath(amx_lib_path)
    amx_lib = ctypes.CDLL(amx_lib_path)

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

    # ─── Initialize AMX hardware ───
    print("--- Initializing AMX hardware ---")
    ffi, runtime_lib, compiler_lib = load_tvm_libraries()
    setup_ffi_api(ffi)

    try:
        h_init = get_global_func(ffi, "runtime.amx_init")
        ret, rv = call_packed_func(ffi, h_init)
        if ret == 0 and rv.v_int64 == 1:
            print("  AMX hardware initialized successfully")
        else:
            print(f"  AMX init returned: ret={ret}, val={rv.v_int64}")
    except Exception as e:
        print(f"  Failed to init AMX: {e}")
        print("  Continuing anyway (AMX might still work)...")

    def run_baseline_inference():
        """Run MHA with baseline (no AMX) C library."""
        # Allocate buffers
        S_mat = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        O_mat = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)

        Q_ptr = Q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        K_ptr = K_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        S_ptr = S_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Step 1: S = Q @ K^T * scale
        S_mat.fill(0.0)
        baseline_lib.baseline_bf16_gemm_nt(
            SEQ_LEN, SEQ_LEN, D_HEAD, Q_ptr, D_HEAD, K_ptr, D_HEAD, S_ptr, SEQ_LEN)
        S_mat *= SCALE

        # Step 2: P = softmax(S) in-place
        baseline_lib.baseline_softmax_fp32(S_ptr, SEQ_LEN, SEQ_LEN)

        # Step 3: O = P @ V
        # Convert P (fp32) to bf16 for the matmul
        P_bf16 = fp32_to_bf16_bytes(S_mat)  # S_mat now contains P
        V_T = np.ascontiguousarray(V_bf16.T)
        P_bf16_ptr = P_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        V_T_ptr = V_T.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

        O_mat.fill(0.0)
        baseline_lib.baseline_bf16_gemm_nt(
            SEQ_LEN, D_HEAD, SEQ_LEN, P_bf16_ptr, SEQ_LEN, V_T_ptr, SEQ_LEN, O_ptr, D_HEAD)

        return O_mat

    def run_amx_inference():
        """Run MHA with AMX-accelerated C library."""
        # Allocate buffers
        S_mat = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        O_mat = np.zeros((SEQ_LEN, D_HEAD), dtype=np.float32)

        Q_ptr = Q_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        K_ptr = K_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        V_ptr = V_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        S_ptr = S_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        O_ptr = O_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Step 1: S = Q @ K^T using AMX
        S_mat.fill(0.0)
        amx_lib.amx_bf16_gemm_nt(
            SEQ_LEN, SEQ_LEN, D_HEAD, Q_ptr, D_HEAD, K_ptr, D_HEAD, S_ptr, SEQ_LEN)
        S_mat *= SCALE

        # Step 2: softmax
        amx_lib.softmax_fp32(S_ptr, SEQ_LEN, SEQ_LEN)

        # Step 3: O = P @ V using AMX
        # Convert P (S after softmax) to bf16
        P_bf16 = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.uint16)
        P_bf16_ptr = P_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
        amx_lib.fp32_to_bf16(S_ptr, P_bf16_ptr, SEQ_LEN * SEQ_LEN)

        # V^T: [D_HEAD, SEQ_LEN]
        V_T = np.ascontiguousarray(V_bf16.T)
        V_T_ptr = V_T.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

        O_mat.fill(0.0)
        amx_lib.amx_bf16_gemm_nt(
            SEQ_LEN, D_HEAD, SEQ_LEN, P_bf16_ptr, SEQ_LEN, V_T_ptr, SEQ_LEN, O_ptr, D_HEAD)

        return O_mat

    # ─── Warmup ───
    print(f"\n--- Warming up ({N_WARMUP} iterations) ---")
    for _ in range(N_WARMUP):
        run_baseline_inference()
        run_amx_inference()

    # ─── Benchmark baseline ───
    print(f"\n--- Benchmarking baseline ({N_ITER} iterations) ---")
    times_baseline = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        run_baseline_inference()
        t1 = time.perf_counter()
        times_baseline.append((t1 - t0) * 1000)  # ms

    times_baseline = np.array(times_baseline)
    baseline_mean = np.mean(times_baseline)
    baseline_std = np.std(times_baseline)
    baseline_min = np.min(times_baseline)

    # ─── Benchmark AMX ───
    print(f"--- Benchmarking AMX ({N_ITER} iterations) ---")
    times_amx = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        run_amx_inference()
        t1 = time.perf_counter()
        times_amx.append((t1 - t0) * 1000)  # ms

    times_amx = np.array(times_amx)
    amx_mean = np.mean(times_amx)
    amx_std = np.std(times_amx)
    amx_min = np.min(times_amx)

    # ─── Verify correctness ───
    O_baseline = run_baseline_inference()
    O_amx = run_amx_inference()

    # For the baseline PV matmul, P is cast directly (bf16 truncation of fp32 P values),
    # which is slightly different from using fp32 values. Let's compare both to reference.
    baseline_error = np.max(np.abs(O_baseline - O_ref))
    amx_error = np.max(np.abs(O_amx - O_ref))
    rel_baseline_error = baseline_error / (np.max(np.abs(O_ref)) + 1e-8)
    rel_amx_error = amx_error / (np.max(np.abs(O_ref)) + 1e-8)

    # ─── Results ───
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"{'':<20} {'Baseline (no AMX)':<25} {'AMX-Optimized':<25} {'Speedup'}")
    print(f"{'-'*70}")
    print(f"{'Mean time (ms)':<20} {baseline_mean:<25.4f} {amx_mean:<25.4f} {baseline_mean/amx_mean:<25.2f}x")
    print(f"{'Std (ms)':<20} {baseline_std:<25.4f} {amx_std:<25.4f}")
    print(f"{'Min time (ms)':<20} {baseline_min:<25.4f} {amx_min:<25.4f} {baseline_min/amx_min:<25.2f}x")
    print(f"{'Max abs error':<20} {baseline_error:<25.6f} {amx_error:<25.6f}")
    print(f"{'Max rel error':<20} {rel_baseline_error:<25.6f} {rel_amx_error:<25.6f}")
    print(f"{'-'*70}")

    # Compute FLOPs: 2 * seq_len^2 * d_head + 2 * seq_len^2 * d_head (two matmuls)
    flops = 4 * SEQ_LEN * SEQ_LEN * D_HEAD
    baseline_gflops = flops / (baseline_mean / 1000) / 1e9
    amx_gflops = flops / (amx_mean / 1000) / 1e9
    print(f"{'GFLOPS':<20} {baseline_gflops:<25.2f} {amx_gflops:<25.2f} {amx_gflops/baseline_gflops:<25.2f}x")

    print(f"\n{'='*70}")
    if amx_mean < baseline_mean:
        print(f"AMX is {baseline_mean/amx_mean:.2f}x faster than baseline!")
    else:
        print(f"AMX did not show speedup ({baseline_mean/amx_mean:.2f}x).")

    return {
        "baseline_mean_ms": baseline_mean,
        "amx_mean_ms": amx_mean,
        "speedup": baseline_mean / amx_mean,
        "baseline_gflops": baseline_gflops,
        "amx_gflops": amx_gflops,
        "baseline_error": float(baseline_error),
        "amx_error": float(amx_error),
    }


if __name__ == "__main__":
    results = benchmark_mha()
    print(f"\nResults JSON: {json.dumps(results, indent=2)}")

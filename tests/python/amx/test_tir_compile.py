"""Test: compile a simple TIR kernel via TVM."""
import os, sys, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "3rdparty", "tvm-ffi", "python"))

BUILD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "build", "lib"))
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi.so"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(BUILD, "libtvm_ffi_testing.so"), mode=ctypes.RTLD_GLOBAL)

import tvm
from tvm.script import tirx as T

@T.prim_func
def simple_gemm(
    A: T.Buffer((16, 32), "uint16"),
    B: T.Buffer((16, 32), "uint16"),
    C: T.Buffer((16, 16), "float32"),
):
    for i, j, k in T.grid(16, 16, 32):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + T.Cast("float32", A[vi, vk]) * T.Cast("float32", B[vj, vk])

print("=== TIR PrimFunc ===")
print(simple_gemm)

# Compile
target = tvm.target.Target("llvm -mcpu=sapphirerapids")
try:
    compiled = tvm.compile(simple_gemm, target=target)
    print(f"\nCompilation SUCCESS: {type(compiled)}")
except Exception as e:
    print(f"\nCompilation failed: {e}")
    # Try alternative
    mod = tvm.IRModule({"main": simple_gemm})
    compiled = tvm.compile(mod, target=target)
    print(f"Compiled from IRModule: {type(compiled)}")

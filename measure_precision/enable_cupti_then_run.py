import os, sys, ctypes

# 1) Make sure the loader can find CUPTI (adjust if CUDA is elsewhere)
os.environ.setdefault("LD_LIBRARY_PATH", "")
cupti_libdir = "/usr/local/cuda/extras/CUPTI/lib64"
cuda_libdir  = "/usr/local/cuda/lib64"
os.environ["LD_LIBRARY_PATH"] = f"{cupti_libdir}:{cuda_libdir}:" + os.environ["LD_LIBRARY_PATH"]

# 2) Load CUDA driver and call cuInit(0) BEFORE any CUDA user libs are imported
libcuda = ctypes.CDLL("libcuda.so.1")
cuInit  = libcuda.cuInit
cuInit.argtypes = [ctypes.c_uint]
cuInit.restype  = ctypes.c_int
assert cuInit(0) == 0, "cuInit(0) failed"

# 3) Load CUPTI and enable activities (this toggles high-res timer domain)
libcupti = ctypes.CDLL("libcupti.so.12")  # change to .so.11 / .so.10 if needed
def cupti_enable(kind):
    fn = libcupti.cuptiActivityEnable
    fn.argtypes = [ctypes.c_int]
    fn.restype  = ctypes.c_int
    rc = fn(kind)
    if rc != 0:  # CUPTI_SUCCESS == 0
        raise RuntimeError(f"cuptiActivityEnable({kind}) failed with {rc}")

# Common kinds that reliably flip high-res
CUPTI_ACTIVITY_KIND_DRIVER  = 0
CUPTI_ACTIVITY_KIND_RUNTIME = 1
CUPTI_ACTIVITY_KIND_KERNEL  = 2
CUPTI_ACTIVITY_KIND_MEMCPY  = 3
CUPTI_ACTIVITY_KIND_MEMSET  = 4

for kind in (CUPTI_ACTIVITY_KIND_DRIVER,
             CUPTI_ACTIVITY_KIND_RUNTIME,
             CUPTI_ACTIVITY_KIND_KERNEL,
             CUPTI_ACTIVITY_KIND_MEMCPY,
             CUPTI_ACTIVITY_KIND_MEMSET):
    cupti_enable(kind)

# Optional: flush once (no harm)
libcupti.cuptiActivityFlushAll(0)

# 4) Now run your real script (which will see high-res if supported)
#    Pass through all original args after this script name.
if len(sys.argv) < 2:
    print("Usage: python enable_cupti_then_run.py <your_script.py> [args...]")
    sys.exit(1)

script = sys.argv[1]
sys.argv = sys.argv[1:]  # so the target script sees its own argv[0]
with open(script, "rb") as f:
    code = compile(f.read(), script, "exec")
exec(code, {"__name__": "__main__"})

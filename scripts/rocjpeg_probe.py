"""Probe whether rocJPEG can run here. Re-run after every AMD driver / ROCm update.

In WSL2 (ROCm 10.0, Sept 2026): HARDWARE fails (no /dev/dri for VA-API), HYBRID = NOT_IMPLEMENTED.
Exit code 0 = HARDWARE backend usable (OmniScan's codec will auto-select it), 1 = not usable.
"""

import ctypes
import ctypes.util
import os
import sys

CANDIDATES = [
    "/opt/rocm/core-10.0/lib/librocjpeg.so.1",
    "/opt/rocm/lib/librocjpeg.so.1",
    ctypes.util.find_library("rocjpeg") or "",
]
STATUS = {
    0: "SUCCESS",
    -1: "NOT_INITIALIZED",
    -2: "INVALID_PARAMETER",
    -3: "BAD_JPEG",
    -4: "JPEG_NOT_SUPPORTED",
    -5: "OUTOF_MEMORY",
    -6: "EXECUTION_FAILED",
    -7: "ARCH_MISMATCH",
    -8: "INTERNAL_ERROR",
    -9: "IMPLEMENTATION_NOT_SUPPORTED",
    -10: "HW_JPEG_DECODER_NOT_SUPPORTED",
    -11: "RUNTIME_ERROR",
    -12: "NOT_IMPLEMENTED",
}
BACKENDS = {"HARDWARE": 0, "HYBRID": 1}


def main() -> int:
    path = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
    if path is None:
        print("librocjpeg not found")
        return 1
    lib = ctypes.CDLL(path)
    lib.rocJpegCreate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.rocJpegCreate.restype = ctypes.c_int
    lib.rocJpegDestroy.argtypes = [ctypes.c_void_p]
    print(f"library: {path}   /dev/dri present: {os.path.exists('/dev/dri')}")
    usable = False
    for name, backend in BACKENDS.items():
        handle = ctypes.c_void_p()
        status = lib.rocJpegCreate(backend, 0, ctypes.byref(handle))
        print(f"rocJpegCreate({name}) -> {status} {STATUS.get(status, '?')}")
        if status == 0:
            lib.rocJpegDestroy(handle)
            usable |= name == "HARDWARE"
    print("rocJPEG HARDWARE usable:", usable)
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())

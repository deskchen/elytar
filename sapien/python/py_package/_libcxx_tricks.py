import ctypes
import os
import platform

if platform.system() == "Linux":
    _search_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
    ]

    def _find(name):
        for d in _search_dirs:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    _cxxabi = _find("libc++abi.so.1")
    _cxx = _find("libc++.so.1")

    if _cxxabi:
        ctypes.CDLL(_cxxabi, ctypes.RTLD_GLOBAL)
    if _cxx:
        ctypes.CDLL(_cxx, ctypes.RTLD_GLOBAL)

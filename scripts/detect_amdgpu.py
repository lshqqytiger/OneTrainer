import ctypes
import ctypes.wintypes
import os
import sys

from modules.util import rocm


class hipDeviceProp(ctypes.Structure):
    _fields_ = [
        ('__front__', ctypes.c_byte * 396),
        ('gcnArchName', ctypes.c_char * 256),
        ('__rear__', ctypes.c_byte * 820)
    ]

class HIP:
    def __init__(self):
        ctypes.windll.kernel32.LoadLibraryA.restype = ctypes.wintypes.HMODULE
        ctypes.windll.kernel32.LoadLibraryA.argtypes = [ctypes.c_char_p]
        path = os.environ.get("windir", "C:\\Windows") + "\\System32\\amdhip64_6.dll"  # noqa: SIM112
        if not os.path.isfile(path):
            path = os.environ.get("windir", "C:\\Windows") + "\\System32\\amdhip64_7.dll"  # noqa: SIM112
        if not os.path.isfile(path):
            sys.exit(1)
        self.handle = ctypes.windll.kernel32.LoadLibraryA(path.encode('utf-8'))
        ctypes.windll.kernel32.GetLastError.restype = ctypes.wintypes.DWORD
        ctypes.windll.kernel32.GetLastError.argtypes = []
        if ctypes.windll.kernel32.GetLastError() != 0:
            sys.exit(1)
        ctypes.windll.kernel32.GetProcAddress.restype = ctypes.c_void_p
        ctypes.windll.kernel32.GetProcAddress.argtypes = [ctypes.wintypes.HMODULE, ctypes.c_char_p]
        self.hipGetDeviceCount = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int))(
            ctypes.windll.kernel32.GetProcAddress(self.handle, b"hipGetDeviceCount"))
        self.hipGetDeviceProperties = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(hipDeviceProp), ctypes.c_int)(
            ctypes.windll.kernel32.GetProcAddress(self.handle, b"hipGetDeviceProperties"))

    def __del__(self):
        # Hopefully this will prevent conflicts with amdhip64_7.dll from ROCm Python packages or HIP SDK
        ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
        ctypes.windll.kernel32.FreeLibrary(self.handle)

    def get_device_count(self):
        count = ctypes.c_int()
        if self.hipGetDeviceCount(ctypes.byref(count)) != 0:
            sys.exit(1)
        return count.value

    def get_device_properties(self, device_id):
        prop = hipDeviceProp()
        if self.hipGetDeviceProperties(ctypes.byref(prop), device_id) != 0:
            sys.exit(1)
        return prop


def detect_amdgpu() -> str:
    if sys.platform != "win32":
        sys.exit(1)

    hip = HIP()
    count = hip.get_device_count()
    amd_gpus = []
    for i in range(count):
        prop = hip.get_device_properties(i)
        name = prop.gcnArchName.decode('utf-8').strip('\x00')
        amd_gpus.append(rocm.Agent(name))
    del hip

    if len(amd_gpus) == 0:
        sys.exit(1)

    index = 0
    for idx, gpu in enumerate(amd_gpus):
        index = idx
        if not gpu.is_apu:
            # although apu was found, there can be a dedicated card. do not break loop.
            # if no dedicated card was found, apu will be used.
            break

    return amd_gpus[index].therock

import sys

from modules.util.config.TrainConfig import TrainConfig
from modules.zluda import ZLUDAInstaller
from modules.zluda.ZLUDAInstaller import core, default_agent  # noqa: F401

import torch
from torch._prims_common import DeviceLikeType

import onnxruntime as ort

PLATFORM = sys.platform
do_nothing = lambda _: None  # noqa: E731


def is_zluda(device: DeviceLikeType):
    device = torch.device(device)
    if device.type in ["cpu", "mps"]:
        return False
    return torch.cuda.get_device_name(device).endswith("[ZLUDA]")


def test(device: DeviceLikeType) -> Exception | None:
    device = torch.device(device)
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        assert out.sum().is_nonzero()
        return None
    except Exception as e:
        return e


def initialize():
    torch.backends.cudnn.enabled = ZLUDAInstaller.MIOpen_enabled
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        if not ZLUDAInstaller.MIOpen_enabled:
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp = do_nothing
    else:
        torch.backends.cuda.enable_cudnn_sdp = do_nothing
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_flash_sdp = torch.backends.cuda.enable_cudnn_sdp
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp = do_nothing

    # ONNX Runtime is not supported
    ort.capi._pybind_state.get_available_providers = lambda: [v for v in ort.get_available_providers() if v != "CUDAExecutionProvider"] # pylint: disable=protected-access
    ort.get_available_providers = ort.capi._pybind_state.get_available_providers # pylint: disable=protected-access


def initialize_devices(config: TrainConfig):
    if not is_zluda(config.train_device) and not is_zluda(config.temp_device):
        return
    devices = [config.train_device, config.temp_device,]
    for i in range(2):
        device = torch.device(devices[i])
        result = test(device)
        if result is not None:
            print(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={torch.cuda.get_device_name(device)}')
            print(result)
            devices[i] = 'cpu'
    config.train_device, config.temp_device = devices

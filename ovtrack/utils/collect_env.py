import cv2
import mmcv
import os.path as osp
import shutil
import subprocess
import sys
import torch
import torchvision
from collections import defaultdict



def collect_env():
    env_info = {}
    env_info["sys.platform"] = sys.platform
    env_info["Python"] = sys.version.replace("\n", "")

    cuda_available = torch.cuda.is_available()
    env_info["CUDA available"] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME

        env_info["CUDA_HOME"] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, "bin", "nvcc")
                if sys.platform == "win32":
                    nvcc += ".exe"
                nvcc = subprocess.check_output(
                    [nvcc, "-V"], stderr=subprocess.STDOUT
                ).decode("utf-8", errors="replace").splitlines()[-1]
            except (OSError, subprocess.SubprocessError, IndexError):
                nvcc = "Not Available"
            env_info["NVCC"] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info["GPU " + ",".join(devids)] = name

    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler is None and sys.platform == "win32":
        compiler = shutil.which("cl")
    if compiler is not None:
        try:
            compiler_info = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.STDOUT
            ).decode("utf-8", errors="replace").splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            compiler_info = "Not Available"
    else:
        compiler_info = "Not Available"
    env_info["Compiler"] = compiler_info

    env_info["PyTorch"] = torch.__version__
    env_info["PyTorch compiling details"] = torch.__config__.show()

    env_info["TorchVision"] = torchvision.__version__

    env_info["OpenCV"] = cv2.__version__

    env_info["MMCV"] = mmcv.__version__

    return env_info


if __name__ == "__main__":
    for name, val in collect_env().items():
        print(f"{name}: {val}")

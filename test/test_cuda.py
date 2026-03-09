import torch
from torch import version as torch_version


def test_cuda():
    available = torch.cuda.is_available()
    print(f"Is CUDA supported by this system? {available}")

    if available:
        print(f"CUDA version: {torch_version.cuda}")

        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")

        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

"""Verify installed components without fetching model weights."""
import os
import sys
from importlib import import_module
from importlib.metadata import distribution
from pathlib import Path

required = {
    "ftgspp-vulkan-community": "ftgspp",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "opencv-python-headless": "cv2",
    "pycolmap": "pycolmap",
    "scipy": "scipy",
    "tqdm": "tqdm",
}
extras = {value for value in os.environ["FTGSPP_INSTALL_EXTRAS"].split(",") if value}
if "birefnet" in extras:
    required.update({
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "safetensors": "safetensors",
        "timm": "timm",
        "kornia": "kornia",
        "einops": "einops",
    })
if "da3" in extras:
    required.update({
        "depth-anything-3": "depth_anything_3.api",
        "torch": "torch",
        "torchvision": "torchvision",
        "safetensors": "safetensors",
        "einops": "einops",
        "addict": "addict",
        "evo": "evo",
        "imageio": "imageio",
        "moviepy": "moviepy",
        "pillow": "PIL",
        "plyfile": "plyfile",
        "trimesh": "trimesh",
        "huggingface-hub": "huggingface_hub",
    })
if "lpips" in extras:
    required.update({
        "torch": "torch",
        "torchvision": "torchvision",
        "torchmetrics": "torchmetrics",
    })
if "remote" in extras:
    required["websockets"] = "websockets"
if "tensorboard" in extras:
    required["tensorboard"] = "tensorboard"

venv = Path(sys.prefix).resolve()
versions = []
for package, module in required.items():
    item = distribution(package)
    location = Path(item.locate_file("")).resolve()
    if not location.is_relative_to(venv):
        raise RuntimeError(
            f"{package} resolved outside .venv: {location}; remove foreign "
            ".pth/PYTHONPATH injection and rerun setup"
        )
    loaded = import_module(module)
    module_file = getattr(loaded, "__file__", None)
    if module_file is not None and not Path(module_file).resolve().is_relative_to(venv):
        raise RuntimeError(f"{module} imported from outside .venv: {module_file}")
    versions.append(f"{package}={item.version}")
print("verified runtime:", " ".join(versions))

if "birefnet" in extras:
    from transformers import AutoModelForImageSegmentation

if "torch" in required:
    import torch
    available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {available}; wheel CUDA: {torch.version.cuda}")
    if available:
        # Execute a kernel: import success alone misses incompatible CUDA wheels.
        value = torch.arange(4, device="cuda", dtype=torch.float32).sum().item()
        if value != 6.0:
            raise RuntimeError("PyTorch CUDA smoke check returned an invalid result")
    else:
        print("warning: optional preparation will need CPU mode or a working CUDA driver", file=sys.stderr)

import_module("ftgspp._ftgspp_vk")
if os.environ.get("FTGSPP_INSTALL_VIEWER") == "1":
    import_module("ftgspp._ftgspp_viewer")

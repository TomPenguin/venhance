"""RIFE inference wrapper: numpy RGB frames in/out, MPS/CPU device handling."""

from __future__ import annotations

import numpy as np
import torch

from .models import MODELS, ensure_model


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Rife:
    def __init__(
        self, model_name: str, device: str = "auto", fp16: bool | None = None
    ) -> None:
        from .vendor.rife_arch import IFNet

        spec = MODELS[model_name]
        weights = ensure_model(model_name)
        self.device = resolve_device(device)
        self.scale_list = list(spec.scale_list)

        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        net = IFNet(arch_ver=spec.arch_ver)
        net.load_state_dict(state, strict=True)
        if fp16 is None:
            fp16 = self.device.type in ("mps", "cuda")
        self.dtype = torch.float16 if fp16 else torch.float32
        self.net = net.eval().to(self.device, self.dtype)

    @property
    def precision(self) -> str:
        return "fp16" if self.dtype == torch.float16 else "fp32"

    def to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 -> 1x3xHxW float32 in [0,1] on device.

        Kept public so the caller can convert each decoded frame once and
        reuse it as img0 of the next pair.
        """
        arr = np.ascontiguousarray(frame)
        if not arr.flags.writeable:
            arr = arr.copy()
        t = torch.from_numpy(arr)
        return t.to(self.device).permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)

    @torch.inference_mode()
    def interpolate(
        self, img0: torch.Tensor, img1: torch.Tensor, t: float
    ) -> np.ndarray:
        """Intermediate frame at blend position t in (0, 1); returns HxWx3 uint8."""
        out = self.net(
            img0,
            img1,
            timestep=t,
            scale_list=self.scale_list,
            training=False,
            fastmode=True,
            ensemble=False,
        )
        return (
            out[0].float().clamp_(0, 1).permute(1, 2, 0).mul_(255.0).round_()
            .to(torch.uint8).cpu().numpy()
        )

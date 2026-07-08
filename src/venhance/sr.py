"""Real-ESRGAN inference wrapper: numpy RGB frames in/out, tiled inference."""

from __future__ import annotations

import numpy as np
import torch

from .models import SR_MODELS, ensure_model
from .rife import resolve_device

TILE_PAD = 16  # overlap fed to the net around each tile to hide seams


class Upscaler:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        tile: int | None = None,
        fp16: bool | None = None,
    ) -> None:
        spec = SR_MODELS[model_name]
        weights = ensure_model(model_name)
        self.device = resolve_device(device)
        self.native_scale = spec.scale

        state = torch.load(weights, map_location="cpu", weights_only=True)
        params = state[spec.state_key]
        if spec.arch == "compact":
            from .vendor.realesrgan_arch import SRVGGNetCompact

            net = SRVGGNetCompact(
                num_feat=spec.num_feat, num_conv=spec.num_conv, upscale=spec.scale
            )
        else:
            from .vendor.realesrgan_arch import RRDBNet

            net = RRDBNet(
                3, 3, scale=spec.scale, num_feat=spec.num_feat, num_block=spec.num_block
            )
        net.load_state_dict(params, strict=True)

        if fp16 is None:
            fp16 = self.device.type in ("mps", "cuda")
        self.dtype = torch.float16 if fp16 else torch.float32
        self.net = net.eval().to(self.device, self.dtype)

        # RRDB won't fit a whole HD frame comfortably; compact models will.
        self.tile = (0 if spec.arch == "compact" else 512) if tile is None else tile

    @property
    def precision(self) -> str:
        return "fp16" if self.dtype == torch.float16 else "fp32"

    @torch.inference_mode()
    def upscale(self, frame: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        """HxWx3 uint8 -> (out_h, out_w)x3 uint8.

        Runs the net at its native scale, then area-resizes to out_size when a
        smaller output scale was requested.
        """
        arr = np.ascontiguousarray(frame)
        x = (
            torch.from_numpy(arr).to(self.device)
            .permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)
        )
        out = self._infer(x)
        oh, ow = out_size
        if out.shape[-2:] != (oh, ow):
            out = torch.nn.functional.interpolate(out.float(), size=(oh, ow), mode="area")
        return (
            out[0].float().clamp_(0, 1).permute(1, 2, 0).mul_(255.0).round_()
            .to(torch.uint8).cpu().numpy()
        )

    def _infer(self, x: torch.Tensor) -> torch.Tensor:
        if self.tile <= 0:
            return self.net(x)
        b, c, h, w = x.shape
        s = self.native_scale
        out = torch.empty((b, c, h * s, w * s), dtype=x.dtype, device=x.device)
        for y0 in range(0, h, self.tile):
            for x0 in range(0, w, self.tile):
                y1, x1 = min(y0 + self.tile, h), min(x0 + self.tile, w)
                py0, px0 = max(y0 - TILE_PAD, 0), max(x0 - TILE_PAD, 0)
                py1, px1 = min(y1 + TILE_PAD, h), min(x1 + TILE_PAD, w)
                tile_out = self.net(x[:, :, py0:py1, px0:px1])
                out[:, :, y0 * s : y1 * s, x0 * s : x1 * s] = tile_out[
                    :, :,
                    (y0 - py0) * s : (y1 - py0) * s,
                    (x0 - px0) * s : (x1 - px0) * s,
                ]
        return out

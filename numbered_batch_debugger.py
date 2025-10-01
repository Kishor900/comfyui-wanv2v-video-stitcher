import math
from typing import Tuple, Any

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# This node takes a batch of images (e.g., ControlNet frames),
# and writes a 1-based index number on the top-right of each image.
# Intended purely for debugging/visual tracing of batched flows.

class NumberedBatchDebugger:
    """
    Inputs:
      - images (IMAGE): A batched tensor of images shaped (B, H, W, C) with float32 in [0,1]
      - start_index (INT): Optional, default 1. 1-based index to start numbering from.
      - margin (INT): Optional, padding from top-right corner.
      - scale (FLOAT): Optional, text size scaling relative to min(H, W).
      - outline (BOOL): Optional, draw a black outline for better visibility.

    Output:
      - IMAGE: Same batch with the index text stamped at top-right.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "start_index": ("INT", {"default": 1, "min": 1, "max": 1_000_000, "step": 1}),
                "margin": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1}),
                "scale": ("FLOAT", {"default": 0.06, "min": 0.02, "max": 0.25, "step": 0.005}),
                "outline": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "stamp_numbers"
    CATEGORY = "Wanv2vControlnetManager/Debug"

    def _to_pil(self, t: torch.Tensor) -> Image.Image:
        """
        Convert a single image tensor [H,W,C], float in [0,1] to PIL Image.
        """
        t = t.clamp(0.0, 1.0)
        arr = (t.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    def _from_pil(self, img: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image back to tensor [H,W,C], float in [0,1].
        """
        arr = np.asarray(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def _draw_number(self, pil_img: Image.Image, idx_text: str, margin: int, scale: float, outline: bool) -> Image.Image:
        draw = ImageDraw.Draw(pil_img)
        W, H = pil_img.size

        # font size proportional to image size; fallback to default bitmap font
        base = int(max(12, round(min(W, H) * scale)))
        try:
            # Use a common font name if available in your environment; default will work everywhere
            font = ImageFont.truetype("arial.ttf", base)
        except Exception:
            font = ImageFont.load_default()

        # Measure text size
        bbox = draw.textbbox((0, 0), idx_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        x = max(0, W - tw - margin)
        y = margin

        if outline:
            # Simple 1px outline around the text (8 directions)
            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((x + ox, y + oy), idx_text, font=font, fill=(0, 0, 0))
        # Foreground (white)
        draw.text((x, y), idx_text, font=font, fill=(255, 255, 255))
        return pil_img

    def stamp_numbers(self,
                      images: torch.Tensor,
                      start_index: int = 1,
                      margin: int = 12,
                      scale: float = 0.06,
                      outline: bool = True) -> Tuple[torch.Tensor]:

        # Normalize input to batched tensor [B,H,W,C]
        if isinstance(images, list):
            # Some nodes might hand lists; convert to a stacked tensor
            images = torch.stack(images, dim=0)
        assert torch.is_tensor(images), "Expected 'images' as a torch Tensor or list of Tensors."

        assert images.ndim == 4 and images.shape[-1] in (3, 4), \
            f"Expected IMAGE tensor with shape [B,H,W,C], got {tuple(images.shape)}"

        B, H, W, C = images.shape
        out_list = []

        for i in range(B):
            pil = self._to_pil(images[i])
            text = str(start_index + i)
            pil = self._draw_number(pil, text, margin=margin, scale=scale, outline=outline)
            t = self._from_pil(pil)

            # Ensure channel count preserved (convert to RGB if needed)
            if t.ndim == 2:
                t = t[..., None].repeat(3, axis=-1)
            if t.shape[-1] != C:
                # If original had alpha, add opaque alpha; if original was RGB, ensure RGB
                if C == 4 and t.shape[-1] == 3:
                    alpha = np.full((H, W, 1), 1.0, dtype=np.float32)
                    t = torch.from_numpy(np.concatenate([t.numpy(), alpha], axis=-1))
                elif C == 3 and t.shape[-1] == 4:
                    t = t[..., :3]

            out_list.append(t)

        out = torch.stack(out_list, dim=0)
        return (out,)

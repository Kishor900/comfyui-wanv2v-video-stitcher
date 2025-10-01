# Merge up to 10 IMAGE batches, removing the first `overlap_size` frames
# from every batch after the first, then concatenating sequentially.

from typing import Tuple, List
import torch

class WanV2VMergeDeOverlap:
    @classmethod
    def INPUT_TYPES(cls):
        # Images are optional so you can wire as many as you have (1..10)
        return {
            "required": {
                "overlap_size": ("INT", {"default": 8, "min": 0, "max": 2048, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "merge"
    CATEGORY = "Wanv2vControlnetManager/WAN"

    # ---- utils ----
    def _ensure_batched(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.is_tensor(t), "Expected torch.Tensor"
        assert t.ndim == 4, f"Expected IMAGE [B,H,W,C], got {tuple(t.shape)}"
        return t

    def _empty_like(self, ref: torch.Tensor) -> torch.Tensor:
        _, H, W, C = ref.shape
        return torch.empty((0, H, W, C), dtype=ref.dtype, device=ref.device)

    # ---- core ----
    def merge(
        self,
        overlap_size: int,
        image_1: torch.Tensor = None,
        image_2: torch.Tensor = None,
        image_3: torch.Tensor = None,
        image_4: torch.Tensor = None,
        image_5: torch.Tensor = None,
        image_6: torch.Tensor = None,
        image_7: torch.Tensor = None,
        image_8: torch.Tensor = None,
        image_9: torch.Tensor = None,
        image_10: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, str]:

        imgs: List[torch.Tensor] = [
            image_1, image_2, image_3, image_4, image_5,
            image_6, image_7, image_8, image_9, image_10
        ]

        # Collect non-None, validate shapes and find a reference
        valid: List[torch.Tensor] = []
        ref = None
        for t in imgs:
            if t is None:
                continue
            t = self._ensure_batched(t)
            if ref is None:
                ref = t
            else:
                # Optional: enforce same H/W/C; comment out if you allow variation
                assert t.shape[1:] == ref.shape[1:], \
                    f"All inputs must share H/W/C. Got {tuple(t.shape[1:])} vs {tuple(ref.shape[1:])}"
            valid.append(t)

        if not valid:
            # Nothing connected → return a tiny empty tensor to be safe
            empty = torch.empty((0, 1, 1, 3), dtype=torch.float32)
            return empty, "No inputs provided; returned empty batch."

        x = max(0, int(overlap_size))

        parts: List[torch.Tensor] = []
        total_in = 0
        total_dropped = 0

        for idx, batch in enumerate(valid, start=1):
            B = batch.shape[0]
            total_in += B

            if idx == 1 or x == 0:
                # First iteration keeps everything (no de-overlap), or x=0
                use = batch
                dropped = 0
            else:
                # Drop the first x frames; if B < x, this contributes empty
                drop = min(x, B)
                use = batch[drop:]
                dropped = drop

            total_dropped += dropped
            parts.append(use)

        merged = torch.cat(parts, dim=0) if parts else self._empty_like(ref)

        info = (
            f"inputs={len(valid)}, total_in_frames={total_in}, "
            f"overlap_size={x}, total_dropped={total_dropped}, "
            f"output_frames={merged.shape[0]}"
        )
        return merged, info

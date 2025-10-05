# Merge up to 10 IMAGE batches, removing the first `overlap_size` frames
# from every batch after the first, then concatenating sequentially.
# If enable_crossfade is True, we do: de-overlap NEXT chunk first → then crossfade
# last K frames of OUT with first K frames of the de-overlapped NEXT chunk.

from typing import Tuple, List
import torch
import torch.nn.functional as F
import math

class WanV2VMergeDeOverlap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_size": ("INT", {"default": 8, "min": 0, "max": 2048, "step": 1}),
                "enable_crossfade": ("BOOLEAN", {"default": False}),
                "crossfade_frames": ("INT", {"default": 8, "min": 0, "max": 2048, "step": 1}),
                "interpolation": ([
                    "linear",
                    "ease_in",
                    "ease_out",
                    "ease_in_out",
                    "bounce",
                    "elastic",
                    "glitchy",
                    "exponential_ease_out",
                ],),
                "start_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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

    # ---- easing functions ----
    def _ease_linear(self, t: torch.Tensor) -> torch.Tensor: return t
    def _ease_in(self, t: torch.Tensor) -> torch.Tensor: return t * t
    def _ease_out(self, t: torch.Tensor) -> torch.Tensor: return 1.0 - (1.0 - t) * (1.0 - t)
    def _ease_in_out(self, t: torch.Tensor) -> torch.Tensor: return t * t * (3.0 - 2.0 * t)

    def _bounce(self, t: torch.Tensor) -> torch.Tensor:
        n1, d1 = 7.5625, 2.75
        out = torch.zeros_like(t)
        c1 = t < 1 / d1
        c2 = (t >= 1 / d1) & (t < 2 / d1)
        c3 = (t >= 2 / d1) & (t < 2.5 / d1)
        c4 = t >= 2.5 / d1
        out = torch.where(c1, n1 * t * t, out)
        out = torch.where(c2, n1 * (t - 1.5 / d1) ** 2 + 0.75, out)
        out = torch.where(c3, n1 * (t - 2.25 / d1) ** 2 + 0.9375, out)
        out = torch.where(c4, n1 * (t - 2.625 / d1) ** 2 + 0.984375, out)
        return out

    def _elastic(self, t: torch.Tensor) -> torch.Tensor:
        c4 = (2 * math.pi) / 3
        tt = t.clamp(0, 1)
        return torch.where(
            tt == 0, torch.zeros_like(tt),
            torch.where(
                tt == 1, torch.ones_like(tt),
                torch.pow(2.0, -10.0 * tt) * torch.sin((tt * 10.0 - 0.75) * c4) + 1.0
            )
        )

    def _glitchy(self, t: torch.Tensor) -> torch.Tensor:
        steps = 8
        q = torch.floor(t * steps) / (steps - 1)
        return self._ease_out(q.clamp(0, 1))

    def _expo_out(self, t: torch.Tensor) -> torch.Tensor:
        tt = t.clamp(0, 1)
        return torch.where(tt == 1, torch.ones_like(tt), 1 - torch.pow(2.0, -10.0 * tt))

    def _get_easer(self, name: str):
        return {
            "linear": self._ease_linear,
            "ease_in": self._ease_in,
            "ease_out": self._ease_out,
            "ease_in_out": self._ease_in_out,
            "bounce": self._bounce,
            "elastic": self._elastic,
            "glitchy": self._glitchy,
            "exponential_ease_out": self._expo_out,
        }[name]

    # ---- utils ----
    def _ensure_batched(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.is_tensor(t), "Expected torch.Tensor"
        assert t.ndim == 4, f"Expected IMAGE [B,H,W,C], got {tuple(t.shape)}"
        return t

    def _empty_like(self, ref: torch.Tensor) -> torch.Tensor:
        _, H, W, C = ref.shape
        return torch.empty((0, H, W, C), dtype=ref.dtype, device=ref.device)

    def _crossfade_pair(
        self,
        prev_tail: torch.Tensor,   # [K,H,W,C]
        next_head: torch.Tensor,   # [K,H,W,C]
        start_level: float,
        end_level: float,
        easing_name: str,
    ) -> torch.Tensor:
        assert prev_tail.shape == next_head.shape, "Crossfade shapes must match"
        K, H, W, C = prev_tail.shape
        if K == 0:
            return prev_tail
        device, dtype = prev_tail.device, prev_tail.dtype
        t = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
        easer = self._get_easer(easing_name)
        eased = easer(t)
        alphas = (1.0 - eased) * start_level + eased * end_level
        alphas = alphas.view(K, 1, 1, 1)
        out = prev_tail * (1.0 - alphas) + next_head * alphas
        return out.clamp(0.0, 1.0)

    # ---- core ----
    def merge(
        self,
        overlap_size: int,
        enable_crossfade: bool,
        crossfade_frames: int,
        interpolation: str,
        start_level: float,
        end_level: float,
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
                assert t.shape[1:] == ref.shape[1:], \
                    f"All inputs must share H/W/C. Got {tuple(t.shape[1:])} vs {tuple(ref.shape[1:])}"
            valid.append(t)

        if not valid:
            empty = torch.empty((0, 1, 1, 3), dtype=torch.float32)
            return empty, "No inputs provided; returned empty batch."

        x = max(0, int(overlap_size))
        k_cfg = max(0, int(crossfade_frames)) if enable_crossfade else 0

        total_in = sum(b.shape[0] for b in valid)
        total_dropped = 0
        total_crossfaded = 0

        # Start with first batch intact (no de-overlap on image_1)
        out = valid[0].clone()

        for idx in range(1, len(valid)):
            nxt = valid[idx]
            Bn = nxt.shape[0]
            if Bn == 0:
                continue

            # --- 1) DE-OVERLAP FIRST ---
            drop = min(x, Bn)
            nxt_rem = nxt[drop:]        # de-overlapped remainder
            total_dropped += drop

            if nxt_rem.shape[0] == 0:
                # Nothing left to append from this chunk
                continue

            # --- 2) THEN CROSSFade between OUT tail and de-overlapped head ---
            if enable_crossfade and k_cfg > 0:
                K = min(k_cfg, out.shape[0], nxt_rem.shape[0])
                if K > 0:
                    prev_tail = out[-K:]           # last K frames of accumulated output
                    next_head = nxt_rem[:K]        # first K frames of de-overlapped next
                    blended = self._crossfade_pair(
                        prev_tail, next_head,
                        float(start_level), float(end_level),
                        interpolation
                    )
                    # Replace tail with blended and append the remainder after K
                    out = torch.cat([out[:-K], blended, nxt_rem[K:]], dim=0)
                    total_crossfaded += K
                else:
                    # Not enough frames to crossfade; just append
                    out = torch.cat([out, nxt_rem], dim=0)
            else:
                # Crossfade disabled; just append de-overlapped remainder
                out = torch.cat([out, nxt_rem], dim=0)

        info = (
            f"inputs={len(valid)}, total_in_frames={total_in}, "
            f"overlap_size={x}, enable_crossfade={bool(enable_crossfade)}, "
            f"crossfade_frames={k_cfg}, total_crossfade_frames={total_crossfaded}, "
            f"total_dropped={total_dropped}, output_frames={out.shape[0]}"
        )
        return out, info

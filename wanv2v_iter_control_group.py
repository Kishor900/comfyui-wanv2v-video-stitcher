from typing import Tuple
import torch

"""
WanV2VIterControlGroup (finalized + masks)

Adds a mask batch output aligned with the built control group:
- Size: chunk_size (B,H,W)
- First `overlap_size` masks are BLACK (0.0 -> non-editable),
- Remaining masks are WHITE (1.0 -> editable).

Everything else remains unchanged.
"""

def _sec_to_hms_ms(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000.0))
    s_total = int(sec)
    h = s_total // 3600
    m = (s_total % 3600) // 60
    s = s_total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

class WanV2VIterControlGroup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_images": ("IMAGE",),  # [T,H,W,C]
                "chunk_size": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 1}),
                "overlap_size": ("INT", {"default": 8, "min": 0, "max": 2048, "step": 1}),
                "iter": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1}),  # 1-based
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.1}),
            },
            "optional": {
                "prev_output": ("IMAGE",),        # previous sampler output [*,H,W,C]
            },
        }

    # Added a MASK batch as the 3rd return
    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("control_group", "info", "edit_mask_batch")
    FUNCTION = "build"
    CATEGORY = "Wanv2vControlnetManager/WAN"

    # ---- helpers ----
    def _ensure_batched(self, t: torch.Tensor) -> torch.Tensor:
        assert torch.is_tensor(t), "Expected torch.Tensor"
        assert t.ndim == 4, f"Expected [B,H,W,C], got {tuple(t.shape)}"
        return t

    def _slice_safe(self, t: torch.Tensor, start: int, end: int) -> torch.Tensor:
        start = max(0, int(start))
        end = max(start, min(int(end), t.shape[0]))
        return t[start:end]

    def _last_n(self, t: torch.Tensor, n: int) -> torch.Tensor:
        n = max(0, int(n))
        if n == 0:
            return t[:0]
        n = min(n, t.shape[0])
        return t[-n:]

    def _empty_like(self, ref: torch.Tensor, B: int = 0) -> torch.Tensor:
        _, H, W, C = ref.shape
        return torch.empty((B, H, W, C), dtype=ref.dtype, device=ref.device)

    def _pad_repeat_last(self, t: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad batch t by repeating its last frame until target_len."""
        if t.shape[0] == 0:
            return t  # cannot pad from empty
        need = target_len - t.shape[0]
        if need <= 0:
            return t[:target_len]
        last = t[-1:].repeat(need, 1, 1, 1)
        return torch.cat([t, last], dim=0)

    def _build_masks(self, H: int, W: int, chunk_size: int, overlap_size: int, device, dtype=torch.float32) -> torch.Tensor:
        """
        Returns a (B,H,W) mask batch:
          first overlap_size are 0.0 (black, non-editable),
          remaining are 1.0 (white, editable).
        """
        B = max(1, int(chunk_size))
        x = max(0, int(overlap_size))
        x = min(x, B)  # cap to B
        black = torch.zeros((x, H, W), dtype=dtype, device=device)
        white = torch.ones((B - x, H, W), dtype=dtype, device=device)
        return torch.cat([black, white], dim=0)

    # ---- core ----
    def build(
        self,
        controlnet_images: torch.Tensor,
        chunk_size: int,
        overlap_size: int,
        iter: int,
        fps: float,
        prev_output: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:

        frames = self._ensure_batched(controlnet_images)
        T, H, W, C = frames.shape

        # Guards
        if chunk_size <= 0:
            chunk_size = 1
        if overlap_size < 0:
            overlap_size = 0
        if iter < 1:
            iter = 1

        x = overlap_size
        drop = (iter - 1) * x  # amount we drop from tail of current group_i in normal case

        # Compute group indices for current and previous groups
        base_start_i = (iter - 1) * chunk_size
        base_end_i   = base_start_i + chunk_size

        base_start_prev = (iter - 2) * chunk_size
        base_end_prev   = base_start_prev + chunk_size

        # duration info
        total_seconds = float(T) / float(fps) if fps > 0 else 0.0

        # --- Iter 1 or no overlap/prev: simple slice + forward pad ---
        if iter == 1 or x == 0 or prev_output is None:
            if base_start_i >= T:
                empty = self._empty_like(frames, 0)
                info = f"(empty) iter={iter}, total={T} ({_sec_to_hms_ms(total_seconds)} @ {fps:.2f})"
                # Even if empty, still emit a mask batch shape (chunk_size,H,W)
                masks = self._build_masks(H, W, chunk_size, overlap_size, frames.device)
                return empty, info, masks

            group_i = self._slice_safe(frames, base_start_i, min(base_end_i, T))
            built = group_i
            # forward pad if short
            need = chunk_size - built.shape[0]
            if need > 0:
                extra = self._slice_safe(frames, base_end_i, base_end_i + need)
                built = torch.cat([built, extra], dim=0)
            built = built[:chunk_size]

            used = f"[{base_start_i}:{min(base_end_i, T)}]"
            info = f"iter={iter}, built={built.shape[0]}, used={used}, total={T} ({_sec_to_hms_ms(total_seconds)} @ {fps:.2f})"

            # Mask batch: first overlap_size black, rest white
            masks = self._build_masks(H, W, built.shape[0], overlap_size, frames.device)
            return built, info, masks

        # --- Iter >= 2: need prev_output tail and residue from group_{i-1} ---
        prev_out = self._ensure_batched(prev_output)
        tail_prev = self._last_n(prev_out, x)  # last x frames from previous sampler output

        # residue from previous control group: last ((iter-2)*x) of group_{i-1}
        group_prev = self._slice_safe(frames, base_start_prev, min(base_end_prev, T))
        residue_len = (iter - 2) * x
        residue_prev = self._last_n(group_prev, residue_len) if residue_len > 0 else group_prev[:0]

        # Case A: we still have frames for group_i (normal overlapped case)
        if base_start_i < T:
            group_i = self._slice_safe(frames, base_start_i, min(base_end_i, T))

            # If drop is too large for this iter, return empty (invalid config)
            if drop >= chunk_size:
                empty = self._empty_like(frames, 0)
                info = (f"(empty) iter={iter}: drop={drop} >= chunk_size={chunk_size}; "
                        f"total={T} ({_sec_to_hms_ms(total_seconds)} @ {fps:.2f})")
                masks = self._build_masks(H, W, chunk_size, overlap_size, frames.device)
                return empty, info, masks

            keep = group_i[:-drop] if drop > 0 else group_i
            built = torch.cat([tail_prev, residue_prev, keep], dim=0)

            # top up after group_i if still short
            if built.shape[0] < chunk_size:
                need = chunk_size - built.shape[0]
                extra = self._slice_safe(frames, base_end_i, base_end_i + need)
                built = torch.cat([built, extra], dim=0)

            built = built[:chunk_size]
            info = (
                f"iter={iter}, NORMAL, x={x}, drop={(iter-1)*x}, "
                f"tail_prev={tail_prev.shape[0]}, residue_prev={residue_prev.shape[0]}, "
                f"keep={keep.shape[0]}, built={built.shape[0]}, total={T} "
                f"({_sec_to_hms_ms(total_seconds)} @ {fps:.2f})"
            )

            masks = self._build_masks(H, W, built.shape[0], overlap_size, frames.device)
            return built, info, masks

        # Case B: FINALIZATION (no group_i left) → use tail(prev) + residue_prev only, pad with last frame
        built = torch.cat([tail_prev, residue_prev], dim=0)
        if built.shape[0] < chunk_size:
            built = self._pad_repeat_last(built, chunk_size)

        built = built[:chunk_size]
        info = (
            f"iter={iter}, FINALIZE, x={x}, "
            f"tail_prev={tail_prev.shape[0]}, residue_prev={residue_prev.shape[0]}, "
            f"built={built.shape[0]} (padded with last prev frame if needed), total={T} "
            f"({_sec_to_hms_ms(total_seconds)} @ {fps:.2f})"
        )

        masks = self._build_masks(H, W, built.shape[0], overlap_size, frames.device)
        return built, info, masks

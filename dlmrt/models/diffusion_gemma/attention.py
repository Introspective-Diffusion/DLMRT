"""DiffusionGemma's attention backends. Sliding layers (head_dim 256): causal rows see their last `sliding_window`
keys, canvas rows [leftpad, end). Full layers (head_dim 512): causal up to the row, or [0, end) for the canvas.

FA3 serves the sliding layers where it runs (Hopper); the Triton kernel (ops.triton.attention) serves the full layers
(beyond FA3's head_dim) and everything else, with tiles tuned per GPU generation. DLMRT_DG_ATTENTION=fa3 / triton
forces the sliding layers' backend.
"""

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

from ... import envs
from ...ops.triton.attention import attention

BACKEND = envs.DG_ATTENTION()


def tiles(major, sliding, kv_heads):
    """The Triton attention tiles of a layer kind on a GPU generation, tuned on B200 and on an RTX 5090
    (benchmarks/attention_sweep.py). Hopper keeps the defaults its runs were validated with. B200 accumulates in TMEM:
    DV + BLOCK_N must stay <= 512. SM 12.x has about 100 KB of shared memory per block; its full layers take the whole
    512-wide value in one program (one Q K^T per key tile instead of one per 256 value columns) with 32-row tiles."""
    if major == 12:
        if not sliding:
            return dict(block_m=32, block_n=32, num_warps=4, num_stages=2, dv=512)
        return dict(block_m=64, block_n=32, num_warps=4, num_stages=2)
    if major != 10:
        return {}
    if not sliding:
        return dict(block_m=128, block_n=32, num_warps=8)
    # With few KV heads per rank (TP), small query tiles leave fewer key splits to combine.
    if kv_heads >= 4:
        return dict(block_m=128, block_n=64, num_warps=8)
    return dict(block_m=64, block_n=32, num_warps=8)


class Attention:
    def __init__(self, config, device, max_seq_len, dtype):
        self.config, self.max_seq_len, self.dtype = config, max_seq_len, dtype
        self.major = torch.cuda.get_device_capability(device)[0]
        self.fa3 = BACKEND == "fa3" or not BACKEND and self.major == 9

    def tiles(self, sliding, kv_heads):
        return tiles(self.major, sliding, kv_heads)

    def schedule(self, rows, end, heads, kv_heads, splits, causal):
        """FA3's scheduler metadata for a causal pass over the sliding layers (computed once per forward), or None."""
        if not (causal and self.fa3):
            return None
        c = self.config
        return get_scheduler_metadata(1, rows, self.max_seq_len, heads, kv_heads, c.head_dim, end,
                                      qkv_dtype=self.dtype, causal=True, window_size=(c.sliding_window - 1, 0),
                                      num_splits=splits)

    def __call__(self, sliding, q, k, v, start, end, leftpad, causal, splits, schedule):
        tiles = self.tiles(sliding, k.shape[2])
        if not sliding:
            return attention(q, k, v, start, end, causal, **tiles)
        w = self.config.sliding_window - 1
        if not self.fa3:
            return attention(q, k, v, start, end, causal, window=w + 1 if causal else 0,
                                left=None if causal else leftpad, **tiles)
        if causal:
            return flash_attn_with_kvcache(q, k, v, cache_seqlens=end, causal=True, window_size=(w, 0),
                                           softmax_scale=1.0, num_splits=splits, scheduler_metadata=schedule)
        # FA3's scheduler metadata does not take cache_leftpad; the canvas pass schedules itself.
        return flash_attn_with_kvcache(q, k, v, cache_seqlens=end, cache_leftpad=leftpad, causal=False,
                                       softmax_scale=1.0, num_splits=splits)

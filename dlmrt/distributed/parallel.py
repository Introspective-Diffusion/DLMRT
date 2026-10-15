"""Tensor parallelism over the GPUs of one NVSwitch node.

One process per GPU runs the same program: the same block graphs and the same host loop.
Every cross-rank reduction returns bitwise-identical values on all ranks, so each rank's
device-side denoise loop takes the same number of rounds and issues the same collectives;
after the prompt is shared no host coordination is needed.

Layout (Megatron): QKV and gate_up column-parallel, O and down row-parallel (heads and FFN
columns split), the tied embedding vocab-parallel for the LM head and the SC GEMM. Partial
sums of row-parallel GEMMs are written into symmetric memory and all-reduced in place.
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from .. import envs
from ..ops.extensions import tp_comm
from ..ops.triton.common import PDL


# multimem: NVSwitch in-network reduction (NVLS); two_shot: reduce-scatter + all-gather over NVLink loads.
ALL_REDUCE = envs.TP_ALL_REDUCE()

# One kernel per row-parallel output: push reduce-scatter + residual + RMSNorm + multicast all-gather.
FUSED = envs.TP_FUSED()

# CTAs that only push partial rows over NVLink (the owner CTAs, one per owned row, come on top).
PUSH_BLOCKS = envs.TP_PUSH_BLOCKS()


def init():
    """Join the torchrun process group; returns None when running on one GPU."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world == 1:
        return None
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return Group(device)


class Group:
    def __init__(self, device):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = device
        self.name = dist.group.WORLD.group_name
        self._buffers = {}
        self._handles = {}
        self._recv = {}
        self._out_peers = {}
        self._last = {}  # per chain: the previous call's output buffer
        self.stamps = 0  # device address of a [grid, 4] int64 buffer to record kernel phase times
        if FUSED:
            tp_comm()

    def symmetric(self, key, shape, dtype=torch.bfloat16, armed=False):
        """A symmetric-memory tensor, allocated once per key (collectively, in the same order on every rank); every
        later call with the key must ask for the same shape and dtype.

        `armed`: filled with the arrival sentinel (every bf16 lane 0xFFFF) before any rank can write into it.
        """
        if key not in self._buffers:
            with torch.inference_mode(False):  # shared by graphs captured in and out of inference mode
                t = symm.empty(*shape, device=self.device, dtype=dtype)
            if armed:  # every rank finishes arming before the rendezvous lets any rank write into it
                t.view(torch.int16).fill_(-1)
                torch.cuda.synchronize(self.device)
            self._handles[key] = symm.rendezvous(t, self.name)
            self._buffers[key] = t
        t = self._buffers[key]
        if tuple(t.shape) != tuple(shape) or t.dtype != dtype:  # a larger request would write past it
            raise ValueError(f"symmetric buffer {key}: {tuple(t.shape)} {t.dtype} allocated, {tuple(shape)} {dtype} "
                             "requested")
        return t

    def _handle(self, t):
        for key, buf in self._buffers.items():
            base = buf.data_ptr()
            if base <= t.data_ptr() < base + buf.numel() * buf.element_size():
                return key, self._handles[key]
        raise ValueError("not a symmetric tensor of this group")

    def multicast(self, t):
        """Multicast address of symmetric tensor `t` (stores reach every rank's copy), or 0 without NVLink multicast."""
        handle = self._handle(t)[1]
        if not handle.multicast_ptr:
            return 0
        return handle.multicast_ptr + (t.data_ptr() - handle.buffer_ptrs[self.rank])

    def peers(self, t):
        """[world] int64 device tensor: the address of `t` in every rank's copy."""
        key, handle = self._handle(t)
        if (key, t.data_ptr()) not in self._out_peers:
            offset = t.data_ptr() - handle.buffer_ptrs[self.rank]
            self._out_peers[key, t.data_ptr()] = torch.tensor([p + offset for p in handle.buffer_ptrs],
                                                              device=self.device, dtype=torch.int64)
        return self._out_peers[key, t.data_ptr()]

    def _receive(self, rows, n, chain):
        """Triple-buffered receive slots (symmetric, armed), every rank's slot address, invocation counters."""
        if (rows, n, chain) not in self._recv:
            key = f"recv{rows}x{n}.{chain}"
            self.symmetric(key, (3, self.world, rows // self.world, n), armed=True)
            ptrs = torch.tensor(self._handles[key].buffer_ptrs, device=self.device, dtype=torch.int64)
            epochs = torch.zeros(rows // self.world + PUSH_BLOCKS, device=self.device, dtype=torch.int32)
            self._recv[rows, n, chain] = ptrs, epochs
        return self._recv[rows, n, chain]

    def partial(self, site, shape):
        """Where a row-parallel GEMM writes its partial product: one buffer shared by every call site (`site` is not
        used), so every call passes the same shape."""
        return self.symmetric("partial", shape)

    def _launch(self, fn, partial, h, weight, site, eps=0.0, scale=1.0, chain=0):
        rows, n = partial.shape
        out = self.symmetric(f"out:{site}", (rows, n), armed=True)
        ptrs, epochs = self._receive(rows, n, chain)
        last = self._last.get(chain)
        if last is out:
            raise ValueError(f"call site {site} ran twice in a row; its buffer would never be re-armed")
        # The previous call's output on this chain: its consumer has finished, so this call re-arms it.
        rearm, rearm_numel = (last.data_ptr(), last.numel()) if last is not None else (0, 0)
        fn(partial, ptrs, h if h is not None else partial, weight if weight is not None else partial, out,
           self.multicast(out), self.peers(out), rearm, rearm_numel, epochs, self.rank, self.world, eps, scale,
           PUSH_BLOCKS, self.stamps, PDL)
        self._last[chain] = out
        return out

    def reduce_norm(self, partial, h, weight, eps, scale=1.0, *, site, chain=0):
        """rmsnorm(h [+= sum of partial over ranks]) * weight * scale, on every rank.

        With `h`, only this rank's rows of `h` are updated (the residual stream becomes sequence-parallel);
        without it the reduced partial itself is normalized. The result lives in this call site's buffer
        until the next call on the same `chain`, which re-arms it. Reductions that may run concurrently
        (different streams) need different `chain`s; calls on one chain are ordered.
        """
        fn = tp_comm().reduce_norm_residual if h is not None else tp_comm().reduce_norm
        return self._launch(fn, partial, h, weight, site, eps, scale, chain)

    def reduce(self, partial, *, site, chain=0):
        """Sum over ranks of `partial`, on every rank, in this call site's buffer."""
        return self._launch(tp_comm().reduce, partial, None, None, site, chain=chain)

    def shard(self, t, dim=0):
        if t.shape[dim] % self.world:
            raise ValueError(f"Dimension {dim} of {tuple(t.shape)} does not split over {self.world} ranks")
        return t.chunk(self.world, dim)[self.rank].contiguous()

    def all_reduce_(self, x):
        """In-place sum over ranks of a symmetric tensor; identical result on every rank."""
        if ALL_REDUCE == "multimem" and self.multicast(x):
            torch.ops.symm_mem.multimem_all_reduce_(x, "sum", self.name)
        elif ALL_REDUCE in ("multimem", "two_shot"):  # two_shot also serves nodes without NVLink multicast
            torch.ops.symm_mem.two_shot_all_reduce_(x, "sum", self.name)
        else:
            raise ValueError(f"Unknown DLMRT_TP_ALL_REDUCE={ALL_REDUCE}")
        return x

    def all_gather(self, x, key):
        """[world, *x.shape]: every rank's `x`, stacked in rank order."""
        out = self.symmetric(key, (self.world, *x.shape), x.dtype)
        if self.multicast(out):
            torch.ops.symm_mem.multimem_all_gather_out(x, self.name, out.view(-1, *x.shape[1:]))
            return out
        # Without multicast: every rank fills its own slot of a zeroed buffer (the other slots stay zero)
        # and a P2P one-shot sum assembles the gather exactly (x + 0 = x).
        slots = self.symmetric(key + ".slots", (self.world, *x.shape), x.dtype)
        if not getattr(slots, "_zeroed", False):
            slots.zero_()
            torch.cuda.synchronize(self.device)
            dist.barrier()
            slots._zeroed = True
        slots[self.rank].copy_(x)
        torch.ops.symm_mem.one_shot_all_reduce_out(slots, "sum", self.name, out)
        return out

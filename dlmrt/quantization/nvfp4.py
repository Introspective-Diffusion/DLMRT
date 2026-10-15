"""ModelOpt NVFP4 expert checkpoints (W4A4 experts, FP8 KV cache).

Weights: E2M1 codes (two per byte, the even element in the low nibble) times an FP8 E4M3 scale per 16 input
elements times the FP32 per-tensor `weight_scale_2`. Activations: each expert GEMM input row is rounded to NVFP4 in
blocks of 16 at run time against the exported `input_scale`.

The experts run on Blackwell FP4 tensor cores through FlashInfer: `trtllm_experts` / `trtllm_moe` (TRT-LLM-gen MoE,
SM 10.x) or `cutlass_experts` / `cutlass_moe` (CUTLASS fused MoE, SM 12.x); `backend` picks one for a GPU.
`triton_experts` lays them out for `dlmrt.ops.triton.nvfp4_moe` (SM 12.x, the TRT-LLM-gen MoE's roundings).
`expert_weights` dequantizes them to BF16 (an export run at bf16). The FP8 KV cache is stored or rounded in
`dlmrt.models.diffusion_gemma.kernels.qkv_rope` and `dlmrt.models.llada2.kernels.qk_rope`.
"""

import ast
import json
import re
from pathlib import Path

import torch

from .. import envs

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Per-expert NVFP4 tensors: DiffusionGemma (model.decoder.layers.{i}.experts.{e}) and LLaDA2
# (model.layers.{i}.mlp.experts.{e}).
EXPERT_KEY = re.compile(r"model\.(?:decoder\.)?layers\.(\d+)\.(?:mlp\.)?experts\.(\d+)\.(gate|up|down)_proj\."
                        r"(weight|weight_scale|weight_scale_2|input_scale)$")


def quant_config(path):
    """(quant_algo, kv_cache_quant_algo) of a ModelOpt export, or (None, None) for a BF16 checkpoint."""
    f = Path(path) / "hf_quant_config.json"
    if not f.exists():
        return None, None
    q = json.loads(f.read_text())["quantization"]
    if q.get("quant_algo") != "NVFP4" or q.get("group_size") != 16 or q.get("kv_cache_quant_algo") not in (None, "FP8"):
        raise ValueError(f"Unsupported ModelOpt quantization: {q.get('quant_algo')}, kv {q.get('kv_cache_quant_algo')}")
    return q["quant_algo"], q.get("kv_cache_quant_algo")


def dequantize(packed, scale, scale2):
    """[N, K/2] uint8 codes, [N, K/16] FP8 E4M3 block scales, FP32 per-tensor scale -> [N, K] BF16."""
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), -1).flatten(-2).long()
    values = table[codes].unflatten(-1, (-1, 16))
    return (values * (scale.float() * scale2.float())[..., None]).flatten(-2).to(torch.bfloat16)


def expert_weights(parts, layer, lo, count, device):
    """Layer `layer`'s experts [lo, lo + count) dequantized: BF16 [count, 2F, H] gate|up and [count, H, F] down.
    `parts[(layer, expert, proj, field)]` are the checkpoint's tensors."""
    def deq(e, proj):
        return dequantize(*(parts[(layer, e, proj, f)].to(device) for f in ("weight", "weight_scale", "weight_scale_2")))

    w13 = torch.stack([torch.cat([deq(e, "gate"), deq(e, "up")]) for e in range(lo, lo + count)])
    w2 = torch.stack([deq(e, "down") for e in range(lo, lo + count)])
    return w13, w2


BACKEND_GPUS = {"trtllm": "SM 10.x", "cutlass": "SM 12.x"}


def backend(device):
    """The NVFP4 MoE backend for `device`: "trtllm" (SM 10.x: B200, B300), "cutlass" (SM 12.x: RTX 50 series) or
    None (no FP4 tensor cores, or no FlashInfer)."""
    major = torch.cuda.get_device_capability(device)[0]
    if major not in (10, 12):
        return None
    try:
        import flashinfer.fused_moe  # noqa: F401
    except ImportError:
        return None
    return "trtllm" if major == 10 else "cutlass"


def _input_scales(parts, layer, lo, count, device):
    """Per-expert FP32 [count] GEMM1 / GEMM2 input scales of layer `layer`; gate and up share theirs and all experts
    one GEMM1 input scale, as one fused GEMM1 over a quantized input needs (ModelOpt exports them so)."""
    def scalars(proj):
        return torch.stack([parts[(layer, e, proj, "input_scale")].to(device).float().reshape(())
                            for e in range(lo, lo + count)])

    a1, a2 = scalars("gate"), scalars("down")
    if not torch.equal(a1, scalars("up")) or not bool((a1 == a1[0]).all()):
        raise ValueError(f"layer {layer}: the experts' GEMM1 input scales differ")
    return a1, a2


def trtllm_experts(parts, layer, lo, count, device):
    """Layer `layer`'s experts [lo, lo + count) for FlashInfer's TRT-LLM-gen NVFP4 MoE (W4A4 on FP4 tensor cores):
    GEMM1 rows [up; gate], both GEMMs' codes and block scales permuted for its 128-row epilogue tiles, and the
    per-expert output scales. Gate and up share one per-tensor scale and all experts one input scale, as one fused
    GEMM1 needs (ModelOpt exports them so)."""
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache

    def get(e, proj, field):
        return parts[(layer, e, proj, field)].to(device)

    cache, w1, s1, w2, s2 = {}, [], [], [], []
    for e in range(lo, lo + count):
        if not torch.equal(get(e, "gate", "weight_scale_2"), get(e, "up", "weight_scale_2")):
            raise ValueError(f"layer {layer} expert {e}: gate and up have different per-tensor scales")
        w = torch.cat([get(e, "up", "weight"), get(e, "gate", "weight")])
        s = torch.cat([get(e, "up", "weight_scale"), get(e, "gate", "weight_scale")]).view(torch.uint8)
        w1.append(w[_maybe_get_cached_w3_w1_permute_indices(cache, w, 128, is_gated_act_gemm=True).to(device)])
        perm = _maybe_get_cached_w3_w1_permute_indices(cache, s, 128, num_elts_per_sf=16, is_gated_act_gemm=True)
        s1.append(nvfp4_block_scale_interleave(s[perm.to(device)].contiguous()).view(s.shape))
        w, s = get(e, "down", "weight"), get(e, "down", "weight_scale").view(torch.uint8)
        w2.append(w[get_w2_permute_indices_with_cache(cache, w, 128).to(device)])
        perm = get_w2_permute_indices_with_cache(cache, s, 128, num_elts_per_sf=16)
        s2.append(nvfp4_block_scale_interleave(s[perm.to(device)].contiguous()).view(s.shape))

    def scalars(proj, field):
        return torch.stack([get(e, proj, field).float().reshape(()) for e in range(lo, lo + count)])

    a1, a2 = _input_scales(parts, layer, lo, count, device)
    g1 = a1 * scalars("gate", "weight_scale_2")
    return {"backend": "trtllm", "w1": torch.stack(w1), "s1": torch.stack(s1).view(torch.float8_e4m3fn),
            "w2": torch.stack(w2), "s2": torch.stack(s2).view(torch.float8_e4m3fn), "g1": g1, "g1_out": g1 / a2,
            "g2": a2 * scalars("down", "weight_scale_2"), "a1_inv": (1 / a1[:1]).contiguous()}


def cutlass_experts(parts, layer, lo, count, device):
    """Layer `layer`'s experts [lo, lo + count) for FlashInfer's CUTLASS NVFP4 MoE (W4A4 on SM 12.x FP4 tensor
    cores): GEMM1 rows [up; gate] (its gated activation multiplies act(second half) by the first), codes as int64
    (16 per element), block scales in the 128 x 4 interleaved layout as int32, and the `quant_scales` it takes: per
    GEMM the input quantization scale, the block scales and the per-expert dequantization scale (input scale x
    weight scale)."""
    from flashinfer import nvfp4_block_scale_interleave

    def get(e, proj, field):
        return parts[(layer, e, proj, field)].to(device)

    def interleaved(s):
        s = s.view(torch.uint8).contiguous()
        return nvfp4_block_scale_interleave(s).view(s.shape)

    w1, s1, w2, s2 = [], [], [], []
    for e in range(lo, lo + count):
        if not torch.equal(get(e, "gate", "weight_scale_2"), get(e, "up", "weight_scale_2")):
            raise ValueError(f"layer {layer} expert {e}: gate and up have different per-tensor scales")
        w1.append(torch.cat([get(e, "up", "weight"), get(e, "gate", "weight")]))
        s1.append(interleaved(torch.cat([get(e, "up", "weight_scale"), get(e, "gate", "weight_scale")])))
        w2.append(get(e, "down", "weight"))
        s2.append(interleaved(get(e, "down", "weight_scale")))

    def scalars(proj):
        return torch.stack([get(e, proj, "weight_scale_2").float().reshape(()) for e in range(lo, lo + count)])

    a1, a2 = _input_scales(parts, layer, lo, count, device)
    s1, s2 = torch.stack(s1).view(torch.int32), torch.stack(s2).view(torch.int32)
    return {"backend": "cutlass", "lo": lo, "w1": torch.stack(w1).view(torch.int64),
            "w2": torch.stack(w2).view(torch.int64),
            "scales": [(1 / a1[0]).contiguous(), s1, a1 * scalars("gate"), (1 / a2).contiguous(), s2,
                       a2 * scalars("down")]}


def triton_experts(parts, layer, lo, count, device):
    """Layer `layer`'s experts [lo, lo + count) for `dlmrt.ops.triton.nvfp4_moe` (W4A4 on SM 12.x FP4 tensor cores
    with the TRT-LLM-gen MoE's roundings): codes and E4M3 block scales as exported (linear), GEMM1 rows [gate; up],
    and per expert the GEMM alphas (input scale x weight scale) and the GEMM2 input quantization scale."""
    def get(e, proj, field):
        return parts[(layer, e, proj, field)].to(device)

    w1, s1, w2, s2 = [], [], [], []
    for e in range(lo, lo + count):
        if not torch.equal(get(e, "gate", "weight_scale_2"), get(e, "up", "weight_scale_2")):
            raise ValueError(f"layer {layer} expert {e}: gate and up have different per-tensor scales")
        w1.append(torch.cat([get(e, "gate", "weight"), get(e, "up", "weight")]))
        s1.append(torch.cat([get(e, "gate", "weight_scale"), get(e, "up", "weight_scale")]))
        w2.append(get(e, "down", "weight"))
        s2.append(get(e, "down", "weight_scale"))

    def scalars(proj):
        return torch.stack([get(e, proj, "weight_scale_2").float().reshape(()) for e in range(lo, lo + count)])

    a1, a2 = _input_scales(parts, layer, lo, count, device)
    return {"backend": "triton", "lo": lo, "w1": torch.stack(w1), "s1": torch.stack(s1).view(torch.float8_e4m3fn),
            "w2": torch.stack(w2), "s2": torch.stack(s2).view(torch.float8_e4m3fn), "g1": a1 * scalars("gate"),
            "g2": a2 * scalars("down"), "a1_inv": (1 / a1[:1]).contiguous(), "a2_inv": (1 / a2).contiguous()}


CUTLASS_ACTIVATION = {"geglu": "Geglu", "swiglu": "Swiglu"}


def cutlass_moe(x, ids, weights, experts, out, num_experts, max_rows, activation="geglu"):
    """out = sum over each row's experts in this rank's [lo, lo + count) of weight * expert(x): x BF16 [rows,
    hidden], quantized per 16 against the GEMM1 input scale inside the kernel; GEMM1 -> GeGLU / SwiGLU -> NVFP4 ->
    GEMM2 in FlashInfer's CUTLASS kernels. `ids` int32 / `weights` FP32 [rows, top_k] hold global expert ids.
    `max_rows` is the largest row count the tactics were tuned for (`cutlass_tune`). Its GeGLU is the erf GELU."""
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    count = experts["w1"].shape[0]
    cutlass_fused_moe(x, ids, weights, experts["w1"], experts["w2"], x.dtype, experts["scales"], output=out,
                      ep_size=num_experts // count, ep_rank=experts["lo"] // count, tune_max_num_tokens=max_rows,
                      activation_type=getattr(ActivationType, CUTLASS_ACTIVATION[activation]))
    return out


def cutlass_tune(experts, num_experts, rows, hidden, top_k, device, activation="geglu"):
    """Profile the CUTLASS MoE's GEMM tactics for up to `rows` rows (FlashInfer's autotuner) before any graph is
    captured, on routing skewed like the router's; the choice is kept in $DLMRT_CACHE_DIR, keyed by GPU, shapes and
    this rank's experts.

    GEMM2 keeps the fastest deterministic tactic of each row bucket: half of its tactics fuse the top-k reduction
    into the epilogue, where the experts' partial rows meet in arbitrary order, and return a different rounding of
    the output on every call."""
    from flashinfer.autotuner import AutoTuner, autotune

    cache = envs.cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(device).replace(" ", "-")
    count = experts["w1"].shape[0]
    path = cache / (f"cutlass-moe-{gpu}-r{rows}-h{hidden}-i{experts['w2'].shape[2] * 16}-e{num_experts}-k{top_k}-"
                    f"{experts['lo']}+{count}-{activation}-det.json")
    fresh = not path.exists()
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, hidden, device=device, generator=g).to(torch.bfloat16)
    popularity = 1.0 / torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    popularity = popularity[torch.randperm(num_experts, device=device, generator=g)]
    ids = torch.multinomial(popularity.expand(rows, -1), top_k, generator=g).sort(1).values.int()
    weights = torch.full((rows, top_k), 1.0 / top_k, device=device)
    out = torch.empty_like(x)
    tuner = AutoTuner.get()
    timed, profile = {}, tuner._profile_single_kernel

    def record(runner, tensors, tactic, config, **kwargs):
        t = profile(runner, tensors, tactic, config, **kwargs)
        if kwargs.get("gemm_idx") == 2:
            timed.setdefault(tensors[0].shape[0], {})[tactic] = t
        return t

    tuner._profile_single_kernel = record
    staging = path.with_suffix(".tuning") if fresh else path
    try:
        with autotune(True, cache=str(staging)):
            cutlass_moe(x, ids, weights, experts, out, num_experts, rows, activation)
    finally:
        tuner._profile_single_kernel = profile
        if fresh:
            staging.unlink(missing_ok=True)
    if fresh:
        entries = [(d, k) for d in (tuner.profiling_cache, tuner._file_configs) for k in d
                   if "fused_moe::gemm2" in str(k)]
        chosen = {(id(d), k): d[k] for d, k in entries}
        verdict = {}

        def deterministic(tactic):
            if tactic not in verdict:
                for d, k in entries:
                    d[k] = (d[k][0], tactic, *d[k][2:])
                runs = []
                for _ in range(3):
                    out.zero_()
                    cutlass_moe(x, ids, weights, experts, out, num_experts, rows, activation)
                    runs.append(out.clone())
                verdict[tactic] = all(torch.equal(r, runs[0]) for r in runs)
            return verdict[tactic]

        for d, k in entries:  # live keys (op, runner, hash, shapes, extras); file keys "(op, runner, shapes)"
            bucket = timed.get(ast.literal_eval(k)[2][0][0] if isinstance(k, str) else k[3][0][0], {})
            ranked = sorted((t for t in bucket if bucket[t] != float("inf")), key=bucket.get)
            pick = next((t for t in ranked if deterministic(t)), chosen[id(d), k][1])
            chosen[id(d), k] = (chosen[id(d), k][0], pick, *chosen[id(d), k][2:])
        for d, k in entries:
            d[k] = chosen[id(d), k]
        tuner._dirty = True
        with autotune(True, cache=str(path)):
            pass
    torch.cuda.synchronize(device)


def pack(ids, weights):
    """FlashInfer's packed top-k: expert << 16 | BF16 weight bits (dlmrt.ops.triton.moe.route_packed in the engine)."""
    return (ids << 16) | (weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF)


def trtllm_moe(x, packed, experts, out, num_experts, lo, intermediate, quantized=None, activation="geglu"):
    """out = sum over each row's experts in [lo, lo + count) of weight * expert(x), all NVFP4: x is quantized per 16
    with its global scale, GEMM1 -> GeGLU / SwiGLU (`activation`) -> NVFP4 -> GEMM2 in FlashInfer's TRT-LLM-gen
    kernels. `packed`: [rows, k] experts and weights, see `pack`. `quantized`: x already quantized as fp4_quantize
    does, (codes [rows, hidden / 2] uint8, scales [rows, hidden / 16] uint8), with x None."""
    from flashinfer import fp4_quantize
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import ActivationType

    act = {"geglu": ActivationType.Geglu, "swiglu": ActivationType.Swiglu}[activation]
    if quantized is None:
        rows, hidden = x.shape
        xq, xs = fp4_quantize(x, experts["a1_inv"], 16, False, False)
    else:
        xq, xs = quantized
        rows, hidden = xq.shape[0], xq.shape[1] * 2
    trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed, routing_bias=None, hidden_states=xq.view(rows, hidden // 2),
        hidden_states_scale=xs.view(torch.float8_e4m3fn).reshape(rows, hidden // 16),
        gemm1_weights=experts["w1"], gemm1_weights_scale=experts["s1"], gemm1_bias=None, gemm1_alpha=None,
        gemm1_beta=None, gemm1_clamp_limit=None, gemm2_weights=experts["w2"], gemm2_weights_scale=experts["s2"],
        gemm2_bias=None, output1_scale_scalar=experts["g1_out"], output1_scale_gate_scalar=experts["g1"],
        output2_scale_scalar=experts["g2"], num_experts=num_experts, top_k=packed.shape[1], n_group=0, topk_group=0,
        intermediate_size=intermediate, local_expert_offset=lo, local_num_experts=experts["w1"].shape[0],
        routed_scaling_factor=None, routing_method_type=1, do_finalize=True,
        activation_type=int(act), tune_max_num_tokens=rows, output=out)
    return out


class Tactics:
    """The autotuner's entry for the TRT-LLM-gen MoE at one row count, the fastest tactics in isolation and the
    cache file the choice persists in (see `tune`, `pin`)."""

    def __init__(self, path, key, candidates):
        self.path, self.key, self.candidates = path, key, candidates


def tune(experts, num_experts, lo, intermediate, rows, hidden, top_k, device, candidates=4, activation="geglu"):
    """Pick the TRT-LLM-gen tactic for `rows` routed rows (FlashInfer's autotuner; its default token tile of 8 reads
    each expert's weights once per 8 rows). Run before capturing graphs; the choice holds for every layer.

    The choice is profiled once (100 timed runs per tactic: with FlashInfer's default 10, tactics ~10% apart swap
    from run to run) and kept in $DLMRT_CACHE_DIR, keyed by GPU, shapes and this rank's experts. Returns a `Tactics`
    whose `candidates` (the fastest few in isolation) are set when it was profiled now rather than loaded: the
    fastest in isolation is not always the fastest next to the concurrent dense MLP (`pin`)."""
    from flashinfer.autotuner import AutoTuner, autotune

    cache = envs.cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(device).replace(" ", "-")
    count = experts["w1"].shape[0]
    act = "" if activation == "geglu" else f"-{activation}"
    path = cache / f"trtllm-moe-{gpu}-r{rows}-h{hidden}-i{intermediate}-e{num_experts}-k{top_k}-{lo}+{count}{act}.json"
    fresh = not path.exists()
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, hidden, device=device, generator=g).to(torch.bfloat16)
    # Routing skewed like the router's (Zipf-like expert popularity): the tactic that wins on uniform routing loses
    # ~0.2 ms per round on the model's.
    popularity = 1.0 / torch.arange(1, num_experts + 1, device=device, dtype=torch.float32)
    popularity = popularity[torch.randperm(num_experts, device=device, generator=g)]
    ids = torch.multinomial(popularity.expand(rows, -1), top_k, generator=g).sort(1).values.int()
    packed = pack(ids, torch.full((rows, top_k), 1.0 / top_k, device=device))
    tuner = AutoTuner.get()
    defaults = tuner.warmup, tuner.repeat
    tuner.warmup, tuner.repeat = 10, 100
    timed, profile = {}, tuner._profile_single_kernel

    def record(runner, tensors, tactic, config, **kwargs):
        t = profile(runner, tensors, tactic, config, **kwargs)
        if tensors[0].shape[0] == rows:
            timed[tactic if isinstance(tactic, int) else tuple(tactic)] = t
        return t

    tuner._profile_single_kernel = record
    try:
        with autotune(True, cache=str(path)):
            trtllm_moe(x, packed, experts, torch.empty_like(x), num_experts, lo, intermediate, activation=activation)
    finally:
        tuner.warmup, tuner.repeat = defaults
        tuner._profile_single_kernel = profile
    torch.cuda.synchronize(device)
    key = next((k for k in tuner.profiling_cache if "trtllm_fp4_block_scale_moe" in str(k) and f"({rows}, " in str(k)),
               None)
    ranked = sorted(timed, key=timed.get)[:candidates] if fresh and key is not None else []
    return Tactics(path, key, [list(t) if isinstance(t, tuple) else t for t in ranked])


def pin(tactics, tactic):
    """Make `tactic` the one FlashInfer uses (graphs captured afterwards) and persist it in the cache file."""
    from flashinfer.autotuner import AutoTuner, autotune

    tuner = AutoTuner.get()
    runner, _, profile = tuner.profiling_cache[tactics.key]
    tuner.profiling_cache[tactics.key] = (runner, tactic, profile)
    tuner._dirty = True
    with autotune(True, cache=str(tactics.path)):
        pass


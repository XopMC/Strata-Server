"""Bounded PyTorch reference operations; no TileLang import or CUDA initialization.

These are numerical oracles and a slow fallback, not performance kernels. Sparse
attention emulates the source's BF16 probability cast in 64-position blocks.
"""
import torch
import torch.nn.functional as F


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if x.shape[-1] % block_size:
        raise ValueError('FP8 activation groups must be complete')
    groups = x.float().unflatten(-1, (-1, block_size))
    scale = groups.abs().amax(-1).clamp_min(1e-4) / 448
    if scale_fmt is not None:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quant = (groups / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((quant.float() * scale.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return quant.flatten(-2), scale.to(scale_dtype)


def fp4_codes(x):
    boundaries = x.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    magnitude = x.abs().contiguous()
    code = torch.bucketize(magnitude, boundaries)
    boundary = boundaries[code.clamp_max(6)]
    code += ((magnitude == boundary) & ((code & 1) != 0)).long()
    return (code | (torch.signbit(x).long() << 3)).to(torch.uint8)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if x.shape[-1] % block_size or scale_dtype not in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        raise ValueError('Unsupported FP4 group or scale format')
    groups = x.float().unflatten(-1, (-1, block_size))
    if scale_dtype == torch.float8_e4m3fn:
        scale = (groups.abs().amax(-1).clamp_min(6 * 2**-9) / 6).to(scale_dtype).float()
    else:
        scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(6 * 2**-126) / 6)))
    codes = fp4_codes((groups / scale.unsqueeze(-1)).clamp(-6, 6)).flatten(-2)
    if inplace:
        table = x.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
        values = table[codes.long()].unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1)
        x.copy_(values.flatten(-2).to(x.dtype))
        return x
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.view(torch.float4_e2m1fn_x2), scale.to(scale_dtype)


def fp8_gemm(x, x_scale, weight, weight_scale, scale_dtype=None, block_size=128):
    x = (x.float().unflatten(-1, (-1, block_size)) * x_scale.float().unsqueeze(-1)).flatten(-2)
    rows = torch.arange(weight.shape[0], device=weight.device) // block_size
    cols = torch.arange(weight.shape[1], device=weight.device) // block_size
    weight = weight.float() * weight_scale.float()[rows[:, None], cols[None, :]]
    return F.linear(x, weight).to(torch.bfloat16)


def fp4_gemm(x, x_scale, weight, weight_scale, scale_dtype=None, act_block_size=128):
    x = (x.float().unflatten(-1, (-1, act_block_size)) * x_scale.float().unsqueeze(-1)).flatten(-2)
    packed = weight.view(torch.uint8)
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2)
    table = x.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    # Native expert scale groups are fixed at32, independent of activation groups.
    weight = (table[codes.long()].unflatten(-1, (-1, 32)) * weight_scale.float().unsqueeze(-1)).flatten(-2)
    return F.linear(x, weight).to(torch.bfloat16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    if sinkhorn_iters < 1:
        raise ValueError('Sinkhorn requires at least one iteration')
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h:2*h] * hc_scale[1] + hc_base[h:2*h])
    comb = (mixes[..., 2*h:] * hc_scale[2] + hc_base[2*h:]).unflatten(-1, (h, h))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    if kv.shape[0] != b or kv.shape[-1] != d or topk_idxs.shape[:2] != (b, s):
        raise ValueError('Sparse attention shapes disagree')
    if torch.any(topk_idxs < -1) or torch.any(topk_idxs >= kv.shape[1]):
        raise ValueError('Sparse index outside KV range')
    output = torch.empty_like(q)
    for batch in range(b):
        for pos in range(s):
            maximum = q.new_full((h,), -1e30, dtype=torch.float32)
            denominator = q.new_zeros(h, dtype=torch.float32)
            numerator = q.new_zeros(h, d, dtype=torch.float32)
            indices = topk_idxs[batch, pos]
            for start in range(0, indices.numel(), 64):
                idx = indices[start:start+64].long()
                values = kv[batch, idx.clamp_min(0)].float()
                values = values.masked_fill((idx == -1).unsqueeze(-1), 0)
                scores = (q[batch, pos].float() @ values.T) * softmax_scale
                scores.masked_fill_((idx == -1).unsqueeze(0), -torch.inf)
                next_max = torch.maximum(maximum, scores.amax(-1))
                rescale = (maximum - next_max).exp()
                probabilities = (scores - next_max.unsqueeze(-1)).exp()
                denominator = denominator * rescale + probabilities.sum(-1)
                numerator = numerator * rescale.unsqueeze(-1) + probabilities.to(torch.bfloat16).float() @ values
                maximum = next_max
            # Stable sink addition; the source represents a virtual zero-valued KV.
            next_max = torch.maximum(maximum, attn_sink)
            rescale = (maximum - next_max).exp()
            denominator = denominator * rescale + (attn_sink - next_max).exp()
            output[batch, pos] = (numerator * rescale.unsqueeze(-1) / denominator.unsqueeze(-1)).to(q.dtype)
    return output

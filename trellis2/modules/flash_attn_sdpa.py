"""Repli quand flash-attn n'est pas installé : même API, calcul via torch SDPA
(qui utilise lui-même un noyau FlashAttention sur GPU récent)."""
import torch
import torch.nn.functional as F

__version__ = "2.7.3"


def _sdpa(q, k, v, mask=None):
    # q,k,v : [B, L, H, C] -> SDPA attend [B, H, L, C]
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return o.transpose(1, 2)


def flash_attn_func(q, k, v, *args, **kwargs):
    return _sdpa(q, k, v)


def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
    q, k, v = qkv.unbind(dim=2)
    return _sdpa(q, k, v)


def flash_attn_kvpacked_func(q, kv, *args, **kwargs):
    k, v = kv.unbind(dim=2)
    return _sdpa(q, k, v)


def _pad(x, cu, b0, b1, lmax):
    # x : [T, H, C] tokens des séquences b0..b1-1 (contigus) -> [B, lmax, H, C]
    lens = (cu[b0 + 1:b1 + 1] - cu[b0:b1])
    seq = torch.repeat_interleave(torch.arange(b1 - b0, device=x.device), lens)
    off = torch.arange(int(cu[b1] - cu[b0]), device=x.device) - (cu[b0:b1] - cu[b0])[seq]
    out = x.new_zeros((b1 - b0, lmax) + tuple(x.shape[1:]))
    out[seq, off] = x[int(cu[b0]):int(cu[b1])]
    return out, seq, off, lens


def _varlen(q, k, v, cu_q, cu_k, max_q=None, max_k=None, **kwargs):
    cu_q = cu_q.long().to(q.device); cu_k = cu_k.long().to(q.device)
    nseq = cu_q.numel() - 1
    if nseq == 1:
        return _sdpa(q[None], k[None], v[None])[0]
    lq = (cu_q[1:] - cu_q[:-1]); lk = (cu_k[1:] - cu_k[:-1])
    out = torch.empty(q.shape[:-1] + (v.shape[-1],), dtype=q.dtype, device=q.device)
    if nseq <= 8:
        for i in range(nseq):
            a, b = int(cu_q[i]), int(cu_q[i + 1]); c, d = int(cu_k[i]), int(cu_k[i + 1])
            out[a:b] = _sdpa(q[None, a:b], k[None, c:d], v[None, c:d])[0]
        return out
    # beaucoup de petites séquences (attention par fenêtres) : batch avec padding, par paquets
    LQ, LK = int(lq.max()), int(lk.max())
    H = q.shape[1]
    budget = 2 ** 27  # nb d'éléments du masque/scores par paquet
    step = max(1, budget // max(1, H * LQ * LK))
    for b0 in range(0, nseq, step):
        b1 = min(nseq, b0 + step)
        qp, sq, oq, _ = _pad(q, cu_q, b0, b1, LQ)
        kp, _, _, lkb = _pad(k, cu_k, b0, b1, LK)
        vp, _, _, _ = _pad(v, cu_k, b0, b1, LK)
        mask = (torch.arange(LK, device=q.device)[None, :] < lkb[:, None])[:, None, None, :]
        o = _sdpa(qp, kp, vp, mask)
        out[int(cu_q[b0]):int(cu_q[b1])] = o[sq, oq]
    return out


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *args, **kwargs):
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, *args, **kwargs):
    k, v = kv.unbind(dim=1)
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, *args, **kwargs):
    q, k, v = qkv.unbind(dim=1)
    return _varlen(q, k, v, cu_seqlens, cu_seqlens)

import torch
from collections import defaultdict


def pos_emb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """Pos embedding for 2D image"""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "dimension must be divisible by 4"

    # 1D pos embedding
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature**omega)

    # 2D pos embedding
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # concat sin and cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def g_k(p, k=1):
    """
    Compute g_k(p) = x_(k) * (1 - x_(k) + x_(k+1))
    where x_(k) and x_(k+1) are k-th and (k+1)-th largest elements per row.
    """
    sorted_p, _ = torch.sort(p, dim=-1, descending=True)
    x_k = sorted_p[..., k - 1]
    x_k1 = sorted_p[..., k] if k < p.size(-1) else torch.zeros_like(x_k)
    return x_k * (1 - x_k + x_k1)


def jasmin_loss(attn_maps, k=0, reduction="mean"):
    """
    Compute JaSMin regularization loss.

    attn_maps: [B, H, N, N] tensor or list of them.
    k=0 -> Eq.(8) from paper
    k>0 -> Eq.(9)
    """
    if isinstance(attn_maps, torch.Tensor):
        attn_maps = [attn_maps]

    losses = []
    for P in attn_maps:
        P = torch.clamp(P, min=1e-12, max=1.0)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)
        g1 = g_k(P, k=1)
        if k == 0:
            loss = torch.log(g1 + 1e-12)
        else:
            gk = g_k(P, k=k)
            loss = torch.log((g1 / (gk + 1e-12)) + 1e-12)

        # max over tokens, mean over heads
        loss = loss.max(dim=-1).values.mean(dim=1)
        losses.append(loss.mean())

    losses = torch.stack(losses)
    return losses.mean() if reduction == "mean" else losses

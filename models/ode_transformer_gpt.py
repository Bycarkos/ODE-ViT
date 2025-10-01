import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class PatchEmbed(nn.Module):
    """
    ViT patch embedding via Conv2d.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/ps, W/ps] -> [B, N, D]
        x = self.proj(x)                           # [B, D, Gh, Gw]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadSelfAttention(nn.Module):
    """
    Thin wrapper around nn.MultiheadAttention to work with [B, N, D] shapes.
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0, bias: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, bias=bias, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> attn_out same shape
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        attn_out = self.proj_drop(attn_out)
        return attn_out


# ---------------------------------
# Parallel block and ODE dynamics
# ---------------------------------

class ParallelAttentionMLP(nn.Module):
    """
    Implements the parallel sublayers: return F(x) + G(x,X) as a *derivative* (no residual add here).
    LayerNorms are applied before the sublayers (pre-norm).
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_drop: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self.attn = MultiheadSelfAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=mlp_drop)
        self.tg = nn.Linear(1, out_features=dim)
        self.tf = nn.Linear(1, out_features=dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tG = self.tg(t.view(-1, 1))
        tF = self.tf(t.view(-1, 1))

        g = self.attn(self.norm_attn(x) + tG)  # G(x)
        f = self.mlp(self.norm_mlp(x) + tF)    # F(x)
        return f + g                    # derivative


class ViT_ODEFunc(nn.Module):
    """
    ODE function f(t, x) that returns dx/dt = F(x) + G(x).
    Optionally time-scales the drift if integrating over [0, 1] but wanting to emulate D discrete layers.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_drop: float = 0.0,
                 emulate_depth: int = 12, time_interval: float = 12.0):
        super().__init__()
        self.dim = dim
        self.block = ParallelAttentionMLP(dim, num_heads, mlp_ratio, attn_drop, proj_drop, mlp_drop)

        # If you integrate over [0, 1] and want to match D layers, multiply by D.
        # If you integrate over [0, D], set scaler=1.0
        if time_interval == 1.0:
            self.scaler = float(emulate_depth)
        else:
            self.scaler = 1.0

    def fourier_encode_time(self, t, dim):
        """
        Args:
            t: scalar float tensor or shape (B,) — time value(s)
            dim: int — dimension of output encoding (must be even)

        Returns:
            Tensor of shape (B, dim), each row is the Fourier encoding of time t
        """
        assert dim % 2 == 0, "Dimension must be even for sin/cos pairs"

        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32) * (-math.log(10000.0) / half_dim)
        ).to(t.device)  # (half_dim,)

        # t can be scalar or vector (B,)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        t = t.unsqueeze(-1)  # (B, 1)

        angles = t * freqs  # (B, half_dim)
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        return torch.cat([sin, cos], dim=-1)  # (B, dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        t = t.to(x.device)
        dx = self.block(x, t) * self.scaler
        return dx #+ x


# -----------------------
# Full ViT-ODE model
# -----------------------

class ViTNeuralODE(nn.Module):
    """
    ViT-like encoder where the "depth" is represented by integrating an ODE whose drift is F(x)+G(x).
    - Patch embedding + cls token + learned positional embeddings
    - ODEBlock over token sequence
    - Classification head over [CLS]
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        emulate_depth: int = 12,           # corresponds to a 12-layer ViT
        time_interval: float = 12.0,       # integrate over [0, 12] to match 12 layers; set to 1.0 if you prefer [0,1]
        num_eval_steps: int = 48,          # solver internal evaluation points (e.g., 4 per unit of time)
        solver: str = "rk4"
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 + num_patches, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # ODE function and integrator
        self.odefunc = ViT_ODEFunc(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            emulate_depth=emulate_depth,
            time_interval=time_interval,
        )

        # Head
        self.norm_head = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dist_head = nn.Linear(embed_dim, num_classes)
        self.norm_dist = nn.LayerNorm(embed_dim)
        self.solver = solver

        # Time grid
        self.time_interval = time_interval
        self.num_eval_steps = num_eval_steps
        self.t_grid = torch.linspace(0.0, time_interval, num_eval_steps)

        self.control_points_idx = (self.num_eval_steps // emulate_depth)
        self.control_points = torch.linspace(0, num_eval_steps-1, steps=12).long()
        self._init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)

        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input images -> token embeddings with cls + pos.
        """
        x = self.patch_embed(x)                    # [B, N, D]
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)     # [B, 1, D]
        dist = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls, dist, x], dim=1)# [B, 1+N, D]
        x = x + self.pos_embed[:, : (N + 2)]
        x = self.pos_drop(x)
        return x

    def forward(
        self,
        pixel_values: torch.Tensor,                           # [B, 3, H, W]
        labels: Optional[torch.Tensor] = None,     # [B]
        teacher_states: Optional[torch.Tensor] = None,  # [L, B, N, D] (optional: baseline ViT block outputs)
        kd_weight: float = 1.0,                    # distillation weight (if teacher_states is provided)
        output_hidden_states: bool = False,
    ):
        """
        Returns:
            dict:
              logits: [B, num_classes]
              loss: scalar (if labels provided)
              states: [T, B, N, D] (if return_all_states)
              ctrl_states: [Q, B, N, D] optional KD anchors
        """
        tokens = self.embed(pixel_values)                     # [B, 1+N, D]
        states = odeint(self.odefunc, tokens, self.t_grid.to(self.device), method=self.solver)

        final = states[-1]                         # [B, 1+N, D]
        cls_final = final[:, 0]                    # [B, D]
        dist_final = final[:, 1]
        logits = self.head(self.norm_head(cls_final))
        logits_dist = self.dist_head(self.norm_dist(dist_final))

        control_points = states[self.control_points]  # [Q, B, 1+N, D]

        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        out = {"logits": logits, "logits_dist": logits_dist, "loss": loss, "control_points": control_points}
        if output_hidden_states:
            out["states"] = states

        return out

# -----------------------
# Minimal usage example
# -----------------------

if __name__ == "__main__":
    # CIFAR-100-like batch
    model = ViTNeuralODE(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=100,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        emulate_depth=12,
        time_interval=1.0,   # match 12 "layers" by integrating over [0,12]
        num_eval_steps=48,
        solver="dopri5",
        distilled_control_points=None,  # set to an int (e.g., 12) if doing KD
    ).cuda()

    x = torch.randn(16, 3, 32, 32).cuda()
    y = torch.randint(0, 100, (16,)).cuda()
    out = model(x, labels=y)
    print(out["logits"].shape, out["loss"])

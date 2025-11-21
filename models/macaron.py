import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


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
        self.init_ivp = nn.Conv2d(in_chans, embed_dim, kernel_size=5, stride=1)
        self.pooler = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor, learn_ivp:bool =False) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D, H/ps, W/ps] -> [B, N, D]
        if learn_ivp:
            x_ivp = self.init_ivp(x)
            x_ivp = F.gelu(x_ivp)
            flattened = self.pooler(x_ivp).flatten(2).squeeze(-1)
        else:
            flattened = None
        x = self.proj(x)                           # [B, D, Gh, Gw]
        x = x.flatten(2).transpose(1, 2).contiguous()
        # [B, N, D]
        return x, flattened


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        #self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        #x = self.fc2(x)
        #x = self.drop(x)
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

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_drop: float = 0.0,
                 bias_init_scale: float = 1e-3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        # Standard FFN: dim -> hidden -> dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(hidden, dim),
            nn.Dropout(mlp_drop),
        )

        self.attn = MultiheadSelfAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        # small init to keep initial IVP close to original CLS
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # small optional scaling parameter for stability (start near 1.0)
        self.res_scale = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, D]
        # half FFN step (pre-norm)
        x_ffn_in = self.norm1(x)
        delta1 = self.ffn(x_ffn_in)
        x1 = x + 0.5 * self.res_scale * delta1

        # attention step (pre-norm)
        x_attn_in = self.norm2(x1)
        delta2 = self.attn(x_attn_in)   # [B, N, D]
        x2 = x1 + self.res_scale * delta2

        # second half FFN (pre-norm)
        x_ffn2_in = self.norm3(x2)
        delta3 = self.ffn(x_ffn2_in)
        x3 = x2 + 0.5 * self.res_scale * delta3

        return x3

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



    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        t = t.to(x.device)
        dx = self.block(x, t) * self.scaler
        return dx #+ x


# -----------------------
# Full ViT-ODE model
# -----------------------

class ViTMacaron(nn.Module):
    """
    ViT-like encoder where the "depth" is represented by integrating an ODE whose drift is F(x)+G(x).
    - Patch embedding + cls token + learned positional embeddings
    - ODEBlock over token sequence
    - Classification head over [CLS]
    """

    ## AD-HOC
    AVG_DISTANCES_CONSECUTIVE_HIDDEN_STATES_VIT = torch.tensor([ 19.9335, 12.61485625,  13.10309922,  14.70024375,  15.15418125,
      17.1821, 14.34054062,  18.23386562,  23.4014875,   14.24714063, 29.36258125, 171.6232875 ])

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
        solver: str = "rk4",
        add_distillation_token: bool = False,
        predict_outher_space: bool = False,
        outher_embedding_dimension: int = 768,
        learn_ivp: bool = False
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        num_extra_tokens = 1

        self.learn_ivp = learn_ivp
        self.add_distillation_token = add_distillation_token
        if learn_ivp:
            self._ivp_projector = nn.Linear(2*embed_dim, embed_dim)



        if add_distillation_token:
            num_extra_tokens += 1
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.dist_head = nn.Linear(embed_dim, num_classes)
            self.norm_dist = nn.LayerNorm(embed_dim)



        # Class token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_extra_tokens + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        if predict_outher_space:
            self.outher_embed = self.init_space_predictor(embed_dim, outher_embedding_dimension)

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
        self.solver = solver

        # Time grid
        self.time_interval = time_interval
        self.num_eval_steps = num_eval_steps
        self.t_grid = torch.linspace(0.0, time_interval, num_eval_steps)

        self._init_weights()

    def get_proportional_control_points_with_temperature(self, temperature, num_eval_steps: Optional[int] = None):

        x = self.AVG_DISTANCES_CONSECUTIVE_HIDDEN_STATES_VIT / temperature
        x_exp = torch.exp(x - torch.max(x))

        x_exp_sum = torch.sum(x_exp)
        x_exp_norm = x_exp / x_exp_sum

        if num_eval_steps is not None:
            steps = torch.round(x_exp_norm*num_eval_steps)
        else:
            steps = torch.round(x_exp_norm* self.num_eval_steps).int()

        checkpoints = torch.cumsum(steps, dim=0).long()

        return checkpoints

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.add_distillation_token:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def init_space_predictor(self, outher_embedding_dimension):
        self.space_predictor = nn.Linear(self.embedding_dim, outher_embedding_dimension)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input images -> token embeddings with cls + pos.
        """
        x, ivp = self.patch_embed(x, self.learn_ivp)                    # [B, N, D]
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)     # [B, 1, D]
        if self.learn_ivp:
            ivp = torch.cat([cls, ivp.unsqueeze(1)], dim=-1)
            cls = self._ivp_projector(ivp) #cls + self.space_predictor(ivp).unsqueeze(1)
            cls = F.gelu(cls)

        if self.add_distillation_token:
            dist = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls, dist, x], dim=1)# [B, 1+N, D]
            extra_tokens = 2
        else:
            x = torch.cat([cls, x], dim=1)# [B, 1+N, D]
            extra_tokens = 1

        x = x + self.pos_embed[:, : (N + extra_tokens)]
        x = self.pos_drop(x)
        return x

    def forward(
        self,
        pixel_values: torch.Tensor,                           # [B, 3, H, W]
        labels: Optional[torch.Tensor] = None,     # [B]
        output_hidden_states: bool = False,
        output_control_points: bool = False,
        t_grid: Optional[torch.Tensor] = None,
        temperature: Optional[float] = 100.0
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

        if t_grid is None:
            num_eval_steps = self.num_eval_steps
            states = odeint(self.odefunc, tokens, self.t_grid.to(self.device), method=self.solver)
        else:
            num_eval_steps = len(t_grid)
            states = odeint(self.odefunc, tokens, t_grid.to(self.device), method=self.solver)

        final = states[-1]                         # [B, 1+N, D]
        cls_final = final[:, 0]                    # [B, D]
        logits = self.head(self.norm_head(cls_final))
        out = {"logits": logits}

        if self.add_distillation_token:
            dist_final = final[:, 1]
            logits_dist = self.dist_head(self.norm_dist(dist_final))
            out["logits_dist"] = logits_dist

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss

        if output_hidden_states:
            out["states"] = states

        if output_control_points:

            control_points = self.get_proportional_control_points_with_temperature(
                temperature=temperature, num_eval_steps=num_eval_steps)
            control_points = states[control_points]  # [Q, B, 1+N, D]
            out["control_points"] = control_points

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
        solver="euler",

    ).cuda()

    x = torch.randn(16, 3, 32, 32).cuda()
    y = torch.randint(0, 100, (16,)).cuda()
    out = model(x, labels=y, output_control_points=True)
    print(out["logits"].shape, out["loss"])

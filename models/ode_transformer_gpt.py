import math
from typing import Optional, Tuple, List
import utils


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class L2SelfAttention(nn.Module):
    """
    L2-based self-attention (Lipschitz-controlled).
    Computes attention weights from negative squared L2 distances instead of dot products.
    """

    def __init__(
        self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, D]
        returns (output, attention)
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        # Compute pairwise L2 distances ||q_i - k_j||^2
        q2 = (q**2).sum(-1, keepdim=True)  # [B, H, N, 1]
        k2 = (k**2).sum(-1).unsqueeze(-2)  # [B, H, 1, N]
        # dist^2 = ||q||^2 + ||k||^2 - 2 q·k^T
        dist2 = q2 + k2 - 2 * torch.matmul(q, k.transpose(-2, -1))  # [B, H, N, N]

        # L2 attention weights (negative distances)
        attn = torch.exp(-dist2 * self.scale)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = torch.matmul(attn, v)  # [B, H, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out, attn


class CenterNorm(nn.Module):
    r"""CenterNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = normalized_shape / (normalized_shape - 1.0)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = self.scale * (x - u)
        x = self.weight[None, None, :] * x + self.bias[None, None, :]
        return x


class PatchEmbed(nn.Module):
    """
    ViT patch embedding via Conv2d.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        add_distillation_token=False,
        register_tokens: int = 4,
        pos_embed_register_tokens: bool = True,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.pos_embed_register_tokens = pos_embed_register_tokens
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        num_patches = self.num_patches
        num_extra_tokens = 1

        self.add_distillation_token = add_distillation_token

        if add_distillation_token:
            num_extra_tokens += 1
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Class token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # register tokens

        self.register_tokens = nn.Parameter(torch.randn(register_tokens, embed_dim))

        self.num_register_tokens = register_tokens

        # if pos_embed_register_tokens:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1 + register_tokens, embed_dim),
            requires_grad=True,
        )
        # else:
        #    self.pos_embed = nn.Parameter(
        #        torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True
        #   )
        self.pos_drop = nn.Dropout(p=0.0)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.add_distillation_token:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x: torch.Tensor, learn_ivp: bool = False) -> torch.Tensor:
        """
        Input images -> token embeddings with cls + pos.
        """

        x = self.proj(x)  # [B, D, Gh, Gw]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, D]

        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]

        register_tokens = self.register_tokens.expand(B, -1, -1)  # [B, R, D]

        if self.add_distillation_token:
            dist = self.dist_token.expand(B, -1, -1)
            x = torch.cat(
                [x[:, [0]], dist, x[:, 1:], register_tokens], dim=1
            )  # [B, 1+N, D]
        else:
            x = torch.cat([x, register_tokens], dim=1)  # [B, 1+N, D]

        if self.pos_embed_register_tokens:
            x += self.pos_embed[
                :, : self.num_patches + 1 + self.num_register_tokens, :
            ].to(x.device)
        else:
            x[:, : self.num_patches + 1, :] += self.pos_embed[
                :, : self.num_patches + 1, :
            ].to(x.device)

        x = self.pos_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
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

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=bias,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> attn_out same shape
        attn_out, attentions = self.mha(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        attn_out = self.proj_drop(attn_out)
        return (attn_out, attentions)


# ---------------------------------
# Parallel block and ODE dynamics
# ---------------------------------


class ParallelAttentionMLP(nn.Module):
    """
    Implements the parallel sublayers: return F(x) + G(x,X) as a *derivative* (no residual add here).
    LayerNorms are applied before the sublayers (pre-norm).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        use_l2: bool = False,
    ):
        super().__init__()
        self.norm_attn = CenterNorm(dim)
        self.norm_mlp = CenterNorm(dim)

        if use_l2:
            self.attn = L2SelfAttention(
                dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop
            )
        else:
            self.attn = MultiheadSelfAttention(
                dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop
            )

        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=mlp_drop)

        # self.tg = nn.Linear(1, out_features=dim)
        # self.tf = nn.Linear(1, out_features=dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        g, self.attentions = self.attn(self.norm_attn(x))  # + tG)  # G(x)
        f = self.mlp(self.norm_mlp(x))  # + tF)  # F(x)
        return f + g  # derivative


class ViT_ODEFunc(nn.Module):
    """
    ODE function f(t, x) that returns dx/dt = F(x) + G(x).
    Optionally time-scales the drift if integrating over [0, 1] but wanting to emulate D discrete layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        emulate_depth: int = 12,
        time_interval: float = 12.0,
        l2_attention: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.block = ParallelAttentionMLP(
            dim,
            num_heads,
            mlp_ratio,
            attn_drop,
            proj_drop,
            mlp_drop,
            use_l2=l2_attention,
        )

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

        # record attentions for this time step
        if not hasattr(self, "attention_trajectory"):
            self.attention_trajectory = []

        self.attention_trajectory.append(
            self.block.attentions.detach()[:, :, :-10, :-10]
        )

        return dx  # + x


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

    ## AD-HOC
    AVG_DISTANCES_CONSECUTIVE_HIDDEN_STATES_VIT = torch.tensor(
        [
            19.9335,
            12.61485625,
            13.10309922,
            14.70024375,
            15.15418125,
            17.1821,
            14.34054062,
            18.23386562,
            23.4014875,
            14.24714063,
            29.36258125,
            171.6232875,
        ]
    )

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
        emulate_depth: int = 12,  # corresponds to a 12-layer ViT
        time_interval: float = 12.0,  # integrate over [0, 12] to match 12 layers; set to 1.0 if you prefer [0,1]
        num_eval_steps: int = 24,  # solver internal evaluation points (e.g., 4 per unit of time)
        solver: str = "rk4",
        add_distillation_token: bool = False,
        l2_attention: bool = False,
        outher_embedding_dimension: int = 768,
        register_tokens: int = 4,
        pos_embed_register_tokens: bool = False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            add_distillation_token,
            register_tokens=register_tokens,
            pos_embed_register_tokens=pos_embed_register_tokens,
        )

        self.emulate_depth = emulate_depth
        self.l2_attention = l2_attention
        self.add_distillation_token = add_distillation_token

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
            l2_attention=l2_attention,
        )

        # Head
        self.head = nn.Linear(embed_dim, num_classes)
        self.solver = solver

        if add_distillation_token:
            self.dist_head = nn.Linear(embed_dim, num_classes)
            self.norm_dist = CenterNorm(embed_dim)

        # Time grid
        self.embed_dim = embed_dim
        self.time_interval = time_interval
        self.num_eval_steps = num_eval_steps
        self.t_grid = torch.linspace(0.0, time_interval, num_eval_steps)

        self.apply(self._spectral_init)

    def g_k(self, p, k=1):
        """
        Compute g_k(p) = x_(k) * (1 - x_(k) + x_(k+1))
        where x_(k) and x_(k+1) are k-th and (k+1)-th largest elements per row.
        """
        sorted_p, _ = torch.sort(p, dim=-1, descending=True)
        x_k = sorted_p[..., k - 1]
        x_k1 = sorted_p[..., k] if k < p.size(-1) else torch.zeros_like(x_k)
        return x_k * (1 - x_k + x_k1)

    def jasmin_loss(self, attn_maps, k=0, reduction="mean"):
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
            g1 = self.g_k(P, k=1)
            if k == 0:
                loss = torch.log(g1 + 1e-12)
            else:
                gk = self.g_k(P, k=k)
                loss = torch.log((g1 / (gk + 1e-12)) + 1e-12)

            # max over tokens, mean over heads
            loss = loss.max(dim=-1).values.mean(dim=1)
            losses.append(loss.mean())

        losses = torch.stack(losses)
        return losses.mean() if reduction == "mean" else losses

    def finite_difference_second_derivative_sequence(self, f_t, delta_t=1e-4):
        """
        Compute second derivative along time dimension (axis=0) using finite differences.

        Args:
            f_t: Tensor of shape [T, B, S, D] — output of attention block evaluated at T time points
            delta_t: scalar float — step size between time evaluations
        Returns:
            Tensor of shape [T-2, B, S, D] — second derivative w.r.t time
        """
        return (f_t[2:] - 2 * f_t[1:-1] + f_t[:-2]) / (delta_t**2)

    def get_proportional_control_points_with_temperature(
        self, temperature, num_eval_steps: Optional[int] = None
    ):
        x = self.AVG_DISTANCES_CONSECUTIVE_HIDDEN_STATES_VIT / temperature
        x_exp = torch.exp(x - torch.max(x))

        x_exp_sum = torch.sum(x_exp)
        x_exp_norm = x_exp / x_exp_sum

        if num_eval_steps is not None:
            steps = torch.round(x_exp_norm * num_eval_steps)
        else:
            steps = torch.round(x_exp_norm * self.num_eval_steps).int()

        checkpoints = torch.cumsum(steps, dim=0).long()
        if checkpoints[-1] == self.num_eval_steps:
            checkpoints[-1] -= 1

        return checkpoints

    @property
    def device(self):
        return next(self.parameters()).device

    def _spectral_init(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

            u, s, v = torch.svd(m.weight)
            m.weight.data = 1.0 * m.weight.data / s[0]

        elif isinstance(m, (nn.Conv2d)):
            # torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.xavier_normal_(m.weight)
            weight = torch.reshape(m.weight.data, (m.weight.data.shape[0], -1))
            u, s, v = torch.svd(weight)
            m.weight.data = m.weight.data / s[0]

        elif isinstance(m, (nn.LayerNorm, CenterNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_upper_bound_by_second_derivative(self, R, L):
        Wq, Wk, Wv = self.odefunc.block.attn.mha.in_proj_weight.reshape(
            3, self.embed_dim, self.embed_dim
        )

        factor1 = R**2 * torch.norm(Wv, p=2)
        factor2 = R * torch.linalg.norm(Wk @ Wq.mT) + (Wk.shape[-1]) ** 0.5
        factor3 = (self.num_eval_steps**2) * (Wq.shape[-1] ** 0.5)
        supremum = (factor1 * factor2) / factor3

        bound = (math.e ** (L) - 1) / (2 * L * self.num_eval_steps) * supremum

        return bound

    @torch.no_grad()
    def compute_upper_bound_by_fininte_difference(self, x, L, N):
        first_factor = (math.e ** (L) - 1) / (2 * L * N)
        second_factor = self.finite_difference_second_derivative_sequence(x, 1 / N)
        curvature_norm = torch.norm(second_factor, p=float("inf"), dim=-1)

        max_sup_per_seq = curvature_norm.max(dim=0)[0]
        max_sup_per_batch = max_sup_per_seq.max(-1)[0]
        sup_global = curvature_norm.max()

        return dict(
            global_upper_bound=(first_factor * sup_global).item(),
            batched_upper_bound=(first_factor * max_sup_per_batch),
            batched_upper_bound_per_seq=(first_factor * max_sup_per_seq),
        )

    def init_space_predictor(self, outher_embedding_dimension):
        self.space_predictor = nn.Linear(self.embedding_dim, outher_embedding_dimension)

    def forward(
        self,
        pixel_values: torch.Tensor,  # [B, 3, H, W]
        labels: Optional[torch.Tensor] = None,  # [B]
        output_hidden_states: bool = False,
        output_control_points: bool = False,
        output_attentions: bool = False,
        output_attention_trajectory: bool = False,
        t_grid: Optional[torch.Tensor] = None,
        temperature: Optional[float] = 100,
        jasmin_k: int = 10,
    ):
        """
        Returns:
            dict:
              logits: [B, num_classes]
              loss: scalar (if labels provided)
              states: [T, B, N, D] (if return_all_states)
              ctrl_states: [Q, B, N, D] optional KD anchors
        """
        tokens = self.patch_embed(pixel_values)  # [B, 1+N, D]
        if t_grid is None:
            num_eval_steps = self.num_eval_steps
            states = odeint(
                self.odefunc, tokens, self.t_grid.to(self.device), method=self.solver
            )
        else:
            num_eval_steps = len(t_grid)
            states = odeint(
                self.odefunc, tokens, t_grid.to(self.device), method=self.solver
            )

        final = states[-1]  # [B, 1+N, D]
        upper_bound_batch = self.compute_upper_bound_by_second_derivative(
            R=jasmin_k, L=1 / 2
        )
        finite_upper_bounds = self.compute_upper_bound_by_fininte_difference(
            states, 0.5, 1 / self.num_eval_steps
        )

        cls_final = final[:, 0]  # [B, D]
        logits = self.head(cls_final)
        out = {
            "logits": logits,
            "second_derivative_upper_bound": upper_bound_batch,
            "finite_difference_upper_bound": finite_upper_bounds,
        }

        if output_attention_trajectory:
            out["attention_trajectory"] = self.odefunc.attention_trajectory

        if output_attentions:
            out["attentions"] = self.odefunc.block.attentions[
                :,
                :,
                : -self.patch_embed.num_register_tokens,
                : -self.patch_embed.num_register_tokens,
            ]

            out["attentions_register_tokens"] = self.odefunc.block.attentions[
                :,
                :,
                -self.patch_embed.num_register_tokens :,
                :,
            ]

            out["jasmin_loss"] = self.jasmin_loss(
                self.odefunc.attention_trajectory[-1],
                k=jasmin_k,
                reduction="mean",
            )

        if self.add_distillation_token:
            dist_final = final[:, 1]
            logits_dist = self.dist_head(dist_final)
            out["logits_dist"] = logits_dist

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            out["loss"] = loss

        if output_hidden_states:
            out["states"] = states

        if output_control_points:
            control_points = self.get_proportional_control_points_with_temperature(
                temperature=temperature, num_eval_steps=num_eval_steps
            )
            control_points = states[
                control_points, :, : -self.patch_embed.num_register_tokens
            ]  # [Q, B, 1+N, D]
            out["control_points"] = control_points

        self.odefunc.attention_trajectory = []  # reset for next forward pass

        return out


# -----------------------
# Minimal usage example
# -----------------------

if __name__ == "__main__":
    # CIFAR-100-like batch
    model = ViTNeuralODE(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=100,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        emulate_depth=12,
        time_interval=1.0,  # match 12 "layers" by integrating over [0,12]
        num_eval_steps=48,
        solver="euler",
    ).cuda()

    x = torch.randn(16, 3, 224, 224).cuda()
    y = torch.randint(0, 100, (16,)).cuda()
    out = model(x, labels=y, output_control_points=True)
    print(out["logits"].shape, out["loss"])

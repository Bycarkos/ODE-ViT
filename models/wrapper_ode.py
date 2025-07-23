from turtle import forward
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

import torch
from torch import nn as nn

from typing import Optional
from .time_emb import TimeEmbedding

class TemporalLinear(nn.Module):
    def __init__(self, sinusoidal_dim, hidden_dim, temporal_dim, output_dim):
        super().__init__()
        self._mlp_temporal = nn.Sequential(nn.Linear(sinusoidal_dim, temporal_dim), nn.Softplus(), nn.Linear(temporal_dim, temporal_dim))
        self.conditional_liner = nn.Linear(hidden_dim + temporal_dim, output_dim, bias=True)

    def forward(self, t, x):
        B, T, d = x.shape
        t = self._mlp_temporal(t)
        hidden_state = self.conditional_liner(torch.cat([x, t.repeat(B, T, 1)], dim=-1))
        return hidden_state

class TemporalLayerNorm(nn.Module):
    def __init__(self,
                 input_dim: int,
                 time_embed_dim: int,
                 eps: float = 1e-5,
                 use_weight: bool = True,
                 use_bias: bool = True):
        """
        Time-dependent LayerNorm implemented via MLP conditioned on time embeddings.
        """
        super().__init__()

        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.input_dim = input_dim

        # MLP to process time embeddings
        self.lin1 = nn.Linear(time_embed_dim, time_embed_dim)
        self.lin2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.activation = nn.SiLU()

        # Time-dependent weights and biases for LayerNorm
        if use_weight:
            self.f_weight = nn.Linear(time_embed_dim, input_dim)
        else:
            self.f_weight = None

        if use_bias:
            self.f_bias = nn.Linear(time_embed_dim, input_dim)
        else:
            self.f_bias = None

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, ..., D] where D is the input_dim
        time_embed: Tensor of shape [B, time_embed_dim]
        """

        # Compute time-dependent modulations
        t = self.lin1(time_embed)
        t = self.activation(t)
        t = self.lin2(t)

        # Compute normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)

        if self.f_weight is not None:
            weight = self.f_weight(t).unsqueeze(1) + 1.0  # [B, 1, D]
            out = out * weight

        if self.f_bias is not None:
            bias = self.f_bias(t).unsqueeze(1)  # [B, 1, D]
            out = out + bias


        return out.squeeze(1)

class TemporalMLP(nn.Module):
    def __init__(self, sinusoidal_dim, temporal_dim,  hidden_dim, output_dim):
        super().__init__()

        self.act = torch.nn.Softplus()

        self.c_fc = TemporalLinear(sinusoidal_dim=sinusoidal_dim,hidden_dim=hidden_dim, temporal_dim=temporal_dim, output_dim=output_dim)

        self.c_proj = TemporalLinear(sinusoidal_dim=sinusoidal_dim,hidden_dim=output_dim, temporal_dim=temporal_dim, output_dim=output_dim)

    def forward(self, t, x):

        x = self.c_fc(t, x)
        x = self.act(x)
        x = self.c_proj(t, x)

        return x

class ConditionalODEfunc(nn.Module):

    def __init__(self, resiudal_block: torch.nn.Module, hidden_dim=768, temporal_dimension:int=768, steps:int=12) -> None:
        super(ConditionalODEfunc, self).__init__()

        self.net = resiudal_block
        self.ln_mlp = TemporalLinear(sinusoidal_dim=temporal_dimension,temporal_dim=temporal_dimension, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.mlp = TemporalMLP(sinusoidal_dim=temporal_dimension, temporal_dim=temporal_dimension, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.temporal_layer_norm = TemporalLayerNorm(input_dim=hidden_dim, time_embed_dim=hidden_dim)
        self.time_embedding = TimeEmbedding(768, 768, learnable_sinusoidal=True)

        
        self.nfe = 0

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, t, x, output_hidden_states: bool = True):
        
        temporal_embedding = self.time_embedding(t.unsqueeze(0))  # Ensure proper shape
        self.nfe += 1

        residual = x.clone()
        for layer in self.net.layer:
            hidden_state_dt = self.ln_mlp(temporal_embedding, x)
            hidden_state_dt = self.temporal_layer_norm(temporal_embedding, hidden_state_dt)
            self_attention_outputs = layer.attention(hidden_state_dt, output_attentions=False)

            x = residual + self_attention_outputs[0]
            residual = x

            #hidden_state_dt = layer.layernorm_after(output)
            #hidden_state_dt = self.temporal_output(temporal_embedding, output)

        mlp_output = self.mlp(temporal_embedding, x)

        return x + mlp_output


class ODEBlock(nn.Module):
    def __init__(self, odefunc, solver: str = 'euler', step_size: float=12, fixed_step_solver: bool = True, atol: Optional[float] = 1e-5, rtol: Optional[float] = 1e-3, ):
        super(ODEBlock, self).__init__()

        self.odefunc = odefunc
        self.atol = atol
        self.rtol = rtol
        self.solver = solver
        
        self.default_integration_time = torch.tensor([0, 1]).float()

        self.ode_solver = odeint_adjoint
        self.options = {}

    def forward(
        self, x: torch.Tensor, t: torch.Tensor = None, save=False
    ) -> torch.Tensor:

        if t is not None:
            self.integration_time = t
        else:
            self.integration_time = self.default_integration_time

        self.integration_time = self.integration_time.type_as(x)
        out = self.ode_solver(
            self.odefunc,
            x,
            self.integration_time,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
            options=self.options,
        )

        # Return only the last time point
        return out

    @property
    def nfe(self) -> int:
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value: int):
        self.odefunc.nfe = value

class ConditionalEDOEncoderWrapper(nn.Module):

    def __init__(self, encoder, n_classes, hidden_dim=768, steps: int= 12, solver:str="rk4"):
        super().__init__()
        self.encoder = encoder.vit  # e.g. a ViT or CNN backbone returning .last_hidden_state [B, N, d]
        self.hidden_dim = hidden_dim
        self.num_queries = n_classes
        self.steps = steps
        self.n_classes = n_classes
        self._act = nn.Softplus()
        self.solver = solver

        # Final classifier: project to vocab
        if hasattr(self.encoder, "classifier"):
            self.projector = self.encoder.classifier
        else:
            self.projector = nn.Linear(hidden_dim, n_classes)

        self.ode_func_wrapper = ConditionalODEfunc(resiudal_block=self.encoder.encoder)
        self.state_update = ODEBlock(odefunc=self.ode_func_wrapper, solver=solver, step_size=steps)
        self.dt = torch.tensor(1.0 / steps, device=self.device)

        self.t_values = torch.arange(steps, dtype=torch.float32, device=self.device)
        self.continuous_time = self.dt * self.t_values

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self,
        pixel_values,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True):

        hidden_state = self.encoder.embeddings(pixel_values)  # [B, N, D]

        states = self.state_update(x=hidden_state, t=self.continuous_time)

        logits = self.projector(states[-1, :, 0])  # CLS token

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": states
        }

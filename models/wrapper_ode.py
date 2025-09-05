from torchdiffeq import odeint_adjoint, odeint

import torch
from torch import nn as nn

from typing import Optional
from .time_emb import TimeEmbedding



class ViT_ODEFunc(nn.Module):

    def __init__(self, resiudal_block: torch.nn.Module, hidden_dim=768, temporal_dimension:int=768, steps:int=12) -> None:
        super(ViT_ODEFunc, self).__init__()

        self.net = resiudal_block
        self.nfe = 0

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, t, x, output_hidden_states: bool = True):

        self.nfe += 1
        dx = x
        for layer in self.net.layer:
            hidden_state_dt = layer.layernorm_before(dx)
            g = layer.attention(hidden_state_dt, output_attentions=False)
            f = layer.intermidiate(hidden_state_dt)

        return g + f


class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = "rk4", rtol: float = 1e-3, atol: float = 1e-5):
        super().__init__()
        self.odefunc = odefunc
        self.solver = solver


    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x0: [B, N, D]
        t:  [T]
        returns: [T, B, N, D]
        """
        # odeint expects (t, batch, ...) ordering as it returns [T, *x_shape]
        states = odeint(self.odefunc, x0, t, method=self.solver, rtol=self.rtol, atol=self.atol)
        # states: [T, B, N, D]
        return states

    @property
    def nfe(self) -> int:
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value: int):
        self.odefunc.nfe = value

class ViTNeuralODE(nn.Module):

    def __init__(self, encoder, n_classes, hidden_dim=768, integration_time: int= 1, solver:str="euler", num_evaluation_steps: int=96):
        super().__init__()
        self.encoder = encoder.vit  # e.g. a ViT or CNN backbone returning .last_hidden_state [B, N, d]
        self.hidden_dim = hidden_dim
        self.num_queries = n_classes

        self.integration_time = integration_time
        self.num_evaluation_steps = num_evaluation_steps

        self.n_classes = n_classes
        self._act = nn.Softplus()
        self.solver = solver

        self.ode_func_wrapper = ViT_ODEFunc(resiudal_block=self.encoder.encoder)
        self.state_update = ODEBlock(odefunc=self.ode_func_wrapper, solver=solver)



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

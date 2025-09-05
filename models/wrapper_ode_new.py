from torchdiffeq import odeint_adjoint, odeint
from transformers import PreTrainedModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTAttention, ViTEmbeddings, ViTIntermediate, ViTOutput, ViTPooler
import torch
from torch import nn as nn

from typing import Optional


class IntegrationFunction(nn.Module):

    def __init__(self, resiudal_block: torch.nn.Module) -> None:
        super(IntegrationFunction, self).__init__()

        self.net = resiudal_block
        self.nfe = 0

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, t, x, output_hidden_states: bool = True):

        self.nfe += 1

        hidden_states = self.net(t, x)

        return hidden_states


class ODEBlock(nn.Module):
    def __init__(self, odefunc,
        solver: str = 'euler',
        step_size: float=(1/12),
        fixed_step_solver: bool = True,
        atol: Optional[float] = 1e-5,
        rtol: Optional[float] = 1e-3,
        ):
        super(ODEBlock, self).__init__()

        self.odefunc = odefunc
        self.atol = atol
        self.rtol = rtol
        self.solver = solver
        self.step_size = step_size
        self.default_integration_time = torch.tensor([0, 1]).float()

        self.ode_solver = odeint
        self.options = {"step_size": self.step_size}

    def forward(
        self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:

        if t is not None:
            self.integration_time = t
        else:
            self.integration_time = self.default_integration_time

        self.integration_time = self.integration_time.type_as(x)

        out = self.ode_solver(
            self.odefunc,
            x,
            self.integration_time,
            method=self.solver
        )

        return out


    @property
    def nfe(self) -> int:
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value: int):
        self.odefunc.nfe = value


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return torch.nn.functional.relu(hidden_states)

class VIT_ODE(nn.Module):
    def __init__(self, vit_config) -> None:
        super().__init__()

        self.temporal_embedding = nn.Parameter(torch.randn(1, 1, vit_config.hidden_size), requires_grad=True)
        self.step_adpater = nn.Parameter(torch.ones(1, 1, vit_config.hidden_size), requires_grad=True)

        self.attention = ViTAttention(config=vit_config)
        self.intermediate = ViTIntermediate(config=vit_config)
        self.output = ViTOutput(config=vit_config)
        self.layernorm_before = nn.LayerNorm(vit_config.hidden_size, eps=vit_config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(vit_config.hidden_size, eps=vit_config.layer_norm_eps)

    def forward(self, t: torch.Tensor, x: torch.Tensor):

        dt = (self.temporal_embedding * t)
        zt = (x * dt)
        x1 = self.layernorm_before(zt)
        g = self.attention(x1, head_mask=None)[0]  # G(x)

        x2 = self.layernorm_before(x1)
        f = self.intermediate(x2)               # F(x)
        f = self.output(f)       # keep raw F(x), not residual

        return self.layernorm_after(f + g)

class NeuralODEIntrepretation(nn.Module):

    def __init__(self,
        vit_config: nn.Module,
        n_classes: int,
        integration_time: int= 12,
        num_step_evaluations: int=50,
        solver:str="euler"):

        super().__init__()

        self.vit_embeddings = ViTEmbeddings(vit_config)
        self.hidden_size = vit_config.hidden_size
        self.pooler= ViTPooler(vit_config)
        self.n_classes = n_classes
        self.integration_time = integration_time
        self.num_step_evaluations = num_step_evaluations
        self.solver = solver

        assert (self.num_step_evaluations % self.integration_time) == 0, "The number of control points is not divisible by the number"

        self.vit = VIT_ODE(vit_config)
        self._ode_func_wrapper = IntegrationFunction(resiudal_block=self.vit)

        self.state_update = ODEBlock(odefunc=self._ode_func_wrapper, solver=solver)

        self.classifier = nn.Linear(self.hidden_size, n_classes)

        self.time = torch.linspace(0., self.integration_time, self.num_step_evaluations).to(self.device)  # shape [12]

    def _load_state_dict(self):
        final_dict = {}
        for module in [self.vit_embeddings, self.pooler, self.vit, self.classifier]:
            final_dict.update(module.state_dict())

        return final_dict


    @property
    def step_size(self) -> float:
        return self._step_size

    @step_size.setter
    def step_size(self, value: float):
        self._step_size = value

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self,
        pixel_values,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True):

        hidden_state = self.vit_embeddings(pixel_values)  # [B, N, D]

        states = self.state_update(x=hidden_state, t=self.time)

        pooler_state = states[-1]

        pooler_output = self.pooler(pooler_state)

        logits = self.classifier(pooler_output)  # CLS token
        loss = None

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": states
        }

if __name__ == "__main__":
    from transformers import AutoConfig
    from safetensors.torch import load_file

    file_path = "checkpoints/Vit_CIFAR100_first_train_reduced.pt/model.safetensors"
    loaded = load_file(file_path)
    config = AutoConfig.from_pretrained("checkpoints/Vit_CIFAR100_first_train_reduced.pt")

    model = NeuralODEIntrepretation(vit_config=config, n_classes=100, num_step_evaluations=120, integration_time=12) #("checkpoints/Vit_CIFAR100_first_train_reduced.pt")
    import pdb; pdb.set_trace()
    model.to("cuda")
    pixel_values = torch.randn(16, 3, 224, 224).cuda()
    labels = torch.randint(0, 100, (16,100)).cuda()
    output = model(pixel_values, labels)

    #get_model_complexity_info(model, (3, 224, 224))

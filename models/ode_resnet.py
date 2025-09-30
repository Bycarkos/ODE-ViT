import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint



# ----- Drift function -----
class ODEFunc(nn.Module):
    """Defines the dynamics dx/dt = f(x, t)."""
    def __init__(self, in_channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out   # derivative dx/dt




class ODEResNet(nn.Module):
    def __init__(self, num_classes=100,
        channels=[64],
        depth_emulations = [1],
        num_eval_steps = 24,
        solver = "euler"):
        super().__init__()

        assert len(depth_emulations) == len(channels), "The number of depth_emulations must match the number of channels"

        self.solver = solver
        self.stages = nn.ModuleList([])
        #self.upstages = nn.ModuleList([])
        
        self.t_grid = torch.linspace(0.0, 1., num_eval_steps)
        self.init_conv = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.depth_emulations = depth_emulations
        
        for idx, chan in enumerate(channels):
            self.stages.append(ODEFunc(chan))

        #for ups in range(1, len(channels)):
        #    self.upstages.append(nn.Conv2d(channels[ups-1], channels[ups], kernel_size=1, stride=1, padding=1, bias=False))

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(channels[-1], num_classes)
        self.act = torch.nn.GELU()
    def forward(self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False ):

        x = self.act(self.init_conv(pixel_values))

        all_states = []
        for idx, stage in enumerate(self.stages):
            intermidiate_states = odeint(stage, x, self.t_grid, method=self.solver)
            x = self.act(intermidiate_states[-1]) # * self.depth_emulations[idx]
            #if idx < len(self.upstages):
            #    x = self.upstages[idx](x).relu()

        x = self.global_pool(x).view(x.size(0), -1)
        logits = self.head(x)


        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = None

        out = {"logits": logits, "loss": loss, }

        if output_hidden_states:
            out["states"] = all_states


        return out

# -----------------------
# Minimal usage example
# -----------------------

if __name__ == "__main__":
    model = ODEResNet(num_classes=100, solver="rk4")
    x = torch.randn(8, 3, 32, 32)
    logits = model(x)["logits"]
    print(logits.shape)  # [8, 100]

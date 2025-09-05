from h11 import Data
import torch
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

class LotkaVolterra(torch.nn.Module):
    """ 
     The Lotka-Volterra equations are a pair of first-order, non-linear, differential equations
     describing the dynamics of two species interacting in a predator-prey relationship.
    """
    def __init__(self,
                 alpha: float = 1.5, # The alpha parameter of the Lotka-Volterra system
                 beta: float = 1.0, # The beta parameter of the Lotka-Volterra system
                 delta: float = 3.0, # The delta parameter of the Lotka-Volterra system
                 gamma: float = 1.0 # The gamma parameter of the Lotka-Volterra system
                 ) -> None:
        super().__init__()
        self.model_params = torch.nn.Parameter(torch.tensor([alpha, beta, delta, gamma]))
        
        
    def forward(self, t, state):
        x = state[...,0]      #variables are part of vector array u 
        y = state[...,1]
        sol = torch.zeros_like(state)
        
        #coefficients are part of tensor model_params
        alpha, beta, delta, gamma = self.model_params    
        sol[...,0] = alpha*x - beta*x*y
        sol[...,1] = -delta*y + gamma*x*y
        return sol
    
    def __repr__(self):
        return f" alpha: {self.model_params[0].item()}, \
            beta: {self.model_params[1].item()}, \
                delta: {self.model_params[2].item()}, \
                    gamma: {self.model_params[3].item()}"
    

def generate_data(sample_trajectories: int= 1000, traj_resolution: int = 384):

    lv_model = LotkaVolterra()
    ts = torch.linspace(0,30.0,traj_resolution) 
    initial_conditions = torch.tensor([[3,3]]) + 0.50*torch.randn((sample_trajectories,2))

    traj = odeint(lv_model, initial_conditions, ts, method='dopri5').detach()

    return traj


class LVtrajDataset(Dataset):
    def __init__(self, trajectories, input_len=96, pred_len=96):
        super().__init__()
        self.traj = trajectories  # (N, T, 2)
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        self.noise_std = 0.01
        self.noise_prob = 0.5
        N, T, C = trajectories.shape
        # number of windows per trajectory
        self.indices = []
        for n in range(N):
            for start in range(0, T - self.total_len, self.total_len):
                # stride = half window (overlapping windows)
                self.indices.append((n, start))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        n, start = self.indices[idx]
        seq = self.traj[n, start:start+self.total_len]  # (L+S, C)
        # Add Gaussian noise on-the-fly
        if torch.rand(1).item() < self.noise_prob:
            seq += self.noise_std * torch.randn_like(seq)

        x = seq[:self.input_len]   # input
        y = seq[(self.input_len - 1):]   # target
        return x.float(), y.float(), seq
    
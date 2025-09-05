import torch
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint



class Lorenz(torch.nn.Module):
    """
    Define the Lorenz system as a PyTorch module.
    """
    def __init__(self,
                 sigma: float =10.0, # The sigma parameter of the Lorenz system
                 rho: float=28.0, # The rho parameter of the Lorenz system
                beta: float=8.0/3, # The beta parameter of the Lorenz system
                ):
        super().__init__()
        self.model_params = torch.nn.Parameter(torch.tensor([sigma, rho, beta]))


    def forward(self, t, state):
        x = state[...,0]      #variables are part of vector array u
        y = state[...,1]
        z = state[...,2]
        sol = torch.zeros_like(state)

        sigma, rho, beta = self.model_params    #coefficients are part of vector array p
        sol[...,0] = sigma*(y-x)
        sol[...,1] = x*(rho-z) - y
        sol[...,2] = x*y - beta*z
        return sol

    def __repr__(self):
        return f" sigma: {self.model_params[0].item()}, \
            rho: {self.model_params[1].item()}, \
                beta: {self.model_params[2].item()}"


class LorenzTrajDataset(Dataset):
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
            for start in range(0, T - self.total_len, self.total_len // 2):
                # stride = half window (overlapping windows)
                self.indices.append((n, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, start = self.indices[idx]
        seq = self.traj[n, start:start+self.total_len]  # (L+S, C)
        # Add Gaussian noise on-the-fly
        #if torch.rand(1).item() < self.noise_prob:
        #    seq += self.noise_std * torch.randn_like(seq)

        x = seq[:self.input_len]   # input
        y = seq[self.input_len-1:]   # target
        return x.float(), y.float(), seq


def generate_data(sample_trajectories: int= 1000, traj_resolution: int = 384):

    lorenz_model = Lorenz()
    ts = torch.linspace(0,50.0,traj_resolution)
    initial_conditions = torch.tensor([[1.0,0.0,0.0]]) + 0.10*torch.randn((sample_trajectories,3))
    sol = odeint(lorenz_model, initial_conditions, ts, method='dopri5').detach()


    return sol

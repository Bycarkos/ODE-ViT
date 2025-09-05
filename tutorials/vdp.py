import torch
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint


class VDP(torch.nn.Module):
    """
    Define the Van der Pol oscillator as a PyTorch module.
    """
    def __init__(self,
                 mu: float, # Stiffness parameter of the VDP oscillator
                 ):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(mu)) # make mu a learnable parameter

    def forward(self,
                t: float, # time index
                state: torch.TensorType, # state of the system first dimension is the batch size
                ) -> torch.Tensor: # return the derivative of the state
        """
            Define the right hand side of the VDP oscillator.
        """
        x = state[..., 0] # first dimension is the batch size
        y = state[..., 1]
        dX = self.mu*(x-1/3*x**3 - y)
        dY = 1/self.mu*x
        # trick to make sure our return value has the same shape as the input
        dfunc = torch.zeros_like(state)
        dfunc[..., 0] = dX
        dfunc[..., 1] = dY
        return dfunc

    def __repr__(self):
        """Print the parameters of the model."""
        return f" mu: {self.mu.item()}"

def generate_data(sample_trajectories: int= 1000, traj_resolution: int = 384):

    vdp_model = VDP(mu=0.3)

    ts = torch.linspace(0, 30, traj_resolution)

    initial_conditions = torch.tensor([0.01, 0.01]) + 0.2*torch.randn((sample_trajectories,2))

    traj = odeint(vdp_model, initial_conditions, ts, method='dopri5').detach()

    return traj


# ---- PyTorch dataset ----
class VDPTrajDataset(Dataset):
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
        if torch.rand(1).item() < self.noise_prob:
            seq += self.noise_std * torch.randn_like(seq)

        x = seq[:self.input_len]   # input
        y = seq[self.input_len-1:]   # target
        return x.float(), y.float(), seq


if __name__ == "__main__":
    # generate raw trajectories
    trajectories = generate_data(sample_trajectories=1000, traj_resolution=384)

    # wrap into dataset
    dataset = VDPTrajDataset(trajectories.permute(1, 0, 2), input_len=96, pred_len=96)

    # train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # check batch
    x, y, seq = next(iter(train_loader))

    print(x.shape, y.shape)  # (32, 96, 2), (32, 96, 2)

    import pdb; pdb.set_trace()

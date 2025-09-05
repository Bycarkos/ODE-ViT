from torch.utils.data import Dataset
import torch



class TrajectoryDataset(Dataset):
    """Holds trajectories of shape (num_traj, T+1, state_dim)
    returns individual trajectories as tensors
    """
    def __init__(self, trajectories, delay: int = 3, future_horizon: int = 2):
        # trajectories: numpy or torch, shape (N, T+1, m)

        self.traj = trajectories
        self.delay = delay + 1 ## This +1 is to get the correct window (t - k - > t) & (t - k + 1 -> t + 1)
        self.future_horizon = future_horizon

    def __len__(self):
        return self.traj.shape[0]

    def update_horizon(self, value):
        self.future_horizon = value

    def __getitem__(self, idx):
        traj = self.traj[idx]

        radom_state_point = torch.randint(0, traj.shape[0] - self.delay, (1,)).item()
        delay_embedding = traj[radom_state_point:radom_state_point + self.delay, :]
        true_horizon = traj[radom_state_point + self.delay : radom_state_point + self.delay + self.future_horizon , :]

        return traj, delay_embedding, true_horizon


    def collate_fn(self, batch):
        """Collate function to return a batch of trajectories and their embeddings."""
        trajs = [b[0].unsqueeze(0) for b in batch]
        delay_embeddings = [b[1].unsqueeze(0) for b in batch]
        next_delay_embeddings = [b[2].unsqueeze(0) for b in batch]

        trajs = torch.cat(trajs, dim=0)  # (batch_size, T+1, state_dim)
        delay_embeddings = torch.cat(delay_embeddings, dim=0)
        next_delay_embeddings = torch.cat(next_delay_embeddings, dim=0)

        return trajs, delay_embeddings, next_delay_embeddings

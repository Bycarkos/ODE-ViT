# type: ignore

"""Learning Koopman Invariant Subspace
Translated to PyTorch from the original Chainer implementation
Original: (c) Naoya Takeishi, 2017.
takeishi@ailab.t.u-tokyo.ac.jp
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import linalg


def ls_solution(g0, g1):
    """Get least-squares solution matrix for regression from rows of g0
    to rows of g1. Both g0 and g1 are torch tensors.
    """
    # Use pseudo-inverse directly - handles singular matrices automatically
    g0pinv = torch.pinverse(g0)
    K = torch.matmul(g0pinv, g1).transpose(0, 1)
    return K


def dmd(y0, y1, eps=1e-6):
    """Do DMD (Dynamic Mode Decomposition). Both y0 and y1 are numpy arrays."""
    Y0 = y0.T
    Y1 = y1.T
    U, S, Vh = linalg.svd(Y0, full_matrices=False)
    r = len(np.where(S >= eps)[0])
    U = U[:, :r]
    invS = np.diag(1.0 / S[:r])
    V = Vh.conj().T[:, :r]
    M = np.dot(np.dot(Y1, V), invS)
    A_til = np.dot(U.conj().T, M)
    lam, z_til, w_til = linalg.eig(A_til, left=True)
    w = np.dot(np.dot(M, w_til), np.diag(1.0 / lam)) + 1j * np.zeros(z_til.shape)
    z = np.dot(U, z_til) + 1j * np.zeros(z_til.shape)
    for i in range(w.shape[1]):
        z[:, i] = z[:, i] / np.dot(w[:, i].conj(), z[:, i])
    return lam, w, z


class KoopmanNetwork(nn.Module):

    def __init__(self, dim_y: int, dim_g_in: int, dim_g_out: int, delay: int) -> None:

        super(KoopmanNetwork, self).__init__()
        self.delay = delay

        self.phi = nn.Sequential(
            nn.BatchNorm1d(dim_y * delay), nn.Linear(dim_y * delay, dim_y), nn.PReLU()
        )

        n_h_round_observation = round((dim_g_in + dim_g_out) * 0.5)
        self.g = nn.Sequential(
            nn.Linear(dim_g_in, n_h_round_observation),
            nn.PReLU(),
            nn.BatchNorm1d(n_h_round_observation),
            nn.Linear(n_h_round_observation, dim_g_out),
        )

        n_h_round_reconstructor = round((dim_y + dim_g_in) * 0.5)

        self.reconstructor = nn.Sequential(
            nn.Linear(dim_g_out, n_h_round_reconstructor),
            nn.PReLU(),
            #nn.BatchNorm1d(n_h_round_reconstructor),
            nn.Linear(n_h_round_reconstructor, dim_y),
        )

        self.act = nn.PReLU()
    def forward(self, inputs: torch.Tensor):
        delay_embedding = self.phi(inputs)

        g = self.observate(delay_embedding)



        h = self.reconstructor(self.act(g))

        return g, h

    def observate(self, y):

        g_x = self.g(y)/(y.shape[-1]**0.5)

        return g_x #F.normalize(g_x, p=2, dim=-1)

    def reconstruct(self, gx):
        return self.reconstructor(gx)


def KoopmanLoss(y0, y1, g0, g1, h0, h1, alpha=1, dim_y=2, epoch: int = 0):

    # Compute the Koopman operator (linear transformation from g0 to g1)
    K = ls_solution(g0, g1)

    # Calculate losses
    loss1 = F.mse_loss(torch.matmul(g0, K.t()), g1)

    # We're assuming y0 and y1 have shape [batch_size, sequence_length, feature_dim]
    # and we want the last step of each sequence
    y0_last = y0[:, -dim_y:]
    y1_last = y1[:, -dim_y:]

    loss2 = F.mse_loss(h0, y0_last)
    loss3 = F.mse_loss(h1, y1_last)

    loss4 = (g0.norm(dim=1, p=2) - 1).relu().mean()
    loss5 = (g1.norm(dim=1, p=2) - 1).relu().mean()

    loss_solution = (loss2 + loss3) * 0.5
    loss_stability = (loss4 + loss5) *0.5
    loss = loss1 + (alpha * loss_solution) + loss_stability

    # For reporting
    metrics = {
        "loss": loss.item(),
        "loss_kpm": loss1.item(),
        "loss_rec": loss_solution.item(),
        "loss_stability": loss_stability.item()
    }

    return loss, metrics


if __name__ == "__main__":
    pass

    # data = np.random.randn(1000, 10)  # Example data: 1000 time points, 10 dimensions
    # dataset = DelayPairDataset([data], dim_delay=5)
    # train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    #
    # dim_y = 10  # Dimension of original data
    # delay = 5   # Number of delay embeddings
    # dim_emb = 50  # Dimension of embedding
    # dim_g = 30   # Dimension of observable
    #
    # phi = Embedder(dim_y, delay, dim_emb)
    # net = Network(dim_emb, dim_g, dim_y)
    # optimizer = torch.optim.Adam(list(phi.parameters()) + list(net.parameters()), lr=0.001)
    #
    # trainer = Trainer(phi, net, optimizer, train_loader)
    # trainer.train(epochs=100, save_path='model_checkpoint.pt')

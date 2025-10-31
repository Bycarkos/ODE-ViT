from typing import Any
from click import Option
import torch  # type: ignore
import os
import pickle

from numpy.typing import ArrayLike
from typing import *

from PIL import Image
from torch_pca import PCA
import matplotlib.pyplot as plt
import numpy as np
import imageio
import io
import scipy.linalg

import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

def _log_metrics(
    wandb_logger, wandb_table, metrics_iter, optimizer, epoch, batch_idx,
    dataloader_len, log_every, data, text, decoded_text, tokens
):
    """Helper function to handle metric logging."""
    if not wandb_logger:
        return

    # Add samples to table
    if wandb_table is not None:
        for idx in range(min(tokens.shape[0], len(text), len(decoded_text))):
            if "raw_images" in data:
                image = wandb.Image(data["raw_images"][idx])
                wandb_table.add_data(image, text[idx], decoded_text[idx])

    # Log iteration metrics
    log_metrics = {
        f"train/iteration_{key}": value / log_every
        for key, value in metrics_iter.items()
    }
    log_metrics["train/lr"] = optimizer.param_groups[0]["lr"]

    #step = (epoch + 1) * (dataloader_len + batch_idx)
    wandb_logger.log(log_metrics)

    if wandb_table is not None:
        wandb_logger.log({"train/iter_table": wandb_table})


def create_image_grid(images, title="Trajectory Steps"):

    N = images.shape[0]
    cols = min(N, 7)
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < N:
            img = upscale_image(images[i])
            if img.max() > 1:
                img = img / 255.0
            ax.imshow(img)
            ax.set_title(f"Step {i}", fontsize=8)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def upscale_image(image: np.ndarray, scale: int = 16, method=Image.BICUBIC):
    """Upscale a (24, 24, 3) image to higher resolution."""
    pil_img = (
        Image.fromarray((image * 255).astype(np.uint8))
        if image.max() <= 1
        else Image.fromarray(image.astype(np.uint8))
    )
    new_size = (image.shape[1] * scale, image.shape[0] * scale)
    upscaled_img = pil_img.resize(new_size, resample=method)
    return np.array(upscaled_img)


def create_animated_gif(images, duration=0.5, figsize=(4, 4), dpi=200):

    frames = []

    for i, img in enumerate(images):
        img = upscale_image(img)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(img if img.max() <= 1 else img / 255.0)
        ax.axis("off")
        ax.set_title(f"Step {i}", fontsize=10)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frame = imageio.v2.imread(buf)
        frames.append(frame)
        plt.close(fig)

    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, frames, format="gif", duration=duration)
    gif_buf.seek(0)
    return gif_buf


def plot_3d_trajectory(points, title="Trajectory", elev=30, azim=135):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.plot(x, y, z, color="blue", linewidth=2)
    ax.scatter(x, y, z, c="red", s=50)
    for i, (xi, yi, zi) in enumerate(points):
        ax.text(xi, yi, zi, str(i), fontsize=8)

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    return fig


def normalize(traj):
    """Min-max normalize a trajectory to [0, 1] range along each axis."""
    min_vals = traj.min(axis=0, keepdims=True)
    max_vals = traj.max(axis=0, keepdims=True)
    return (traj - min_vals) / (max_vals - min_vals + 1e-8)


def plot_merged_3d_trajectories(
    teacher_points, student_points, title="Teacher vs Student", elev=30, azim=135
):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    teacher_points = normalize(teacher_points)
    student_points = normalize(student_points)

    # Teacher
    x_t, y_t, z_t = teacher_points[:, 0], teacher_points[:, 1], teacher_points[:, 2]
    ax.plot(x_t, y_t, z_t, color="blue", linewidth=2, label="Teacher")
    ax.scatter(x_t, y_t, z_t, c="blue", s=50)
    for i, (xi, yi, zi) in enumerate(teacher_points):
        ax.text(xi, yi, zi, f"T{i}", fontsize=8, color="blue")

    # Student
    x_s, y_s, z_s = student_points[:, 0], student_points[:, 1], student_points[:, 2]
    ax.plot(x_s, y_s, z_s, color="orange", linewidth=2, label="Student")
    ax.scatter(x_s, y_s, z_s, c="orange", s=50)
    for i, (xi, yi, zi) in enumerate(student_points):
        ax.text(xi, yi, zi, f"S{i}", fontsize=8, color="orange")

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    return fig


def update_and_save_model_pt(
    previous_metric,
    actual_metric,
    model,
    optimizer,
    lr_scheduler,
    checkpoint_path: str,
    compare: str = "<",
):
    saver = {"optimizer": optimizer.state_dict(), "state_dict": model.state_dict(), "lr_scheduler": lr_scheduler}

    if eval(str(actual_metric)+compare+str(previous_metric)):
        torch.save(saver, checkpoint_path)
        return True, actual_metric


    return False, previous_metric

def load_model_pt(student_model, optimizer, checkpoint_path: str, device):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)


    student_model.load_state_dict(checkpoint["state_dict"])
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except:
        print("Optimizer state could not be loaded.")
        
    lr = checkpoint["lr_scheduler"]


    return student_model, optimizer, lr

def update_and_save_model(
    previous_metric,
    actual_metric,
    model,
    checkpoint_path: str,
    processor: Optional = None,
    compare: str = "<",
):

    if compare == "<":

        if actual_metric < previous_metric:
            previous_metric = actual_metric
            model.save_pretrained(checkpoint_path, push_to_hub=False, from_pt=True)
            if processor:
                processor.save_pretrained(checkpoint_path)

            return True, previous_metric

    elif compare == ">":

        if actual_metric > previous_metric:
            previous_metric = actual_metric
            model.save_pretrained(checkpoint_path, push_to_hub=False, from_pt=True)
            if processor:
                processor.save_pretrained(checkpoint_path)

            return True, previous_metric

    return False, previous_metric


def write_pickle(info: Any, filepath: str) -> None:
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)

    with open(filepath, "wb") as file:
        pickle.dump(info, file)


def read_pickle(filepath: str) -> Any:

    with open(filepath, "rb") as file:
        obj = pickle.load(file=file)

    return obj


def project_onto_subspace(A: torch.Tensor, Vt: torch.Tensor, k: int = 1):
    """
    matmul(A, V[:, :k]) projects data to the first k principal components

    """
    return torch.matmul(A, Vt[:, :k])


def perform_pca_lowrank(A: torch.Tensor, n_eigenvectors: int = 6, center: bool = True, reg_eps: float = 1e-5):
    """
    Perform PCA on the input tensor A and return the low-rank approximation.

        - U is m x q matrix
        - S is q-vector
        - V is n x q matrix
    """
    # Centering
    if center:
        A = A - A.mean(dim=0, keepdim=True)

    # Tikhonov regularization: add small noise to the covariance matrix
    # Instead of modifying A directly, we regularize through covariance
    cov = A.T @ A
    cov += reg_eps * torch.eye(cov.size(0), device=A.device, dtype=A.dtype)

    # Do SVD on the regularized covariance
    U_cov, S, Vt = torch.svd(cov)

    Vt = Vt[:, :n_eigenvectors]  # shape (n_eigenvectors, features)

    return U_cov, S[:n_eigenvectors], Vt


def align_trajectories(X, Y):
    # Align Y to X using orthogonal Procrustes
    R, _ = scipy.linalg.orthogonal_procrustes(Y, X)
    Y_aligned = Y @ R
    return Y_aligned

def perform_PCA_return_components(A, n_components: int = 10):

    A = A / A.norm(dim=-1, keepdim=True)

    pca_model = PCA(n_components=n_components, svd_solver="full")

    pca_model.fit(A)

    print("Explaining Capability: ", pca_model.explained_variance_ratio_)

    return pca_model.components_, pca_model.explained_variance_ratio_

def predict_trajectory(gx:torch.Tensor, zh : torch.Tensor, lamb: torch.Tensor, wh: torch.Tensor, horizon: int):
    horizon_steps = [gx]

    g_prev = gx.clone()
    # precompute W * lambda
    K = w @ (torch.diag(lam).to(zh.conj().dtype) @ zh.conj().T)
    for i in range(horizon):
        g_next = g_prev @ K.T
        horizon_steps.append(g_next)
        g_prev = g_next.clone()

    return horizon_steps

def analyse_PCA(
    A_full: torch.Tensor,
    n_components: int = 10,
    sample_size: list = [1000, 2000, 3000, 4000, 5000, 10000, 15000],
):
    angles = []
    explained_variances = []
    components_full, _ = perform_PCA_return_components(
        A_full, n_components=n_components
    )

    for n in sample_size:

        idx = torch.randperm(A_full.shape[0])[:n]
        X_subset = A_full[idx]

        components_subset, var_ratio_subset = perform_PCA_return_components(
            X_subset, n_components=n_components
        )

        # Cosine similarity between first component of full and subset PCA
        angle = torch.nn.functional.cosine_similarity(
            (components_full[0]).float(),
            (components_subset[0]).float(),
            dim=0,
        ).item()

        angles.append(angle)
        explained_variances.append(var_ratio_subset[None, :])

    explained_variances = torch.cat(explained_variances, dim=0)
    # === Cosine Similarity Plot ===
    plt.figure(figsize=(7, 5))
    plt.plot(
        sample_size, angles, marker="o", linestyle="-", color="#1f77b4", linewidth=2
    )
    plt.title("Stability of the Top PCA Component")
    plt.xlabel("Number of Samples Used for PCA")
    plt.ylabel("Cosine Similarity with Full Dataset PCA")
    plt.ylim(0, 1.05)
    plt.xticks(sample_size)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("pca_plots/pca_cosine_similarity_non_normalized.png")

    # === Explained Variance Plot ===
    plt.figure(figsize=(8, 6))

    for i in range(n_components):
        plt.plot(
            sample_size,
            explained_variances[:, i].numpy(),
            label=f"PC {i+1}",
            marker="o",
            linewidth=2,
            alpha=0.8,
        )

    plt.title("Explained Variance Ratio of PCA Components")
    plt.xlabel("Number of Samples Used for PCA")
    plt.ylabel("Explained Variance Ratio")
    plt.legend(title="Principal Component")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(sample_size)
    plt.tight_layout()
    plt.savefig("pca_plots/pca_explained_variance_non_normalized.png")




class GenerationVocab:
    #START_TAG = "<START>"
    #STOP_TAG = "<STOP>"
    BLANK = "_"
    PAD_TAG = "<PAD>"
    UNKNOWN_TAG = "<UNK>"

    AUX_TAGS = [BLANK, PAD_TAG, UNKNOWN_TAG]

    def __init__(self, VOCAB:List) -> None:
        self.tokens = self.AUX_TAGS + VOCAB

        self.token2index = {tok: ii for ii, tok in enumerate(self.tokens)}
        self.index2token = {ii: tok for tok, ii in self.token2index.items()}

    @property
    def pad_token_id(self):
        return self.token2index[self.PAD_TAG]

    @property
    def blank_token_id(self):
        return self.token2index[self.BLANK]

    def __len__(self) -> int:
        return len(self.token2index)

    def __call__(self, line: List[str], add_special_tokens:bool = False):
        return {"input_ids":self.tokenise(line=line)}

    def tokenise(self, line: List[str]) -> List[int]:

        return [
            self.token2index[tok]
            if tok in self.token2index
            else self.token2index[self.UNKNOWN_TAG]
            for tok in line
        ]

    def pad(self, tokenised: List[int], size: int) -> ArrayLike:
        padded = np.full((size,), self.token2index[self.PAD_TAG])
        max_index = min(len(tokenised), size - 2)
        padded[1 : max_index + 1] = tokenised[:max_index]
        padded[0] = self.token2index[self.START_TAG]
        padded[max_index + 1] = self.token2index[self.STOP_TAG]
        return padded

    def prepare(self, line: List[str], size: int) -> ArrayLike:
        return self.pad(self.tokenise(line), size)

    def unpad(self, padded: ArrayLike) -> List[int]:
        output: List[int] = []
        for tok in padded:
            if tok not in {
                self.token2index[self.START_TAG],
                self.token2index[self.PAD_TAG],
            }:
                if tok == self.token2index[self.STOP_TAG]:
                    return output
                output.append(tok)

        return output


    def decode(self, tokenised: List[int]) -> List[str]:
        return "".join(self.index2token.get(ind, self.UNKNOWN_TAG) for ind in tokenised)

    def unprepare(self, padded: ArrayLike) -> List[str]:
        return self.detokenise(self.unpad(padded))

if __name__ == "__main__":

    B, N = 16, 3
    g0 = torch.randn(B, N) + 1j*np.random.randn(B, N)
    z = torch.randn(N, N) + 1j*np.random.randn(N, N)
    w = torch.randn(N, N) + 1j*np.random.randn(N, N)
    lam = torch.exp(2j * torch.pi * torch.rand(N))  # eigenvalues on the unit circle

    g_preds = predict_trajectory(g0, z, lam, w, horizon=20)
    print(g_preds[0].shape)

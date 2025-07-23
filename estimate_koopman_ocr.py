import torch
import tqdm
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm

from datasets import Collator
import lkis as LKIS
import utils
from models.wrapper_models import EncoderWrapperLearnedQueries

from transformers import VisionEncoderDecoderModel, AutoProcessor  # type: ignore
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra import initialize, compose


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_dmd_results(dmd_results_path, save_dir="./dmd_analysis_plots"):
    """
    Load and analyze DMD eigenvalues from saved .npy files and generate enhanced plots.

    Parameters:
        dmd_results_path (str): Path to folder containing .npy files with (ww, np.diag(lamb), zh).
        save_dir (str): Path to save the generated plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    eigenvalues_dict = {}

    # Load eigenvalues from all files
    for fname in sorted(os.listdir(dmd_results_path)):
        if fname.endswith(".npy"):
            try:
                ww, lamb_diag, zh = np.load(os.path.join(dmd_results_path, fname), allow_pickle=True)
                lamb = np.diag(lamb_diag)
                eigenvalues_dict[fname] = lamb
            except Exception as e:
                print(f"[Warning] Skipping {fname}: {e}")

    if not eigenvalues_dict:
        print("❌ No valid DMD files found.")
        return

    # Color map
    colors = cm.tab10(np.linspace(0, 1, len(eigenvalues_dict)))

    # -------- Plot 1: Complex Plane --------
    fig, ax = plt.subplots(figsize=(10, 10))
    unit_circle = Circle((0, 0), radius=1.0, color='gray', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(unit_circle)

    for i, (fname, eigvals) in enumerate(eigenvalues_dict.items()):
        ax.scatter(
            eigvals.real, eigvals.imag, s=40, alpha=0.8,
            color=colors[i], edgecolor='black', label=fname.replace('.npy','')
        )

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title("Complex Plane of DMD Eigenvalues\n(With Unit Circle for Stability Reference)", fontsize=15)
    ax.set_xlabel("Real Part of Eigenvalue", fontsize=13)
    ax.set_ylabel("Imaginary Part of Eigenvalue", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dmd_eigenvalues_complex_plane.png"), dpi=300)
    plt.close()

    # -------- Plot 2: Magnitude Distribution --------
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (fname, eigvals) in enumerate(eigenvalues_dict.items()):
        magnitudes = abs(eigvals)
        ax.scatter(np.arange(len(magnitudes)),magnitudes, alpha=0.8, color=colors[i], label=fname.replace('.npy',''))

    ax.axvline(1.0, color='red', linestyle='--', label="Stability Boundary (|λ|=1)")
    ax.set_title("Distribution of DMD Eigenvalue Magnitudes", fontsize=15)
    ax.set_xlabel("Eigenvalue Magnitude |λ|", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dmd_eigenvalue_magnitude_histogram.png"), dpi=300)
    plt.close()

    print(f"✅ Saved enhanced plots to: '{save_dir}'")

    # Save summary statistics
    summary_stats = ""
    for label, eigs in eigenvalues_dict.items():
        magnitudes = np.abs(eigs)
        spectral_radius = np.max(magnitudes)
        mean_magnitude = np.mean(magnitudes)
        num_unstable = np.sum(magnitudes > 1)
        num_real = np.sum(np.isclose(eigs.imag, 0.0))

        summary_stats += (
            f"File: {label}\n"
            f"  - Spectral Radius: {spectral_radius:.4f}\n"
            f"  - Mean Magnitude: {mean_magnitude:.4f}\n"
            f"  - # Unstable Eigenvalues (|λ| > 1): {num_unstable}\n"
            f"  - # Purely Real Eigenvalues: {num_real}\n\n"
        )

    summary_path = os.path.join(save_dir, "summary_stats.txt")
    with open(summary_path, "w") as f:
        f.write(summary_stats)


@torch.no_grad()
def get_dmd(
    k_model,
    encoder_model,
    dloader,
    full_grid:bool=True
):

    acumulative_g0, acumulative_g1 = [], []

    for batch_idx, data in tqdm.tqdm(
        enumerate(dloader),
        desc="Extracting eigenfunctions",
        total=len(dloader),
        leave=True,
        position=1
    ):
        inputs = data["pixel_values"].to(device)

        features = encoder_model(**inputs, output_hidden_states=True)["hidden_states"]
        features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)[:, :, 0]
        if full_grid:
            features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)
        else:
            features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)[:, :, 0]

        k = k_model.delay
        for i in range(0, features.shape[1] - k + 1):
            if i + k + 1 <= features.shape[1]:
                y0 = features[:, i:i+k, :].reshape(-1, features.shape[-1] * k)
                y1 = features[:, i+1:i+k+1, :].reshape(-1, features.shape[-1] * k)

                g0, h0 = k_model(y0)
                g1, h1 = k_model(y1)

                acumulative_g0.append(g0.cpu())
                acumulative_g1.append(g1.cpu())

    Y0 = torch.cat(acumulative_g0, dim=0).numpy()
    Y1 = torch.cat(acumulative_g1, dim=0).numpy()

    lmb, w, z = LKIS.dmd(Y0, Y1)

    return lmb, w, z

@torch.no_grad()
def evaluate_K(k_model,
                encoder_model,
                dloader,
                K,
                full_grid:bool = False):

    ww, lamb, zh = K
    ww = torch.from_numpy(ww)
    lamb = torch.from_numpy(lamb)
    zh = torch.from_numpy(zh)

    K = (ww @ (lamb.to(zh.dtype) @ zh.conj().T)).to(device)
    K = K.real.float()
    batch_error = 0
    mae_batch_error = 0
    batch_accumulation = []
    mae_batch_accumulation = []
    for batch_idx, data in tqdm.tqdm(
        enumerate(dloader),
        desc="Extracting eigenfunctions",
        total=len(dloader),
        leave=True,
        position=1,
    ):
        inputs = data["pixel_values"].to(device)

        features = encoder_model(**inputs, output_hidden_states=True)["hidden_states"]
        if full_grid:
            features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)#[:, :, 0]
        else:
            features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)[:, :, 0]
        accumulation_error = 0
        mae_accumulation_error = 0
        k = k_model.delay
        for i in range(0, features.shape[1] - k + 1):
            if i + k + 1 <= features.shape[1]:
                y0 = features[:, i:i+k, :].reshape(-1, features.shape[-1] * k)
                y1 = features[:, i+1:i+k+1, :].reshape(-1, features.shape[-1] * k)

                g0, h0 = k_model(y0)
                g1, h1 = k_model(y1)

                accumulation_error += (torch.mean((g0@K.mT - g1)**2))
                mae_accumulation_error += (torch.mean(torch.abs(g0@K.mT - g1)))

        batch_error += (accumulation_error/len(range(1, features.shape[1] - 1)))
        mae_batch_error += (mae_accumulation_error/len(range(1, features.shape[1] - 1)))
        batch_accumulation.append(accumulation_error.item())
        mae_batch_accumulation.append(mae_accumulation_error.item())


    print("\n")
    print("MSE in the prediction=", batch_error.item()/len(dloader))
    print("MAE in the prediction=", mae_batch_error.item()/len(dloader))

    print("RMSE", np.sqrt(batch_error.item()/len(dloader)))
    print("Standard Deviation Error in the prediction=", np.std(np.array(batch_accumulation)))
    print("Standard Deviation MA Error in the prediction=", np.std(np.array(mae_batch_accumulation)))



def main(cfg: DictConfig):

    #analyze_dmd_results(cfg.setup.save_path)
    #exit()
    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    checkpoint_path = cfg.models.teacher.checkpoint_path
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    print("loading teacher model")
    teacher_model_encoder_model = VisionEncoderDecoderModel.from_pretrained(
        checkpoint_path
    ).encoder.to(device)
    teacher_model = EncoderWrapperLearnedQueries(encoder=teacher_model_encoder_model, n_classes=170)
    teacher_model.load_state_dict(torch.load("checkpoints/TrOCR_Esposalles_CTC_With_Q_2.pt", weights_only=True))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    teacher_model.eval()

    ## Remove this if not CTC
    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "?", "\u00e0", "\u00e1", "\u00e2", "\u00e3", "\u00e4", "\u00e5", "\u0101", "\u00e6", "\u00e7", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "\u00ec", "\u00ed", "\u00ee", "\u00ef", "\u00f0", "\u00f1", "\u00f2", "\u00f3", "\u00f4", "\u00f5", "\u00f6", "\u014d", "\u00f8", "\u00f9", "\u00fa", "\u00fb", "\u00fc", "\u00fd", "\u00fe", "\u00ff", "\u00c0", "\u00c1", "\u00c2", "\u00c3", "\u00c4", "\u00c5", "\u00c6", "\u00c7", "\u00c8", "\u00c9", "\u00ca", "\u00cb", "\u00cc", "\u00cd", "\u00ce", "\u00cf", "\u00d0", "\u00d1", "\u00d2", "\u00d3", "\u00d4", "\u00d5", "\u00d6", "\u00d8", "\u00d9", "\u00da", "\u00db", "\u00dc", "\u00dd", "\u00de", "\u0178", "\u00ab", "\u00bb", "\u2014", "\u2019", "\u00b0", "\u2013", "\u0153", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", " "]
    tokenizer = utils.GenerationVocab(VOCAB=vocab) #AutoTokenizer.from_pretrained("google/canine-s")
    dataset = instantiate(cfg.data.dataset.Esposalles, transforms=transforms, tokenizer=tokenizer)

    dataset = instantiate(
        cfg.data.dataset.Esposalles,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    generator = torch.Generator().manual_seed(2)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=generator
    )

    collate_fn = Collator(
        processor=processor, pad_token=processor.tokenizer.pad_token_id
    )

    dloader = DataLoader(
        train_dataset, collate_fn=collate_fn.ocr_collate_fn, **cfg.data.collator
    )

    test_dloader = DataLoader(
        test_dataset, collate_fn=collate_fn.ocr_collate_fn, **cfg.data.collator
    )

    print("loading Koopman Network")
    student_model = LKIS.KoopmanNetwork(**cfg.models.student.inputs)
    student_model.load_state_dict(torch.load(cfg.models.student.checkpoint_path, weights_only=True))
    student_model = student_model.to(device)
    student_model.eval()
    os.makedirs(cfg.setup.save_path, exist_ok=True)
    save_path = os.path.join(cfg.setup.save_path, "K_" + cfg.setup.save_file)

    print("Getting the Koopman Operator")
    if os.path.exists(save_path+".npy"):
        ww, lamb, zh = np.load(save_path+".npy")
        print("Koopman Operator already exists")
    else:
        lamb, ww, zh = get_dmd(k_model=student_model, encoder_model=teacher_model, dloader=dloader)
        np.save(save_path, (ww, np.diag(lamb), zh))
        lamb = np.diag(lamb)

    K = [ww, lamb, zh]

    print("Evaluating Koopman Operator on test")
    evaluate_K(student_model, teacher_model, test_dloader, K)

    analyze_dmd_results(cfg.setup.save_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        required=True,
        help="Yaml Config file with the configuration for the models to run, for now the models to run are [defoDeTR, DeTR, CondDetr, Yolo]",
    )
    parser.add_argument(
        "-cp",
        "--config_path",
        required=True,
        help="path where the yaml configs are stored",
    )

    args = parser.parse_args()

    with initialize(version_base="1.3.2", config_path=args.config_path):
        cfg = compose(config_name=args.config_file)

    main(cfg=cfg)

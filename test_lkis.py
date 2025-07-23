import torch
import tqdm
import wandb
import os



import utils
import lkis as LKIS
from datasets import Collator

from torch_pca import PCA

from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

from models.wrapper_models import ResNetForKoopmanEstimation
from transformers import ViTForImageClassification, AutoImageProcessor , ViTImageProcessor

from omegaconf import DictConfig
from hydra import initialize, compose
import matplotlib.pyplot as plt
import numpy as np

# Set Matplotlib style
plt.style.use("seaborn-v0_8-deep")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 120,
    }
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(cfg: DictConfig):

    if cfg.log_wandb == True:

        wandb.login()
        wandb_logger = wandb.init(**cfg.setup.wandb
        )
    else:
        wandb_logger = None

    checkpoint_path = cfg.modeling.teacher.checkpoint_path

    print("loading teacher model")
    if cfg.data.dataset.name == "cifar100":
        train_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=True)
        validation_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=False)
    else:
        train_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/train")
        validation_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/val")


    checkpoint_path = cfg.modeling.teacher.checkpoint_path
    teacher_model = ResNetForKoopmanEstimation(checkpoint_path, out_channels=768, out_size=(4, 4)) if cfg.modeling.type == "resnet" else ViTForImageClassification.from_pretrained(cfg.modeling.teacher.checkpoint_path)
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True) if cfg.modeling.type == "resnet" else ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    collator = Collator(processor)
    teacher_model.to(device)
    train_dloader = DataLoader(train_dataset, **cfg.data.collator, collate_fn=collator.classification_collate_fn)

    test_dloader = DataLoader(
        validation_dataset,
        collate_fn=collator.classification_collate_fn,
        **cfg.data.collator,
    )

    print("loading Koopman Network")
    student_model = LKIS.KoopmanNetwork(**cfg.modeling.student.inputs)
    student_model.load_state_dict(torch.load(cfg.modeling.student.checkpoint_path, weights_only=True))
    student_model = student_model.to(device)
    student_model.eval()


    #Vt = utils.read_pickle(filepath="dmd_results/Vt_Esposalles.pkl").to(device)
    Vt_dictionary = {}
    embeddings_to_keep = []
    for batch_idx, data in tqdm.tqdm(
        enumerate(train_dloader),
        desc="Extracting the PCA Vt for computing the trajectories",
        leave=True,
        position=1,
        total=len(train_dloader),
    ):
        inputs: dict = data["pixel_values"].to(device)
        with torch.no_grad():
            features = teacher_model(**inputs, output_hidden_states=True)["hidden_states"]
            embeddings_to_keep.append(features.view(-1, 768))

        if (batch_idx * features.shape[0]) >= 2000:
            break

    embeddings_to_compute_pca = torch.cat(embeddings_to_keep, dim=0)
    U, S, Vt = utils.perform_pca_lowrank(embeddings_to_compute_pca, n_eigenvectors=3, center=True)

    if wandb_logger:
        table = wandb.Table(
            columns=[
                "Input Image",
                "Merged Trajectory",
                "Teacher Trajectory",
                "Student Trajectory",
                "Colored Patches Teacher",
                "Colored Patches Student",
                "Gif Teacher",
                "Gif Students",
            ]
        )

    student_model.eval()
    teacher_model.eval()
    patch_h = patch_w = 384 // 16

    for batch_idx, data in tqdm.tqdm(
        enumerate(test_dloader),
        desc="Training Procedure",
        leave=True,
        position=1,
        total=len(test_dloader),
    ):

        inputs: dict = data["pixel_values"].to(device)
        images = data["raw_images"]

        with torch.no_grad():

            features = teacher_model(**inputs, output_hidden_states=True)["hidden_states"]


            ## Trajectories
            features_cls_trajectory = torch.cat(
                [feat.unsqueeze(1) for feat in features], dim=1
            )[:, :, 0]
            all_features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)[
                :, :, 1:
            ]

            ## getting colors of images
            mask_images = []
            for i in range(all_features.size(1)):
                block = all_features[:, i].squeeze(0)
                block_observation = student_model.observate(block).cpu().numpy()
                image_result = utils.normalize(block_observation).reshape(
                    1, patch_h, patch_w, -1
                )

                mask_images.append(image_result)

            coloring_predictions = np.concatenate(mask_images, axis=0)
            coloring_original = (
                utils.project_onto_subspace(A=all_features, Vt=Vt, k=3)
                .reshape(-1, patch_h, patch_w, 3)
                .cpu()
                .numpy()
            )

            coloring_original = utils.normalize(coloring_original)

            image_grid_original = utils.create_image_grid(coloring_original)
            image_grid_predictions = utils.create_image_grid(coloring_predictions)

            gif_grid_original = utils.create_animated_gif(coloring_original)
            gif_grid_predictions = utils.create_animated_gif(coloring_predictions)

            ### get the trajectories and the observations
            b, seq, d = features_cls_trajectory.shape

            trajectory_from_vision_encoder = utils.project_onto_subspace(
                A=features_cls_trajectory, Vt=Vt, k=3
            )
            observations_from_model = student_model.observate(
                features_cls_trajectory.view(b * seq, d)
            )
            observations_from_model = observations_from_model.view(b, seq, 3)

            # Get only the first sample in the batch
            for i in range(observations_from_model.size(0)):
                traj_teacher = (
                    trajectory_from_vision_encoder[i].cpu().numpy()
                )  # (12, 3)
                traj_student = observations_from_model[i].cpu().numpy()  # (12, 3)
                fig_merged = utils.plot_merged_3d_trajectories(
                    traj_teacher, traj_student
                )
                fig_teacher = utils.plot_3d_trajectory(traj_teacher, title="Teacher")
                fig_student = utils.plot_3d_trajectory(traj_student, title="Student")

                table.add_data(
                    wandb.Image(images[i]),
                    wandb.Image(fig_merged),
                    wandb.Image(fig_teacher),
                    wandb.Image(fig_student),
                    wandb.Image(image_grid_original),
                    wandb.Image(image_grid_predictions),
                    wandb.Video(gif_grid_original),
                    wandb.Video(gif_grid_predictions),
                )

                plt.close(fig_teacher)
                plt.close(fig_student)
                plt.close(fig_merged)

        plt.close(image_grid_original)
        plt.close(image_grid_predictions)
        if batch_idx >= 9:
            break

    wandb.log({"Trajectory Comparison Table": table})


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

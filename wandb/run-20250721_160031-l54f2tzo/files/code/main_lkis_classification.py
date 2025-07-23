import torch
import tqdm
import wandb
import os


import utils
from train import train_lkis_task
from datasets import Collator
import lkis as LKIS

from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets import CIFAR100

from transformers import get_scheduler
from models.wrapper_models import ResNetForKoopmanEstimation
from transformers import ViTForImageClassification, AutoImageProcessor , ViTImageProcessor
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from hydra import initialize, compose
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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


@torch.no_grad()
def get_embeddings_by_step(
    model, dloader, number_of_blocks: int = 12, sample_number: int = 2000
):

    model.eval()
    # embeddings = {f"features_{i}": [] for i in range(number_of_blocks)}
    cls_embeddings = {f"cls_{i}": [] for i in range(number_of_blocks)}
    grouped_cls_embeddings = []
    # grouped_sequence_embeddings = []

    cumulative_sampled = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dloader),
        desc="Extracting Embeddings",
        total=sample_number // dloader.batch_size,
        leave=True,
        position=1,
    ):
        if cumulative_sampled >= sample_number:
            break
        inputs = data["pixel_values"].to(device)

        outputs = model(**inputs, output_hidden_states=True).hidden_states

        for block_level in range(number_of_blocks):
            # embeddings[f"features_{block_level}"].append(outputs[block_level][:, 1:, :].cpu())
            cls_embeddings[f"cls_{block_level}"].append(
                outputs[block_level][:, 0, :].cpu()
            )

            # grouped_sequence_embeddings.append(embeddings[f"features_{block_level}"][-1])
            grouped_cls_embeddings.append(cls_embeddings[f"cls_{block_level}"][-1])

        cumulative_sampled += outputs[0].shape[0]

    return dict(
        cls_embeddings=cls_embeddings,
        grouped_cls_embeddings=torch.cat(grouped_cls_embeddings, dim=0),
    )


def main(cfg: DictConfig):

    if cfg.log_wandb == True:
        wandb.login()
        wandb_logger = wandb.init(config=dict(cfg.setup.dict), **cfg.setup.wandb
        )
    else:
        wandb_logger = None

    if cfg.data.dataset.name == "cifar100":
        train_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=True)
        validation_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=False)
    else:
        train_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/train")
        validation_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/val")


    model = ResNetForKoopmanEstimation(cfg.modeling.teacher.checkpoint_path, out_size=(4, 4)) if cfg.modeling.type == "resnet" else ViTForImageClassification.from_pretrained(cfg.modeling.teacher.checkpoint_path)



    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True) if cfg.modeling.type == "resnet" else ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    collator = Collator(processor)

    train_dloader = DataLoader(train_dataset, **cfg.data.collator.train, collate_fn=collator.classification_collate_fn)

    val_dloader = DataLoader(
        validation_dataset,
        collate_fn=collator.classification_collate_fn,
        **cfg.data.collator.val,
    )

    model.to(device)

    student_model = LKIS.KoopmanNetwork(**cfg.modeling.student.inputs).to(device)
    dim_y = cfg.modeling.student.inputs.dim_y

    if cfg.modeling.type == "resnet":
        params = list(student_model.parameters()) + list(model.parameters())

    else:
        params = list(student_model.parameters())

    ## TODO Tenir en compte això
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    # Number of training steps
    num_epochs = cfg.setup.dict.epochs
    # Assuming you have a DataLoader called train_dataloader
    num_training_steps = num_epochs * len(train_dloader)

    # Create scheduler
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=1000,  # Number of warmup steps
        num_training_steps=num_training_steps,
    )

    optimal_loss = 1e20
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{cfg.modeling.student.checkpoint_name}.pt"
    criterion = LKIS.KoopmanLoss

    for epoch in tqdm.tqdm(
        range(1, cfg.setup.dict.epochs), desc="Training Procedure", position=0, leave=False
    ):
        _, metrics = train_lkis_task(
            teacher_model_encoder=model,
            student_model=student_model,
            optimizer=optimizer,
            dataloader=train_dloader,
            criterion=criterion,
            epoch=epoch,
            num_accumulation_steps=cfg.setup.dict.num_accumulation_steps,
            log_every=cfg.setup.dict.log_every,
            wandb=wandb_logger,
            scheduler=scheduler,
            K=dim_y,
            full_grid=cfg.setup.dict.full_grid
        )

        train_loss = metrics["loss"] / len(train_dloader)
        print(f"Loss Epoch: {epoch} Value: {train_loss}")

        updated, optimal_loss = utils.update_and_save_model_pt(
            previous_metric=optimal_loss,
            actual_metric=train_loss,
            model=student_model,
            checkpoint_path=checkpoint_name,
            compare="<",
        )
        if updated:
            print("Updated")

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

# type: ignore
## Dataset Things
from curses import use_default_colors
import torch.utils
import torch.utils.data
from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets import CIFAR100, CIFAR10
from datasets.collator import Collator
import utils


from train import train_classification_task
from test import test_classification_task

from transformers import (
    ViTForImageClassification,
    ResNetForImageClassification,
    ViTImageProcessor,
    AutoImageProcessor,
    ViTModel,
    Dinov2ForImageClassification,
    Dinov2WithRegistersForImageClassification,
)

## Common packages
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import torchvision.transforms.v2 as T

## Typing Packages

## Configuration Package
from omegaconf import DictConfig
from hydra import initialize, compose

## Experiment Tracking packages
import tqdm

## Common packages
import os
import wandb

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if cfg.log_wandb == True:
        wandb.login()
        wandb_logger = wandb.init(
            project=cfg.setup.wandb.project,
            group=cfg.setup.wandb.group,
            name=cfg.setup.wandb.name,
            settings=wandb.Settings(code_dir="./models/"),
        )
    else:
        wandb_logger = False

    if cfg.data.dataset.name == "cifar100":
        train_dataset = CIFAR100(
            root=cfg.data.dataset.dataset_path, download=False, train=True
        )
        validation_dataset = CIFAR100(
            root=cfg.data.dataset.dataset_path, download=False, train=False
        )
    elif cfg.data.dataset.name == "cifar10":
        train_dataset = CIFAR10(
            root=cfg.data.dataset.dataset_path, download=False, train=True
        )
        validation_dataset = CIFAR10(
            root=cfg.data.dataset.dataset_path, download=False, train=False
        )
    else:
        train_dataset = ImageFolder(root=cfg.data.dataset.dataset_path + "/train")
        validation_dataset = ImageFolder(root=cfg.data.dataset.dataset_path + "/val")

    if cfg.modeling.type == "resnet":
        model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            ignore_mismatched_sizes=True,
            num_labels=len(train_dataset.classes),
        )
    elif cfg.modeling.type == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(train_dataset.classes),
            num_hidden_layers=cfg.setup.dict.num_hidden_layers,
        )
    elif cfg.modeling.type == "dino":
        model = ViTForImageClassification.from_pretrained(
            "facebook/dino-vitb16",
            num_labels=len(train_dataset.classes),
            num_hidden_layers=cfg.setup.dict.num_hidden_layers,
        )

    elif cfg.modeling.type == "dinov2":
        model = Dinov2WithRegistersForImageClassification.from_pretrained(
            "facebook/dinov2-with-registers-base",
            num_labels=len(train_dataset.classes),
            num_hidden_layers=cfg.setup.dict.num_hidden_layers,
        )

    else:
        raise ValueError(f"Unknown model type: {cfg.modeling.type}")

    if cfg.modeling.type == "resnet":
        model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(2048, len(train_dataset.classes)),
        )
    elif cfg.modeling.type in ["vit", "dino", "dinov2"]:
        model.classifier = torch.nn.Linear(
            model.classifier.in_features, len(train_dataset.classes)
        )

        if cfg.setup.dict.classifier_only:
            for name, param in model.named_parameters():
                print(name)
                if name.startswith("classifier") or name.startswith("pooler"):
                    print(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    else:
        raise ValueError("Invalid model type")

    model.set_attn_implementation("eager")

    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        "Training Model with a total parameters of", model_parameters / 1e6, "Millions"
    )

    model_dist = [
        (name, p.numel() / 1e6)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    print("The distribution of the parameters is: ")
    for name, num_params in model_dist:
        print(f"{name}: {num_params:.2f} Million")

    processor = (
        AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
        if cfg.modeling.type == "resnet"
        else ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    )

    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-with-registers-base", use_fast=False
    )

    collator = Collator(processor)

    train_dloader = DataLoader(
        train_dataset,
        **cfg.data.collator.train,
        collate_fn=collator.classification_collate_fn,
    )

    val_dloader = DataLoader(
        validation_dataset,
        collate_fn=collator.classification_collate_fn,
        **cfg.data.collator.val,
    )
    model.to(device)

    initial_lr = 1e-5
    optimizer = AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4,
    )

    ## ** Scheduler
    steps_per_epoch = len(train_dloader)  # Number of batches per epoch
    total_steps = cfg.setup.dict.epochs * steps_per_epoch

    # Warmup steps (e.g., 10% of total steps)
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    optimal_loss = 0.0
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{cfg.modeling.checkpoint_name}.pt"

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(
        range(1, cfg.setup.dict.epochs),
        desc="Training Procedure",
        position=0,
        leave=False,
    ):
        model, train_loss = train_classification_task(
            dataloader=train_dloader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            wandb_logger=wandb_logger,
            scheduler=scheduler,
            epoch=epoch,
            num_accumulation_steps=cfg.setup.dict.accumulation_steps,
            log_every=cfg.setup.dict.log_every,
        )

        print(f"Loss Epoch: {epoch} Value: {train_loss}")

        if (epoch) == 5:
            for name, param in model.named_parameters():
                if "vit.encoder.layer":
                    param.requires_grad = True
                    print(f"Unfreezing parameter: {name}")

        if ((epoch) % 1) == 0:
            _, loss_validation, acc_validation = test_classification_task(
                dataloader=val_dloader,
                model=model,
                criterion=criterion,
                wandb_logger=wandb_logger,
            )

            updated, optimal_loss = utils.update_and_save_model(
                previous_metric=optimal_loss,
                actual_metric=acc_validation,
                model=model,
                checkpoint_path=checkpoint_name,
                processor=processor,
                compare=">",
            )

            if updated:
                print(
                    f"Model Updated: Validation Metric Epoch: {0} Value: {acc_validation} Optimal_Metric: {optimal_loss}"
                )

    print("End of training")


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

    import os
    from PIL import Image, UnidentifiedImageError

    def delete_corrupted_images(root_dir):
        num_deleted = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()  # Verifies that it's an image
                    except (UnidentifiedImageError, IOError, OSError) as e:
                        print(f"Deleting corrupted image: {file_path} — Reason: {e}")
                        os.remove(file_path)
                        num_deleted += 1
        print(f"\nFinished. Deleted {num_deleted} corrupted images.")

    # delete_corrupted_images("/data/users/cboned/data/Generic/Imagenet1k/train")
    # delete_corrupted_images("/data/users/cboned/data/Generic/Imagenet1k/val")

    with initialize(version_base="1.3.2", config_path=args.config_path):
        cfg = compose(config_name=args.config_file)

    main(cfg=cfg)

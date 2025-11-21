# type: ignore
## Dataset Things
from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets import CIFAR100, CIFAR10

from datasets.collator import Collator
import utils


from train import train_classification_task
from test import test_classification_task

from transformers import (
    AutoConfig,
    ViTForImageClassification,
    ViTConfig,
    ResNetForImageClassification,
    ViTImageProcessor,
)
from models.wrapper_ode_new import NeuralODEIntrepretation
from models.ode_transformer_gpt import ViTNeuralODE
from models.macaron import ViTMacaron
from models.ode_resnet import ODEResNet

## Common packages
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

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

    if cfg.log_wandb == True:
        wandb.login()
        config = dict(cfg.setup.dict)
        config.update(dict(cfg.modeling.inputs))
        wandb_logger = wandb.init(
            project=cfg.setup.wandb.project,
            group=cfg.setup.wandb.group,
            name=cfg.setup.wandb.name,
            config=config,
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

    if cfg.modeling.type == "vit":
        model = ViTNeuralODE(**cfg.modeling.inputs)
    elif cfg.modeling.type == "macaron":
        model = ViTMacaron(**cfg.modeling.inputs)
    else:
        model = ODEResNet(**cfg.modeling.inputs)

    teacher_model = ViTForImageClassification.from_pretrained("facebook/dino-vitb16")

    model.patch_embed.cls_token = teacher_model.vit.embeddings.cls_token
    model.patch_embed.cls_token.requires_grad = False

    # model.patch_embed.proj.weight.data.copy_(
    #   teacher_model.vit.embeddings.patch_embeddings.projection.weight.data
    #        )
    # for param in model.patch_embed.proj.parameters():
    #    param.requires_grad = False

    save_path = "/data/users/cboned/checkpoints"
    checkpoint_name = f"{save_path}/VIT_ODE_CIFAR100.pt"

    if cfg.infer_from_checkpoint:
        weight_to_update = torch.load(checkpoint_name, weights_only=True)
        for w in weight_to_update.keys():
            if model.state_dict().get(w) is not None:
                model.state_dict()[w].data.copy_(weight_to_update[w])

    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if wandb_logger:
        wandb_logger.log({"model_parameters": model_parameters})

    print(
        "Training Model with a total parameters of", model_parameters / 1e6, "Millions"
    )

    model = model.to(device)

    processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")

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

    initial_lr = 1e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=5e-2,
    )

    ## ** Scheduler
    steps_per_epoch = len(train_dloader)  # Number of batches per epoch
    total_steps = cfg.setup.dict.epochs * steps_per_epoch

    # Warmup steps (e.g., 10% of total steps)
    warmup_steps = int(0.1 * total_steps)
    num_cycles = cfg.setup.dict.epochs // 100

    optimal_loss = 0.0
    save_path = "/data/users/cboned/checkpoints"
    os.makedirs(save_path, exist_ok=True)
    checkpoint_name = f"{save_path}/{cfg.modeling.checkpoint_name}.pt"

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=num_cycles,
    )

    optimal_loss = 0.0
    save_path = "/data/users/cboned/checkpoints"
    os.makedirs(save_path, exist_ok=True)
    checkpoint_name = f"{save_path}/{cfg.modeling.checkpoint_name}.pt"

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.01)

    if cfg.log_wandb == True:
        wandb_logger.watch(model, log="all")

    for epoch in tqdm.tqdm(
        range(1, cfg.setup.dict.epochs),
        desc="Training Procedure",
        position=0,
        leave=False,
    ):
        if epoch > 100:
            for p in model.patch_embed.proj.parameters():
                p.requires_grad = True

        _, train_loss = train_classification_task(
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

        if ((epoch) % 1) == 0:
            model, loss_validation, acc_validation = test_classification_task(
                dataloader=val_dloader,
                model=model,
                criterion=criterion,
                wandb_logger=wandb_logger,
            )

            updated, optimal_loss = utils.update_and_save_model_pt(
                previous_metric=optimal_loss,
                actual_metric=acc_validation,
                model=model,
                optimizer=optimizer,
                lr_scheduler=optimizer.param_groups[0]["lr"],
                checkpoint_path=checkpoint_name,
                compare=">",
            )
            if updated:
                print(
                    f"Model Updated: Validation Loss Epoch: {0} Value: {acc_validation} Optimal_loss: {optimal_loss}"
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

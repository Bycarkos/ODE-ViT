import torch.nn.functional as F
from torchvision.datasets.imagenet import ImageFolder
from datasets.collator import Collator
import utils
from collections import defaultdict

from torchvision.datasets import CIFAR100, CIFAR10


from train import train_classification_task_distillation
from test import test_classification_task
from loss_trainer import ImageDistilTrainer
from models.ode_transformer_gpt import ViTNeuralODE
from transformers import ViTForImageClassification, ViTImageProcessor, ViTModel


## Common packages
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import (
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
        config.update(dict(cfg.modeling.student.inputs))
        config.update(dict(checkpoint=cfg.modeling.student.checkpoint_name))
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

    student_model = ViTNeuralODE(**cfg.modeling.student.inputs)

    teacher_model_checkpoint = cfg.modeling.teacher.checkpoint_path
    teacher_model = ViTForImageClassification.from_pretrained(
        teacher_model_checkpoint, attn_implementation="eager"
    )

    print("Initializing student with teacher’s patch embedding and head weights...")

    student_model.patch_embed.proj.weight.data.copy_(
        teacher_model.vit.embeddings.patch_embeddings.projection.weight.data
    )
    for param in student_model.patch_embed.proj.parameters():
        param.requires_grad = False

    student_model.patch_embed.cls_token = teacher_model.vit.embeddings.cls_token
    student_model.patch_embed.cls_token.requires_grad = False

    student_model.head = teacher_model.classifier
    for param in student_model.head.parameters():
        param.requires_grad = False

    student_model.patch_embed.pos_embed = (
        teacher_model.vit.embeddings.position_embeddings)

    #student_model.patch_embed.pos_embed.requires_grad = False

    model_parameters = sum(
        p.numel() for p in student_model.parameters() if p.requires_grad
    )
    model_dist = [
        (name, p.numel() / 1e6)
        for name, p in student_model.named_parameters()
        if p.requires_grad
    ]
    if wandb_logger:
        wandb_logger.log({"model_parameters": model_parameters})

    print(
        "Training Model with a total parameters of", model_parameters / 1e6, "Millions"
    )

    print("The distribution of the parameters is: ")
    for name, num_params in model_dist:
        print(f"{name}: {num_params:.2f} Million")

    print("Teacher Model Loaded Correctly")

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
    optimizer = AdamW(
        student_model.parameters(), #filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=initial_lr,
        weight_decay=5e-2,
    )

    ## ** Scheduler
    steps_per_epoch = len(train_dloader)  # Number of batches per epoch
    total_steps = cfg.setup.dict.epochs * steps_per_epoch

    # Warmup steps (e.g., 10% of total steps)
    warmup_steps = int(0.1 * total_steps)
    num_cycles = cfg.setup.dict.epochs // 30

    optimal_loss = 0.0
    save_path = "/data/users/cboned/checkpoints"
    os.makedirs(save_path, exist_ok=True)
    checkpoint_name = f"{save_path}/{cfg.modeling.student.checkpoint_name}.pt"

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=num_cycles,
    )

    teacher_model.set_attn_implementation("eager")

    criterion = ImageDistilTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        temperature=cfg.setup.dict.temperature,
        lambda_param=cfg.setup.dict.lambda_param,
        mse_full_path=cfg.setup.dict.mse_full_path,
        use_mse_loss=cfg.setup.dict.use_mse_loss,
        use_supervision=cfg.setup.dict.use_supervision,
        jasmin_k=cfg.setup.dict.jasmin_k,
        use_distillation=cfg.setup.dict.use_distillation,
        curriculum=cfg.setup.dict.curriculum,
        patience_factor=cfg.setup.dict.patience_factor,
    )

    if cfg.finetunne.recover_training:
        print("Recovering training from checkpoint:", cfg.finetunne.checkpoint)
        student_model, _, lr = utils.load_model_pt(
            student_model,
            optimizer,
            cfg.finetunne.checkpoint,
            device=device,
        )

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if cfg.log_wandb:
        wandb_logger.watch(student_model, log="all")


    for epoch in tqdm.tqdm(
        range(1, cfg.setup.dict.epochs),
        desc="Training Procedure",
        position=0,
        leave=False,
    ):

        if (epoch == 25) and (cfg.setup.dict.curriculum):
            for param in student_model.head.parameters():
                param.requires_grad = True

        _, train_loss = train_classification_task_distillation(
            dataloader=train_dloader,
            student_model=student_model,
            teacher_model=teacher_model,
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
            student_model, loss_validation, acc_validation = test_classification_task(
                dataloader=val_dloader,
                model=student_model,
                criterion=criterion,
                wandb_logger=wandb_logger,
            )

            updated, optimal_loss = utils.update_and_save_model_pt(
                previous_metric=optimal_loss,
                actual_metric=acc_validation,
                model=student_model,
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

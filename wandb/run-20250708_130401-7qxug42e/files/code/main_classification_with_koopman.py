# type: ignore
## Dataset Things
from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets import CIFAR100

from datasets.collator import Collator
import utils
import lkis as LKIS

from train import  train_classification_with_koopman
from test import test_classification_task

from transformers import ViTForImageClassification, ViTImageProcessor
from models.wrapper_ode import ConditionalEDOEncoderWrapper
## Common packages
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


## Configuration Package
from omegaconf import DictConfig
from hydra import initialize, compose

## Experiment Tracking packages
import tqdm

## Common packages
import os
import wandb
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_lkis_model(cfg: DictConfig):
    lkis_model = LKIS.KoopmanNetwork(**cfg.model.inputs)
    lkis_model.to(device)

    loaded_keys = lkis_model.load_state_dict(
        torch.load(cfg.model.checkpoint_path, weights_only=True), strict=False
    )
    print(f"Loaded keys: {loaded_keys}")

    for param in lkis_model.parameters():
        param.requires_grad = False

    return lkis_model


def load_koopman_operator(path):
    koopman_operator = np.load(path)
    ww, lamb, zh = koopman_operator
    ww = torch.from_numpy(ww)
    lamb = torch.from_numpy(lamb)
    zh = torch.from_numpy(zh)
    K = (ww @ (lamb.to(zh.dtype) @ zh.conj().T)).to(device)
    K = K.real.float()

    return K

def main(cfg: DictConfig):

    if cfg.log_wandb == True:
        wandb.login()
        wandb_logger = wandb.init(
            project=cfg.setup.wandb.project,
            group=cfg.setup.wandb.group,
            name=cfg.setup.wandb.name,
        )
    else:
        wandb_logger = False


    if cfg.data.dataset.name == "cifar100":
        train_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=True)
        validation_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=False)
    else:
        train_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/train")
        validation_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/val")


    base_checkpoint_path = cfg.modeling.base
    model = ViTForImageClassification.from_pretrained(base_checkpoint_path)

    model.classifier = torch.nn.Linear(model.classifier.in_features, len(train_dataset.classes))

    model = ConditionalEDOEncoderWrapper(encoder=model, n_classes=len(train_dataset.classes), steps=12)
    model = model.to(device)

    ## getting the koopman model
    koopman_model = get_lkis_model(cfg.modeling.koopman)
    koopman_operator = load_koopman_operator(cfg.modeling.koopman.operator.checkpoint_path)

    ## Processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    collator = Collator(processor)

    train_dloader = DataLoader(train_dataset, **cfg.data.collator.train, collate_fn=collator.classification_collate_fn)

    val_dloader = DataLoader(
        validation_dataset,
        collate_fn=collator.classification_collate_fn,
        **cfg.data.collator.val,
    )
    model.to(device)

    initial_lr = 1e-5
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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

    optimal_loss = 1e10
    save_path = "/data/users/cboned/checkpoints"
    os.makedirs(save_path, exist_ok=True)
    checkpoint_name = f"{save_path}/{cfg.modeling.checkpoint_name}.pt"

    criterion = torch.nn.CrossEntropyLoss()

    print("CREATING THE BASELINE METRIC VALUE\n STARTING TO EVALUATE FO THE FIRST TIME")
    _, loss_validation = test_classification_task(
        dataloader=val_dloader,
        model=model,
        criterion=criterion,
        wandb_logger=wandb_logger,
    )

    _, optimal_loss = utils.update_and_save_model_pt(
        previous_metric=optimal_loss,
        actual_metric=loss_validation,
        model=model,
        checkpoint_path=checkpoint_name,
        compare="<",
    )

    print(
        f"Validation Loss Epoch: {0} Value: {loss_validation} Optimal_loss: {optimal_loss}"
    )


    for epoch in tqdm.tqdm(
        range(1, cfg.setup.dict.epochs), desc="Training Procedure", position=0, leave=False
    ):

        _, train_loss = train_classification_with_koopman(
            dataloader=train_dloader,
            model=model,
            koopman_operator=koopman_operator,
            koopman_model=koopman_model,
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
            _, loss_validation = test_classification_task(
                dataloader=val_dloader,
                model=model,
                criterion=criterion,
                wandb_logger=wandb_logger,
            )

            updated, optimal_loss = utils.update_and_save_model_pt(
                previous_metric=optimal_loss,
                actual_metric=loss_validation,
                model=model,
                checkpoint_path=checkpoint_name,
                compare="<",
            )

            if updated:
                print(
                    f"Model Updated: Validation Loss Epoch: {0} Value: {loss_validation} Optimal_loss: {optimal_loss}"
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

    with initialize(version_base="1.3.2", config_path=args.config_path):
        cfg = compose(config_name=args.config_file)

    main(cfg=cfg)

# type: ignore
## Dataset Things
import torch.utils
import torch.utils.data
import utils

from train import train_ocr_task, train_ocr_task_ctc
from test import test_ocr_task, test_ocr_task_ctc
from models.HTR_VT import MaskedAutoencoderViT

from datasets import EsposallesDatasetForHtr
from datasets import Collator
from datasets.decoders import GreedyTextDecoder
from transformers.models.trocr import TrOCRProcessor
from transformers import AutoTokenizer

## Common packages
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import torchvision.transforms.v2 as T

## Typing Packages

## Configuration Package
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra import initialize, compose

## Experiment Tracking packages
import tqdm

## Common packages
import os
import wandb


def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.log_wandb == True:
        wandb.login()
        wandb_logger = wandb.init(
            project=cfg.setup.wandb.project,
            group=cfg.setup.wandb.group,
            name=cfg.setup.wandb.name,
        )
    else:
        wandb_logger = False

    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained("google/canine-s")

    dataset = instantiate(cfg.data.dataset.Esposalles, transforms=transforms, tokenizer=tokenizer)

    generator = torch.Generator().manual_seed(2)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=generator
    )

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten", use_fast=True
    )
    
    collate_fn = Collator(
        processor=processor, pad_token=tokenizer.pad_token_id
    )

    train_dloader = DataLoader(
        train_dataset, collate_fn=collate_fn.ocr_collate_fn, **cfg.data.collator.train
    )
    val_dloader = DataLoader(
        validation_dataset,
        collate_fn=collate_fn.ocr_collate_fn,
        **cfg.data.collator.val,
    )
    test_dloader = DataLoader(
        test_dataset, collate_fn=collate_fn.ocr_collate_fn, **cfg.data.collator.test
    )

    model = MaskedAutoencoderViT(**cfg.modeling.inputs, nb_cls=len(tokenizer), img_size=(384, 384), patch_size=(16, 16))#VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
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
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{cfg.modeling.checkpoint_name}.pt"

    criterion = torch.nn.CTCLoss(
        reduction="mean", zero_infinity=True, blank=dataset._blank_token_id
    )

    decoder = GreedyTextDecoder(blank_index=dataset._blank_token_id, tokenizer=tokenizer)

    print("CREATING THE BASELINE METRIC VALUE\n STARTING TO EVALUATE FO THE FIRST TIME")
    _, loss_validation = test_ocr_task_ctc(
        dataloader=val_dloader,
        model=model,
        criterion=criterion,
        ctc_decoder=decoder,
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

        _, train_loss = train_ocr_task_ctc(
            dataloader=train_dloader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            wandb=wandb_logger,
            scheduler=scheduler,
            epoch=epoch,
            ctc_decoder=decoder,
            num_accumulation_steps=cfg.setup.dict.accumulation_steps,
            log_every=cfg.setup.dict.log_every,
        )

        print(f"Loss Epoch: {epoch} Value: {train_loss}")

        if ((epoch) % 1) == 0:
            _, loss_validation = test_ocr_task_ctc(
                dataloader=val_dloader,
                model=model,
                criterion=criterion,
                ctc_decoder=decoder,
                wandb=wandb_logger,
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

    ## TODO test Això
    model.from_pretrained(checkpoint_name)

    _, loss_test = test_ocr_task_ctc(
        dataloader=test_dloader,
        model=model,
        criterion=criterion,
        ctc_decoder=decoder,
        wandb=wandb_logger,
    )

    updated, optimal_loss = utils.update_and_save_model_pt(
        previous_metric=optimal_loss,
        actual_metric=loss_test,
        model=model,
        checkpoint_path=checkpoint_name,
        compare="<",
    )

    if updated:
        print(f"Model Updated on Test Loss: {loss_test}")


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

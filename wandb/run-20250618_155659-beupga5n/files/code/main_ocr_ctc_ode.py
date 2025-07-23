# type: ignore
## Dataset Things
import torch.utils
import torch.utils.data
import utils


from train import train_ocr_task, train_ocr_task_ctc
from test import test_ocr_task, test_ocr_task_ctc
from models.wrapper_models import EDOEncoderWrapperLearnedQueries

from datasets import EsposallesDatasetForHtr
from datasets import Collator
from datasets.decoders import GreedyTextDecoder
from transformers.models.trocr import TrOCRProcessor
from transformers import VisionEncoderDecoderModel

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

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "?", "\u00e0", "\u00e1", "\u00e2", "\u00e3", "\u00e4", "\u00e5", "\u0101", "\u00e6", "\u00e7", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "\u00ec", "\u00ed", "\u00ee", "\u00ef", "\u00f0", "\u00f1", "\u00f2", "\u00f3", "\u00f4", "\u00f5", "\u00f6", "\u014d", "\u00f8", "\u00f9", "\u00fa", "\u00fb", "\u00fc", "\u00fd", "\u00fe", "\u00ff", "\u00c0", "\u00c1", "\u00c2", "\u00c3", "\u00c4", "\u00c5", "\u00c6", "\u00c7", "\u00c8", "\u00c9", "\u00ca", "\u00cb", "\u00cc", "\u00cd", "\u00ce", "\u00cf", "\u00d0", "\u00d1", "\u00d2", "\u00d3", "\u00d4", "\u00d5", "\u00d6", "\u00d8", "\u00d9", "\u00da", "\u00db", "\u00dc", "\u00dd", "\u00de", "\u0178", "\u00ab", "\u00bb", "\u2014", "\u2019", "\u00b0", "\u2013", "\u0153", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", " "]
    tokenizer = utils.GenerationVocab(VOCAB=vocab) #AutoTokenizer.from_pretrained("google/canine-s")
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

    model = VisionEncoderDecoderModel.from_pretrained("checkpoints/TrOCR_Esposalles_reduced.pt")#("microsoft/trocr-base-handwritten")

    model = EDOEncoderWrapperLearnedQueries(model.encoder, n_classes = len(tokenizer))

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
        reduction="mean", blank=dataset.blank_token_id)

    decoder = GreedyTextDecoder(blank_index=dataset.blank_token_id, tokenizer=tokenizer)

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
            wandb_logger=wandb_logger,
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

    #model.from_pretrained(checkpoint_name)

    _, loss_test = test_ocr_task_ctc(
        dataloader=test_dloader,
        model=model,
        criterion=criterion,
        ctc_decoder=decoder,
        wandb_logger=wandb_logger,
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

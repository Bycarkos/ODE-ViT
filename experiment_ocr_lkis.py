import torch
import tqdm
import os
import numpy as np

from typing import Dict, List

from transformers import get_scheduler
from transformers import VisionEncoderDecoderModel, AutoProcessor  # type: ignore
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchmetrics.functional.text import char_error_rate, word_error_rate

from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra import initialize, compose
import wandb

import utils
from datasets import Collator
from datasets import EsposallesDatasetForHtr #type: ignore
import lkis as LKIS
from models.reduced_encoder_decoder import REncoderDecoderModel #type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_reduced_model(cfg: DictConfig):

    teacher_model = VisionEncoderDecoderModel.from_pretrained(
        cfg.base.checkpoint_path
    ).to(device)

    reduced_model = instantiate(
        cfg.reduced_model.model,
        cfg=teacher_model.config,
        encoder=teacher_model.encoder,
        decoder=teacher_model.decoder,
    )
    for param in reduced_model.decoder.parameters():

        param.requires_grad = False
    reduced_model.to(device)

    return reduced_model


def get_lkis_model(cfg: DictConfig):
    lkis_model = LKIS.KoopmanNetwork(**cfg.lkis_model.inputs)
    lkis_model.to(device)

    loaded_keys = lkis_model.load_state_dict(
        torch.load(cfg.lkis_model.checkpoint_path, weights_only=True), strict=False
    )
    print(f"Loaded keys: {loaded_keys}")

    for param in lkis_model.parameters():
        param.requires_grad = False

    return lkis_model


def get_dataset(cfg: DictConfig, processor, transforms=None):
    datasets = []
    for var_dataset in cfg.data.dataset.keys():

        dataset = instantiate(
            cfg.data.dataset[var_dataset],
            tokenizer=processor.tokenizer,
            transforms=transforms,
        )
        datasets.append(dataset)

    generator = torch.Generator().manual_seed(2)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=generator
    )

    return train_dataset, validation_dataset, test_dataset


def get_collate_fn(processor):
    collate_fn = Collator(
        processor=processor, pad_token=processor.tokenizer.pad_token_id
    )

    return collate_fn


def get_dataloader(dataset, **kwargs):

    dloader = DataLoader(dataset, **kwargs)

    return dloader


def train_one_epoch(
    learning_model: torch.nn.Module,
    koopman_model: torch.nn.Module,
    Koopman_operator: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    ctc_decoder: torch.nn.Module,
    wandb_logger=None,
    scheduler=None,
    num_accumulation_steps: int = 1,  # Default to 1 instead of -1
    log_every: int = 250,
    epoch: int = 1,
) -> tuple[torch.nn.Module, Dict[str, float]]:
    """
    Train the model for one epoch with gradient accumulation support.

    Args:
        learning_model: Main model to train
        koopman_model: Koopman operator model (kept in eval mode)
        optimizer: Optimizer for training
        dataloader: DataLoader for training data
        criterion: Loss criterion
        ctc_decoder: Decoder for text generation
        device: Device to run training on
        wandb_logger: Weights & Biases logger (optional)
        scheduler: Learning rate scheduler (optional)
        num_accumulation_steps: Number of steps to accumulate gradients
        log_every: Log metrics every N batches
        K: Unused parameter (consider removing)
        epoch: Current epoch number

    Returns:
        Tuple of (trained_model, averaged_metrics)
    """

    # Set model modes
    learning_model.train()
    koopman_model.eval()

    # Initialize metrics tracking
    metric_keys = ["loss_kmp", "loss_rec", "guiding_loss", "loss", "recognition_loss", "cer", "wer"]
    metrics = {key: 0.0 for key in metric_keys}
    metrics_iter = {key: 0.0 for key in metric_keys}

    ww, lamb, zh = Koopman_operator
    ww = torch.from_numpy(ww)
    lamb = torch.from_numpy(lamb)
    zh = torch.from_numpy(zh)
    K = (ww @ (lamb.to(zh.dtype) @ zh.conj().T)).to(device)
    K = K.real.float()

    # Initialize wandb table if logging enabled
    wandb_table = None
    if wandb_logger:
        try:
            import wandb
            wandb_table = wandb.Table(columns=["image", "ground_truth", "transcription"])
        except ImportError:
            print("Warning: wandb not available, skipping table logging")

    accumulation_counter = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}",
        leave=True,
        total=len(dataloader),
    ):

        inputs: dict = data["pixel_values"].to(device, non_blocking=True)
        text = data["text"]
        tokens = data["tokens"].to(device, non_blocking=True)
        labels = tokens.clone()

        # Forward pass through learning model
        outputs = learning_model(
            **inputs,
            labels=labels,
            output_hidden_states=True
        )

        # Extract and concatenate hidden states
        features = torch.cat([
            feat.unsqueeze(1) for feat in outputs.encoder_hidden_states
        ], dim=1)


        # Initialize loss
        koopman_loss = 0.0
        k = koopman_model.delay

        # Koopman operator loss computation
        for i in range(1, features.shape[1] - k + 1):
            # Ensure we don't go out of bounds for both y0 and y1
            if i + k + 1 <= features.shape[1]:
                y0 = features[:, i:i+k, :].reshape(-1, features.shape[-1] * k)
                y1 = features[:, i+1:i+k+1, :].reshape(-1, features.shape[-1] * k)

            # Forward through Koopman model
            g0 = koopman_model.observate(y0)
            g1 = koopman_model.observate(y1)

            guiding_loss = torch.mean((g0@K.mT - g1)**2)

        # Accumulate losses
        koopman_loss += guiding_loss
        # Update metrics
        metrics_iter["guiding_loss"] += (guiding_loss.item())

        # Recognition loss
        recognition_loss = outputs.loss

        # Total loss
        total_loss = recognition_loss + koopman_loss

        # Backward pass
        total_loss.backward()

        # Update accumulation counter
        accumulation_counter += 1

        # Generate text for evaluation
        with torch.no_grad():
            generated_ids = learning_model.generate(**inputs, evaluate_at_t=False)
            decoded_text = ctc_decoder.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        # Calculate metrics
        batch_cer = float(char_error_rate(decoded_text, text))
        batch_wer = float(word_error_rate(decoded_text, text))

        # Update metrics
        metrics["recognition_loss"] += recognition_loss.item()
        metrics["loss"] += total_loss.item()
        metrics["cer"] += batch_cer
        metrics["wer"] += batch_wer
        metrics["guiding_loss"] += guiding_loss.item()

        metrics_iter["recognition_loss"] += recognition_loss.item()
        metrics_iter["loss"] += total_loss.item()
        metrics_iter["cer"] = batch_cer
        metrics_iter["wer"] = batch_wer

        # Gradient accumulation and optimizer step
        if accumulation_counter >= num_accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Reset counter
            accumulation_counter = 0

        # Logging
        if (batch_idx + 1) % log_every == 0:
            utils._log_metrics(
                wandb_logger, wandb_table, metrics_iter, optimizer,
                epoch, batch_idx, len(dataloader), log_every,
                data, text, decoded_text, tokens
            )

            # Reset iteration metrics
            metrics_iter = {key: 0.0 for key in metric_keys}

    # Final epoch logging
    if wandb_logger and wandb_table:
        epoch_metrics = {f"train/{key}": value / len(dataloader) for key, value in metrics.items()}
        epoch_metrics["train/epoch"] = epoch
        wandb_logger.log(epoch_metrics)
        wandb_logger.log({"train/epoch_table": wandb_table})

    # Return averaged metrics
    final_metrics = {key: value / len(dataloader) for key, value in metrics.items()}

    return learning_model, final_metrics

def main(cfg: DictConfig):

    if cfg.log_wandb == True:

        wandb.login()
        wandb_logger = wandb.init(cfg=dict(cfg.setup.dict), **cfg.setup.wandb
        )
    else:
        wandb_logger = None

    transforms = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    processor_checkpoint_path = cfg.modeling.processor.checkpoint_path
    processor = AutoProcessor.from_pretrained(processor_checkpoint_path)
    model = get_reduced_model(cfg.modeling)
    print("Reduced model loaded")
    lkis_model = get_lkis_model(cfg.modeling)
    print("LKIS model loaded")

    params_to_update = []

    for param in model.parameters():
        if param.requires_grad:
            params_to_update.append(param)

    optimizer = torch.optim.AdamW(params_to_update, lr=1e-5)

    train_dset, val_dset, test_dset = get_dataset(cfg, processor, transforms)
    val_dloader = get_dataloader(
        val_dset,
        collate_fn=get_collate_fn(processor).ocr_collate_fn,
        **cfg.data.collator.val,
    )

    test_dloader = get_dataloader(
        test_dset,
        collate_fn=get_collate_fn(processor).ocr_collate_fn,
        **cfg.data.collator.test,
    )

    train_dloader = get_dataloader(
        train_dset,
        collate_fn=get_collate_fn(processor).ocr_collate_fn,
        **cfg.data.collator.train,
    )

    # Number of training steps
    num_epochs = cfg.setup.dict.epochs
    # Assuming you have a DataLoader called train_dataloader
    num_training_steps = num_epochs * len(train_dloader)

    # Create scheduler
    scheduler = get_scheduler(
        name="cosine",  # Scheduler type (options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup)
        optimizer=optimizer,
        num_warmup_steps=100,  # Number of warmup steps
        num_training_steps=num_training_steps,
    )

    optimal_loss = 1e20
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{cfg.modeling.reduced_model.checkpoint_name}.pt"
    criterion = LKIS.KoopmanLoss

    print("Starting training procedure")
    koopman_operator =  np.load(cfg.modeling.lkis_model.koopman_operator_path+".npy")


    print("Evaluating to get the baseline")
    _, loss_validation = utils.test_ocr_task(
        dataloader=val_dloader,
        model=model,
        criterion=criterion,
        ctc_decoder=processor.tokenizer,
        wandb_logger=wandb_logger,
        log_every=25
    )

    updated, optimal_loss = utils.update_and_save_model(
        previous_metric=optimal_loss,
        actual_metric=loss_validation,
        processor=processor,
        model=model,
        checkpoint_path=checkpoint_name,
        compare="<",
    )

    for epoch in tqdm.tqdm(
        range(1, cfg.setup.epochs), desc="Training Procedure", position=0, leave=False
    ):
        _, metrics = train_one_epoch(
            learning_model=model,
            koopman_model=lkis_model,
            Koopman_operator=koopman_operator,
            dataloader=train_dloader,
            optimizer=optimizer,
            criterion=criterion,
            ctc_decoder=processor.tokenizer,
            epoch=epoch,
            num_accumulation_steps=cfg.setup.num_accumulation_steps,
            log_every=cfg.setup.log_every,
            wandb_logger=wandb_logger,
            scheduler=scheduler,
        )

        train_loss = metrics["loss"] / len(train_dloader)
        print(f"Loss Epoch: {epoch} Value: {train_loss}")

        if ((epoch) % 1) == 0:

            _, loss_validation = utils.test_ocr_task(
                dataloader=val_dloader,
                model=model,
                criterion=criterion,
                ctc_decoder=processor.tokenizer,
                wandb_logger=wandb_logger,
                log_every=25
            )

            updated, optimal_loss = utils.update_and_save_model(
                previous_metric=optimal_loss,
                actual_metric=loss_validation,
                processor=processor,
                model=model,
                checkpoint_path=checkpoint_name,
                compare="<",
            )

            if updated:
                print("Updated")


    _, loss_validation = utils.test_ocr_task(
        dataloader=test_dloader,
        model=model,
        criterion=criterion,
        ctc_decoder=processor.tokenizer,
        wandb_logger=wandb_logger,
        log_every=25,
        mode="test"
    )

    updated, optimal_loss = utils.update_and_save_model(
        previous_metric=optimal_loss,
        actual_metric=loss_validation,
        processor=processor,
        model=model,
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

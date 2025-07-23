import torch
import tqdm
import os
import numpy as np

from typing import Dict, List

from transformers import get_scheduler
from transformers import VisionEncoderDecoderModel, AutoProcessor  # type: ignore
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from jiwer import cer, wer  # Make sure to import these

from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra import initialize, compose
import wandb

import utils
from test import test_ocr_task_ctc
from datasets import Collator
from datasets import EsposallesDatasetForHtr #type: ignore
import lkis as LKIS
from models.wrapper_models import EDOEncoderWrapperLearnedQueries #type: ignore
from datasets.decoders import GreedyTextDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_reduced_model(cfg: DictConfig):

    teacher_encoder_model = VisionEncoderDecoderModel.from_pretrained(
        cfg.base.checkpoint_path
    ).to(device)

    reduced_model = EDOEncoderWrapperLearnedQueries(encoder=teacher_encoder_model.encoder, n_classes=170)
    
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


def get_dataset(cfg: DictConfig, tokenizer, transforms=None):


    dataset = instantiate(cfg.data.dataset.Esposalles, transforms=transforms, tokenizer=tokenizer)

    generator = torch.Generator().manual_seed(2)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=generator
    )

    return train_dataset, validation_dataset, test_dataset, dataset


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

    cumulative = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}",
        leave=True,
        total=len(dataloader),
    ):
        
        decoded_text = []
        cumulative += 1

        inputs = data["pixel_values"].to(device)
        text = data["text"]
        tokens = data["tokens"].to(device)

        # Forward pass through learning model
        with torch.autocast(device_type="cuda", enabled=True):
            output = learning_model(**inputs)

            preds = output["logits"]
            final_preds = preds.permute(1, 0, 2).log_softmax(2)
            pred_size = torch.IntTensor([preds.size(1)] * tokens.shape[0]).to(tokens.device)
            target_lengths = torch.sum(tokens != ctc_decoder.tokenizer.pad_token_id, dim=1) # 0 because pad token id is 0, handcrafted
            loss_recognition = criterion(final_preds, tokens, pred_size, target_lengths)


            # Extract and concatenate hidden states
            features = torch.cat([
                feat.unsqueeze(1) for feat in output["hidden_states"]
            ], dim=1)


            # Initialize loss
            koopman_loss = 0.0
            k = koopman_model.delay

            with torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32):
                # Koopman operator loss computation
                for i in range(1, features.shape[1] - k + 1):
                    # Ensure we don't go out of bounds for both y0 and y1
                    if i + k + 1 <= features.shape[1]:
                        y0 = features[:, i:i+k, :].reshape(-1, features.shape[-1] * k)
                        y1 = features[:, i+1:i+k+1, :].reshape(-1, features.shape[-1] * k)

                    # Forward through Koopman model
                    g0, h0 = koopman_model(y0)
                    g1, h1 = koopman_model(y1)

                    guiding_loss = torch.mean((g0@K.mT - g1)**2)

                # Accumulate losses
                koopman_loss += guiding_loss
                # Update metrics
                metrics_iter["guiding_loss"] += (guiding_loss.item())

        # Total loss
        total_loss = loss_recognition + koopman_loss

        # Backward pass
        total_loss.backward()

        # Update accumulation counter

        # Generate text for evaluation
        to_generate = preds.clone()
        generated_ids = ctc_decoder(to_generate.detach().cpu().numpy())
        generated_text = [ctc_decoder.tokenizer.decode(get["text"]) for get in generated_ids]
        
        decoded_text.extend(generated_text)

        # Calculate metrics
        batch_cer = float(cer(text, decoded_text))
        batch_wer = float(wer(text, decoded_text))

        # Update metrics
        metrics["recognition_loss"] += loss_recognition.item()
        metrics["loss"] += total_loss.item()
        metrics["cer"] += batch_cer
        metrics["wer"] += batch_wer
        metrics["guiding_loss"] += guiding_loss.item()

        metrics_iter["recognition_loss"] += loss_recognition.item()
        metrics_iter["loss"] += total_loss.item()
        metrics_iter["cer"] = batch_cer
        metrics_iter["wer"] = batch_wer

        # Gradient accumulation and optimizer step
        if cumulative >= num_accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Reset counter
            cumulative = 0

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
        wandb_logger = wandb.init(config=dict(cfg.setup.dict), **cfg.setup.wandb
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
    
    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "?", "\u00e0", "\u00e1", "\u00e2", "\u00e3", "\u00e4", "\u00e5", "\u0101", "\u00e6", "\u00e7", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "\u00ec", "\u00ed", "\u00ee", "\u00ef", "\u00f0", "\u00f1", "\u00f2", "\u00f3", "\u00f4", "\u00f5", "\u00f6", "\u014d", "\u00f8", "\u00f9", "\u00fa", "\u00fb", "\u00fc", "\u00fd", "\u00fe", "\u00ff", "\u00c0", "\u00c1", "\u00c2", "\u00c3", "\u00c4", "\u00c5", "\u00c6", "\u00c7", "\u00c8", "\u00c9", "\u00ca", "\u00cb", "\u00cc", "\u00cd", "\u00ce", "\u00cf", "\u00d0", "\u00d1", "\u00d2", "\u00d3", "\u00d4", "\u00d5", "\u00d6", "\u00d8", "\u00d9", "\u00da", "\u00db", "\u00dc", "\u00dd", "\u00de", "\u0178", "\u00ab", "\u00bb", "\u2014", "\u2019", "\u00b0", "\u2013", "\u0153", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", " "]
    tokenizer = utils.GenerationVocab(VOCAB=vocab)

    train_dset, val_dset, test_dset, dataset = get_dataset(cfg, tokenizer, transforms)
    
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

    initial_lr = 1e-5
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        weight_decay=1e-4,
    )

    ## ** Scheduler
    steps_per_epoch = len(train_dloader)  # Number of batches per epoch
    total_steps = cfg.setup.dict.epochs * steps_per_epoch

    # Warmup steps (e.g., 10% of total steps)
    warmup_steps = int(0.05 * total_steps)

    # Create scheduler
    scheduler = get_scheduler(
        name="cosine",  # Scheduler type (options: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup)
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,  # Number of warmup steps
        num_training_steps=total_steps,
    )

    optimal_loss = 1e10
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{cfg.modeling.reduced_model.checkpoint_name}.pt"
    criterion = torch.nn.CTCLoss(
        reduction="mean", zero_infinity=True, blank=dataset._pad_token_id
    )
    decoder = GreedyTextDecoder(blank_index=dataset._pad_token_id, tokenizer=tokenizer)

    print("Starting training procedure")
    koopman_operator =  np.load(cfg.modeling.lkis_model.koopman_operator_path+".npy")


    print("Evaluating to get the baseline")
    _, loss_validation = test_ocr_task_ctc(
        dataloader=val_dloader,
        model=model,
        criterion=criterion,
        ctc_decoder=decoder,
        wandb_logger=wandb_logger,
        log_every=1
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
        _, metrics = train_one_epoch(
            learning_model=model,
            koopman_model=lkis_model,
            Koopman_operator=koopman_operator,
            dataloader=train_dloader,
            optimizer=optimizer,
            criterion=criterion,
            ctc_decoder=decoder,
            epoch=epoch,
            num_accumulation_steps=cfg.setup.dict.num_accumulation_steps,
            log_every=cfg.setup.dict.log_every,
            wandb_logger=wandb_logger,
            scheduler=scheduler,
        )

        train_loss = metrics["loss"] / len(train_dloader)
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

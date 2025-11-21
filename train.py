# type: ignore

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
from collections import defaultdict

from torchmetrics.functional.text import char_error_rate, word_error_rate  # type: ignore
from wandb import Table, Image  # type: ignore


import models.utils as utils

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_classification_task(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    wandb_logger=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5_000,
):
    metrics_epoch = defaultdict(float)
    metrics_iter = defaultdict(float)
    cumulative = 0
    params = model.parameters()
    model.train()

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader),
        desc="Training Procedure",
        leave=True,
        position=1,
        total=len(dataloader),
    ):
        cumulative += 1

        pixel_values = data["pixel_values"].to(device)
        labels = data["labels"].to(device)

        output = model(
            **pixel_values,
            labels=labels,
            output_attentions=True,
            # jasmin_k=10,
            output_attention_trajectory=True,
        )

        preds = output["logits"]
        soft_pred = preds.softmax(dim=-1).argmax(dim=-1)
        loss = output["loss"]  # criterion(preds, labels)

        jasmin_loss = output.get("jasmin_loss", None)
        if jasmin_loss is not None:
            loss += jasmin_loss  # * 0.1
        else:
            jasmin_loss = utils.jasmin_loss(output["attention_trajectory"][-1:], k=10)

        loss += jasmin_loss

        loss.backward()

        metrics_epoch["epoch_loss"] += loss.item()
        metrics_iter["iteration_loss"] += loss.item()

        metrics_iter["iteration_acc"] += (soft_pred == labels).float().mean(-1)
        metrics_epoch["epoch_acc"] += (soft_pred == labels).float().mean(-1)

        metrics_iter["jasmin_loss"] += jasmin_loss.item()
        metrics_epoch["jasmin_loss"] += jasmin_loss.item()

        if cumulative >= num_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            cumulative = 0

        metrics_iter.update({"train/lr": optimizer.param_groups[0]["lr"]})

        if ((batch_idx + 1) % log_every) == 0:
            if wandb_logger:
                metrics_iter = {
                    f"train/{key}": value / log_every
                    for key, value in metrics_iter.items()
                }
                wandb_logger.log(metrics_iter)
                metrics_iter = defaultdict(float)

    loss_to_return = metrics_epoch["epoch_loss"] / len(dataloader)

    if wandb_logger:
        metrics_epoch = {
            f"train/{key}": value / len(dataloader)
            for key, value in metrics_epoch.items()
        }
        metrics_epoch.update({"train/epoch": epoch})
        wandb_logger.log(metrics_epoch)

    return model, loss_to_return


def train_one_sample_classification_task_distillation(
    dataloader: DataLoader,
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    wandb_logger=None,
    epoch: int = 0,
    log_every: int = 10,
):
    metrics_epoch = defaultdict(float)
    metrics_iter = defaultdict(float)
    params = student_model.parameters()

    data = next(iter(dataloader))
    pixel_values = data["pixel_values"].to(device)

    labels = data["labels"].to(device)

    output = criterion.compute_loss_test_one_sample(inputs=pixel_values, labels=labels)
    loss = output["loss"]

    if output.get("mse_losses", None) is not None:
        for key, value in output["mse_losses"].items():
            metrics_epoch[key] += value.item()
            metrics_iter[key] += value.item()

    student_output = output["student_output"]
    if output.get("mse_loss", None):
        mse_loss = output["mse_loss"]
        metrics_epoch["mse loss"] += mse_loss.item()
    #    metrics_iter["mse loss"] += mse_loss.item()

    if output.get("jasmin_loss", None):
        jasmin_loss = output["jasmin_loss"]
        metrics_epoch["jasmin loss"] += jasmin_loss.item()
    #    metrics_iter["jasmin loss"] += jasmin_loss.item()

    preds = student_output["logits"]
    soft_pred = preds.softmax(dim=-1).argmax(dim=-1)

    metrics_epoch["epoch_loss"] += loss.item()
    # metrics_iter["iteration_loss"] += loss.item()

    # metrics_iter["iteration_acc"] += (soft_pred == labels).float().mean(-1)
    metrics_epoch["epoch_acc"] += (soft_pred == labels).float().mean(-1)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    optimizer.zero_grad()
    if scheduler:
        scheduler.step()

    loss_to_return = metrics_epoch["epoch_loss"]

    if wandb_logger:
        metrics_epoch = {f"train/{key}": value for key, value in metrics_epoch.items()}
        metrics_epoch.update({"train/epoch": epoch})
        wandb_logger.log(metrics_epoch)

    if epoch % 1 == 0:
        print(
            f"Epoch {epoch}: Loss {loss_to_return:.4f}, Accuracy {metrics_epoch['epoch_acc']:.4f}"
        )
        print(f"Upper bound: {output['second_derivative_upper_bound']:.8f}")
        for key, value in output["finite_difference_upper_bound"].items():
            if isinstance(value, float):
                print(f"Finite Difference Upper Bound {key}: {value:.8f}")

    return student_model, loss_to_return


def train_classification_task_distillation(
    dataloader: torch.utils.data.DataLoader,
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler=None,
    wandb_logger=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    metrics_epoch = defaultdict(float)
    metrics_iter = defaultdict(float)

    for batch_idx, batch in tqdm.tqdm(
        enumerate(dataloader),
        desc=f"Epoch {epoch} — Training",
        total=len(dataloader),
        leave=True,
        position=1,
        unit="batch",
    ):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Forward + loss computation
        output = criterion(inputs=pixel_values, labels=labels, epoch=epoch)

        for key, value in output.items():
            if not isinstance(value, dict):
                v = value.item()
                metrics_epoch[key] += v
                metrics_iter[key] += v

        student_output = output["student_output"]
        preds = student_output["logits"]
        soft_pred = preds.softmax(dim=-1).argmax(dim=-1)
        acc = (soft_pred == labels).float().mean().item()
        metrics_iter["iteration_acc"] += acc
        metrics_epoch["epoch_acc"] += acc

        logits_dist = student_output.get("logits_dist", None)
        
        if logits_dist is not None:
            soft_pred_dist = logits_dist.argmax(dim=-1)
            mixed_pred = ((logits_dist * criterion.lambda_param + preds) / 2).argmax(
                dim=-1
            )

            metrics_epoch["epoch_acc_dist"] += (
                (soft_pred_dist == labels).float().mean().item()
            )
            metrics_epoch["mixed_acc"] += (mixed_pred == labels).float().mean().item()

        metrics_iter["lr"] = optimizer.param_groups[0]["lr"]

        # Logging
        if (batch_idx + 1) % log_every == 0 and wandb_logger:
            iter_metrics = {
                f"train/{k}": v / log_every for k, v in metrics_iter.items()
            }
            iter_metrics["Bounds/second_derivative"] = output.get(
                "second_derivative_upper_bound", 0
            )
            finite_diff_bounds = output.get("finite_difference_upper_bound", {})
            for k, v in finite_diff_bounds.items():
                if isinstance(v, (float, int)):
                    iter_metrics[f"Bounds/{k}"] = v
            wandb_logger.log(iter_metrics)
            metrics_iter.clear()

    loss_to_return = metrics_epoch["epoch_loss"] / len(dataloader)
    if wandb_logger:
        epoch_metrics = {
            f"train_epoch/{k}": v / len(dataloader) for k, v in metrics_epoch.items()
        }
        epoch_metrics["train_epoch/epoch"] = epoch
        wandb_logger.log(epoch_metrics)

    return student_model, loss_to_return


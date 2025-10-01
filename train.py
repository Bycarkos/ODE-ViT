# type: ignore

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
from collections import defaultdict

from torchmetrics.functional.text import char_error_rate, word_error_rate  # type: ignore
from wandb import Table, Image  # type: ignore
device = "cuda" if torch.cuda.is_available() else "cpu"



def train_classification_with_koopman(
    dataloader: DataLoader,
    model: torch.nn.Module,
    koopman_model: torch.nn.Module,
    koopman_operator: torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    wandb_logger=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5_000):

    model.train()
    koopman_model.eval()
    params = model.parameters()

    metrics_epoch = defaultdict(float)
    metrics_iter = defaultdict(float)
    cumulative = 0

    K = koopman_operator

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc="Training Procedure", leave=True, position=1, total=len(dataloader),
    ):

        cumulative += 1

        pixel_values = data["pixel_values"].to(device)
        labels = data["labels"].to(device)

        output = model(**pixel_values, labels=labels)

        preds = output["logits"]
        soft_pred = preds.softmax(dim=-1).argmax(dim=-1)
        loss = output["loss"]

        features = output["hidden_states"][:, :, 0]
        features = features.permute(1, 0,2)

        koopman_loss = 0.0
        k = koopman_model.delay
        ce_losses = []
        for i in range(features.shape[1] - k - 1):
            y0 = features[:, i:i+k, :].flatten(1)
            g, h = koopman_model(y0)
            state = g.clone()
            for j in range(i+k, features.shape[1] -1):
                state = (state @ K.mT)

            h_final_state = koopman_model.reconstruct(state)
            cls_state = model.projector(h_final_state)

            loss_state = torch.nn.functional.cross_entropy(cls_state, labels)

            ce_losses.append(loss_state)

        ce_losses.append(loss)
        #ce_losses = torch.tensor(ce_losses, requires_grad=True)
        weighting = torch.arange(1, len(ce_losses)+1)
        weighting = weighting/weighting.shape[-1]
        total_loss = sum([ce_losses[i] * weighting[i] for i in range(len(ce_losses))])

        for i, value in enumerate(ce_losses):
            st = f"ce_state_{i}"
            metrics_iter[st] += value.item()

        #metrics_iter["guiding_loss"] += koopman_loss.item()
        #metrics_epoch["epoch_guiding_loss"] += koopman_loss.item()

        metrics_epoch["epoch_loss"] += total_loss.item()
        metrics_iter["iteration_loss"] += total_loss.item()

        metrics_iter["iteration_acc"] += (soft_pred == labels).float().mean(-1)
        metrics_epoch["epoch_acc"] += (soft_pred == labels).float().mean(-1)

        total_loss.backward()

        if cumulative >= num_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            cumulative = 0

        metrics_iter.update({"train/lr": optimizer.param_groups[0]["lr"]})

        if ((batch_idx + 1) % log_every) == 0:

            if wandb_logger:
                metrics_iter = {
                    f"train/{key}": value / log_every for key, value in metrics_iter.items()
                }
                wandb_logger.log(metrics_iter)
                metrics_iter = defaultdict(float)

    loss_to_return = metrics_epoch["epoch_loss"]/len(dataloader)

    if wandb_logger:
        metrics_epoch = {
            f"train/{key}": value / len(dataloader) for key, value in metrics_epoch.items()
        }
        metrics_epoch.update({"train/epoch": epoch})
        wandb_logger.log(metrics_epoch)

    return model, loss_to_return

def train_lkis_task(
    teacher_model_encoder: torch.nn.Module,
    student_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    epoch: int,
    num_accumulation_steps: int,
    log_every: int,
    wandb=None,
    scheduler=None,
    K: int = 2,
    full_grid: bool = True):

    teacher_model_encoder.eval()
    student_model.train()
    # params = student_model.parameters()

    cumulative = 0
    metrics = {"loss_stability":0.0, "loss_kpm": 0.0, "loss_rec": 0.0, "loss": 0.0}
    metrics_iter = {"loss_stability_iter":0.0, "loss_kpm_iter": 0.0, "loss_rec_iter": 0.0, "loss_iter": 0.0}

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader),
        desc="Training Procedure",
        leave=True,
        position=1,
        total=len(dataloader),
    ):

        inputs: dict = data["pixel_values"].to(device)

        with torch.no_grad():
            features = teacher_model_encoder(**inputs, output_hidden_states=True)["hidden_states"]  # Shape [batch_size, Seq + 1, 768] -> Shape [batch_size, 768] Get the cls

        if full_grid:
            features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)
        else:
            features = torch.cat([feat.unsqueeze(1) for feat in features], dim=1)[:, :, 0]

        k = student_model.delay
        loss = 0

        for i in range(0, features.shape[1] - k):
            if i + k <= features.shape[1]:
                y0 = features[:, i:i+k, :].flatten(1)
                y1 = features[:, i+1:i+k+1, :].flatten(1)

                g0, h0 = student_model(y0)
                g1, h1 = student_model(y1)
                tmp_loss, losses_desglosed = criterion(y0, y1, g0, g1, h0, h1, dim_y=h0.shape[1])
                loss += tmp_loss

                metrics = {
                    "loss_kpm": metrics["loss_kpm"] + losses_desglosed["loss_kpm"],
                    "loss_stability": metrics["loss_stability"] + losses_desglosed["loss_stability"],
                    "loss_rec": metrics["loss_rec"] + losses_desglosed["loss_rec"],
                    "loss": metrics["loss"] + tmp_loss.item(),
                }
                metrics_iter = {
                    "loss_kpm_iter": metrics_iter["loss_kpm_iter"]
                    + losses_desglosed["loss_kpm"],
                    "loss_stability_iter": metrics_iter["loss_stability_iter"] + losses_desglosed["loss_stability"],
                    "loss_rec_iter": metrics_iter["loss_rec_iter"]
                    + losses_desglosed["loss_rec"],
                    "loss_iter": metrics_iter["loss_iter"] + tmp_loss.item(),
                }
        loss.backward()

        if cumulative >= num_accumulation_steps:
            # torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            cumulative = 0

        if ((batch_idx + 1) % log_every) == 0:

            if wandb:
                wandb.log(
                    {
                        f"train/{key}": value / log_every
                        for key, value in metrics_iter.items()
                    }
                )
            metrics_iter = {"loss_stability_iter":0.0, "loss_kpm_iter": 0.0, "loss_rec_iter": 0.0, "loss_iter": 0.0}


        cumulative += 1

    return student_model, metrics

def train_classification_task(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    wandb_logger=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5_000):


    metrics_epoch = defaultdict(float)
    metrics_iter = defaultdict(float)
    cumulative = 0
    params = model.parameters()
    model.train()

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc="Training Procedure", leave=True, position=1, total=len(dataloader),
    ):

        cumulative += 1

        pixel_values = data["pixel_values"].to(device)
        labels = data["labels"].to(device)

        output = model(**pixel_values, labels=labels)

        preds = output["logits"]
        soft_pred = preds.softmax(dim=-1).argmax(dim=-1)
        loss = output["loss"] #criterion(preds, labels)
        loss.backward()

        metrics_epoch["epoch_loss"] += (loss.item())
        metrics_iter["iteration_loss"] += (loss.item())

        metrics_iter["iteration_acc"] += (soft_pred == labels).float().mean(-1)
        metrics_epoch["epoch_acc"] += (soft_pred == labels).float().mean(-1)

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
                    f"train/{key}": value / log_every for key, value in metrics_iter.items()
                }
                wandb_logger.log(metrics_iter)
                metrics_iter = defaultdict(float)

    loss_to_return = metrics_epoch["epoch_loss"]/len(dataloader)

    if wandb_logger:
        metrics_epoch = {
            f"train/{key}": value / len(dataloader) for key, value in metrics_epoch.items()
        }
        metrics_epoch.update({"train/epoch": epoch})
        wandb_logger.log(metrics_epoch)

    return model, loss_to_return

def train_classification_task_distillation(
    dataloader: DataLoader,
    teacher_model: torch.nn.Module,
    student_model:torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    wandb_logger=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5_000):


    metrics_epoch = defaultdict(float)
    metrics_iter = defaultdict(float)
    cumulative = 0
    params = student_model.parameters()



    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc="Training Procedure", leave=True, position=1, total=len(dataloader), unit="batch", unit_scale=True
    ):
        cumulative += 1

        pixel_values = data["pixel_values"].to(device)

        labels = data["labels"].to(device)

        loss, student_target_loss, mse_loss, student_output = criterion.compute_loss(inputs = pixel_values, labels=labels)

        preds = student_output["logits"]
        soft_pred = preds.softmax(dim=-1).argmax(dim=-1)
        soft_pred_dist = student_output["logits_dist"].argmax(dim=-1)
        mixed_pred = ((student_output["logits_dist"] + preds)/2).argmax(dim=-1)

        loss.backward()



        metrics_epoch["epoch_loss"] += loss.item()
        metrics_iter["iteration_loss"] += loss.item()

        metrics_epoch["kd_loss"] += student_target_loss.item()
        metrics_iter["kd_loss"] += student_target_loss.item()

        metrics_epoch["mse loss"] += mse_loss.item()
        metrics_iter["mse loss"] += mse_loss.item()

        metrics_iter["iteration_acc"] += (soft_pred == labels).float().mean(-1)
        metrics_epoch["epoch_acc"] += (soft_pred == labels).float().mean(-1)

        metrics_epoch["epoch_acc_dist"] += (soft_pred_dist == labels).float().mean(-1)

        metrics_epoch["mixed_acc"] += (mixed_pred == labels).float().mean(-1)

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
                    f"train/{key}": value / log_every for key, value in metrics_iter.items()
                }
                wandb_logger.log(metrics_iter)
                metrics_iter = defaultdict(float)

        loss_to_return = metrics_epoch["epoch_loss"]/len(dataloader)

    if wandb_logger:
        metrics_epoch = {
            f"train/{key}": value / len(dataloader) for key, value in metrics_epoch.items()
        }
        metrics_epoch.update({"train/epoch": epoch})
        wandb_logger.log(metrics_epoch)



    return student_model, loss_to_return

def train_ocr_task_ctc_distillation(
    dataloader: DataLoader,
    teacher_model: torch.nn.Module,
    student_model:torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    ctc_decoder,
    wandb_logger=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5_000,):

    metrics_epoch = {"epoch_loss": 0.0, "epoch_cer": 0.0, "epoch_wer": 0.0, "epoch_mse": 0.0, "epoch_distill": 0.0}
    metrics_iter = {"iteration_loss": 0.0, "iteration_cer": 0.0, "iteration_wer": 0.0, "iteration_mse": 0.0, "iteration_distill": 0.0}

    student_model.train()

    if wandb_logger:
        table: Table = Table(columns=["image", "ground_truth", "transcription"])

    params = student_model.parameters()
    cumulative = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc="Training Procedure", leave=True, position=1, total=len(dataloader), unit="batch", unit_scale=True
    ):
        decoded_text = []
        cumulative += 1

        inputs = data["pixel_values"].to(device)
        text = data["text"]
        tokens = data["tokens"].to(device)


        output = student_model(**inputs)
        hidden_states_students = output["hidden_states"]
        hidden_states_students = torch.cat([feat.unsqueeze(1) for feat in hidden_states_students], dim=1)
        with torch.no_grad():
            output_teacher = teacher_model(**inputs)
            hidden_states_teacher = output_teacher["hidden_states"]
            hidden_states_teacher = torch.cat([feat.unsqueeze(1) for feat in hidden_states_teacher], dim=1)
            logits_teacher = output_teacher["logits"]
        ## Loss recognition
        preds = output["logits"]


        final_preds = preds.permute(1, 0, 2).log_softmax(2)
        pred_size = torch.IntTensor([preds.size(1)] * tokens.shape[0]).to(tokens.device)
        target_lengths = torch.sum(tokens != ctc_decoder.tokenizer.pad_token_id, dim=1) # 0 because pad token id is 0, handcrafted
        loss = criterion(final_preds, tokens, pred_size, target_lengths)

        ## loss MSE

        loss_mse = torch.nn.functional.mse_loss(hidden_states_students[:, -1], hidden_states_teacher[:, -1])

        ## loss Dist
        #loss_dist = torch.nn.functional.cross_entropy(preds, logits_teacher.softmax(dim=-1))
        log_probs_student = torch.nn.functional.log_softmax(preds / 1., dim=-1)
        probs_teacher = torch.nn.functional.softmax(logits_teacher / 1., dim=-1)
        loss_dist = torch.nn.functional.kl_div(log_probs_student, probs_teacher, reduction='batchmean') * (1. ** 2)

        loss_dist *= 0.05

        #import pdb; pdb.set_trace()
        loss_final = loss + 0.1 * loss_mse #+ loss_dist
        loss_final.backward()
        to_generate = preds.clone()
        generated_ids = ctc_decoder(to_generate.detach().cpu().numpy())
        generated_text = [ctc_decoder.tokenizer.decode(get["text"]) for get in generated_ids]

        decoded_text.extend(generated_text)

        metrics_epoch["epoch_loss"] += loss.item()
        metrics_epoch["epoch_distill"] += loss_dist.item()
        metrics_epoch["epoch_mse"] += loss_mse.item()

        metrics_iter["iteration_loss"] += loss.item()
        metrics_iter["iteration_distill"] += loss_dist.item()
        metrics_iter["iteration_mse"] += loss_mse.item()

        metrics_iter["iteration_cer"] = float(char_error_rate(decoded_text, text))
        metrics_iter["iteration_wer"] = float(word_error_rate(decoded_text, text))

        metrics_epoch["epoch_cer"] += metrics_iter["iteration_cer"]
        metrics_epoch["epoch_wer"] += metrics_iter["iteration_wer"]

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

                for _idx in range(tokens.shape[0]):

                    ## Wandb Logging
                    original_text = text[_idx]
                    predicted_text = decoded_text[_idx]
                    image = data["raw_images"][_idx]
                    ground_truth_image = Image(image)
                    table.add_data(ground_truth_image, original_text, predicted_text)

                wandb_logger.log(metrics_iter)
                #wandb_logger.log({"train/table": table})
                metrics_iter = {"iteration_loss": 0.0, "iteration_cer": 0.0, "iteration_wer": 0.0, "iteration_mse": 0.0, "iteration_distill": 0.0}

    loss_to_return = metrics_epoch["epoch_loss"]/len(dataloader)

    if wandb_logger:
        metrics_epoch = {
            f"train/{key}": value / log_every for key, value in metrics_epoch.items()
        }
        metrics_epoch.update({"train/epoch": epoch})
        wandb_logger.log(metrics_epoch)
        wandb_logger.log({"train/table": table})

    return student_model, loss_to_return

def train_ocr_task_ctc(
            dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    ctc_decoder,
    wandb_logger=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5_000,):

    metrics_epoch = {"epoch_loss": 0.0, "epoch_cer": 0.0, "epoch_wer": 0.0}
    metrics_iter = {"iteration_loss": 0.0, "iteration_cer": 0.0, "iteration_wer": 0.0}

    model.train()

    if wandb_logger:
        table: Table = Table(columns=["image", "ground_truth", "transcription"])

    params = model.parameters()
    cumulative = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc="Training Procedure", leave=True, position=1
    ):
        decoded_text = []
        cumulative += 1

        inputs = data["pixel_values"].to(device)
        text = data["text"]
        tokens = data["tokens"].to(device)


        output = model(**inputs)
        preds = output["logits"]


        final_preds = preds.permute(1, 0, 2).log_softmax(2)
        pred_size = torch.IntTensor([preds.size(1)] * tokens.shape[0]).to(tokens.device)
        target_lengths = torch.sum(tokens != ctc_decoder.tokenizer.pad_token_id, dim=1) # 0 because pad token id is 0, handcrafted
        loss = criterion(final_preds, tokens, pred_size, target_lengths)

        loss.backward()

        to_generate = preds.clone()
        generated_ids = ctc_decoder(to_generate.detach().cpu().numpy())
        generated_text = [ctc_decoder.tokenizer.decode(get["text"]) for get in generated_ids]

        decoded_text.extend(generated_text)

        metrics_epoch["epoch_loss"] += loss.item()
        metrics_iter["iteration_loss"] += loss.item()

        metrics_iter["iteration_cer"] = float(char_error_rate(decoded_text, text))
        metrics_iter["iteration_wer"] = float(word_error_rate(decoded_text, text))

        metrics_epoch["epoch_cer"] += metrics_iter["iteration_cer"]
        metrics_epoch["epoch_wer"] += metrics_iter["iteration_wer"]

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

                for _idx in range(tokens.shape[0]):

                    ## Wandb Logging
                    original_text = text[_idx]
                    predicted_text = decoded_text[_idx]
                    image = data["raw_images"][_idx]
                    ground_truth_image = Image(image)
                    table.add_data(ground_truth_image, original_text, predicted_text)

                wandb_logger.log(metrics_iter)
                #wandb_logger.log({"train/table": table})
                metrics_iter = {"iteration_loss": 0.0, "iteration_cer": 0.0, "iteration_wer": 0.0}

    loss_to_return = metrics_epoch["epoch_loss"]/len(dataloader)

    if wandb_logger:
        metrics_epoch = {
            f"train/{key}": value / log_every for key, value in metrics_epoch.items()
        }
        metrics_epoch.update({"train/epoch": epoch})
        wandb_logger.log(metrics_epoch)
        wandb_logger.log({"train/table": table})

    return model, loss_to_return

def train_ocr_task(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer,
    criterion: torch.nn.Module,
    scheduler,
    ctc_decoder,
    wandb=None,
    epoch: int = 0,
    num_accumulation_steps: int = 16,
    log_every: int = 5_000,):

    metrics_epoch = {"epoch_loss": 0.0, "epoch_cer": 0.0, "epoch_wer": 0.0}
    metrics_iter = {"iteration_loss": 0.0, "iteration_cer": 0.0, "iteration_wer": 0.0}

    model.train()

    if wandb:
        table: Table = wandb.Table(columns=["image", "ground_truth", "transcription"])

    params = model.parameters()
    cumulative = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc="Training Procedure", leave=True, position=1
    ):
        decoded_text = []
        cumulative += 1

        pixel_values = data["pixel_values"].to(device)
        text = data["text"]
        tokens = data["tokens"].to(device)

        output = model(**pixel_values, decoder_input_ids=tokens, labels=tokens)

        loss = output.loss
        # pred_size = torch.IntTensor([preds.size(1)] * pixel_values.shape[0]).cuda()
        # target_lengths = torch.sum(tokens != ctc_decoder.pad_token_id, dim=1)
        # preds = preds.permute(1, 0, 2).log_softmax(2)

        generated_ids = model.generate(pixel_values)
        generated_text = ctc_decoder.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        decoded_text.extend(generated_text)

        loss = output.loss
        loss.backward()

        metrics_epoch["epoch_loss"] += loss.item()
        metrics_iter["iteration_loss"] += loss.item()

        metrics_iter["iteration_cer"] = float(char_error_rate(decoded_text, text))
        metrics_iter["iteration_wer"] = float(word_error_rate(decoded_text, text))

        metrics_epoch["epoch_cer"] += metrics_iter["iteration_cer"]
        metrics_epoch["epoch_wer"] += metrics_iter["iteration_wer"]

        if cumulative >= num_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            cumulative = 0

        metrics_iter.update({"train/lr": optimizer.param_groups[0]["lr"]})

        if ((batch_idx + 1) % log_every) == 0:

            if wandb:

                for _idx in range(pixel_values.shape[0]):

                    ## Wandb Logging
                    original_text = text[_idx]
                    predicted_text = decoded_text[_idx]
                    image = data["raw_images"][_idx]
                    ground_truth_image = wandb.Image(image)
                    table.add_data(ground_truth_image, original_text, predicted_text)
                    wandb.log(
                        metrics_iter, step=(epoch + 1) * (len(dataloader) + batch_idx)
                    )
                    wandb.log({"train/table": table})

    if wandb:
        metrics_epoch = {
            f"train/{key}": value / log_every for key, value in metrics_epoch.items()
        }
        metrics_epoch.update({"train/epoch": epoch})
        wandb.log(metrics_epoch, step=(epoch + 1) * len(dataloader))
        wandb.log({"train/table": table})

    return model, (metrics_epoch["epoch_loss"] / log_every)

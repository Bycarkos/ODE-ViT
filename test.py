# type: ignore
from secrets import token_bytes
import torch
from torch.utils.data import DataLoader
import tqdm

from torchmetrics.functional.text import char_error_rate, word_error_rate
from torchmetrics.functional.classification import accuracy

import torchvision.transforms.v2 as T
from wandb import Table, Image
import PIL as pil

device = "cuda" if torch.cuda.is_available() else "cpu"

UNNORMALIZE = T.Compose(
    [
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


@torch.no_grad()
def infer_one_sample(filepath: str, model: torch.nn.Module, processor):
    image = pil.Image.open(filepath).convert("RGB")
    image = processor(image, return_tensors="pt").to(device)
    # labels = processor.tokenizer("Pro", return_tensors="pt").input_ids.to(device)
    # outputs = model(**image,decoder_input_ids=labels, labels=labels, output_hidden_states=True)
    generated_ids = model.generate(**image)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_text[0]

@torch.no_grad()
def test_classification_task(
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    wandb_logger=None,
    mode: str = "val",
    log_every: int = 25
):

    metrics = {"epoch_loss": 0.0, "epoch_acc": 0.0}

    model.eval()
    cumulative = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc=f"{mode} Procedure", leave=True, position=1
    ):
        cumulative += 1

        pixel_values = data["pixel_values"].to(device)
        labels = data["labels"].to(device)

        output = model(**pixel_values, labels=labels)

        preds = output["logits"]
        soft_pred = preds.softmax(dim=-1).argmax(dim=-1)
        loss = output["loss"] #criterion(preds, labels)
        metrics["epoch_loss"] += loss.item()
        metrics["epoch_acc"] += (soft_pred == labels).float().mean(-1) #(accuracy(soft_pred, labels, task="multiclass"))


        if (batch_idx + 1) % log_every == 0:
            break

    if wandb_logger:
        wandb_logger.log(
            {f"{mode}/{key}": value / log_every for key, value in metrics.items()}
        )

    return model, (metrics["epoch_loss"] / log_every)

@torch.no_grad()
def test_ocr_task_ctc(
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    ctc_decoder,
    wandb_logger=None,
    mode: str = "val",
    log_every: int = 10
):

    metrics = {"epoch_loss": 0.0, "epoch_cer": 0.0, "epoch_wer": 0.0}

    model.eval()

    if wandb_logger:
        table = Table(columns=["image", "ground_truth", "transcription"])

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc=f"{mode} Procedure", leave=True, position=1
    ):

        decoded_text = []
        inputs = data["pixel_values"].to(device)
        text = data["text"]

        tokens = data["tokens"].to(device)
        output = model(**inputs)["logits"]
        final_preds = output.permute(1, 0, 2).log_softmax(2)
        pred_size = torch.IntTensor([output.size(1)] * tokens.shape[0]).to(tokens.device)
        target_lengths = torch.sum(tokens != ctc_decoder.tokenizer.pad_token_id, dim=1) # 0 because pad token id is 0, handcrafted
        loss = criterion(final_preds, tokens, pred_size, target_lengths)

        generated_ids = ctc_decoder(output.cpu().numpy())
        generated_text = [ctc_decoder.tokenizer.decode(get["text"]) for get in generated_ids]

        metrics["epoch_loss"] += loss.item()

        decoded_text.extend(generated_text)
        metrics["epoch_cer"] += float(char_error_rate(decoded_text, text).item())
        metrics["epoch_wer"] += float(word_error_rate(decoded_text, text).item())

        if wandb_logger:
            for _idx in range(len(decoded_text)):

                ## Wandb Logging
                original_text = text[_idx]
                image = data["raw_images"][_idx]
                ground_truth_image = Image(image)
                table.add_data(ground_truth_image, original_text, generated_text[_idx])

        if (batch_idx + 1) % log_every == 0:
            break

    if wandb_logger:
        wandb_logger.log(
            {f"{mode}/{key}": value / log_every for key, value in metrics.items()}
        )
        wandb_logger.log({f"{mode}/table": table})

    return model, (metrics["epoch_loss"] / log_every)



@torch.no_grad()
def test_ocr_task(
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    ctc_decoder,
    wandb_logger=None,
    mode: str = "val",
    log_every: int = 100,
):

    metrics = {"epoch_loss": 0.0, "epoch_cer": 0.0, "epoch_wer": 0.0}

    model.eval()

    if wandb_logger:
        table = Table(columns=["image", "ground_truth", "transcription"])

    for batch_idx, data in tqdm.tqdm(
        enumerate(dataloader), desc=f"{mode} Procedure", leave=True, position=1
    ):

        decoded_text = []
        inputs = data["pixel_values"].to(device)
        text = data["text"]
        tokens = data["tokens"].to(device)

        output = model(**inputs, labels=tokens)
        loss = output.loss

        generated_ids = model.generate(**inputs)
        generated_text = ctc_decoder.batch_decode(generated_ids, skip_special_tokens=True
        )

        metrics["epoch_loss"] += loss.item()

        decoded_text.extend(generated_text)

        metrics["epoch_cer"] += float(char_error_rate(decoded_text, text).item())
        metrics["epoch_wer"] += float(word_error_rate(decoded_text, text).item())

        if wandb_logger:
            for _idx in range(len(decoded_text)):

                ## Wandb Logging
                original_text = text[_idx]
                image = data["raw_images"][_idx]
                ground_truth_image = Image(image)
                table.add_data(ground_truth_image, original_text, generated_text[_idx])

        if (batch_idx + 1) % log_every == 0:
            break

    if wandb_logger:
        wandb_logger.log(
            {f"{mode}/{key}": value / log_every for key, value in metrics.items()}
        )
        wandb_logger.log({f"{mode}/table": table})

    return model, (metrics["epoch_loss"] / log_every)


if __name__ == "__main__":
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor

    example = "/data/users/cboned/data/HTR/Esposalles/IEHHR_training_part1/idPage10354_Record1/words/idPage10354_Record1_Line0_Word0.png"
    model = VisionEncoderDecoderModel.from_pretrained(
        "checkpoints/TrOCR_Esposalles.pt"
    ).to(device)
    processor = TrOCRProcessor.from_pretrained("checkpoints/TrOCR_Esposalles.pt")

    print(infer_one_sample(example, model, processor))

import torch
import tqdm
import wandb
import os


import utils
from train import train_lkis_task
from datasets import Collator
from datasets.esposalles_dataset import EsposallesDatasetForHtr
import lkis as LKIS
from models.wrapper_models import EncoderWrapperLearnedQueries

from transformers import get_scheduler
from transformers import VisionEncoderDecoderModel, AutoProcessor  # type: ignore
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra import initialize, compose
import matplotlib.pyplot as plt

# Set Matplotlib style
plt.style.use("seaborn-v0_8-deep")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 120,
    }
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@torch.no_grad()
def get_embeddings_by_step(
    model, dloader, number_of_blocks: int = 12, sample_number: int = 2000
):

    model.eval()
    # embeddings = {f"features_{i}": [] for i in range(number_of_blocks)}
    cls_embeddings = {f"cls_{i}": [] for i in range(number_of_blocks)}
    grouped_cls_embeddings = []
    # grouped_sequence_embeddings = []

    cumulative_sampled = 0

    for batch_idx, data in tqdm.tqdm(
        enumerate(dloader),
        desc="Extracting Embeddings",
        total=sample_number // dloader.batch_size,
        leave=True,
        position=1,
    ):
        if cumulative_sampled >= sample_number:
            break
        inputs = data["pixel_values"].to(device)

        outputs = model(**inputs, output_hidden_states=True).hidden_states

        for block_level in range(number_of_blocks):
            # embeddings[f"features_{block_level}"].append(outputs[block_level][:, 1:, :].cpu())
            cls_embeddings[f"cls_{block_level}"].append(
                outputs[block_level][:, 0, :].cpu()
            )

            # grouped_sequence_embeddings.append(embeddings[f"features_{block_level}"][-1])
            grouped_cls_embeddings.append(cls_embeddings[f"cls_{block_level}"][-1])

        cumulative_sampled += outputs[0].shape[0]

    return dict(
        cls_embeddings=cls_embeddings,
        grouped_cls_embeddings=torch.cat(grouped_cls_embeddings, dim=0),
    )





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

    checkpoint_path = cfg.models.teacher.checkpoint_path
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    teacher_model_encoder_model = VisionEncoderDecoderModel.from_pretrained(
        checkpoint_path
    ).encoder

    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "?", "\u00e0", "\u00e1", "\u00e2", "\u00e3", "\u00e4", "\u00e5", "\u0101", "\u00e6", "\u00e7", "\u00e8", "\u00e9", "\u00ea", "\u00eb", "\u00ec", "\u00ed", "\u00ee", "\u00ef", "\u00f0", "\u00f1", "\u00f2", "\u00f3", "\u00f4", "\u00f5", "\u00f6", "\u014d", "\u00f8", "\u00f9", "\u00fa", "\u00fb", "\u00fc", "\u00fd", "\u00fe", "\u00ff", "\u00c0", "\u00c1", "\u00c2", "\u00c3", "\u00c4", "\u00c5", "\u00c6", "\u00c7", "\u00c8", "\u00c9", "\u00ca", "\u00cb", "\u00cc", "\u00cd", "\u00ce", "\u00cf", "\u00d0", "\u00d1", "\u00d2", "\u00d3", "\u00d4", "\u00d5", "\u00d6", "\u00d8", "\u00d9", "\u00da", "\u00db", "\u00dc", "\u00dd", "\u00de", "\u0178", "\u00ab", "\u00bb", "\u2014", "\u2019", "\u00b0", "\u2013", "\u0153", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", " "]
    tokenizer = utils.GenerationVocab(VOCAB=vocab)

    teacher_model = EncoderWrapperLearnedQueries(encoder=teacher_model_encoder_model, n_classes=len(tokenizer))
    teacher_model.load_state_dict(torch.load("checkpoints/TrOCR_Esposalles_CTC_With_Q_25.pt", weights_only=True))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    dataset = instantiate(
        cfg.data.dataset.Esposalles,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    generator = torch.Generator().manual_seed(2)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=generator
    )

    collate_fn = Collator(
        processor=processor, pad_token=processor.tokenizer.pad_token_id
    )

    train_dloader = DataLoader(
        train_dataset, collate_fn=collate_fn.ocr_collate_fn, **cfg.data.collator.train
    )

    student_model = LKIS.KoopmanNetwork(**cfg.models.student.inputs).to(device)
    dim_y = cfg.models.student.inputs.dim_y

    if cfg.models.student.finetune:
        load_checkpoint = os.path.join(f"checkpoints/{cfg.models.student.checkpoint_name}.pt")
        student_model.load_state_dict(torch.load(load_checkpoint, weights_only=True))

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    # Number of training steps
    num_epochs = cfg.setup.dict.epochs
    # Assuming you have a DataLoader called train_dataloader
    num_training_steps = num_epochs * len(train_dloader)

    # Create scheduler
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1000,  # Number of warmup steps
        num_training_steps=num_training_steps,
    )

    optimal_loss = 1e20
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{cfg.models.student.checkpoint_name}.pt"
    criterion = LKIS.KoopmanLoss

    for epoch in tqdm.tqdm(
        range(1, cfg.setup.dict.epochs), desc="Training Procedure", position=0, leave=False
    ):
        _, metrics = train_lkis_task(
            teacher_model_encoder=teacher_model,
            student_model=student_model,
            optimizer=optimizer,
            dataloader=train_dloader,
            criterion=criterion,
            epoch=epoch,
            num_accumulation_steps=cfg.setup.dict.num_accumulation_steps,
            log_every=cfg.setup.dict.log_every,
            wandb=wandb_logger,
            scheduler=scheduler,
            K=dim_y,
            full_grid=cfg.setup.dict.full_grid
        )

        train_loss = metrics["loss"] / len(train_dloader)
        print(f"Loss Epoch: {epoch} Value: {train_loss}")

        updated, optimal_loss = utils.update_and_save_model_pt(
            previous_metric=optimal_loss,
            actual_metric=train_loss,
            model=student_model,
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

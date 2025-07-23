from gc import enable
import torch
import tqdm
import os

from typing import Dict

from transformers import get_scheduler
from transformers import VisionEncoderDecoderModel, AutoProcessor, ViTModel, ViTConfig  # type: ignore
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from jiwer import cer, wer  # Make sure to import these

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
        dataset, [0.8, 0.1, 0.1], generator=generator
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




def main(cfg: DictConfig):

    if cfg.log_wandb == True:

        wandb.login()
        wandb_logger = wandb.init(
            project="Koopman",
            group="TrOCR",
            name="Evaluation at K: Ocr Guided by Koopman Operator",
            tags=["Koopman", "Invariant Subspace", "Learning"],
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
    model = REncoderDecoderModel.from_pretrained(cfg.modeling.base.checkpoint_path).to(device)
    print("Model loaded")

    train_dset, val_dset, test_dset = get_dataset(cfg, processor, transforms)
    test_dloader = get_dataloader(
        test_dset,
        collate_fn=get_collate_fn(processor).ocr_collate_fn,
        **cfg.data.collator.train,
    )


    for k in range(24):
    # Initialize metrics storage
        all_predictions = []
        all_ground_truths = []
        exact_matches = 0

        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dloader):
                inputs: dict = data["pixel_values"].to(device, non_blocking=True)
                text = data["text"]

                generated_ids = model.generate(**inputs, evaluate_at_t=True, t=k)
                decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)


                # Store predictions and ground truths for metric calculation
                for pred, gt in zip(decoded_text, text):
                    pred = pred.strip()
                    gt = gt.strip()

                    all_predictions.append(pred)
                    all_ground_truths.append(gt)

                    # Exact match accuracy
                    if pred == gt:
                        exact_matches += 1

                    #else:
                    #    print(f"Prediction: {pred}")
                    #    print(f"Ground Truth: {gt}")
                    #    print("\n" + "="*50)

                if batch_idx % 100 == 0:
                    print(f"Processed batch {batch_idx}")


        # Calculate final statistics
        total_samples = len(all_predictions)
        exact_match_accuracy = exact_matches / total_samples

        # Calculate CER and WER using jiwer
        character_error_rate = cer(all_ground_truths, all_predictions)
        word_error_rate = wer(all_ground_truths, all_predictions)

        results = {
            f'exact_match_accuracy_{k}': exact_match_accuracy,
            f'character_error_rate_{k}': character_error_rate,
            f'word_error_rate_{k}': word_error_rate,
            f'total_samples_{k}': total_samples
        }

        # Print results
        print("\n" + "="*50)
        print("OCR EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {results[f'total_samples_{k}']}")
        print(f"Exact match accuracy at k={k}: {results[f'exact_match_accuracy_{k}']:.4f}")
        print(f"Character error rate (CER): {results[f'character_error_rate_{k}']:.4f}")
        print(f"Word error rate (WER): {results[f'word_error_rate_{k}']:.4f}")
        print("="*50)

        # Log to wandb if enabled
        if wandb_logger is not None:
            wandb.log(results)


    wandb.finish()

    return results

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

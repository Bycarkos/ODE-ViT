# type: ignore
## Dataset Things
import torch.nn.functional as F
from torchvision.datasets.imagenet import ImageFolder
from datasets.collator import Collator
import utils


from torchvision.datasets import CIFAR100


from train import  train_classification_task_distillation
from test import test_classification_task
from models.ode_transformer_gpt import ViTNeuralODE
from transformers import ViTForImageClassification,  ViTImageProcessor


## Common packages
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

## Typing Packages

## Configuration Package
from omegaconf import DictConfig
from hydra import initialize, compose

## Experiment Tracking packages
import tqdm

## Common packages
import os
import wandb

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDistilTrainer(torch.nn.Module):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = torch.nn.KLDivLoss(reduction="batchmean")
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.teacher.eval()
        self.student.train()
        self.temperature = temperature
        self.lambda_param = lambda_param

        self.alpha_param = (1-self.lambda_param)/10


    def compute_loss(self, inputs, labels, return_outputs=True):
        student_output = self.student(**inputs, labels=labels, output_hidden_states=True)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs, output_hidden_states=True)

        last_cls_token = teacher_output.hidden_states[-1][:,0]
        last_state = student_output["states"][-1, :, 0]


        mse_loss = self.mse_loss(last_cls_token, last_state)
        soft_teacher = F.softmax(teacher_output["logits"] / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output["logits_dist"] / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_target_loss = student_output["loss"]

        ## Losses
        student_target_loss = (1. - self.lambda_param) * student_target_loss
        distillation_loss = self.lambda_param * distillation_loss

        # Calculate final loss
        loss = student_target_loss + distillation_loss + (mse_loss * self.alpha_param)

        return (loss, student_target_loss, mse_loss, student_output) if return_outputs else (loss, student_target_loss, mse_loss)



def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.log_wandb == True:
        wandb.login()
        wandb_logger = wandb.init(
            project=cfg.setup.wandb.project,
            group=cfg.setup.wandb.group,
            name=cfg.setup.wandb.name,
            config=dict(cfg.setup.dict).update(dict(cfg.modeling.student.inputs))
        )
    else:
        wandb_logger = False


    if cfg.data.dataset.name == "cifar100":
        train_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=True)
        validation_dataset = CIFAR100(root=cfg.data.dataset.dataset_path, download=False, train=False)
    else:
        train_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/train")
        validation_dataset = ImageFolder(root=cfg.data.dataset.dataset_path+"/val")


    model = ViTNeuralODE(**cfg.modeling.student.inputs).cuda()

    """
    save_path = "/data/users/cboned/checkpoints"
    checkpoint_name = f"{save_path}/EDO_DISTILLATION_VIT_ON_CIFAR100_mse_last_step.pt"

    weight_to_update = torch.load(checkpoint_name, weights_only=True)
    for w in weight_to_update.keys():
        if model.state_dict().get(w) is not None:
            model.state_dict()[w].data.copy_(weight_to_update[w])

    """

    teacher_model_checkpoint = cfg.modeling.teacher.checkpoint_path
    teacher_model = ViTForImageClassification.from_pretrained(teacher_model_checkpoint)

    model_parameters = (sum(p.numel() for p in model.parameters() if p.requires_grad))
    if wandb_logger:
        wandb_logger.log({"model_parameters": model_parameters})

    print("Training Model with a total parameters of", model_parameters/1e6, "Millions")

    print("Teacher Model Loaded Correctly")

    student_model = model.to(device)
    teacher_model = teacher_model.to(device)


    criterion = ImageDistilTrainer(teacher_model=teacher_model,
                                student_model=student_model,
                                temperature=cfg.setup.dict.temperature,
                                lambda_param=cfg.setup.dict.lambda_param)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    collator = Collator(processor)

    train_dloader = DataLoader(train_dataset, **cfg.data.collator.train, collate_fn=collator.classification_collate_fn)

    val_dloader = DataLoader(
        validation_dataset,
        collate_fn=collator.classification_collate_fn,
        **cfg.data.collator.val,
    )
    model.to(device)

    initial_lr = 1e-3
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=initial_lr,
        weight_decay=5e-2,
    )

    ## ** Scheduler
    steps_per_epoch = len(train_dloader)  # Number of batches per epoch
    total_steps = cfg.setup.dict.epochs * steps_per_epoch

    # Warmup steps (e.g., 10% of total steps)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    optimal_loss = 0.0
    save_path = "/data/users/cboned/checkpoints"
    os.makedirs(save_path, exist_ok=True)
    checkpoint_name = f"{save_path}/{cfg.modeling.student.checkpoint_name}.pt"



    if cfg.log_wandb == True:
        wandb_logger.watch(model, log="all")


    print("CREATING THE BASELINE METRIC VALUE\n STARTING TO EVALUATE FO THE FIRST TIME")
    _, loss_validation, acc_validation = test_classification_task(
        dataloader=val_dloader,
        model=student_model,
        criterion=criterion,
        wandb_logger=wandb_logger,
    )

    _, optimal_loss = utils.update_and_save_model_pt(
        previous_metric=optimal_loss,
        actual_metric=acc_validation,
        model=student_model,
        checkpoint_path=checkpoint_name,
        compare=">",
    )

    print(
        f"Validation Loss Epoch: {0} Value: {acc_validation} Optimal_loss: {optimal_loss}"
    )



    for epoch in tqdm.tqdm(
        range(1, cfg.setup.dict.epochs), desc="Training Procedure", position=0, leave=False
    ):

        _, train_loss = train_classification_task_distillation(
            dataloader=train_dloader,
            student_model=student_model,
            teacher_model=teacher_model,
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
            _, loss_validation, acc_validation = test_classification_task(
                dataloader=val_dloader,
                model=student_model,
                criterion=criterion,
                wandb_logger=wandb_logger,
            )

            updated, optimal_loss = utils.update_and_save_model_pt(
                previous_metric=optimal_loss,
                actual_metric=acc_validation,
                model=student_model,
                checkpoint_path=checkpoint_name,
                compare=">",
            )

            if updated:
                print(
                    f"Model Updated: Validation Loss Epoch: {0} Value: {acc_validation} Optimal_loss: {optimal_loss}"
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


    import os
    from PIL import Image, UnidentifiedImageError

    def delete_corrupted_images(root_dir):
        num_deleted = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()  # Verifies that it's an image
                    except (UnidentifiedImageError, IOError, OSError) as e:
                        print(f"Deleting corrupted image: {file_path} — Reason: {e}")
                        os.remove(file_path)
                        num_deleted += 1
        print(f"\nFinished. Deleted {num_deleted} corrupted images.")

    #delete_corrupted_images("/data/users/cboned/data/Generic/Imagenet1k/train")
    #delete_corrupted_images("/data/users/cboned/data/Generic/Imagenet1k/val")

    with initialize(version_base="1.3.2", config_path=args.config_path):
        cfg = compose(config_name=args.config_file)

    main(cfg=cfg)

import torch
import tqdm
import os
from collections import defaultdict


import utils
from datasets import Collator

from torch_pca import PCA
import os

from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

from transformers import ViTForImageClassification, AutoImageProcessor , ViTImageProcessor

import matplotlib.pyplot as plt
import numpy as np
import scienceplots


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

TEACHER_CHECKPOINT_PATH = "/data/users/cboned/checkpoints/Vit_CIFAR100_first_train.pt"
DATASET_PATH = "/data/users/cboned/data/Generic/cifar"


train_dataset = CIFAR100(root=DATASET_PATH, download=False, train=True)
validation_dataset = CIFAR100(root=DATASET_PATH, download=False, train=False)


teacher_model = ViTForImageClassification.from_pretrained(TEACHER_CHECKPOINT_PATH)


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
collator = Collator(processor)
teacher_model.to(device)
train_dloader = DataLoader(train_dataset,
                           shuffle=True,
                           batch_size=1,
                           pin_memory=False,
                           num_workers=0,
                           collate_fn=collator.classification_collate_fn
                           )


test_dloader = DataLoader(validation_dataset,
                          batch_size=1,
                          pin_memory=False,
                          num_workers=0,
                          shuffle=True,
                          collate_fn=collator.classification_collate_fn

)


traj_per_classes = defaultdict(list)

for batch_idx, data in tqdm.tqdm(
    enumerate(train_dloader),
    desc="Extracting CLS Embeddings",
    total=len(train_dloader),
):

    inputs: dict = data["pixel_values"]
    labels: list = data["labels"]

    with torch.no_grad():
        features = teacher_model(**inputs, output_hidden_states=True)["hidden_states"]

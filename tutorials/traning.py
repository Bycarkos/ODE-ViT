import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from model import Model
import vdp as vp
from vdp import VDPTrajDataset
import loka_volterra as lv
from loka_volterra import LVtrajDataset

import lorenz as lz
from lorenz import LorenzTrajDataset

import Koopa as kp

import time
from tqdm import tqdm


import matplotlib.pyplot as plt


class Config():

    def __init__(self,
        mask_spectrum:torch.Tensor,
        enc_in: int,
        input_len: int,
        pred_len: int,
        seg_len: int,
        num_blocks: int,
        dynamic_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        multistep: bool):

        self.mask_spectrum = mask_spectrum
        self.enc_in = enc_in
        self.seq_len = input_len
        self.seg_len = seg_len
        self.pred_len = pred_len
        self.num_blocks = num_blocks
        self.dynamic_dim = dynamic_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.multistep = False


if __name__ == "__main__":

    DYNAMICS_DIM = 8
    HIDDEN_DIM = 128
    INPUT_DIM = 3
    NUM_LAYERS = 1
    TRAINING_SERIES_LENGTH = 350
    SEGMENT_LENGTH = 35
    GLOBAL_SERIES_LENGTH = 1024
    SAMPLE_TRAJECTORY = 1000
    PRED_LENGTH = 400

    CONTEXTUAL_MODULES = TRAINING_SERIES_LENGTH // SEGMENT_LENGTH

    # generate raw trajectories
    trajectories = lz.generate_data(sample_trajectories=SAMPLE_TRAJECTORY, traj_resolution=GLOBAL_SERIES_LENGTH)

    # wrap into datasetlz
    dataset = LorenzTrajDataset(
        trajectories.permute(1, 0, 2),
        input_len=TRAINING_SERIES_LENGTH,
        pred_len= PRED_LENGTH
    )

    # train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    amps = 0.0
    for data in train_loader:
        time_series = data[0]
        amps += abs(torch.fft.rfft(time_series, dim=1)).mean(dim=0).mean(dim=1)

    mask_spectrum = amps.topk(int(amps.shape[0]*0.2)).indices

    model = Model(  input_dim=INPUT_DIM,
                    num_layers=NUM_LAYERS,
                    hidden_dim=HIDDEN_DIM,
                    segment_length=SEGMENT_LENGTH,
                    number_of_segments=CONTEXTUAL_MODULES,
                    dynamics_dim=DYNAMICS_DIM
    )
#    conf = Config(mask_spectrum=mask_spectrum,
#    enc_in=3,
#    input_len=1024 + SEGMENT_LENGTH,
#    pred_len=1024,
#    seg_len=SEGMENT_LENGTH,
#    num_blocks=NUM_LAYERS,
#    dynamic_dim=3,
#    hidden_dim=256,
#    hidden_layers=3,
#    multistep=False

#    )

    model = model.to('cuda')

    epochs = 1500
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logs = []
    horizon = 50
    best_model = 1e4
    horizon_consistency = 100
    for ep in tqdm(range(1, epochs)):
        model.train()
        ep_loss = 0.0
        ep_kpm = 0.0
        ep_rec = 0.0
        ep_pred = 0.0
        n_batches = 0

        if (ep % 100) == 0 :
            horizon += 10

        for batch_idx, batch in enumerate(train_loader):
            #print(f"Epoch {ep}/{epochs}  Batch {batch_idx+1}/{len(train_loader)}", end="\r")
            optimizer.zero_grad()


            x, Y, seq = batch
            x = x.to('cuda')
            Y = Y.to('cuda')

            output = model(x, pred_length=horizon, labels=Y[:, :horizon, :])

            loss = output["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_rec += output["loss_rec"].item()
            ep_pred += output["loss_pred"].item()
            ep_kpm += output["loss_k"].item()
            n_batches += 1


        ep_loss /= n_batches
        ep_rec /= n_batches
        ep_pred /= n_batches
        ep_kpm /= n_batches

        logs.append((ep, ep_loss, ep_rec, ep_pred))
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep}/{epochs}  Loss {ep_loss:.6f}  rec {ep_rec:.6f}  pred {ep_pred:.6f} kpm {ep_kpm:.6f}  horizon {horizon}")

#            torch.save(model.state_dict(), f"model_ep{ep}.pt")
            if ep_loss < best_model:
                torch.save(model.state_dict(), f"model_final.pt")
                best_model = ep_loss


    # evaluate on validation set
    model.eval()
    val_loss = 0.0
    val_kpm = 0.0
    val_rec = 0.0
    val_pred = 0.0
    n_batches = 0
    horizon = 16
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x, Y, seq = batch
            x = x.to('cuda')
            Y = Y.to('cuda')

            output = model(x, pred_length=horizon, labels=Y[:, :horizon, :])

            loss = output["loss"]

            val_loss += loss.item()
            val_rec += output["loss_rec"].item()
            val_pred += output["loss_pred"].item()
            n_batches += 1

        val_loss /= n_batches
        val_rec /= n_batches
        val_pred /= n_batches

        print(f"Validation  Loss {val_loss:.6f}  rec {val_rec:.6f}  pred {val_pred:.6f}  horizon {horizon}")

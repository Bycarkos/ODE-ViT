import math
from sympy import im
import torch
import torch.nn as nn
from Koopa import TimeVarKP, KPLayerApprox, KPLayer, MLP
from attentions import ScaledDotProductAttention
from typing import Optional

class SegmentEncoder(nn.Module):

    def __init__(self,
        input_dim,
        hidden_dim,
        num_layers=2,
        segment_length=10,
        number_of_segments: int=4,
        K_dim: int=8):
        super(SegmentEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.segment_length = segment_length
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.number_of_segments = number_of_segments

        self.q, self.k, self.v = nn.Linear(input_dim, hidden_dim), nn.Linear(input_dim, hidden_dim), nn.Linear(input_dim, hidden_dim)
        self.memory = nn.ModuleList([nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)])

        for i in range(number_of_segments-1):
            self.memory.append(nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True))

        self.time_var_encoding =  MLP(f_in=segment_length*input_dim, f_out=hidden_dim, hidden_layers=3, hidden_dim=hidden_dim, activation="tanh")

        self.observation_encoder = MLP(f_in=3*hidden_dim, f_out=K_dim, hidden_layers=2, hidden_dim=hidden_dim)
        nn.Linear(3*hidden_dim, K_dim)

        self.activation = nn.Tanh()

        self.att = ScaledDotProductAttention()

    def forward(self, x_enc: torch.Tensor):
        # x_enc: (batch_size, seq_len, input_dim)
        B, T, _ = x_enc.shape
        x_enc_chunked = x_enc.chunk(self.number_of_segments, dim=1)
        x_enc_chunked = torch.stack(x_enc_chunked)

        memory_out, (hn, cn) = self.memory[0](x_enc_chunked[0])
        memories = [memory_out]
        for i in range(1, len(self.memory)):
            memory_out, (hn, cn) = self.memory[i](x_enc_chunked[i], (hn, cn))
            memories.append(memory_out)

        memories = torch.stack(memories)
        memories = memories.permute(1, 0, 2, 3)
        memories = memories.reshape(B, T, -1)


        q, k , v = self.q(x_enc), self.k(x_enc), self.v(x_enc)

        out, att = self.att(q, k, v)
        series_sample = torch.cat([memories, out], dim=-1) # (batch_size, 3*hidden_dim) /// 3*hidden_dim = forward || backward || combination encoding

        observation = self.observation_encoder(series_sample)
        observation = self.activation(observation)

        return observation


class SegmentDecoder(nn.Module):
    def __init__(self, encoder_input_dim: int, hidden_dim:int,  K_dim: int=8):
        super().__init__()
        self.encoder_input_dim = encoder_input_dim
        self.K_dim = K_dim

        self.decoder = MLP(f_in=K_dim, f_out=encoder_input_dim, hidden_layers=4, hidden_dim=hidden_dim, activation="tanh")


    def forward(self, encoded_time_series: torch.Tensor):

        return self.decoder(encoded_time_series)

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                num_layers=2,
                segment_length=10,
                dynamics_dim=8,
                number_of_segments: int = 4,
                multistep=False):
        super(Model, self).__init__()


        # input_dim, hidden_dim, num_layers=2, segment_length=10, K_dim: int=8
        self.encoder = SegmentEncoder(input_dim, hidden_dim, num_layers, number_of_segments=number_of_segments, segment_length=segment_length, K_dim=dynamics_dim)
        self.seg_len = segment_length

        self.multistep = multistep
        if not multistep:
            self.kp_layer = KPLayer()
        else:
            self.kp_layer = KPLayerApprox()

        self.decoder = SegmentDecoder(encoder_input_dim=input_dim, hidden_dim=hidden_dim, K_dim=dynamics_dim)

    def forward(self, time_series: torch.Tensor, pred_length: int=1, labels: Optional[torch.Tensor]=None):

        # x: (batch_size, seq_len, input_dim)

        B, L, C = time_series.shape

        observation = self.encoder(time_series)


        kp_out_rec, kp_out_horizon = self.kp_layer(observation, pred_len=pred_length)

        rec_series = self.decoder(kp_out_rec)

        rec_prediction_series = self.decoder(kp_out_horizon)



        if labels is not None:
            loss_rec = nn.MSELoss(reduction="none")(time_series, rec_series).mean(-1)
            loss_rec = (loss_rec).mean()

            loss_pred = nn.MSELoss(reduction="none")(labels, rec_prediction_series).mean(-1)
            loss_pred = (loss_pred).mean()

            #loss = loss_rec + loss_pred
            # Koopman consistency loss: K g(x_t) ≈ g(x_{t+1})
            g_xt = observation[:, :-1, :]          # (B, L-1, E)
            g_xt1 = observation[:, 1:, :]          # (B, L-1, E)
            K = self.kp_layer.K                   # (B, E, E)

            g_xt_proj = torch.bmm(g_xt, K)        # (B, L-1, E)
            loss_k = nn.MSELoss(reduction="none")(g_xt_proj, g_xt1).mean(-1)
            loss_k = (loss_k).mean()

            # Total loss
            loss = loss_rec + loss_pred #+ loss_k

        return {
            "rec_series": rec_series,
            "pred_series": rec_prediction_series,
            "loss": loss if labels is not None else None,
            "loss_rec": loss_rec if labels is not None else None,
            "loss_pred": loss_pred if labels is not None else None,
            "loss_k": loss_k if labels is not None else None
        }



if __name__ == "__main__":
    model = Model(input_dim=2, hidden_dim=64, segment_length=20, dynamics_dim=8, multistep=False)
    x = torch.randn(16, 20, 2)
    model(x, pred_length=5)

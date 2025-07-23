import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence, Callable, Dict
from dataclasses import dataclass
import numpy as np


from time_emb import TimeEmbeding


class TemporalLinear(nn.Module):
    def __init__(self, input_dim,
    output_dim,
    time_embed_dim, sinusoidal_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_embed_dim = time_embed_dim
        self.use_bias = use_bias

        # Time embedding MLP
        self.lin1 = nn.Linear(sinusoidal_dim, time_embed_dim)
        self.lin2 = nn.Linear(time_embed_dim, time_embed_dim)

        # Project time embedding into weights for input -> output
        self.f_W = nn.Linear(time_embed_dim, input_dim * output_dim)

        if use_bias:
            self.f_b = nn.Linear(time_embed_dim, output_dim)
        else:
            self.f_b = None

    def forward(self, time_embed, x):
        """
        time_embed: Tensor of shape [sinusoidal_dim]
        x: Tensor of shape [B, input_dim]
        """

        # MLP for time embedding
        t = F.silu(self.lin1(time_embed))
        t = self.lin2(t)  # shape: [B, time_embed_dim]

        # Generate dynamic weights: shape [B, input_dim * output_dim]
        W = self.f_W(t)  # shape: [B, input_dim * output_dim]
        W = W.view(-1, self.input_dim, self.output_dim).repeat(x.size(0), 1, 1)  # [B, input_dim, output_dim]

        # Apply dynamic linear transformation: x @ W
        out = torch.bmm(x, W).squeeze(1)  # [B, output_dim]

        if self.f_b is not None:
            b = self.f_b(t)  # [B, output_dim]
            out = out + b

        return out


class TemporalLayerNorm(nn.Module):
    def __init__(self,
                 input_dim: int,
                 time_embed_dim: int,
                 eps: float = 1e-5,
                 use_weight: bool = True,
                 use_bias: bool = True):
        """
        Time-dependent LayerNorm implemented via MLP conditioned on time embeddings.
        """
        super().__init__()

        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.input_dim = input_dim

        # MLP to process time embeddings
        self.lin1 = nn.Linear(time_embed_dim, time_embed_dim)
        self.lin2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.activation = nn.SiLU()

        # Time-dependent weights and biases for LayerNorm
        if use_weight:
            self.f_weight = nn.Linear(time_embed_dim, input_dim)
        else:
            self.f_weight = None

        if use_bias:
            self.f_bias = nn.Linear(time_embed_dim, input_dim)
        else:
            self.f_bias = None

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, ..., D] where D is the input_dim
        time_embed: Tensor of shape [B, time_embed_dim]
        """

        # Compute time-dependent modulations
        t = self.lin1(time_embed)
        t = self.activation(t)
        t = self.lin2(t)

        # Compute normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)

        if self.f_weight is not None:
            weight = self.f_weight(t).unsqueeze(1) + 1.0  # [B, 1, D]
            out = out * weight

        if self.f_bias is not None:
            bias = self.f_bias(t).unsqueeze(1)  # [B, 1, D]
            out = out + bias


        return out.squeeze(1)


def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0):
    """Scaled dot-product attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)

    output = torch.matmul(attention_weights, value)
    return output


class Attention(nn.Module):
    """Multi-head attention with time-dependent weights"""

    def __init__(self, sinusoidal_dim: int,
                        time_embed_dim: int,
                        embedding_dim: int,
                        n_head: int = 1,
                        use_bias: bool = True):

        super().__init__()

        self.head_size = embedding_dim
        self.n_head = n_head

        """
        def __init__(self, input_dim,
        output_dim,
        time_embed_dim, sinusoidal_dim, use_bias=True)
        """
        # Time-dependent attention projections
        self.c_attn = TemporalLinear(
            sinusoidal_dim=sinusoidal_dim,
            time_embed_dim=time_embed_dim,
            input_dim=embedding_dim,
            output_dim=3 * embedding_dim * n_head,  # qkv combined
            use_bias=use_bias
        )

        self.c_proj = TemporalLinear(
            sinusoidal_dim=sinusoidal_dim,
            time_embed_dim=time_embed_dim,
            input_dim=embedding_dim,
            output_dim=embedding_dim,  # qkv combined
            use_bias=use_bias
        )

        self.attn_dropout = nn.Dropout(0.2)

    def forward(self, time_embed: torch.Tensor, x: torch.Tensor, mask=None) -> torch.Tensor:


        batch_size, seq_len, _ = x.size()
        # Generate Q, K, V using time-dependent weights
        #
        qkv = self.c_attn(time_embed, x)

        qkv = qkv.view(batch_size, seq_len, 3, self.n_head, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_head, seq_len, head_size]

        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(
            query, key, value, mask,
            dropout_p=0.2 if self.training else 0.0
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.head_size)
        attn_output = self.c_proj(time_embed, attn_output)

        return attn_output

class MLP(nn.Module):
    """Multi-layer perceptron with time-dependent weights"""

    def __init__(self, sinusoidal_dim: int, time_embed_dim: int, embedding_dim: int, use_bias:bool = True):
        super().__init__()

        self.c_fc = TemporalLinear(
            sinusoidal_dim=sinusoidal_dim,
            time_embed_dim=time_embed_dim,
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            use_bias=use_bias
        )

        self.c_proj = TemporalLinear(
            sinusoidal_dim=sinusoidal_dim,
            time_embed_dim=time_embed_dim,
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            use_bias=use_bias
        )


        self.act = F.gelu


    def forward(self, time_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(time_embed, x)
        x = self.act(x)
        x = self.c_proj(time_embed, x)
        return x



class Block(nn.Module):
    """Transformer block with time-dependent components"""

    def __init__(self,
    sinusoidal_dim: int,
    time_embed_dim: int, embedding_dim: int,
    n_head: int= 1,
    use_bias: bool=True):


        super().__init__()

        self.attn_ln = TemporalLayerNorm(
            input_dim=sinusoidal_dim,
            time_embed_dim=time_embed_dim,
            use_bias=use_bias
        )

        self.attn = Attention(sinusoidal_dim, time_embed_dim, embedding_dim, use_bias=use_bias, n_head=n_head)

        self.mlp_ln = TemporalLayerNorm(
            input_dim=sinusoidal_dim,
            time_embed_dim=time_embed_dim,
            use_bias=use_bias
        )

        self.mlp = MLP(sinusoidal_dim, time_embed_dim, embedding_dim)
        self.resid_dropout = nn.Dropout(0.2)

    def forward(self, time_embed: torch.Tensor, x: torch.Tensor, mask=None) -> torch.Tensor:
        # Self-attention with residual connection

        attn_output = self.attn(time_embed, self.attn_ln(time_embed, x), mask)
        attn_output = self.resid_dropout(attn_output)

        # MLP with residual connection
        mlp_output = self.mlp(time_embed, self.mlp_ln(time_embed, x))
        mlp_output = self.resid_dropout(mlp_output)

        return attn_output + mlp_output


class NeuralOdeEncoder(nn.Module):
    """Neural ODE Transformer implementation"""

    def __init__(self,
                 time_embed_dim: int = 768,
                 sinusoidal_dim: int = 768,
                 embedding_dim: int = 768,
                 steps: int= 12,
                 n_head: int = 1,
                 ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.sinusoidal_dim = sinusoidal_dim

        self._head = n_head

        # Time embedding (assuming you have this implemented)
        self.time_embedding = TimeEmbeding(sinusoidal_dim, time_embed_dim, learnable_sinusoidal=True)

        # Single block that will be applied multiple times
        self.block = Block(sinusoidal_dim, time_embed_dim, embedding_dim=embedding_dim)

        # Final layer norm
        self.ln_f = nn.LayerNorm(time_embed_dim)
        self.steps = steps
        # Step size for ODE integration
        self.dt = 1.0 / steps

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:

        # Generate time points
        #
        t = torch.arange(1, self.steps + 1, dtype=x.dtype, device=x.device) * self.dt

        time_embeds = self.time_embedding(t)  # [n_layer, time_embed_dim]

        # Neural ODE integration using Euler method
        for i in range(self.steps):
            time_embed = time_embeds[i]

            # Compute derivative
            dx_dt = self.block(time_embed, x, attn_mask)

            # Euler step: x_{t+1} = x_t + dt * dx_dt
            x = x + self.dt * dx_dt

        x = self.ln_f(x).tanh()

        return x


# Example usage:
if __name__ == "__main__":
    # Create configuration
    # Create model
    model = NeuralOdeEncoder(time_embed_dim=768, sinusoidal_dim=768, steps=12, n_head=8)

    # Example forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.rand((batch_size, seq_len, 768), dtype=torch.float32)

    # Forward pass
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]

    # Compute loss
    labels = input_ids.clone()
    loss = model.compute_loss(input_ids, labels)
    print(f"Loss: {loss.item()}")

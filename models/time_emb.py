import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, x, max_period=10000, scale=1000):
        """Return [x, sin(w*x), cos(w*x)]"""
        # time interval solving differential is small, rescale it
        x = x * scale

        # Create frequency values
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=x.dtype, device=x.device) / half_dim
        )

        # Broadcast x and freqs for element-wise multiplication
        # x shape: (...) -> (..., 1)
        # freqs shape: (half_dim,) -> (1, ..., 1, half_dim)
        x_expanded = x.unsqueeze(-1)  # Add dimension for broadcasting
        freqs = freqs.view(*([1] * x.ndim), -1)  # Reshape freqs for broadcasting

        args = x_expanded * freqs

        # Create sinusoidal embeddings
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)

        # Concatenate [x/scale, sin(args), cos(args)]
        x_normalized = (x / scale).unsqueeze(-1)  # Original x scaled back
        fouriered = torch.cat([x_normalized, sin_emb, cos_emb], dim=-1)

        return fouriered


class LearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.weights = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        """Return [x, sin(w*x), cos(w*x)]"""
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # (..., 1)
        weights = self.weights.view(*([1] * x.ndim), -1)  # (1, ..., 1, dim)

        freqs = x_expanded * weights * 2 * math.pi

        # Create sinusoidal embeddings
        sin_emb = torch.sin(freqs)
        cos_emb = torch.cos(freqs)

        # Concatenate [x, sin(freqs), cos(freqs)]
        x_expanded_for_concat = x.unsqueeze(-1)  # Keep original x
        import pdb; pdb.set_trace()
        fouriered = torch.cat([x_expanded_for_concat, sin_emb, cos_emb], dim=-1)
        return fouriered


class TimeEmbedding(nn.Module):
    """Alternative architecture that shares Sinusoidal and MLP"""

    def __init__(
        self,
        sinusoidal_dim,
        embed_dim,
        multiplier=1,
        dropout=0.1,
        learnable_sinusoidal=False
    ):
        super().__init__()

        # Initialize sinusoidal embedding
        if learnable_sinusoidal:
            self.sinusoidal = LearnedSinusoidalPosEmb(sinusoidal_dim)
        else:
            self.sinusoidal = SinusoidalPosEmb(sinusoidal_dim)

        # Calculate input dimension after sinusoidal embedding
        # [x, sin(args), cos(args)] = 1 + sinusoidal_dim + sinusoidal_dim = 2*sinusoidal_dim + 1
        fourier_dim = 2 * sinusoidal_dim + 1
        intermediate_dim = embed_dim * multiplier

        # Linear layers
        self.lin1 = nn.Linear(fourier_dim, intermediate_dim)
        self.lin2 = nn.Linear(intermediate_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t):
        # Transform t into sinusoidal embeddings

        x = self.sinusoidal(t)

        # Pass through MLP
        x = self.lin1(x)
        x = F.silu(x)  # SiLU activation
        x = self.dropout(x)
        x = self.lin2(x)

        return x


class ScaleShift(nn.Module):

    def __init__(self, embed_dim, out_dim):
        super().__init__()
        # Output 2 values per out_dim (scale and shift)
        self.lin = nn.Linear(embed_dim, out_dim * 2)

    def forward(self, x):
        x = F.silu(x)
        x = self.lin(x)

        # Split into scale and shift
        # Reshape to (..., out_dim, 2) then split along last dimension
        *batch_dims, total_dim = x.shape
        out_dim = total_dim // 2
        x = x.view(*batch_dims, out_dim, 2)
        scale, shift = x.unbind(-1)  # Split along the last dimension

        return scale, shift


# Example usage and testing
if __name__ == "__main__":
    # Test SinusoidalPosEmb
    sinusoidal_emb = SinusoidalPosEmb(dim=64)
    t = torch.randn(10)
    sinusoidal_out = sinusoidal_emb(t)
    print(f"Sinusoidal embedding shape: {sinusoidal_out.shape}")  # Should be (10, 129)

    # Test LearnedSinusoidalPosEmb
    learned_emb = LearnedSinusoidalPosEmb(dim=64)
    learned_out = learned_emb(t)
    print(f"Learned sinusoidal embedding shape: {learned_out.shape}")  # Should be (10, 129)

    # Test AlternativeTimeEmbedding
    time_emb = TimeEmbeding(
        sinusoidal_dim=64,
        embed_dim=128,
        multiplier=4,
        learnable_sinusoidal=False
    )
    time_out = time_emb(t)
    print(f"Time embedding shape: {time_out.shape}")  # Should be (10, 128)

    # Test ScaleShift
    scale_shift = ScaleShift(embed_dim=128, out_dim=64)
    scale, shift = scale_shift(time_out)
    print(f"Scale shape: {scale.shape}, Shift shape: {shift.shape}")  # Both should be (10, 64)

## Common packages
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForImageClassification
from transformers.models.resnet.modeling_resnet import ResNetBasicLayer, ResNetBottleNeckLayer

from typing import Optional
from .time_emb import TimeEmbedding

class EEMTransformerBlock(nn.Module):
    def __init__(self, transformer_block, hidden_dim):
        super().__init__()
        self.transformer = transformer_block  # transformer block ja preentrenat
        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):

        for layer in self.transformer.layer:
            self_attention_outputs = layer.attention(layer.layernorm_before(x), output_attentions=False)
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]

            ## first hidden_state residual connection dynamics
            x = x + self.alpha * attention_output

            layer_output = layer.layernorm_after(x)
            layer_output = layer.intermediate(layer_output)
            layer_output = layer.output(layer_output, x)

            ## Second MLP
            x = x + (self.beta*layer_output)
            outputs = (layer_output,) + outputs

        return outputs[0]

class NM_ODE_Gamma(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, steps=10, T=1.0):
        super().__init__()
        self.steps =  steps
        self.dt = T / steps
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.I = nn.Parameter(torch.zeros(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x = CLS token
        p = self.I
        Wx = self.linear(x)
        for _ in range(self.steps):
            dp = -p + F.gelu(Wx + self.gamma * p)
            p = p + self.dt * dp
        return self.classifier(p)

class EDOEncoderWrapperWithMemory(nn.Module):
    def __init__(self, encoder, n_classes, hidden_dim=768, steps: int= 12):
        super().__init__()



        self.encoder = encoder.vit  # e.g. a ViT or CNN backbone returning .last_hidden_state [B, N, d]
        self.hidden_dim = hidden_dim
        self.num_queries = n_classes
        self.EEM = EEMTransformerBlock(encoder.vit.encoder, hidden_dim=hidden_dim)
        self.NM = NM_ODE_Gamma(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_classes, steps=steps//2)

        self.steps = steps
        self.n_classes = n_classes
        self.dt = torch.tensor(1.0 / steps, device=self.device)
        self.time_embedding = TimeEmbedding(768, 768, learnable_sinusoidal=True)
        t_values = torch.arange(1, steps + 1, dtype=torch.float32, device=self.encoder.device)
        self.register_buffer('t_values', t_values)
        self._time_conditioning = nn.Linear(2*hidden_dim, hidden_dim)
        self._act = nn.PReLU()

        # Final classifier: project to vocab
        if hasattr(self.encoder, "classifier"):
            self.projector = self.encoder.classifier
        else:
            self.projector = nn.Linear(hidden_dim, n_classes)

        ## freezing parameters
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, pixel_values, labels: Optional[torch.Tensor] = None, output_hidden_states: bool = True):
        device = pixel_values.device
        dt = self.dt
        t = self.t_values.to(device) * dt
        temporal_embedding = self.time_embedding(t)

        hidden_state = self.encoder.embeddings(pixel_values)  # [B, N, D]
        outputs_hidden_states = [hidden_state]

        B, T, D = hidden_state.shape
        for i in range(self.steps):
            hidden_state = self._time_conditioning(torch.cat([hidden_state, temporal_embedding[i].repeat(B, T, 1)], dim=-1))
            hidden_state = self._act(hidden_state)
            hidden_state = self.EEM(hidden_state)

            outputs_hidden_states.append(hidden_state)

        last_hidden_state = self.encoder.layernorm(hidden_state)
        logits = self.NM(last_hidden_state[:, 0])

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs_hidden_states
        }

class EDOEncoderWrapperClassification(nn.Module):
    def __init__(self, encoder, n_classes, hidden_dim=768, steps: int= 12):
        super().__init__()
        self.encoder = encoder.vit  # e.g. a ViT or CNN backbone returning .last_hidden_state [B, N, d]
        self.hidden_dim = hidden_dim
        self.num_queries = n_classes

        self.steps = steps
        self.n_classes = n_classes
        self.dt = torch.tensor(1.0 / steps, device=self.device)
        self.time_embedding = TimeEmbedding(768, 768, learnable_sinusoidal=True)
        self.encoder.pooler = torch.nn.Linear(768, 768)
        t_values = torch.arange(1, steps + 1, dtype=torch.float32, device=self.encoder.device)
        self.register_buffer('t_values', t_values)
        self._time_conditioning = nn.Linear(2*hidden_dim, hidden_dim)
        self._act = nn.PReLU()
        # Final classifier: project to vocab
        if hasattr(self.encoder, "classifier"):
            self.projector = self.encoder.classifier
        else:
            self.projector = nn.Linear(hidden_dim, n_classes)

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, pixel_values, labels: Optional[torch.Tensor] = None, output_hidden_states: bool = True):
        device = pixel_values.device
        dt = self.dt
        t = self.t_values.to(device) * dt
        temporal_embedding = self.time_embedding(t)

        hidden_state = self.encoder.embeddings(pixel_values)  # [B, N, D]
        outputs_hidden_states = [hidden_state]
        intermediate_losses = []
        B, T, D = hidden_state.shape
        for i in range(self.steps):
            hidden_state = self._time_conditioning(torch.cat([hidden_state, temporal_embedding[i].repeat(B, T, 1)], dim=-1))
            hidden_state = self._act(hidden_state)
            encoder_output = self.encoder.encoder(hidden_state, output_hidden_states=False, return_dict=True)
            euler_step = encoder_output.last_hidden_state
            hidden_state = hidden_state + (euler_step * dt)
            outputs_hidden_states.append(hidden_state)
            logits_step = self.projector(hidden_state[:, 0])  # Classification at step i
            if (labels is not None) and (i >= 6):
                loss_step = torch.nn.functional.cross_entropy(logits_step, labels)
                intermediate_losses.append(loss_step)

        last_hidden_state = self.encoder.layernorm(hidden_state)
        logits = self.projector(last_hidden_state[:, 0])  # CLS token

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs_hidden_states
        }

class EncoderWrapperLearnedQueries(nn.Module):
    def __init__(self, encoder, n_classes, hidden_dim=768):
        super().__init__()
        self.encoder = encoder  # e.g. a ViT or CNN backbone returning .last_hidden_state [B, N, d]
        self.hidden_dim = hidden_dim
        self.num_queries = n_classes
        self.n_classes = n_classes
        # Learnable queries (T, d) —> expand to (B, T, d)
        self.learned_queries = nn.Embedding(n_classes, hidden_dim)

        # Cross-attention: queries attend to encoder output
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Final classifier: project to vocab
        self.projector = nn.Linear(hidden_dim, n_classes)

    def forward(self, pixel_values, topk:int=25, output_hidden_states:bool=True):
        encoder_output = self.encoder(pixel_values,  output_hidden_states=output_hidden_states)  # assumed shape: [B, N, d]

        if hasattr(encoder_output, 'last_hidden_state'):
            encoder_feats = encoder_output.last_hidden_state  # [B, N, d]
        else:
            encoder_feats = encoder_output  # in case encoder just returns tensor

        hidden_states = encoder_output.get("hidden_states", None)
        B = encoder_feats.size(0)

        # Expand queries: [T, d] → [B, T, d]
        queries = self.learned_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, T, d]

        # Cross-attention (query attends to encoder_feats as key and value)
        attended, attn_weights = self.cross_attention(query=queries, key=encoder_feats, value=encoder_feats)

        # Project to vocab (logits for each character position)
        logits = self.projector(attended)[:, :topk]

        return dict(logits=logits,
                    mask_logits=None,
                    hidden_states=hidden_states)

class EDOEncoderWrapperLearnedQueries(nn.Module):

    def __init__(self, encoder, n_classes, hidden_dim=768, steps: int= 12):
        super().__init__()
        self.encoder = encoder  # e.g. a ViT or CNN backbone returning .last_hidden_state [B, N, d]
        self.hidden_dim = hidden_dim
        self.num_queries = n_classes

        self.steps = steps
        self.n_classes = n_classes
        self.dt = torch.tensor(1.0 / steps, device=self.device)
        self.time_embedding = TimeEmbedding(768, 768, learnable_sinusoidal=True)
        self.encoder.pooler = torch.nn.Linear(768, 768)
        t_values = torch.arange(1, steps + 1, dtype=torch.float32, device=self.encoder.device)
        self.register_buffer('t_values', t_values)

        # Learnable queries (T, d) —> expand to (B, T, d)
        self.learned_queries = nn.Embedding(n_classes, hidden_dim)

        # Cross-attention: queries attend to encoder output
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Final classifier: project to vocab
        self.projector = nn.Linear(hidden_dim, n_classes)

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, pixel_values, topk:int=25, output_hidden_states:bool=True):
        t = self.t_values * self.dt
        temporal_embedding = self.time_embedding(t)


        hidden_state = self.encoder.embeddings(pixel_values)
        outputs_hidden_states = [hidden_state.clone()]

        for i in range(self.steps):
            encoder_output = self.encoder.encoder(hidden_state)  # assumed shape: [B, N, d]
            euler_step = encoder_output.last_hidden_state
            euler_step += temporal_embedding[i]
            hidden_state = (hidden_state + (euler_step * self.dt))
            outputs_hidden_states.append(hidden_state.clone())

        last_hidden_state = hidden_state.clone()
        last_hidden_state = self.encoder.layernorm(last_hidden_state)
        B = hidden_state.size(0)

        # Expand queries: [T, d] → [B, T, d]
        queries = self.learned_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, T, d]

        # Cross-attention (query attends to encoder_feats as key and value)
        attended, attn_weights = self.cross_attention(query=queries, key=last_hidden_state, value=last_hidden_state)

        # Project to vocab (logits for each character position)
        logits = self.projector(attended)[:, :topk]  # [B, T, vocab_size]

        return dict(logits=logits,
                    hidden_states=outputs_hidden_states)


class EncoderWrapper(torch.nn.Module):

    def __init__(self, encoder, n_classes):
        super().__init__()
        self.encoder = encoder
        self.projector = torch.nn.Linear(768, n_classes)

    def forward(self, pixel_values):
        output = self.encoder(pixel_values)
        output = self.projector(output.last_hidden_state.relu())
        return output



class ResNetForKoopmanEstimation(torch.nn.Module):
    def __init__(self, model_name="microsoft/resnet-50", out_size: tuple= (1, 1)):
        super().__init__()
        # Load pretrained model
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        # For capturing residual states
        self.pre_residuals = []
        self.post_residuals = []
        self.residuals = []

        self.out_size = out_size

        self.pooling = nn.AdaptiveMaxPool2d(out_size)
        # Register hooks on all residual blocks
        for module in self.model.resnet.encoder.modules():
            if isinstance(module, (ResNetBasicLayer, ResNetBottleNeckLayer)):
                module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if len(self.residuals) == 0:
            self.residuals.append(input[0].detach())
            self.residuals.append(output.detach())
        else:
            self.residuals.append(output.detach())

        self.pre_residuals.append(input[0].detach().cpu())
        self.post_residuals.append(output.detach().cpu())

    def forward(self, pixel_values, labels=None, output_hidden_states=False):
        # Clear previous states
        self.pre_residuals.clear()
        self.post_residuals.clear()
        self.residuals.clear()
        output = self.model(pixel_values=pixel_values, labels=labels, return_dict=True)

        residual_states = []

        for residual in self.residuals:

            residual = residual.max(dim=1).values
            out = self.pooling(residual)
            out = out.flatten(1)
            out = F.normalize(out, p=2, dim=1) # / (out.shape[1]**0.5)
            residual_states.append(out)



        output["hidden_states"] = residual_states

        return output

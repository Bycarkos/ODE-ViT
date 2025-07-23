import torch
from typing import Optional, Tuple
from transformers import ViTConfig, ViTModel, VisionEncoderDecoderModel

from  dataclasses import dataclass
from .time_emb import TimeEmbeding

import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"



def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids



def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

@dataclass
class Seq2SeqLMOutput():
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.EncoderDecoderCache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    def keys(self):
        return vars(self).keys()




class REncoderDecoderModel(VisionEncoderDecoderModel):
    def __init__(self, cfg, encoder=None, decoder=None, steps: int= 12, **kwargs):
        super(REncoderDecoderModel, self).__init__(cfg)
        self.steps = steps
        self.dt = torch.tensor(1.0 / steps, device=self.device)
        self.time_embedding = TimeEmbeding(768, 768, learnable_sinusoidal=True)
        self.encoder.pooler = torch.nn.Linear(768, 768)
        t_values = torch.arange(1, steps + 1, dtype=torch.float32, device=self.encoder.device)
        self.register_buffer('t_values', t_values)


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @property
    def device(self) -> torch.device:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        evaluate_at_t: bool=True,
        t: int = 1,
        max_length: int = 20,
        min_length: int = 0,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bad_words_ids: Optional[list] = None,
        num_return_sequences: int = 1,
        attention_mask: Optional[torch.BoolTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: bool = True,
        **model_kwargs
    ):
        """
        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values of input images.
            max_length (`int`, *optional*, defaults to 20):
                Maximum length of the sequence to be generated.
            min_length (`int`, *optional*, defaults to 0):
                Minimum length of the sequence to be generated.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
            repetition_penalty (`float`, *optional*, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.
            early_stopping (`bool`, *optional*, defaults to False):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            pad_token_id (`int`, *optional*):
                The id of the padding token.
            bos_token_id (`int`, *optional*):
                The id of the beginning-of-sequence token.
            eos_token_id (`int`, *optional*):
                The id of the end-of-sequence token.
            bad_words_ids (`List[List[int]]`, *optional*):
                List of token ids that are not allowed to be generated.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            decoder_start_token_id (`int`, *optional*):
                If provided, the model will use this token as the starting token for decoder.
            use_cache (`bool`, *optional*, defaults to True):
                Whether or not to use cache for faster generation.

        Returns:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`:
                The generated sequences. Each sequence is an array of token ids.
        """
        # Get encoder outputs
        batch_size = pixel_values.shape[0]

        if evaluate_at_t == True:
            hidden_state, outputs_hidden_states = self.evaluate_at_t(pixel_values=pixel_values, t_evaluate=t)
        else:
            hidden_state, outputs_hidden_states = self.encode(pixel_values=pixel_values)

        encoder_hidden_states = hidden_state
        encoder_attention_mask = None

        # Prepare for decoder generation
        if decoder_start_token_id is None:
            decoder_start_token_id = self.decoder.config.decoder_start_token_id or self.decoder.config.bos_token_id

        if pad_token_id is None:
            pad_token_id = self.decoder.config.pad_token_id

        if eos_token_id is None:
            eos_token_id = self.decoder.config.eos_token_id

        # Initialize the decoder input with the start token
        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=pixel_values.device
        )


        expanded_encoder_hidden_states = encoder_hidden_states

        # If decoder doesn't have generate, implement basic generation logic
        # This is a simple greedy search as fallback
        cur_len = input_ids.shape[1]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=pixel_values.device)

        while cur_len < max_length:
            model_inputs = self.decoder.prepare_inputs_for_generation(
                input_ids,
                encoder_hidden_states=expanded_encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                **model_kwargs
            )

            outputs = self.decoder(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in input_ids[i]:
                        if previous_token.item() >= 0:
                            next_token_logits[i, previous_token.item()] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = -float("Inf")

            # Sample next token
            if temperature == 0.0:  # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Update input_ids and model_kwargs
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            # Update generated length
            cur_len = cur_len + 1

            # Check if EOS token was generated
            if eos_token_id is not None:
                eos_in_sents = (next_token == eos_token_id)
                # if sentence is unfinished and the token to add is eos, finish it
                unfinished_sequences = unfinished_sequences.mul((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sequences.max() == 0:
                    break

        return input_ids

    def encode(self, pixel_values: torch.FloatTensor):

        t = self.t_values * self.dt

        temporal_emb = self.time_embedding(t)

        hidden_state = self.encoder.embeddings(pixel_values)
        outputs_hidden_states = [hidden_state.clone()]

        # Afegir la part temporal del ODE transformer.
        for i in range(self.steps):
            encoder_output = self.encoder.encoder(hidden_state)
            euler_step = encoder_output.last_hidden_state
            temporal_emb_batch = temporal_emb[i]
            euler_step = euler_step + temporal_emb_batch
            hidden_state = hidden_state + (euler_step * self.dt)

            outputs_hidden_states.append(hidden_state.clone())

        hidden_state = self.encoder.layernorm(hidden_state)
        hidden_state = self.encoder.pooler(hidden_state).tanh()

        return hidden_state, outputs_hidden_states


    def evaluate_at_t(self,
                    pixel_values: Optional[torch.FloatTensor] = None,
                    t_evaluate: int = 1):

        t = self.t_values * self.dt


        temporal_emb = self.time_embedding(t)

        hidden_state = self.encoder.embeddings(pixel_values)
        outputs_hidden_states = [hidden_state.clone()]

        # Afegir la part temporal del ODE transformer.
        for i in range(self.steps)[:t_evaluate]:
            encoder_output = self.encoder.encoder(hidden_state)
            euler_step = encoder_output.last_hidden_state
            temporal_emb_batch = temporal_emb[i]
            euler_step = euler_step + temporal_emb_batch
            hidden_state = hidden_state + (euler_step * self.dt)

            outputs_hidden_states.append(hidden_state.clone())

        hidden_state = self.encoder.layernorm(hidden_state)
        hidden_state = self.encoder.pooler(hidden_state).tanh()

        return hidden_state, outputs_hidden_states

    def forward(self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):


        num_items_batch = pixel_values.shape[0]

        last_hidden_state, outputs_hidden_states  = self.encode(pixel_values=pixel_values)

        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id)

        decoder_input_ids = labels.clone()
        decoder_input_ids[:, 0] = self.decoder.config.decoder_start_token_id
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       encoder_hidden_states=last_hidden_state,
                                       encoder_attention_mask=encoder_attention_mask,
                                       attention_mask=decoder_attention_mask,
                                       labels=labels,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states)

        loss = ForCausalLMLoss(logits=decoder_outputs.logits,
                                labels=labels,
                                vocab_size=self.decoder.config.vocab_size,
                                num_items_in_batch=num_items_batch,
                                ignore_index=self.decoder.config.pad_token_id)


        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=last_hidden_state,
            encoder_hidden_states=outputs_hidden_states)


if __name__ == "__main__":
    from transformers import ViTModel, ViTConfig
    from transformers import VisionEncoderDecoderModel

    example_image = torch.randn(1, 3, 384, 384)
    example_text = torch.randint(0, 1000, (1, 20))

    model = VisionEncoderDecoderModel.from_pretrained("checkpoints/TrOCR_Esposalles.pt")

    vit_config = ViTConfig(
        num_hidden_layers=1,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=384,
        patch_size=16,
        num_channels=3,
    )

    # Encoder inicialitzat des de zero
    vit_encoder = ViTModel(vit_config)

    final_model = REncoderDecoderModel(
        encoder=vit_encoder,
        decoder=model.decoder,
        steps=12,
        loss_function=model.loss_function
    )

    outputs = final_model(pixel_values=example_image, decoder_input_ids=example_text, labels=example_text)
    import pdb; pdb.set_trace()

"""ViT-GPT2 image-captioning model with flexible regular fine-tuning.

Encoder fine-tuning modes (--encoder-mode):
  frozen    – all encoder parameters frozen
  last_n    – last N ViT transformer blocks + final LayerNorm trainable
  full      – all encoder parameters trainable

Decoder fine-tuning modes (--decoder-mode):
  frozen      – all decoder parameters frozen
  cross_attn  – only cross-attention weights (crossattention + ln_cross_attn)
                trainable in every decoder block; self-attention and MLP frozen.
                This is the recommended default for decoder fine-tuning: it lets
                the model learn to attend to visual features without disturbing
                the language model's generation behaviour (which would cause
                repetition loops).
  last_n      – last N GPT-2 blocks fully trainable + ln_f + lm_head
  full        – all decoder parameters trainable
"""
import torch
from torch import nn
from transformers import VisionEncoderDecoderModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENCODER_MODES = ('frozen', 'last_n', 'full')
DECODER_MODES = ('frozen', 'cross_attn', 'last_n', 'full')


class VisionEncoderDecoderCaptioningModel(nn.Module):

    def __init__(
        self,
        pretrained_model_name: str = 'nlpconnect/vit-gpt2-image-captioning',
        encoder_mode: str = 'frozen',
        decoder_mode: str = 'cross_attn',
        encoder_ft_layers: int = 4,
        decoder_ft_layers: int = 4,
    ):
        super().__init__()
        assert encoder_mode in ENCODER_MODES, f'encoder_mode must be one of {ENCODER_MODES}'
        assert decoder_mode in DECODER_MODES, f'decoder_mode must be one of {DECODER_MODES}'

        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name)

        # --- Encoder ---
        self._freeze_module(self.model.encoder)
        if encoder_mode == 'full':
            self._unfreeze_module(self.model.encoder)
        elif encoder_mode == 'last_n':
            self._unfreeze_last_n_vit_layers(self.model.encoder, encoder_ft_layers)

        # --- Decoder ---
        self._freeze_module(self.model.decoder)
        if decoder_mode == 'full':
            self._unfreeze_module(self.model.decoder)
        elif decoder_mode == 'last_n':
            self._unfreeze_last_n_gpt2_layers(self.model.decoder, decoder_ft_layers)
        elif decoder_mode == 'cross_attn':
            self._unfreeze_decoder_cross_attn(self.model.decoder)

    def configure_generation(self, tokenizer, max_length: int = 64, num_beams: int = 4,
                             repetition_penalty: float = 1.0):
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = (tokenizer.cls_token_id if tokenizer.cls_token_id is not None
                            else tokenizer.eos_token_id)

        self.model.config.pad_token_id             = pad_token_id
        self.model.config.decoder_start_token_id   = bos_token_id
        self.model.config.eos_token_id             = tokenizer.eos_token_id

        self.model.generation_config.pad_token_id           = pad_token_id
        self.model.generation_config.decoder_start_token_id = bos_token_id
        self.model.generation_config.eos_token_id           = tokenizer.eos_token_id
        self.model.generation_config.max_length             = max_length
        self.model.generation_config.num_beams              = num_beams
        self.model.generation_config.repetition_penalty     = repetition_penalty

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def generate(self, pixel_values: torch.Tensor, **kwargs):
        return self.model.generate(pixel_values=pixel_values, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def _unfreeze_module(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    @classmethod
    def _unfreeze_last_n_vit_layers(cls, encoder: nn.Module, n: int):
        """Unfreeze the last N ViTLayer blocks and the final LayerNorm."""
        layers = encoder.encoder.layer
        n = min(n, len(layers))
        for layer in layers[len(layers) - n:]:
            cls._unfreeze_module(layer)
        if hasattr(encoder, 'layernorm'):
            cls._unfreeze_module(encoder.layernorm)

    @classmethod
    def _unfreeze_last_n_gpt2_layers(cls, decoder: nn.Module, n: int):
        """Unfreeze the last N GPT-2 blocks, ln_f, and lm_head."""
        blocks = decoder.transformer.h
        n = min(n, len(blocks))
        for block in blocks[len(blocks) - n:]:
            cls._unfreeze_module(block)
        if hasattr(decoder.transformer, 'ln_f'):
            cls._unfreeze_module(decoder.transformer.ln_f)
        if hasattr(decoder, 'lm_head'):
            cls._unfreeze_module(decoder.lm_head)

    @classmethod
    def _unfreeze_decoder_cross_attn(cls, decoder: nn.Module):
        """Unfreeze only the cross-attention modules in every GPT-2 block.

        Each block exposes `crossattention` (GPT2Attention) and
        `ln_cross_attn` (LayerNorm) when add_cross_attention=True.
        These are the only weights the decoder needs to update in order to
        learn to attend to visual encoder features; keeping self-attention
        and the MLP frozen preserves the pre-trained language generation
        behaviour and prevents repetition collapse.
        """
        found = False
        for block in decoder.transformer.h:
            if hasattr(block, 'crossattention'):
                cls._unfreeze_module(block.crossattention)
                found = True
            if hasattr(block, 'ln_cross_attn'):
                cls._unfreeze_module(block.ln_cross_attn)
        if not found:
            raise RuntimeError(
                'No crossattention modules found in decoder. '
                'Ensure the model was loaded with add_cross_attention=True '
                '(VisionEncoderDecoderModel does this automatically).'
            )

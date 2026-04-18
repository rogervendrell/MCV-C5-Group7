"""Task 2 models: frozen ViT encoder + LoRA-tuned Qwen3 decoder.

Two public classes
------------------
FrozenViTEncoder
    Any HuggingFace ViT-family model (or the encoder extracted from a
    VisionEncoderDecoderModel) used as a frozen image feature extractor.
    Swapping the backbone is as simple as changing ``vit_model_name``.

ViTCausalLMCaptioningModel
    Frozen ViT → linear projection → LoRA-fine-tuned causal LM decoder
    (Qwen3-1.7B, Qwen3-4B, or any AutoModelForCausalLM-compatible model).
    Visual patch embeddings are projected into the LM's embedding space and
    prepended as soft visual tokens before the text tokens.  Only the
    projection layer and the LoRA adapters are trainable.
"""

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Frozen ViT encoder
# ---------------------------------------------------------------------------

class FrozenViTEncoder(nn.Module):
    """Load any HuggingFace ViT model as a frozen image encoder.

    Supports two source formats automatically:
    - Standard ViT models (``model_type != 'vision-encoder-decoder'``),
      e.g. ``'google/vit-base-patch16-224'``.
    - The encoder extracted from a VisionEncoderDecoderModel,
      e.g. ``'nlpconnect/vit-gpt2-image-captioning'``.

    Fine-tuned Task1 checkpoint weights can be loaded via
    ``load_task1_encoder_weights(checkpoint_path)``.
    """

    def __init__(self, model_name: str):
        super().__init__()

        from transformers import AutoConfig, AutoModel, VisionEncoderDecoderModel

        cfg = AutoConfig.from_pretrained(model_name)
        if cfg.model_type == 'vision-encoder-decoder':
            # Extract only the ViT encoder from the full model.
            full = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.vit = full.encoder
            del full
        else:
            self.vit = AutoModel.from_pretrained(model_name)

        for param in self.vit.parameters():
            param.requires_grad = False

        self.hidden_size: int = self.vit.config.hidden_size

    # ------------------------------------------------------------------
    # Optional: load fine-tuned encoder weights from a Task1 checkpoint
    # ------------------------------------------------------------------

    def load_task1_encoder_weights(self, checkpoint_path: str):
        """Load ViT encoder weights from a Task1 best_model.pt checkpoint.

        Task1 saves the full VisionEncoderDecoderCaptioningModel state dict.
        Keys belonging to the encoder start with ``'model.encoder.'``.
        """
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        enc_prefix = 'model.encoder.'
        encoder_state = {
            k[len(enc_prefix):]: v
            for k, v in state.items()
            if k.startswith(enc_prefix)
        }

        # If LoRA was used, PEFT adds 'base_model.model.' in front of each key.
        missing, _ = self.vit.load_state_dict(encoder_state, strict=False)
        if missing:
            peft_prefix = 'base_model.model.'
            encoder_state = {
                k[len(peft_prefix):] if k.startswith(peft_prefix) else k: v
                for k, v in encoder_state.items()
            }
            self.vit.load_state_dict(encoder_state, strict=False)

        print(f'[FrozenViTEncoder] Loaded encoder weights from {checkpoint_path}', flush=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings including the CLS token.

        Shape: ``[B, num_patches + 1, hidden_size]``
        """
        return self.vit(pixel_values=pixel_values).last_hidden_state


# ---------------------------------------------------------------------------
# ViT + causal LM captioning model (LoRA fine-tuning)
# ---------------------------------------------------------------------------

# Standard LoRA target modules for Qwen3 / Llama-family models.
# These cover the attention projections in each transformer block.
_CAUSAL_LM_LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


class ViTCausalLMCaptioningModel(nn.Module):
    """Frozen ViT encoder + LoRA-tuned causal LM decoder for image captioning.

    Architecture
    ------------
    1. ``pixel_values``  →  FrozenViTEncoder  →  ``[B, V, vit_dim]``
    2.                   →  ``visual_projection`` (Linear, *trainable*)
                         →  ``[B, V, lm_dim]``
    3. Text tokens       →  LM embedding table
                         →  ``[B, T, lm_dim]``
    4. Concatenate visual + text embeddings, pass through LoRA-LM.

    Only the projection layer and LoRA adapters are trainable.

    Parameters
    ----------
    vit_model_name:
        Any HuggingFace ViT hub name (or VisionEncoderDecoder name to extract
        the encoder from).
    lm_model_name:
        HuggingFace name of the causal LM to use as decoder,
        e.g. ``'Qwen/Qwen3-1.7B'`` or ``'Qwen/Qwen3-4B'``.
    """

    def __init__(
        self,
        vit_model_name:       str,
        lm_model_name:        str,
        lora_r:               int   = 8,
        lora_alpha:           int   = 16,
        lora_dropout:         float = 0.1,
        lora_target_modules:  list[str] | None = None,
    ):
        super().__init__()

        # --- Frozen ViT encoder ------------------------------------------
        self.encoder = FrozenViTEncoder(vit_model_name)

        # --- Causal LM decoder -------------------------------------------
        from transformers import AutoModelForCausalLM
        self.decoder = AutoModelForCausalLM.from_pretrained(
            lm_model_name,
            torch_dtype='auto',
        )
        lm_hidden_size = self.decoder.config.hidden_size

        # --- Visual projection (trainable) --------------------------------
        self.visual_projection = nn.Linear(
            self.encoder.hidden_size,
            lm_hidden_size,
            bias=False,
        )

        # --- Apply LoRA to the decoder ------------------------------------
        target_modules = lora_target_modules or _CAUSAL_LM_LORA_TARGET_MODULES
        self.decoder = _wrap_with_lora(
            self.decoder,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

        # Gradient checkpointing: recomputes activations during backward
        # instead of storing them, trading ~30% compute for large memory savings.
        self.decoder.enable_input_require_grads()
        self.decoder.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor | None = None,
    ):
        """Compute cross-entropy loss over the text tokens.

        Labels should be the token IDs with padding replaced by -100;
        visual prefix positions are masked automatically.
        """
        # Visual features → LM embedding space
        with torch.no_grad():
            visual_feats = self.encoder(pixel_values)           # [B, V, vit_dim]
        visual_embeds = self.visual_projection(visual_feats)    # [B, V, lm_dim]
        B, V, _ = visual_embeds.shape

        # Text embeddings
        text_embeds = self.decoder.get_input_embeddings()(input_ids)  # [B, T, lm_dim]

        # Concatenate: visual prefix first, then text.
        # Cast to the decoder's dtype (e.g. bfloat16) in case the projection
        # layer is still float32 (matters outside AMP autocast contexts).
        decoder_dtype = text_embeds.dtype
        inputs_embeds = torch.cat(
            [visual_embeds.to(decoder_dtype), text_embeds], dim=1
        )  # [B, V+T, lm_dim]

        # Extend attention mask to cover visual tokens
        visual_attn = torch.ones(B, V, device=pixel_values.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([visual_attn, attention_mask], dim=1)

        # Prepend -100 labels for all visual positions (never predicted)
        if labels is not None:
            visual_labels = torch.full(
                (B, V), -100, device=labels.device, dtype=labels.dtype
            )
            full_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            full_labels = None

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    # ------------------------------------------------------------------
    # Generate (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        pixel_values:      torch.Tensor,
        tokenizer,
        generation_kwargs: dict | None = None,
    ) -> list[str]:
        """Auto-regressively generate captions conditioned on visual tokens.

        Returns a list of decoded strings (one per image in the batch).
        """
        generation_kwargs = generation_kwargs or {}

        visual_feats  = self.encoder(pixel_values)
        visual_embeds = self.visual_projection(visual_feats)  # [B, V, lm_dim]
        B, V, _       = visual_embeds.shape

        # Seed the decoder with the BOS token so it knows to start generating.
        bos_id     = tokenizer.bos_token_id or tokenizer.eos_token_id
        bos_ids    = torch.full((B, 1), bos_id, device=pixel_values.device, dtype=torch.long)
        bos_embeds = self.decoder.get_input_embeddings()(bos_ids)   # [B, 1, lm_dim]

        # Ensure visual embeddings match the decoder's dtype.
        decoder_dtype = bos_embeds.dtype
        inputs_embeds = torch.cat(
            [visual_embeds.to(decoder_dtype), bos_embeds], dim=1
        )  # [B, V+1, lm_dim]
        attention_mask = torch.ones(B, V + 1, device=pixel_values.device, dtype=torch.long)

        generated_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# Backward-compatibility alias
ViTLlamaCaptioningModel = ViTCausalLMCaptioningModel


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def _wrap_with_lora(
    module:         nn.Module,
    lora_r:         int,
    lora_alpha:     int,
    lora_dropout:   float,
    target_modules: list[str],
) -> nn.Module:
    try:
        import peft
    except ImportError as exc:
        raise ImportError('Install peft: pip install peft') from exc

    config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias='none',
    )
    return peft.get_peft_model(module, config)

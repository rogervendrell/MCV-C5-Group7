"""Image-captioning model: ResNet encoder + Transformer decoder."""
import math
import torch
from torch import nn
from transformers import ResNetModel

from vocabulary import TEXT_MAX_LEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NHEAD = 8  # must divide hidden_size; 1024/8=128 ✓


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (batch-first)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = TEXT_MAX_LEN + 2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaselineModel(nn.Module):
    """ResNet image encoder + multi-layer Transformer decoder.

    The encoder pooler output is used as a single-token memory sequence for
    cross-attention inside the Transformer decoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        sos_idx: int,
        freeze_encoder: bool = False,
        num_decoder_layers: int = 2,
        encoder_name: str = 'microsoft/resnet-18',
        unfreeze_last_stage: bool = False,
        hidden_size: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size

        # Encoder
        self.encoder = ResNetModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_stage:
                for param in self.encoder.encoder.stages[-1].parameters():
                    param.requires_grad = True

        # Project encoder output (always 512) into hidden_size
        self.img_proj = nn.Linear(512, hidden_size)

        # Decoder
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=NHEAD,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        self.proj = nn.Linear(hidden_size, vocab_size)

    # Forward

    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """Return image memory tensor of shape (B, 1, hidden_size)."""
        feat = self.encoder(img)
        img_feat = feat.pooler_output.view(img.size(0), 512)  # (B, 512)
        return self.img_proj(img_feat).unsqueeze(1)            # (B, 1, hidden_size)

    def forward(
        self, img: torch.Tensor, ground_truth: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            img:          (B, 3, H, W)
            ground_truth: (B, T) token indices including leading SOS.
                          When provided, teacher-forcing is used (training).
                          When None, autoregressive decoding is used (inference).
        Returns:
            logits: (B, vocab_size, T-1)
        """
        memory = self._encode_image(img)  # (B, 1, hidden_size)

        if ground_truth is not None:
            return self._teacher_forcing(memory, ground_truth)
        else:
            return self._autoregressive(memory, img.device)

    def _teacher_forcing(
        self, memory: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Full parallel forward pass with a causal mask (fast, used during training)."""
        tgt = ground_truth[:, :-1]               # (B, T-1) — drop EOS/last pad
        T = tgt.size(1)

        tgt_embeds = self.pos_enc(self.embed(tgt))  # (B, T-1, hidden_size)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=memory.device
        )
        tgt_key_padding_mask = (tgt == 0)  # True where PAD token

        out = self.transformer_decoder(
            tgt_embeds,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, T-1, hidden_size)

        logits = self.proj(out)  # (B, T-1, vocab_size)
        return logits.permute(0, 2, 1)  # (B, vocab_size, T-1)

    def _autoregressive(
        self, memory: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Token-by-token generation with temperature sampling (used during eval)."""
        batch_size = memory.size(0)
        curr_tokens = torch.full(
            (batch_size, 1), self.sos_idx, device=device, dtype=torch.long
        )
        all_logits = []

        for t in range(TEXT_MAX_LEN - 1):
            embeds = self.pos_enc(self.embed(curr_tokens))   # (B, t+1, hidden_size)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                t + 1, device=device
            )
            out = self.transformer_decoder(embeds, memory, tgt_mask=tgt_mask)
            logits = self.proj(out[:, -1, :])                # (B, vocab_size)
            all_logits.append(logits.unsqueeze(2))           # (B, vocab_size, 1)

            temp = 0.5
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)         # (B, 1)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

        return torch.cat(all_logits, dim=2)  # (B, vocab_size, T-1)

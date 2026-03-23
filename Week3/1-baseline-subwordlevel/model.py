"""Baseline image-captioning model: ResNet-18 encoder + GRU decoder."""
import torch
from torch import nn
from transformers import ResNetModel

import tokenizer as tok

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaselineModel(nn.Module):
    """ResNet-18 image encoder + multi-layer GRU subword-level decoder.

    The encoder produces a 512-d feature vector that seeds all GRU hidden
    states. At each decoding step the previous output is fed back as input
    (open-loop teacher-free, matching the original notebook).
    """

    def __init__(self, freeze_encoder: bool = False, num_decoder_layers: int = 1):
        super().__init__()
        self.encoder = ResNetModel.from_pretrained('microsoft/resnet-18')
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.num_decoder_layers = num_decoder_layers
        self.gru   = nn.GRU(512, 512, num_layers=num_decoder_layers)
        self.proj  = nn.Linear(512, tok.VOCAB_SIZE)
        self.embed = nn.Embedding(tok.VOCAB_SIZE, 512)

    def forward(self, img: torch.Tensor, ground_truth: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            img:          (B, 3, 224, 224)
            ground_truth: (B, TEXT_MAX_LEN) token indices, optional.
                          When provided, teacher forcing is used (single GRU call, no loop).
                          When None, autoregressive decoding is used.
        Returns:
            logits: (B, VOCAB_SIZE, TEXT_MAX_LEN)
        """
        batch_size = img.shape[0]

        # Encode image → initial hidden state (repeated for each GRU layer)
        feat = self.encoder(img)
        hidden = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # (1, B, 512)
        hidden = hidden.repeat(self.num_decoder_layers, 1, 1)              # (num_layers, B, 512)

        sos = torch.tensor(tok.SOS_ID, device=img.device)
        sos_embed = self.embed(sos).expand(batch_size, -1)  # (B, 512)

        if ground_truth is not None: # and torch.rand(1).item() < 0.5:
            # Teacher forcing: single GRU call over embedded ground-truth tokens.
            # Input:  [embed(gt[0]), ..., embed(gt[T-2])]  — gt already starts with <SOS>
            # Output: T-1 GRU hidden vectors for positions 1..T-1.
            # Position 0 is always the bare <SOS> embedding (no GRU step), matching
            # the autoregressive path where outputs[0] = inp = SOS before the loop.
            gt_embed  = self.embed(ground_truth[:, :-1])          # (B, T-1, 512)
            gru_out, _ = self.gru(gt_embed.permute(1, 0, 2), hidden)  # (T-1, B, 512)
            gru_out   = gru_out.permute(1, 0, 2)                  # (B, T-1, 512)
            res = torch.cat([sos_embed.unsqueeze(1), gru_out], dim=1)  # (B, T, 512)
        else:
            # Autoregressive: one token at a time, carrying the hidden state forward.
            token = sos_embed.unsqueeze(0)  # (1, B, 512)
            outputs = [token]
            for _ in range(tok.TEXT_MAX_LEN - 1):
                out, hidden = self.gru(token, hidden)  # out: (1, B, 512)
                outputs.append(out)
                token = out  # feed current output as next input
            res = torch.cat(outputs, dim=0).permute(1, 0, 2)  # (B, T, 512)

        res = self.proj(res)           # (B, T, VOCAB_SIZE)
        res = res.permute(0, 2, 1)     # (B, VOCAB_SIZE, T)
        return res
